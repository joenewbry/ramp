#!/usr/bin/env python3
"""Ramp — Acceleration-focused dashboard for API usage and data collection.

Tracks position, velocity, and acceleration across:
  - Anthropic (Claude) API usage & spend
  - OpenAI API usage & spend
  - xAI (Grok) API usage & spend
  - Screen-Self-Driving GCS training pipeline

Runs as a FastAPI server on the Jetson, exposed via Cloudflare tunnel
at ramp.digitalsurfacelabs.com.

Usage:
    python ramp.py                   # default port 8081
    python ramp.py --port 8097
"""

import argparse
import json
import os
import sqlite3
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_env(path: str = None):
    """Load key=value pairs from a .env file."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config.env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if v and k not in os.environ:
                    os.environ[k] = v

_load_env()

ANTHROPIC_ADMIN_KEY = os.environ.get("ANTHROPIC_ADMIN_KEY", "")
OPENAI_ADMIN_KEY = os.environ.get("OPENAI_ADMIN_KEY", "")
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "ai-therapist-ssd-benchmark")
GCS_PROJECT = os.environ.get("GCS_PROJECT", "ai-therapist-3e55e")

DEFAULT_PORT = 8097
STATS_DB_PATH = Path("/ssd/ramp/stats.db")

KNOWN_DEVICES = [
    {"id": "jetson-orin-nano",      "label": "Jetson Orin Nano",      "color": "#7c8aff"},
    {"id": "nvidia-spark",          "label": "NVIDIA Spark",           "color": "#60a5fa"},
    {"id": "rtx4080-workstation",   "label": "RTX 4080 Workstation",   "color": "#a78bfa"},
]

app = FastAPI(title="Ramp Dashboard", version="0.1.0")

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_cache = {}
_cache_lock = threading.Lock()

def cached(key: str, fn, ttl: int = 60):
    with _cache_lock:
        if key in _cache:
            val, ts = _cache[key]
            if time.time() - ts < ttl:
                return val
    result = fn()
    with _cache_lock:
        _cache[key] = (result, time.time())
    return result

# ---------------------------------------------------------------------------
# Velocity / Acceleration math
# ---------------------------------------------------------------------------

def compute_metrics(daily_values: list[dict]) -> dict:
    """Given a list of {date, value} dicts sorted by date, compute metrics.

    Returns:
        position: total cumulative
        velocity: average daily value over last 7 days
        acceleration: change in daily value averaged over last 5 days
        daily: list with per-day velocity and acceleration
    """
    if not daily_values:
        return {"position": 0, "velocity": 0, "acceleration": 0, "daily": []}

    result = []
    cumulative = 0
    for i, entry in enumerate(daily_values):
        val = entry["value"]
        cumulative += val
        # velocity = difference from previous day
        vel = val - daily_values[i - 1]["value"] if i > 0 else 0
        # acceleration = change in velocity from previous day
        if i >= 2:
            prev_vel = daily_values[i - 1]["value"] - daily_values[i - 2]["value"]
            acc = vel - prev_vel
        else:
            acc = 0
        result.append({
            "date": entry["date"],
            "value": val,
            "cumulative": cumulative,
            "velocity": vel,
            "acceleration": acc,
        })

    # Summary stats
    position = cumulative
    last_7 = [r["value"] for r in result[-7:]]
    velocity = sum(last_7) / len(last_7) if last_7 else 0
    # Acceleration: change in daily average over last 5 days
    last_5 = result[-5:]
    if len(last_5) >= 2:
        accels = [last_5[i]["velocity"] - last_5[i - 1]["velocity"]
                  for i in range(1, len(last_5))]
        acceleration = sum(accels) / len(accels) if accels else 0
    else:
        acceleration = 0

    return {
        "position": position,
        "velocity": round(velocity, 1),
        "acceleration": round(acceleration, 2),
        "daily": result,
    }

# ---------------------------------------------------------------------------
# Anthropic API
# ---------------------------------------------------------------------------

def fetch_anthropic_usage() -> dict:
    """Fetch daily usage from Anthropic Admin API (last 14 days)."""
    if not ANTHROPIC_ADMIN_KEY:
        return {"configured": False, "error": "No admin key", "daily_tokens": [], "daily_cost": []}

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=14)

    try:
        # Token usage
        with httpx.Client(timeout=15) as client:
            resp = client.get(
                "https://api.anthropic.com/v1/organizations/usage_report/messages",
                params={
                    "starting_at": start.strftime("%Y-%m-%dT00:00:00Z"),
                    "ending_at": end.strftime("%Y-%m-%dT00:00:00Z"),
                    "bucket_width": "1d",
                    "limit": 14,
                },
                headers={
                    "anthropic-version": "2023-06-01",
                    "x-api-key": ANTHROPIC_ADMIN_KEY,
                },
            )
            resp.raise_for_status()
            usage_data = resp.json()

        # Cost data
        with httpx.Client(timeout=15) as client:
            cost_resp = client.get(
                "https://api.anthropic.com/v1/organizations/cost_report",
                params={
                    "starting_at": start.strftime("%Y-%m-%dT00:00:00Z"),
                    "ending_at": end.strftime("%Y-%m-%dT00:00:00Z"),
                    "bucket_width": "1d",
                    "limit": 14,
                },
                headers={
                    "anthropic-version": "2023-06-01",
                    "x-api-key": ANTHROPIC_ADMIN_KEY,
                },
            )
            cost_resp.raise_for_status()
            cost_data = cost_resp.json()

        # Parse token usage
        daily_tokens = []
        for bucket in usage_data.get("data", []):
            date_str = bucket.get("bucket_start_time", bucket.get("start_time", ""))[:10]
            total_input = 0
            total_output = 0
            for r in bucket.get("results", []):
                total_input += r.get("input_tokens", 0) + r.get("cache_creation_input_tokens", 0)
                total_output += r.get("output_tokens", 0)
            daily_tokens.append({
                "date": date_str,
                "value": total_input + total_output,
                "input": total_input,
                "output": total_output,
            })

        # Parse cost
        daily_cost = []
        for bucket in cost_data.get("data", []):
            date_str = bucket.get("bucket_start_time", bucket.get("start_time", ""))[:10]
            total_cost = 0
            for r in bucket.get("results", []):
                total_cost += float(r.get("cost_cents", 0)) / 100.0
            daily_cost.append({
                "date": date_str,
                "value": round(total_cost, 4),
            })

        return {
            "configured": True,
            "daily_tokens": daily_tokens,
            "daily_cost": daily_cost,
            "token_metrics": compute_metrics(daily_tokens),
            "cost_metrics": compute_metrics(daily_cost),
        }
    except Exception as e:
        return {"configured": True, "error": str(e), "daily_tokens": [], "daily_cost": []}

# ---------------------------------------------------------------------------
# OpenAI API
# ---------------------------------------------------------------------------

def fetch_openai_usage() -> dict:
    """Fetch daily usage from OpenAI Admin API (last 14 days)."""
    if not OPENAI_ADMIN_KEY:
        return {"configured": False, "error": "No admin key", "daily_tokens": [], "daily_cost": []}

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=14)
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())

    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(
                "https://api.openai.com/v1/organization/usage/completions",
                params={
                    "start_time": start_ts,
                    "end_time": end_ts,
                    "bucket_width": "1d",
                    "limit": 14,
                },
                headers={
                    "Authorization": f"Bearer {OPENAI_ADMIN_KEY}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        daily_tokens = []
        for bucket in data.get("data", []):
            ts = bucket.get("start_time", 0)
            date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d") if ts else ""
            total_input = 0
            total_output = 0
            for r in bucket.get("results", []):
                total_input += r.get("input_tokens", 0)
                total_output += r.get("output_tokens", 0)
            daily_tokens.append({
                "date": date_str,
                "value": total_input + total_output,
                "input": total_input,
                "output": total_output,
            })

        # OpenAI doesn't have a separate cost endpoint in the same way;
        # we estimate from tokens (rough pricing)
        daily_cost = []
        for d in daily_tokens:
            # Very rough estimate — will be improved
            est_cost = (d.get("input", 0) * 2.5 + d.get("output", 0) * 10) / 1_000_000
            daily_cost.append({"date": d["date"], "value": round(est_cost, 4)})

        return {
            "configured": True,
            "daily_tokens": daily_tokens,
            "daily_cost": daily_cost,
            "token_metrics": compute_metrics(daily_tokens),
            "cost_metrics": compute_metrics(daily_cost),
        }
    except Exception as e:
        return {"configured": True, "error": str(e), "daily_tokens": [], "daily_cost": []}

# ---------------------------------------------------------------------------
# xAI / Grok API
# ---------------------------------------------------------------------------

def fetch_xai_usage() -> dict:
    """Fetch xAI usage. No official usage API yet — placeholder."""
    if not XAI_API_KEY:
        return {"configured": False, "error": "No API key", "daily_tokens": [], "daily_cost": []}

    # xAI doesn't have a public usage API yet.
    # When they add one, implement here.
    return {
        "configured": True,
        "error": "xAI usage API not yet available. Track via console.",
        "daily_tokens": [],
        "daily_cost": [],
    }

# ---------------------------------------------------------------------------
# GCS Training Pipeline
# ---------------------------------------------------------------------------

def _run_gsutil(args: list[str], timeout: int = 30) -> str | None:
    try:
        result = subprocess.run(
            ["gsutil"] + args,
            capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        return result.stdout if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

def _run_gcloud(args: list[str], timeout: int = 15) -> str | None:
    try:
        result = subprocess.run(
            ["gcloud"] + args,
            capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        return result.stdout if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

def fetch_gcs_pipeline() -> dict:
    """Fetch GCS training pipeline data."""
    def _fetch():
        timeline = {}

        # Count data/sessions by date prefix
        out = _run_gsutil(["ls", f"gs://{GCS_BUCKET}/data/sessions/"], timeout=30)
        if out:
            for line in out.strip().split("\n"):
                name = line.rstrip("/").split("/")[-1]
                if name and len(name) >= 8:
                    day = f"{name[:4]}-{name[4:6]}-{name[6:8]}"
                    timeline[day] = timeline.get(day, 0) + 1

        # Count generated sessions
        gen_out = _run_gsutil(["ls", f"gs://{GCS_BUCKET}/generated/"], timeout=10)
        if gen_out:
            for gen_line in gen_out.strip().split("\n"):
                gen_id = gen_line.rstrip("/").split("/")[-1]
                if not gen_id:
                    continue
                sess_out = _run_gsutil(
                    ["ls", f"gs://{GCS_BUCKET}/generated/{gen_id}/sessions/"], timeout=30
                )
                if sess_out:
                    for line in sess_out.strip().split("\n"):
                        name = line.rstrip("/").split("/")[-1]
                        if name and len(name) >= 8:
                            day = f"{name[:4]}-{name[4:6]}-{name[6:8]}"
                            timeline[day] = timeline.get(day, 0) + 1

        daily = [{"date": d, "value": timeline[d]} for d in sorted(timeline.keys())]

        # Active VMs
        vms = []
        vm_out = _run_gcloud([
            "compute", "instances", "list",
            "--project", GCS_PROJECT,
            "--format", "json",
        ])
        if vm_out:
            try:
                for vm in json.loads(vm_out):
                    vms.append({
                        "name": vm["name"],
                        "status": vm["status"],
                        "type": vm["machineType"].split("/")[-1],
                        "zone": vm["zone"].split("/")[-1],
                    })
            except (json.JSONDecodeError, KeyError):
                pass

        # Training runs
        training = []
        train_out = _run_gsutil(["ls", f"gs://{GCS_BUCKET}/training/"], timeout=10)
        if train_out:
            for line in train_out.strip().split("\n"):
                train_id = line.rstrip("/").split("/")[-1]
                if not train_id:
                    continue
                status_json = _run_gsutil(
                    ["cat", f"gs://{GCS_BUCKET}/training/{train_id}/STATUS.json"]
                )
                if status_json:
                    try:
                        s = json.loads(status_json)
                        training.append({
                            "id": train_id,
                            "status": s.get("status", "unknown"),
                            "models": s.get("models", "?"),
                            "gpu": s.get("gpu", "?"),
                        })
                    except json.JSONDecodeError:
                        pass

        return {
            "daily_sessions": daily,
            "metrics": compute_metrics(daily),
            "active_vms": [v for v in vms if v["status"] == "RUNNING"],
            "all_vms": vms,
            "training_runs": training,
        }

    return cached("gcs_pipeline", _fetch, ttl=60)

# ---------------------------------------------------------------------------
# Compute fleet stats (SQLite)
# ---------------------------------------------------------------------------

def fetch_compute_stats() -> dict:
    """Read latest stats + 24h history from SQLite for all known devices."""
    def _fetch():
        devices = []
        if not STATS_DB_PATH.exists():
            for dev in KNOWN_DEVICES:
                devices.append({"id": dev["id"], "label": dev["label"],
                                 "color": dev["color"], "online": False,
                                 "current": None, "history": []})
            return {"devices": devices}

        try:
            conn = sqlite3.connect(str(STATS_DB_PATH))
            conn.row_factory = sqlite3.Row
            cutoff_24h = int(time.time()) - 86400

            for dev in KNOWN_DEVICES:
                did = dev["id"]
                row = conn.execute(
                    "SELECT * FROM device_stats WHERE device_id=? ORDER BY timestamp DESC LIMIT 1",
                    (did,),
                ).fetchone()
                current = dict(row) if row else None

                # Consider online if last sample is within 3 minutes
                online = bool(current and (time.time() - current["timestamp"]) < 180)

                history_rows = conn.execute(
                    "SELECT timestamp, cpu_pct, gpu_pct FROM device_stats "
                    "WHERE device_id=? AND timestamp > ? ORDER BY timestamp ASC",
                    (did, cutoff_24h),
                ).fetchall()
                history = [{"ts": r["timestamp"], "cpu": r["cpu_pct"], "gpu": r["gpu_pct"]}
                           for r in history_rows]

                devices.append({
                    "id": did,
                    "label": dev["label"],
                    "color": dev["color"],
                    "online": online,
                    "current": current,
                    "history": history,
                })
            conn.close()
        except Exception as e:
            for dev in KNOWN_DEVICES:
                devices.append({"id": dev["id"], "label": dev["label"],
                                 "color": dev["color"], "online": False,
                                 "current": None, "history": [], "error": str(e)})

        return {"devices": devices}

    return cached("compute", _fetch, ttl=30)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/data")
async def api_data():
    """Full dashboard data payload."""
    # Fetch all sources (API calls cached)
    anthropic = cached("anthropic", fetch_anthropic_usage, ttl=300)
    openai = cached("openai", fetch_openai_usage, ttl=300)
    xai = cached("xai", fetch_xai_usage, ttl=300)
    gcs = fetch_gcs_pipeline()

    return JSONResponse({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "providers": {
            "anthropic": anthropic,
            "openai": openai,
            "xai": xai,
        },
        "gcs_pipeline": gcs,
    })


@app.get("/api/compute")
async def api_compute():
    """Compute fleet stats — latest metrics + 24h history per device."""
    return JSONResponse(fetch_compute_stats())


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ramp — Acceleration Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace;
    background: #06060f;
    color: #e0e0e0;
    padding: 20px 24px;
    min-height: 100vh;
}

/* Header */
.header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 24px;
    border-bottom: 1px solid #1a1a3a;
    padding-bottom: 12px;
}
h1 { color: #7c8aff; font-size: 1.6em; letter-spacing: -0.5px; }
h1 span { color: #ff6b6b; font-weight: 400; }
.subtitle { color: #555; font-size: 0.8em; }
.refresh-info { color: #444; font-size: 0.72em; }

/* Section */
h2 {
    color: #5c6bc0;
    font-size: 1.05em;
    margin: 28px 0 14px;
    border-bottom: 1px solid #1a1a3a;
    padding-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

/* Acceleration hero cards */
.accel-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 16px;
    margin-bottom: 8px;
}
.accel-card {
    background: linear-gradient(135deg, #0e0e22 0%, #141430 100%);
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 20px;
    position: relative;
    overflow: hidden;
}
.accel-card.not-configured {
    opacity: 0.4;
    border-style: dashed;
}
.accel-card .provider {
    font-size: 0.75em;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #888;
    margin-bottom: 12px;
}
.accel-card .provider .dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.accel-card .metrics-row {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
}
.metric-box {
    flex: 1;
    min-width: 80px;
    text-align: center;
    padding: 10px 6px;
    background: rgba(0,0,0,0.3);
    border-radius: 6px;
}
.metric-box .label {
    font-size: 0.65em;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #666;
    margin-bottom: 4px;
}
.metric-box .value {
    font-size: 1.5em;
    font-weight: 700;
}
.metric-box .unit {
    font-size: 0.6em;
    color: #555;
}
.positive { color: #4ade80; }
.negative { color: #f87171; }
.neutral { color: #7c8aff; }

/* Acceleration emphasis */
.metric-box.accel-emphasis {
    background: rgba(124, 138, 255, 0.08);
    border: 1px solid rgba(124, 138, 255, 0.2);
}
.metric-box.accel-emphasis .value {
    font-size: 1.8em;
}

/* Chart */
.chart-section {
    background: #0e0e22;
    border: 1px solid #1a1a3a;
    border-radius: 10px;
    padding: 20px;
    margin: 16px 0;
}
.chart-container { position: relative; height: 260px; }
.chart-tabs {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
}
.chart-tab {
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 0.75em;
    cursor: pointer;
    background: #1a1a3a;
    color: #888;
    border: 1px solid #2a2a4a;
    font-family: inherit;
}
.chart-tab.active {
    background: #2a2a5a;
    color: #7c8aff;
    border-color: #5c6bc0;
}

/* Status pills */
.status-bar {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 12px;
}
.status-pill {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    background: #12122a;
    border: 1px solid #2a2a4a;
    border-radius: 20px;
    font-size: 0.75em;
}
.status-pill .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
}
.dot-green { background: #4ade80; }
.dot-red { background: #f87171; }
.dot-yellow { background: #fbbf24; }
.dot-gray { background: #555; }

/* Error msg */
.error-msg { color: #f87171; font-size: 0.8em; padding: 8px; }
.config-hint { color: #666; font-size: 0.75em; font-style: italic; padding: 8px 0; }

/* Acceleration table */
.accel-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82em;
    margin-top: 12px;
}
.accel-table th {
    text-align: left;
    padding: 8px 10px;
    background: #12122a;
    color: #5c6bc0;
    border-bottom: 2px solid #2a2a5a;
    font-weight: 600;
}
.accel-table td {
    padding: 6px 10px;
    border-bottom: 1px solid #1a1a2a;
}
.accel-table tr:hover { background: #121230; }
.accel-table .num { text-align: right; font-variant-numeric: tabular-nums; }

/* Compute fleet */
.compute-card {
    background: linear-gradient(135deg, #0e0e22 0%, #141430 100%);
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 20px;
    position: relative;
}
.compute-card.offline {
    border-style: dashed;
    opacity: 0.35;
}
.compute-card .device-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 14px;
}
.compute-card .device-name {
    font-size: 0.78em;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #aaa;
    font-weight: 600;
}
.stat-pills {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 10px;
}
.stat-pill {
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.75em;
    font-weight: 700;
    background: rgba(0,0,0,0.35);
    border: 1px solid #2a2a4a;
    min-width: 56px;
    text-align: center;
}
.stat-pill .pill-label {
    color: #666;
    font-size: 0.75em;
    font-weight: 400;
    display: block;
    letter-spacing: 0.5px;
}
.stat-green { color: #4ade80; }
.stat-yellow { color: #fbbf24; }
.stat-red { color: #f87171; }
.compute-meta {
    font-size: 0.72em;
    color: #555;
    margin-top: 6px;
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
}
.sparkline-wrap {
    margin-top: 12px;
    height: 60px;
    position: relative;
}
</style>
</head>
<body>

<div class="header">
    <div>
        <h1>Ramp <span>// acceleration</span></h1>
        <div class="subtitle">Focus on what's improving, not just what is.</div>
    </div>
    <div class="refresh-info" id="refresh-info">Loading...</div>
</div>

<div id="app"><div style="color:#555">Loading data...</div></div>

<script>
let charts = {};

function fmtNum(n) {
    if (typeof n !== 'number' || isNaN(n)) return '—';
    if (Math.abs(n) >= 1e9) return (n/1e9).toFixed(1) + 'B';
    if (Math.abs(n) >= 1e6) return (n/1e6).toFixed(1) + 'M';
    if (Math.abs(n) >= 1e3) return (n/1e3).toFixed(1) + 'K';
    return n.toLocaleString();
}

function fmtDollar(n) {
    if (typeof n !== 'number' || isNaN(n)) return '—';
    return '$' + n.toFixed(2);
}

function signClass(n) {
    if (n > 0) return 'positive';
    if (n < 0) return 'negative';
    return 'neutral';
}

function signPrefix(n) {
    return n > 0 ? '+' : '';
}

function providerColor(name) {
    return {anthropic: '#d4a574', openai: '#10b981', xai: '#60a5fa', gcs: '#a78bfa'}[name] || '#888';
}

function renderAccelCard(name, label, data, isCost) {
    if (!data || !data.configured) {
        return `<div class="accel-card not-configured">
            <div class="provider"><span class="dot" style="background:${providerColor(name)}"></span>${label}</div>
            <div class="config-hint">Not configured. Add admin key to config.env</div>
        </div>`;
    }
    if (data.error && !data.token_metrics) {
        return `<div class="accel-card">
            <div class="provider"><span class="dot" style="background:${providerColor(name)}"></span>${label}</div>
            <div class="error-msg">${data.error}</div>
        </div>`;
    }

    const tm = data.token_metrics || {position: 0, velocity: 0, acceleration: 0};
    const cm = data.cost_metrics || {position: 0, velocity: 0, acceleration: 0};
    const m = isCost ? cm : tm;
    const fmt = isCost ? fmtDollar : fmtNum;

    return `<div class="accel-card">
        <div class="provider"><span class="dot" style="background:${providerColor(name)}"></span>${label}</div>
        <div class="metrics-row">
            <div class="metric-box">
                <div class="label">Position</div>
                <div class="value neutral">${fmt(m.position)}</div>
                <div class="unit">${isCost ? 'total spend' : 'total tokens'}</div>
            </div>
            <div class="metric-box">
                <div class="label">Velocity</div>
                <div class="value ${signClass(m.velocity)}">${signPrefix(m.velocity)}${fmt(m.velocity)}</div>
                <div class="unit">7-day avg/day</div>
            </div>
            <div class="metric-box accel-emphasis">
                <div class="label">Acceleration</div>
                <div class="value ${signClass(m.acceleration)}">${signPrefix(m.acceleration)}${isCost ? fmtDollar(m.acceleration) : fmtNum(m.acceleration)}</div>
                <div class="unit">5-day &Delta; velocity</div>
            </div>
        </div>
    </div>`;
}

function renderGcsCard(gcs) {
    if (!gcs || !gcs.metrics) {
        return `<div class="accel-card not-configured">
            <div class="provider"><span class="dot" style="background:${providerColor('gcs')}"></span>Training Pipeline</div>
            <div class="config-hint">No GCS data found</div>
        </div>`;
    }

    const m = gcs.metrics;
    return `<div class="accel-card">
        <div class="provider"><span class="dot" style="background:${providerColor('gcs')}"></span>Training Pipeline (GCS)</div>
        <div class="metrics-row">
            <div class="metric-box">
                <div class="label">Position</div>
                <div class="value neutral">${fmtNum(m.position)}</div>
                <div class="unit">total sessions</div>
            </div>
            <div class="metric-box">
                <div class="label">Velocity</div>
                <div class="value ${signClass(m.velocity)}">${signPrefix(m.velocity)}${fmtNum(m.velocity)}</div>
                <div class="unit">7-day avg/day</div>
            </div>
            <div class="metric-box accel-emphasis">
                <div class="label">Acceleration</div>
                <div class="value ${signClass(m.acceleration)}">${signPrefix(m.acceleration)}${fmtNum(m.acceleration)}</div>
                <div class="unit">5-day &Delta; velocity</div>
            </div>
        </div>
    </div>`;
}

function renderAccelTable(data) {
    const rows = [];
    for (const [name, label] of [['anthropic','Anthropic'],['openai','OpenAI'],['xai','xAI/Grok']]) {
        const p = data.providers[name];
        if (!p || !p.configured || !p.token_metrics) continue;
        const tm = p.token_metrics;
        const cm = p.cost_metrics || {position:0, velocity:0, acceleration:0};
        rows.push({name: label, color: providerColor(name), tm, cm});
    }
    if (data.gcs_pipeline && data.gcs_pipeline.metrics) {
        const m = data.gcs_pipeline.metrics;
        rows.push({name: 'GCS Pipeline', color: providerColor('gcs'),
            tm: m, cm: {position: 0, velocity: 0, acceleration: 0}});
    }

    if (!rows.length) return '';

    // Daily acceleration breakdown for last 5 days
    let dailyHtml = '';
    for (const r of rows) {
        const daily = r.tm.daily || [];
        const last5 = daily.slice(-5);
        if (!last5.length) continue;
        dailyHtml += `<tr style="border-top:2px solid #2a2a4a"><td colspan="5" style="color:${r.color};font-weight:bold;padding-top:10px">${r.name}</td></tr>`;
        for (const d of last5) {
            dailyHtml += `<tr>
                <td>${d.date}</td>
                <td class="num">${fmtNum(d.value)}</td>
                <td class="num">${fmtNum(d.cumulative)}</td>
                <td class="num ${signClass(d.velocity)}">${signPrefix(d.velocity)}${fmtNum(d.velocity)}</td>
                <td class="num ${signClass(d.acceleration)}">${signPrefix(d.acceleration)}${fmtNum(d.acceleration)}</td>
            </tr>`;
        }
    }

    return `<table class="accel-table">
        <thead><tr>
            <th>Date</th><th class="num">Value</th><th class="num">Cumulative</th>
            <th class="num">Velocity</th><th class="num">Acceleration</th>
        </tr></thead>
        <tbody>${dailyHtml}</tbody>
    </table>`;
}

function renderChart(data) {
    const datasets = [];
    const colors = {anthropic: '#d4a574', openai: '#10b981', xai: '#60a5fa', gcs: '#a78bfa'};

    // Get all dates
    let allDates = new Set();
    for (const [name, p] of Object.entries(data.providers)) {
        if (p.token_metrics && p.token_metrics.daily) {
            p.token_metrics.daily.forEach(d => allDates.add(d.date));
        }
    }
    if (data.gcs_pipeline && data.gcs_pipeline.metrics && data.gcs_pipeline.metrics.daily) {
        data.gcs_pipeline.metrics.daily.forEach(d => allDates.add(d.date));
    }
    allDates = [...allDates].sort();
    if (!allDates.length) return '';

    // Build acceleration datasets
    for (const [name, label] of [['anthropic','Anthropic'],['openai','OpenAI'],['xai','xAI']]) {
        const p = data.providers[name];
        if (!p || !p.token_metrics || !p.token_metrics.daily) continue;
        const dateMap = {};
        p.token_metrics.daily.forEach(d => dateMap[d.date] = d.acceleration);
        datasets.push({
            label: label + ' Accel',
            data: allDates.map(d => dateMap[d] || 0),
            borderColor: colors[name],
            backgroundColor: colors[name] + '33',
            borderWidth: 2,
            pointRadius: 3,
            tension: 0.3,
        });
    }
    if (data.gcs_pipeline && data.gcs_pipeline.metrics && data.gcs_pipeline.metrics.daily) {
        const dateMap = {};
        data.gcs_pipeline.metrics.daily.forEach(d => dateMap[d.date] = d.acceleration);
        datasets.push({
            label: 'GCS Accel',
            data: allDates.map(d => dateMap[d] || 0),
            borderColor: colors.gcs,
            backgroundColor: colors.gcs + '33',
            borderWidth: 2,
            pointRadius: 3,
            tension: 0.3,
        });
    }

    return `<div class="chart-section">
        <div style="font-size:0.85em;color:#5c6bc0;font-weight:600;margin-bottom:12px">ACCELERATION OVER TIME</div>
        <div class="chart-container"><canvas id="accelChart"></canvas></div>
    </div>`;
}

function updateAccelChart(data) {
    const canvas = document.getElementById('accelChart');
    if (!canvas) return;

    const colors = {anthropic: '#d4a574', openai: '#10b981', xai: '#60a5fa', gcs: '#a78bfa'};
    let allDates = new Set();
    const sources = [];

    for (const [name, label] of [['anthropic','Anthropic'],['openai','OpenAI'],['xai','xAI']]) {
        const p = data.providers[name];
        if (!p || !p.token_metrics || !p.token_metrics.daily) continue;
        p.token_metrics.daily.forEach(d => allDates.add(d.date));
        sources.push({name, label, daily: p.token_metrics.daily, color: colors[name]});
    }
    if (data.gcs_pipeline?.metrics?.daily) {
        data.gcs_pipeline.metrics.daily.forEach(d => allDates.add(d.date));
        sources.push({name:'gcs', label:'GCS Pipeline', daily: data.gcs_pipeline.metrics.daily, color: colors.gcs});
    }

    allDates = [...allDates].sort();
    if (!allDates.length) return;

    const datasets = sources.map(s => {
        const dateMap = {};
        s.daily.forEach(d => dateMap[d.date] = d.acceleration);
        return {
            label: s.label,
            data: allDates.map(d => dateMap[d] || 0),
            borderColor: s.color,
            backgroundColor: s.color + '22',
            borderWidth: 2,
            pointRadius: 3,
            pointBackgroundColor: s.color,
            fill: true,
            tension: 0.3,
        };
    });

    if (charts.accel) {
        charts.accel.data.labels = allDates;
        charts.accel.data.datasets = datasets;
        charts.accel.update('none');
        return;
    }

    charts.accel = new Chart(canvas, {
        type: 'line',
        data: { labels: allDates, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { labels: { color: '#888', font: { family: "'SF Mono', monospace", size: 11 } } },
                tooltip: {
                    backgroundColor: '#1a1a3a',
                    titleColor: '#e0e0e0',
                    bodyColor: '#bbb',
                    borderColor: '#2a2a4a',
                    borderWidth: 1,
                },
            },
            scales: {
                x: {
                    ticks: { color: '#555', font: { size: 10 } },
                    grid: { color: 'rgba(42,42,74,0.3)' },
                },
                y: {
                    title: { display: true, text: 'Acceleration (Δ velocity)', color: '#5c6bc0', font: { size: 11 } },
                    ticks: { color: '#666', font: { size: 10 } },
                    grid: { color: 'rgba(42,42,74,0.3)' },
                },
            },
        },
    });
}

function renderStatusBar(data) {
    const pills = [];
    for (const [name, label] of [['anthropic','Anthropic'],['openai','OpenAI'],['xai','xAI/Grok']]) {
        const p = data.providers[name];
        const ok = p && p.configured && !p.error;
        const dotClass = !p || !p.configured ? 'dot-gray' : p.error ? 'dot-red' : 'dot-green';
        pills.push(`<div class="status-pill"><span class="dot ${dotClass}"></span>${label}</div>`);
    }
    // GCS
    const gcsOk = data.gcs_pipeline && data.gcs_pipeline.metrics && data.gcs_pipeline.metrics.position > 0;
    pills.push(`<div class="status-pill"><span class="dot ${gcsOk ? 'dot-green' : 'dot-yellow'}"></span>GCS Pipeline</div>`);

    // Active VMs
    const activeVms = data.gcs_pipeline?.active_vms || [];
    if (activeVms.length) {
        pills.push(`<div class="status-pill"><span class="dot dot-green"></span>${activeVms.length} VM${activeVms.length>1?'s':''} running</div>`);
    }

    return `<div class="status-bar">${pills.join('')}</div>`;
}

function render(data) {
    const el = document.getElementById('app');
    document.getElementById('refresh-info').textContent =
        `Last: ${new Date().toLocaleTimeString()} — refreshes every 30s`;

    el.innerHTML = `
        <div id="compute-fleet-section"></div>

        ${renderStatusBar(data)}

        <h2>Acceleration Overview — Tokens</h2>
        <div class="accel-grid">
            ${renderAccelCard('anthropic', 'Anthropic (Claude)', data.providers.anthropic, false)}
            ${renderAccelCard('openai', 'OpenAI', data.providers.openai, false)}
            ${renderAccelCard('xai', 'xAI (Grok)', data.providers.xai, false)}
            ${renderGcsCard(data.gcs_pipeline)}
        </div>

        <h2>Acceleration Overview — Spend</h2>
        <div class="accel-grid">
            ${renderAccelCard('anthropic', 'Anthropic (Claude)', data.providers.anthropic, true)}
            ${renderAccelCard('openai', 'OpenAI', data.providers.openai, true)}
            ${renderAccelCard('xai', 'xAI (Grok)', data.providers.xai, true)}
        </div>

        ${renderChart(data)}

        <h2>Daily Acceleration Breakdown (Last 5 Days)</h2>
        ${renderAccelTable(data)}
    `;

    requestAnimationFrame(() => updateAccelChart(data));
}

// ---------------------------------------------------------------------------
// Compute Fleet
// ---------------------------------------------------------------------------

let computeCharts = {};

function statColor(pct) {
    if (pct == null) return '#555';
    if (pct < 50) return '#4ade80';
    if (pct < 80) return '#fbbf24';
    return '#f87171';
}

function statClass(pct) {
    if (pct == null) return '';
    if (pct < 50) return 'stat-green';
    if (pct < 80) return 'stat-yellow';
    return 'stat-red';
}

function fmtPct(v) {
    return v != null ? v.toFixed(0) + '%' : '—';
}

function fmtUptime(secs) {
    if (!secs) return '—';
    const d = Math.floor(secs / 86400);
    const h = Math.floor((secs % 86400) / 3600);
    const m = Math.floor((secs % 3600) / 60);
    if (d > 0) return `${d}d ${h}h`;
    if (h > 0) return `${h}h ${m}m`;
    return `${m}m`;
}

function renderDeviceSparkline(canvasId, history, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !history || !history.length) return;

    const labels = history.map(h => {
        const d = new Date(h.ts * 1000);
        return d.getHours() + ':' + String(d.getMinutes()).padStart(2, '0');
    });
    const cpuData = history.map(h => h.cpu);
    const gpuData = history.map(h => h.gpu);

    const datasets = [{
        label: 'CPU',
        data: cpuData,
        borderColor: color,
        backgroundColor: color + '22',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
        tension: 0.3,
    }];
    if (gpuData.some(v => v != null)) {
        datasets.push({
            label: 'GPU',
            data: gpuData,
            borderColor: '#f87171',
            backgroundColor: '#f8717122',
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false,
            tension: 0.3,
        });
    }

    if (computeCharts[canvasId]) {
        computeCharts[canvasId].data.labels = labels;
        computeCharts[canvasId].data.datasets = datasets;
        computeCharts[canvasId].update('none');
        return;
    }

    computeCharts[canvasId] = new Chart(canvas, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false },
            },
            scales: {
                x: { display: false },
                y: {
                    display: false,
                    min: 0,
                    max: 100,
                },
            },
            animation: false,
        },
    });
}

function renderComputeFleet(data) {
    const el = document.getElementById('compute-fleet-section');
    if (!el) return;
    if (!data || !data.devices) {
        el.innerHTML = '';
        return;
    }

    const cards = data.devices.map(dev => {
        if (!dev.online || !dev.current) {
            return `<div class="compute-card offline">
                <div class="device-header">
                    <span class="dot dot-gray" style="width:8px;height:8px;border-radius:50%;display:inline-block"></span>
                    <span class="device-name" style="color:${dev.color}">${dev.label}</span>
                </div>
                <div style="color:#444;font-size:0.8em">Not connected</div>
            </div>`;
        }

        const c = dev.current;
        const sparkId = `spark-${dev.id}`;

        return `<div class="compute-card">
            <div class="device-header">
                <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#4ade80;box-shadow:0 0 6px #4ade80"></span>
                <span class="device-name" style="color:${dev.color}">${dev.label}</span>
            </div>
            <div class="stat-pills">
                <div class="stat-pill">
                    <span class="pill-label">CPU</span>
                    <span class="${statClass(c.cpu_pct)}">${fmtPct(c.cpu_pct)}</span>
                </div>
                <div class="stat-pill">
                    <span class="pill-label">GPU</span>
                    <span class="${statClass(c.gpu_pct)}">${fmtPct(c.gpu_pct)}</span>
                </div>
                <div class="stat-pill">
                    <span class="pill-label">RAM</span>
                    <span class="${statClass(c.ram_pct)}">${fmtPct(c.ram_pct)}</span>
                </div>
                <div class="stat-pill">
                    <span class="pill-label">Disk</span>
                    <span class="${statClass(c.disk_pct)}">${fmtPct(c.disk_pct)}</span>
                </div>
            </div>
            <div class="compute-meta">
                ${c.cpu_temp != null ? `<span>CPU ${c.cpu_temp.toFixed(0)}°C</span>` : ''}
                ${c.gpu_temp != null ? `<span>GPU ${c.gpu_temp.toFixed(0)}°C</span>` : ''}
                ${c.power_mw != null ? `<span>${(c.power_mw/1000).toFixed(1)}W</span>` : ''}
                <span>up ${fmtUptime(c.uptime_secs)}</span>
                ${c.disk_used_gb != null ? `<span>${c.disk_used_gb}/${c.disk_total_gb}GB disk</span>` : ''}
            </div>
            ${dev.history.length > 1 ? `<div class="sparkline-wrap"><canvas id="${sparkId}"></canvas></div>` : ''}
        </div>`;
    });

    el.innerHTML = `
        <h2>Compute Fleet</h2>
        <div class="accel-grid">${cards.join('')}</div>
    `;

    // Draw sparklines after DOM update
    requestAnimationFrame(() => {
        data.devices.forEach(dev => {
            if (dev.online && dev.history.length > 1) {
                renderDeviceSparkline(`spark-${dev.id}`, dev.history, dev.color);
            }
        });
    });
}

async function fetchCompute() {
    try {
        const resp = await fetch('/api/compute');
        const data = await resp.json();
        renderComputeFleet(data);
    } catch (e) {
        const el = document.getElementById('compute-fleet-section');
        if (el) el.innerHTML = '';
    }
}

// ---------------------------------------------------------------------------
// Main refresh
// ---------------------------------------------------------------------------

async function refresh() {
    try {
        const resp = await fetch('/api/data');
        const data = await resp.json();
        render(data);
    } catch (e) {
        document.getElementById('refresh-info').textContent = 'Error: ' + e.message;
    }
}

refresh();
fetchCompute();
setInterval(refresh, 30000);
setInterval(fetchCompute, 30000);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ramp — Acceleration Dashboard")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    print(f"Ramp Dashboard: http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
