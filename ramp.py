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
import asyncio
import json
import os
import sqlite3
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
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
    {"id": "jetson-orin-nano",      "label": "Jetson Orin Nano",      "color": "#7c8aff",  "type": "jetson",      "alias": "Prometheus",   "ip": "192.168.0.18",  "user": "prometheus",  "password": "rising"},
    {"id": "nvidia-dgx-spark",     "label": "NVIDIA DGX Spark",       "color": "#60a5fa",  "type": "dgx",         "alias": "Spark",        "ip": "192.168.0.234", "user": "macro",       "password": ""},
    {"id": "rtx4080-workstation",   "label": "RTX 4080 Workstation",   "color": "#a78bfa",  "type": "workstation", "alias": "Workstation"},
    {"id": "atlas",                 "label": "Jetson Orin Nano Super", "color": "#34d399",  "type": "jetson",      "alias": "Atlas",        "ip": "192.168.0.101",  "user": "atlas",       "password": "shrugged",   "legacy_ids": ["orin-2", "orin-nano-2"]},
    {"id": "epimetheus",            "label": "Jetson Orin Nano Super", "color": "#fbbf24",  "type": "jetson",      "alias": "Epimetheus",   "ip": "192.168.0.202", "user": "epimetheus",  "password": "reflecting", "legacy_ids": ["orin-3", "orin-nano-3"]},
]

APP_VERSION = Path("VERSION").read_text().strip() if Path("VERSION").exists() else "dev"

app = FastAPI(title="Ramp Dashboard", version=APP_VERSION)

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
# SSH helpers for remote device polling
# ---------------------------------------------------------------------------

def _ssh_cmd(device: dict, cmd: str, timeout: int = 10) -> str | None:
    """Run a command on a remote device via SSH. Returns stdout or None."""
    ip = device.get("ip")
    user = device.get("user")
    if not ip or not user:
        return None

    ssh_opts = [
        "-o", "ConnectTimeout=5",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
    ]
    target = f"{user}@{ip}"

    # Try key-based auth first
    try:
        result = subprocess.run(
            ["ssh", *ssh_opts, target, cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fall back to sshpass if password is set
    pw = device.get("password")
    if pw:
        try:
            result = subprocess.run(
                ["sshpass", f"-p{pw}", "ssh",
                 "-o", "ConnectTimeout=5",
                 "-o", "StrictHostKeyChecking=no",
                 target, cmd],
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode == 0:
                return result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    return None


def _local_or_ssh_cmd(device: dict, cmd: str, timeout: int = 10) -> str | None:
    """Run locally for Prometheus (where dashboard runs), SSH for others."""
    if device["id"] == "jetson-orin-nano":
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode == 0:
                return result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None
    return _ssh_cmd(device, cmd, timeout=timeout)


def _find_device(device_id: str) -> dict | None:
    """Look up a device by ID or legacy ID."""
    for d in KNOWN_DEVICES:
        if d["id"] == device_id:
            return d
        if device_id in d.get("legacy_ids", []):
            return d
    return None


_SKIP_IFACES = {"lo", "docker0", "br-", "veth", "virbr"}


def _parse_proc_net_dev(text: str) -> list[dict]:
    """Parse /proc/net/dev output into [{interface, rx_bytes, tx_bytes}, ...]."""
    results = []
    for line in text.strip().splitlines():
        if ":" not in line:
            continue
        iface_part, stats_part = line.split(":", 1)
        iface = iface_part.strip()
        if any(iface.startswith(s) for s in _SKIP_IFACES):
            continue
        cols = stats_part.split()
        if len(cols) < 10:
            continue
        rx_bytes = int(cols[0])
        tx_bytes = int(cols[8])
        if rx_bytes < 1_000_000:  # skip interfaces with <1MB received
            continue
        results.append({
            "interface": iface,
            "rx_bytes": rx_bytes,
            "tx_bytes": tx_bytes,
        })
    return results


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
                                 "color": dev["color"], "type": dev["type"],
                                 "alias": dev["alias"], "online": False,
                                 "current": None, "history": [], "processes": [],
                                 "disks": []})
            return {"devices": devices}

        try:
            conn = sqlite3.connect(str(STATS_DB_PATH))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            cutoff_24h = int(time.time()) - 86400

            for dev in KNOWN_DEVICES:
                did = dev["id"]
                candidate_ids = [did, *(dev.get("legacy_ids") or [])]
                placeholders = ",".join(["?"] * len(candidate_ids))
                row = conn.execute(
                    f"SELECT * FROM device_stats WHERE device_id IN ({placeholders}) "
                    "ORDER BY timestamp DESC LIMIT 1",
                    tuple(candidate_ids),
                ).fetchone()
                current = dict(row) if row else None

                # Consider online if last sample is within 3 minutes
                online = bool(current and (time.time() - current["timestamp"]) < 180)

                history_rows = conn.execute(
                    "SELECT timestamp, cpu_pct, gpu_pct, ram_pct, cpu_temp, gpu_temp, power_mw "
                    "FROM device_stats "
                    f"WHERE device_id IN ({placeholders}) AND timestamp > ? ORDER BY timestamp ASC",
                    tuple(candidate_ids) + (cutoff_24h,),
                ).fetchall()
                history = [{"ts": r["timestamp"], "cpu": r["cpu_pct"], "gpu": r["gpu_pct"],
                            "ram": r["ram_pct"], "cpu_temp": r["cpu_temp"],
                            "gpu_temp": r["gpu_temp"], "power": r["power_mw"]}
                           for r in history_rows]

                # Fetch latest processes for this device
                processes = []
                try:
                    proc_rows = conn.execute(
                        "SELECT name, cpu_pct, mem_pct, kind FROM device_processes "
                        f"WHERE device_id IN ({placeholders}) ORDER BY snapshot_ts DESC, cpu_pct DESC LIMIT 10",
                        tuple(candidate_ids),
                    ).fetchall()
                    processes = [{"name": r["name"], "cpu": r["cpu_pct"],
                                  "mem": r["mem_pct"], "kind": r["kind"]}
                                 for r in proc_rows]
                except sqlite3.OperationalError:
                    pass  # table may not exist yet

                # Fetch latest disk info for this device
                disks = []
                try:
                    disk_rows = conn.execute(
                        "SELECT mount, label, used_gb, total_gb, pct FROM device_disks "
                        f"WHERE device_id IN ({placeholders}) ORDER BY snapshot_ts DESC, total_gb DESC",
                        tuple(candidate_ids),
                    ).fetchall()
                    disks = [{"mount": r["mount"], "label": r["label"],
                              "used_gb": r["used_gb"], "total_gb": r["total_gb"],
                              "pct": r["pct"]}
                             for r in disk_rows]
                except sqlite3.OperationalError:
                    pass  # table may not exist yet

                devices.append({
                    "id": did,
                    "label": dev["label"],
                    "color": dev["color"],
                    "type": dev["type"],
                    "alias": dev["alias"],
                    "ip": dev.get("ip"),
                    "online": online,
                    "current": current,
                    "history": history,
                    "processes": processes,
                    "disks": disks,
                })
            conn.close()
        except Exception as e:
            for dev in KNOWN_DEVICES:
                devices.append({"id": dev["id"], "label": dev["label"],
                                 "color": dev["color"], "type": dev["type"],
                                 "alias": dev["alias"], "ip": dev.get("ip"), "online": False,
                                 "current": None, "history": [], "processes": [],
                                 "disks": [], "error": str(e)})

        return {"devices": devices}

    return cached("compute", _fetch, ttl=30)


# ---------------------------------------------------------------------------
# Network stats
# ---------------------------------------------------------------------------

def fetch_network_stats() -> dict:
    """Compute per-device daily network usage from cumulative byte counters."""
    def _fetch():
        if not STATS_DB_PATH.exists():
            return {"devices": []}

        try:
            conn = sqlite3.connect(str(STATS_DB_PATH))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            cutoff = int(time.time()) - (14 * 86400)

            results = []
            for dev in KNOWN_DEVICES:
                did = dev["id"]
                candidate_ids = [did, *(dev.get("legacy_ids") or [])]
                placeholders = ",".join(["?"] * len(candidate_ids))

                rows = conn.execute(
                    f"""SELECT timestamp, interface, rx_bytes, tx_bytes
                        FROM device_network
                        WHERE device_id IN ({placeholders}) AND timestamp > ?
                        ORDER BY interface, timestamp ASC""",
                    tuple(candidate_ids) + (cutoff,),
                ).fetchall()

                if not rows:
                    continue

                # Group by interface
                by_iface = {}
                for r in rows:
                    iface = r["interface"]
                    if iface not in by_iface:
                        by_iface[iface] = []
                    by_iface[iface].append({
                        "ts": r["timestamp"],
                        "rx": r["rx_bytes"],
                        "tx": r["tx_bytes"],
                    })

                # Compute deltas and bucket into days
                daily = {}  # date_str -> {rx, tx}
                total_rx = 0
                total_tx = 0
                last_pair_gap = None
                last_pair_rx_rate = 0
                last_pair_tx_rate = 0

                for iface, readings in by_iface.items():
                    for i in range(1, len(readings)):
                        prev = readings[i - 1]
                        curr = readings[i]
                        dt = curr["ts"] - prev["ts"]
                        if dt <= 0:
                            continue

                        drx = curr["rx"] - prev["rx"]
                        dtx = curr["tx"] - prev["tx"]

                        # Counter reset (reboot): use current value as delta
                        if drx < 0:
                            drx = curr["rx"]
                        if dtx < 0:
                            dtx = curr["tx"]

                        day = datetime.fromtimestamp(curr["ts"], tz=timezone.utc).strftime("%Y-%m-%d")
                        if day not in daily:
                            daily[day] = {"rx": 0, "tx": 0}
                        daily[day]["rx"] += drx
                        daily[day]["tx"] += dtx
                        total_rx += drx
                        total_tx += dtx

                        # Track most recent pair for current rate
                        if last_pair_gap is None or curr["ts"] > readings[i - 1]["ts"]:
                            last_pair_gap = dt
                            last_pair_rx_rate = drx / dt if dt > 0 else 0
                            last_pair_tx_rate = dtx / dt if dt > 0 else 0

                # Sort daily into list
                daily_list = [{"date": d, "rx": daily[d]["rx"], "tx": daily[d]["tx"]}
                              for d in sorted(daily.keys())]

                # Today's totals
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                today_data = daily.get(today, {"rx": 0, "tx": 0})

                # Current rate — only show if last reading gap <= 5 minutes
                rate_rx = 0
                rate_tx = 0
                if last_pair_gap is not None and last_pair_gap <= 300:
                    rate_rx = last_pair_rx_rate
                    rate_tx = last_pair_tx_rate

                results.append({
                    "id": did,
                    "alias": dev["alias"],
                    "color": dev["color"],
                    "today_rx": today_data["rx"],
                    "today_tx": today_data["tx"],
                    "rate_rx": rate_rx,
                    "rate_tx": rate_tx,
                    "daily": daily_list,
                })

            conn.close()

            # Fleet roll-up: aggregate all devices into a total card
            if len(results) > 1:
                fleet_daily = {}  # date -> {rx, tx}
                fleet_today_rx = 0
                fleet_today_tx = 0
                fleet_rate_rx = 0
                fleet_rate_tx = 0
                for r in results:
                    fleet_today_rx += r["today_rx"]
                    fleet_today_tx += r["today_tx"]
                    fleet_rate_rx += r["rate_rx"]
                    fleet_rate_tx += r["rate_tx"]
                    for d in r["daily"]:
                        if d["date"] not in fleet_daily:
                            fleet_daily[d["date"]] = {"rx": 0, "tx": 0}
                        fleet_daily[d["date"]]["rx"] += d["rx"]
                        fleet_daily[d["date"]]["tx"] += d["tx"]

                fleet_daily_list = [{"date": d, "rx": fleet_daily[d]["rx"],
                                     "tx": fleet_daily[d]["tx"]}
                                    for d in sorted(fleet_daily.keys())]
                results.insert(0, {
                    "id": "_total",
                    "alias": "Fleet Total",
                    "color": "#ffffff",
                    "today_rx": fleet_today_rx,
                    "today_tx": fleet_today_tx,
                    "rate_rx": fleet_rate_rx,
                    "rate_tx": fleet_rate_tx,
                    "daily": fleet_daily_list,
                    "is_total": True,
                })

            return {"devices": results}
        except Exception as e:
            return {"devices": [], "error": str(e)}

    return cached("network", _fetch, ttl=30)


# ---------------------------------------------------------------------------
# Remote network polling (SSH-based)
# ---------------------------------------------------------------------------

_poll_shutdown = threading.Event()


def _poll_remote_network():
    """Background thread: SSH into remote devices, read /proc/net/dev, insert into DB."""
    # Devices with IP set, excluding Prometheus (has local stats_collector)
    targets = [d for d in KNOWN_DEVICES if d.get("ip") and d["id"] != "jetson-orin-nano"]

    while not _poll_shutdown.is_set():
        for dev in targets:
            try:
                output = _ssh_cmd(dev, "cat /proc/net/dev")
                if not output:
                    continue

                counters = _parse_proc_net_dev(output)
                if not counters:
                    continue

                ts = int(time.time())
                conn = sqlite3.connect(str(STATS_DB_PATH))
                conn.execute("PRAGMA journal_mode=WAL")
                for nc in counters:
                    conn.execute(
                        """INSERT INTO device_network
                           (timestamp, device_id, interface, rx_bytes, tx_bytes)
                           VALUES (?, ?, ?, ?, ?)""",
                        (ts, dev["id"], nc["interface"],
                         nc["rx_bytes"], nc["tx_bytes"]),
                    )
                conn.commit()
                conn.close()
            except Exception:
                pass  # silently skip unreachable devices

        # Invalidate network cache after a polling round
        with _cache_lock:
            _cache.pop("network", None)

        # Sleep in 1s increments for graceful shutdown
        for _ in range(60):
            if _poll_shutdown.is_set():
                break
            time.sleep(1)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/network")
async def api_network():
    """Per-device network traffic — daily totals and current rates."""
    return JSONResponse(fetch_network_stats())


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


def _ensure_ingest_tables(conn):
    """Ensure stats and process tables exist for ingest."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS device_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            device_id TEXT NOT NULL,
            cpu_pct REAL, gpu_pct REAL,
            ram_pct REAL, ram_used_mb INTEGER, ram_total_mb INTEGER,
            disk_pct REAL, disk_used_gb REAL, disk_total_gb REAL,
            cpu_temp REAL, gpu_temp REAL,
            uptime_secs INTEGER, power_mw INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_ts ON device_stats (timestamp, device_id);
        CREATE TABLE IF NOT EXISTS device_processes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_ts INTEGER NOT NULL,
            device_id TEXT NOT NULL,
            name TEXT NOT NULL,
            cpu_pct REAL,
            mem_pct REAL,
            kind TEXT DEFAULT 'process'
        );
        CREATE INDEX IF NOT EXISTS idx_proc_ts ON device_processes (device_id, snapshot_ts);
        CREATE TABLE IF NOT EXISTS device_disks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_ts INTEGER NOT NULL,
            device_id TEXT NOT NULL,
            mount TEXT NOT NULL,
            label TEXT,
            used_gb REAL,
            total_gb REAL,
            pct REAL
        );
        CREATE INDEX IF NOT EXISTS idx_disk_ts ON device_disks (device_id, snapshot_ts);
        CREATE TABLE IF NOT EXISTS device_network (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            device_id TEXT NOT NULL,
            interface TEXT NOT NULL,
            rx_bytes INTEGER NOT NULL,
            tx_bytes INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_net_ts ON device_network (device_id, timestamp);
    """)
    conn.commit()


@app.post("/api/ingest/stats")
async def ingest_stats(request: Request):
    """Accept pushed stats from remote device collectors."""
    body = await request.json()
    device_id = body.get("device_id")
    if not device_id:
        return JSONResponse({"error": "device_id required"}, status_code=400)

    try:
        STATS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(STATS_DB_PATH))
        conn.execute("PRAGMA journal_mode=WAL")
        _ensure_ingest_tables(conn)

        # Insert device stats
        row = {
            "timestamp": body.get("timestamp", int(time.time())),
            "device_id": device_id,
            "cpu_pct": body.get("cpu_pct"),
            "gpu_pct": body.get("gpu_pct"),
            "ram_pct": body.get("ram_pct"),
            "ram_used_mb": body.get("ram_used_mb"),
            "ram_total_mb": body.get("ram_total_mb"),
            "disk_pct": body.get("disk_pct"),
            "disk_used_gb": body.get("disk_used_gb"),
            "disk_total_gb": body.get("disk_total_gb"),
            "cpu_temp": body.get("cpu_temp"),
            "gpu_temp": body.get("gpu_temp"),
            "uptime_secs": body.get("uptime_secs"),
            "power_mw": body.get("power_mw"),
        }
        conn.execute(
            """INSERT INTO device_stats
               (timestamp, device_id, cpu_pct, gpu_pct,
                ram_pct, ram_used_mb, ram_total_mb,
                disk_pct, disk_used_gb, disk_total_gb,
                cpu_temp, gpu_temp, uptime_secs, power_mw)
               VALUES
               (:timestamp, :device_id, :cpu_pct, :gpu_pct,
                :ram_pct, :ram_used_mb, :ram_total_mb,
                :disk_pct, :disk_used_gb, :disk_total_gb,
                :cpu_temp, :gpu_temp, :uptime_secs, :power_mw)""",
            row,
        )

        # Insert processes if provided
        processes = body.get("processes", [])
        ts = row["timestamp"]
        if processes:
            # Clear old processes for this device
            conn.execute(
                "DELETE FROM device_processes WHERE device_id=?", (device_id,))
            for proc in processes[:10]:
                conn.execute(
                    """INSERT INTO device_processes
                       (snapshot_ts, device_id, name, cpu_pct, mem_pct, kind)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (ts, device_id, proc.get("name", "?"),
                     proc.get("cpu", 0), proc.get("mem", 0),
                     proc.get("kind", "process")),
                )

        # Insert disks if provided
        disks = body.get("disks", [])
        if disks:
            conn.execute(
                "DELETE FROM device_disks WHERE device_id=?", (device_id,))
            for dk in disks:
                conn.execute(
                    """INSERT INTO device_disks
                       (snapshot_ts, device_id, mount, label, used_gb, total_gb, pct)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (ts, device_id, dk.get("mount", "?"),
                     dk.get("label", dk.get("mount", "?")),
                     dk.get("used_gb", 0), dk.get("total_gb", 0),
                     dk.get("pct", 0)),
                )

        # Insert network counters if provided
        net_counters = body.get("net_counters", [])
        for nc in net_counters:
            conn.execute(
                """INSERT INTO device_network
                   (timestamp, device_id, interface, rx_bytes, tx_bytes)
                   VALUES (?, ?, ?, ?, ?)""",
                (ts, device_id, nc.get("interface", "?"),
                 nc.get("rx_bytes", 0), nc.get("tx_bytes", 0)),
            )

        conn.commit()
        conn.close()

        # Invalidate caches so next fetch picks up new data
        with _cache_lock:
            _cache.pop("compute", None)
            _cache.pop("network", None)

        return JSONResponse({"status": "ok", "device_id": device_id})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Log viewer API
# ---------------------------------------------------------------------------

@app.get("/api/logs/{device_id}")
async def get_logs(
    device_id: str,
    lines: int = Query(100, ge=1, le=500),
    service: str | None = Query(None),
    since: str | None = Query(None),
):
    dev = _find_device(device_id)
    if not dev:
        return JSONResponse({"error": "Unknown device"}, status_code=404)
    if not dev.get("ip"):
        return JSONResponse({"error": "Device has no IP"}, status_code=400)

    def _fetch():
        parts = ["journalctl", "--no-pager", "-o", "short-iso"]
        if service:
            parts += ["-u", service]
        if since:
            parts += ["--since", since]
            parts += ["-n", str(lines)]
        else:
            parts += ["-n", str(lines)]
        cmd = " ".join(parts)
        return _local_or_ssh_cmd(dev, cmd, timeout=15)

    try:
        raw = await asyncio.to_thread(_fetch)
    except Exception:
        raw = None

    if raw is None:
        return JSONResponse({"error": "Device unreachable"}, status_code=502)

    result_lines = [l for l in raw.splitlines() if l.strip()]
    last_ts = ""
    if result_lines:
        # short-iso format: "2026-02-28T10:15:30-0800 hostname ..."
        first_token = result_lines[-1].split(" ", 1)[0]
        if "T" in first_token:
            last_ts = first_token

    return {"device_id": device_id, "lines": result_lines, "last_timestamp": last_ts, "count": len(result_lines)}


@app.get("/api/logs/{device_id}/services")
async def get_log_services(device_id: str):
    dev = _find_device(device_id)
    if not dev:
        return JSONResponse({"error": "Unknown device"}, status_code=404)
    if not dev.get("ip"):
        return JSONResponse({"error": "Device has no IP"}, status_code=400)

    def _fetch():
        cmd = "systemctl list-units --type=service --state=running --no-pager --plain --no-legend"
        raw = _local_or_ssh_cmd(dev, cmd, timeout=10)
        if not raw:
            return []
        services = []
        for line in raw.strip().splitlines():
            parts = line.split()
            if parts:
                name = parts[0]
                # Strip .service suffix for display
                if name.endswith(".service"):
                    name = name[:-8]
                services.append(name)
        return sorted(services)

    cache_key = f"log_services_{device_id}"
    try:
        services = await asyncio.to_thread(lambda: cached(cache_key, _fetch, ttl=60))
    except Exception:
        return JSONResponse({"error": "Device unreachable"}, status_code=502)

    return {"device_id": device_id, "services": services}


@app.get("/VERSION", response_class=PlainTextResponse)
async def version_file():
    return APP_VERSION


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML.replace("__VERSION__", APP_VERSION)


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ramp — Factory Floor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace;
    background: #06060f;
    color: #e0e0e0;
    padding: 16px 20px;
    min-height: 100vh;
}

/* Header */
.header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 12px;
    border-bottom: 1px solid #1a1a3a;
    padding-bottom: 10px;
}
h1 { color: #7c8aff; font-size: 1.4em; letter-spacing: -0.5px; }
h1 span { color: #ff6b6b; font-weight: 400; }
.refresh-info { color: #444; font-size: 0.7em; }

/* Fleet summary bar */
.fleet-summary {
    display: flex;
    align-items: center;
    gap: 24px;
    padding: 10px 16px;
    background: linear-gradient(90deg, #0e0e22 0%, #141430 100%);
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    margin-bottom: 16px;
    font-size: 0.8em;
    flex-wrap: wrap;
}
.fleet-summary .fleet-label {
    color: #5c6bc0;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 0.85em;
}
.fleet-summary .fleet-stat {
    color: #888;
}
.fleet-summary .fleet-stat .val {
    font-weight: 700;
    margin-right: 2px;
}

/* Fleet grid — 3 columns */
.fleet-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 20px;
}
@media (max-width: 1100px) {
    .fleet-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 700px) {
    .fleet-grid { grid-template-columns: 1fr; }
}

/* Device card */
.dev-card {
    background: linear-gradient(135deg, #0e0e22 0%, #141430 100%);
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 16px 18px;
    min-height: 340px;
    display: flex;
    flex-direction: column;
}
.dev-card.offline {
    border-style: dashed;
    opacity: 0.35;
    min-height: 160px;
}
.dev-card.vm-card {
    min-height: 160px;
}
.dev-card .dev-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
}
.dev-card .dev-alias {
    font-size: 1em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.dev-card .dev-type {
    font-size: 0.7em;
    color: #666;
    margin-bottom: 14px;
}
.online-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
}
.online-dot.on {
    background: #4ade80;
    box-shadow: 0 0 8px #4ade80;
}
.online-dot.off { background: #555; }

/* Utilization bars */
.util-bars {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-bottom: 12px;
}
.util-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.78em;
}
.util-label {
    width: 32px;
    color: #888;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.85em;
}
.util-track {
    flex: 1;
    height: 16px;
    background: rgba(0,0,0,0.4);
    border-radius: 3px;
    overflow: hidden;
    position: relative;
}
.util-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
}
.util-pct {
    width: 38px;
    text-align: right;
    font-weight: 700;
    font-size: 0.9em;
}

/* Device chart */
.dev-chart-wrap {
    flex: 1;
    min-height: 100px;
    max-height: 150px;
    position: relative;
    margin-bottom: 10px;
}

/* Device stats footer */
.dev-stats {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    font-size: 0.72em;
    color: #666;
    margin-bottom: 8px;
}
.dev-stats span { white-space: nowrap; }

/* Disk bar (compact) */
.disk-row {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.7em;
    color: #666;
    margin-bottom: 4px;
}
.disk-row span:first-child {
    min-width: 70px;
    white-space: nowrap;
}
.disk-row span:last-child {
    min-width: 95px;
    text-align: right;
    white-space: nowrap;
}
.disk-track {
    flex: 1;
    height: 6px;
    background: rgba(0,0,0,0.4);
    border-radius: 3px;
    overflow: hidden;
}
.disk-fill {
    height: 100%;
    background: #5c6bc0;
    border-radius: 3px;
}

/* Services section */
.dev-services {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    font-size: 0.7em;
}
.svc-pill {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    background: rgba(74,222,128,0.08);
    border: 1px solid rgba(74,222,128,0.2);
    border-radius: 10px;
    color: #4ade80;
}
.svc-pill.docker {
    background: rgba(96,165,250,0.08);
    border-color: rgba(96,165,250,0.2);
    color: #60a5fa;
}
.svc-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: currentColor;
}

/* Collapsible sections */
.collapsible-header {
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    user-select: none;
    color: #5c6bc0;
    font-size: 1.05em;
    margin: 24px 0 10px;
    border-bottom: 1px solid #1a1a3a;
    padding-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
}
.collapsible-header:hover { color: #7c8aff; }
.collapsible-header .arrow {
    transition: transform 0.2s;
    font-size: 0.8em;
}
.collapsible-header.collapsed .arrow {
    transform: rotate(-90deg);
}
.collapsible-body {
    overflow: hidden;
    transition: max-height 0.3s ease;
}
.collapsible-body.hidden {
    max-height: 0 !important;
    overflow: hidden;
}

/* Acceleration cards (kept from original, slightly compressed) */
.accel-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 12px;
    margin-bottom: 8px;
}
.accel-card {
    background: linear-gradient(135deg, #0e0e22 0%, #141430 100%);
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 16px;
    position: relative;
    overflow: hidden;
}
.accel-card.not-configured {
    opacity: 0.4;
    border-style: dashed;
}
.accel-card .provider {
    font-size: 0.72em;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #888;
    margin-bottom: 10px;
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
    gap: 12px;
    flex-wrap: wrap;
}
.metric-box {
    flex: 1;
    min-width: 75px;
    text-align: center;
    padding: 8px 4px;
    background: rgba(0,0,0,0.3);
    border-radius: 6px;
}
.metric-box .label {
    font-size: 0.6em;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #666;
    margin-bottom: 3px;
}
.metric-box .value {
    font-size: 1.3em;
    font-weight: 700;
}
.metric-box .unit {
    font-size: 0.55em;
    color: #555;
}
.positive { color: #4ade80; }
.negative { color: #f87171; }
.neutral { color: #7c8aff; }
.metric-box.accel-emphasis {
    background: rgba(124, 138, 255, 0.08);
    border: 1px solid rgba(124, 138, 255, 0.2);
}
.metric-box.accel-emphasis .value { font-size: 1.5em; }

/* Chart section */
.chart-section {
    background: #0e0e22;
    border: 1px solid #1a1a3a;
    border-radius: 10px;
    padding: 16px;
    margin: 12px 0;
}
.chart-container { position: relative; height: 220px; }

/* Table */
.accel-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.78em;
    margin-top: 10px;
}
.accel-table th {
    text-align: left;
    padding: 6px 8px;
    background: #12122a;
    color: #5c6bc0;
    border-bottom: 2px solid #2a2a5a;
    font-weight: 600;
}
.accel-table td {
    padding: 5px 8px;
    border-bottom: 1px solid #1a1a2a;
}
.accel-table tr:hover { background: #121230; }
.accel-table .num { text-align: right; font-variant-numeric: tabular-nums; }

/* Error / hints */
.error-msg { color: #f87171; font-size: 0.8em; padding: 8px; }
.config-hint { color: #666; font-size: 0.75em; font-style: italic; padding: 6px 0; }

/* Status pills */
.status-bar {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 8px;
    margin-bottom: 4px;
}
.status-pill {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 3px 8px;
    background: #12122a;
    border: 1px solid #2a2a4a;
    border-radius: 20px;
    font-size: 0.7em;
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

/* Network traffic section */
.net-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 14px;
    margin-bottom: 8px;
}
.net-card {
    background: linear-gradient(135deg, #0e0e22 0%, #141430 100%);
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 16px 18px;
}
.net-card.net-total {
    border: 2px solid rgba(255,255,255,0.25);
    background: linear-gradient(135deg, #12122e 0%, #1a1a3e 100%);
}
.net-card .net-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
}
.net-card .net-alias {
    font-size: 1em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.net-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.82em;
    padding: 3px 0;
}
.net-stat .net-label { color: #888; }
.net-stat .net-val { font-weight: 700; font-variant-numeric: tabular-nums; }
.net-rate {
    font-size: 0.75em;
    color: #666;
    margin-top: 6px;
    padding-top: 6px;
    border-top: 1px solid #1a1a3a;
}
.net-chart-wrap {
    height: 120px;
    position: relative;
    margin-top: 10px;
}
.net-today-badge {
    font-size: 0.82em;
    color: #888;
    margin-left: auto;
    font-weight: 400;
}
.net-today-badge .val { font-weight: 700; color: #e0e0e0; }

/* Log viewer */
.log-overlay {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.6); z-index: 9000;
    display: none; opacity: 0; transition: opacity 0.2s;
}
.log-overlay.open { display: block; opacity: 1; }
.log-panel {
    position: fixed; top: 0; right: 0; bottom: 0;
    width: min(70%, 900px); background: #0a0a1a;
    border-left: 1px solid #222; z-index: 9001;
    display: flex; flex-direction: column;
    transform: translateX(100%); transition: transform 0.25s ease;
}
.log-overlay.open .log-panel { transform: translateX(0); }
.log-toolbar {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 14px; background: #111128;
    border-bottom: 1px solid #222; flex-shrink: 0;
}
.log-toolbar .log-title {
    font-weight: 700; font-size: 0.95em; margin-right: auto;
}
.log-toolbar select, .log-toolbar button {
    background: #1a1a3a; color: #ccc; border: 1px solid #333;
    border-radius: 4px; padding: 4px 8px; font-size: 0.8em;
    cursor: pointer;
}
.log-toolbar select:hover, .log-toolbar button:hover { border-color: #555; }
.log-toolbar button.active { background: #2a2a5a; color: #fff; }
.log-body {
    flex: 1; overflow-y: auto; padding: 10px 14px;
    font-family: 'SF Mono', Consolas, 'Courier New', monospace;
    font-size: 0.78em; line-height: 1.5; color: #b0b0b0;
    white-space: pre-wrap; word-break: break-all;
}
.log-body .log-line { padding: 1px 0; }
.log-body .log-line:hover { background: rgba(255,255,255,0.03); }
.log-status {
    padding: 6px 14px; background: #111128;
    border-top: 1px solid #222; font-size: 0.75em;
    color: #555; flex-shrink: 0;
    display: flex; justify-content: space-between;
}
.log-btn-icon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 24px; height: 24px; border-radius: 4px;
    background: transparent; border: 1px solid transparent;
    color: #555; cursor: pointer; font-size: 0.85em;
    transition: all 0.15s; margin-left: auto;
}
.log-btn-icon:hover { background: rgba(255,255,255,0.08); color: #aaa; border-color: #333; }
</style>
</head>
<body>

<div class="header">
    <div>
        <h1>Ramp <span>// acceleration</span></h1>
    </div>
    <div class="refresh-info" id="refresh-info">Loading...</div>
</div>

<div id="fleet-summary"></div>
<div id="fleet-grid"></div>
<div id="network-section"></div>
<div id="api-section"></div>

<!-- Log viewer panel -->
<div class="log-overlay" id="log-overlay" onclick="if(event.target===this)closeLogViewer()">
    <div class="log-panel">
        <div class="log-toolbar">
            <span class="log-title" id="log-title">Logs</span>
            <select id="log-service" onchange="onLogServiceChange()">
                <option value="">All services</option>
            </select>
            <select id="log-lines" onchange="onLogLinesChange()">
                <option value="50">50</option>
                <option value="100" selected>100</option>
                <option value="200">200</option>
                <option value="500">500</option>
            </select>
            <button id="log-pause-btn" onclick="toggleLogPause()">Pause</button>
            <button onclick="closeLogViewer()">Close</button>
        </div>
        <div class="log-body" id="log-body"></div>
        <div class="log-status">
            <span id="log-status-left">Ready</span>
            <span id="log-status-right"></span>
        </div>
    </div>
</div>

<script>
let charts = {};
let computeCharts = {};
let networkCharts = {};
let _lastApiData = null;
let _lastComputeData = null;
let _lastNetworkData = null;
let _apiSectionCollapsed = true;
let _networkSectionCollapsed = false;

// -----------------------------------------------------------------------
// Formatters
// -----------------------------------------------------------------------

function fmtNum(n) {
    if (typeof n !== 'number' || isNaN(n)) return '\u2014';
    if (Math.abs(n) >= 1e9) return (n/1e9).toFixed(1) + 'B';
    if (Math.abs(n) >= 1e6) return (n/1e6).toFixed(1) + 'M';
    if (Math.abs(n) >= 1e3) return (n/1e3).toFixed(1) + 'K';
    return n.toLocaleString();
}
function fmtDollar(n) {
    if (typeof n !== 'number' || isNaN(n)) return '\u2014';
    return '$' + n.toFixed(2);
}
function signClass(n) { return n > 0 ? 'positive' : n < 0 ? 'negative' : 'neutral'; }
function signPrefix(n) { return n > 0 ? '+' : ''; }
function providerColor(name) {
    return {anthropic:'#d4a574', openai:'#10b981', xai:'#60a5fa', gcs:'#a78bfa'}[name] || '#888';
}
function fmtPct(v) { return v != null ? v.toFixed(0) + '%' : '\u2014'; }
function fmtUptime(secs) {
    if (!secs) return '\u2014';
    const d = Math.floor(secs / 86400);
    const h = Math.floor((secs % 86400) / 3600);
    const m = Math.floor((secs % 3600) / 60);
    if (d > 0) return d + 'd ' + h + 'h';
    if (h > 0) return h + 'h ' + m + 'm';
    return m + 'm';
}

function fmtBytes(b) {
    if (typeof b !== 'number' || isNaN(b) || b === 0) return '0 B';
    if (b >= 1e12) return (b / 1e12).toFixed(1) + ' TB';
    if (b >= 1e9) return (b / 1e9).toFixed(1) + ' GB';
    if (b >= 1e6) return (b / 1e6).toFixed(0) + ' MB';
    if (b >= 1e3) return (b / 1e3).toFixed(0) + ' KB';
    return b + ' B';
}
function fmtRate(bps) {
    if (typeof bps !== 'number' || isNaN(bps) || bps === 0) return '0 B/s';
    if (bps >= 1e9) return (bps / 1e9).toFixed(1) + ' GB/s';
    if (bps >= 1e6) return (bps / 1e6).toFixed(1) + ' MB/s';
    if (bps >= 1e3) return (bps / 1e3).toFixed(0) + ' KB/s';
    return bps.toFixed(0) + ' B/s';
}

function utilColor(pct) {
    if (pct == null) return '#555';
    if (pct < 50) return '#4ade80';
    if (pct < 80) return '#fbbf24';
    return '#f87171';
}

// -----------------------------------------------------------------------
// Fleet summary bar
// -----------------------------------------------------------------------

function renderFleetSummary(devices, gcsVms) {
    const el = document.getElementById('fleet-summary');
    const online = devices.filter(d => d.online);
    const total = devices.length;
    const avgCpu = online.length ? (online.reduce((s, d) => s + (d.current?.cpu_pct || 0), 0) / online.length).toFixed(0) : 0;
    const avgGpu = online.length ? (online.reduce((s, d) => s + (d.current?.gpu_pct || 0), 0) / online.length).toFixed(0) : 0;
    const totalPower = online.reduce((s, d) => s + (d.current?.power_mw || 0), 0);
    const powerStr = totalPower ? (totalPower / 1000).toFixed(1) + 'W' : '\u2014';
    const vmCount = (gcsVms || []).filter(v => v.status === 'RUNNING').length;

    el.innerHTML = `<div class="fleet-summary">
        <span class="fleet-label">Fleet</span>
        <span class="fleet-stat"><span class="val" style="color:${online.length > 0 ? '#4ade80' : '#f87171'}">${online.length}/${total}</span> online</span>
        <span class="fleet-stat">Avg CPU <span class="val" style="color:${utilColor(+avgCpu)}">${avgCpu}%</span></span>
        <span class="fleet-stat">Avg GPU <span class="val" style="color:${utilColor(+avgGpu)}">${avgGpu}%</span></span>
        <span class="fleet-stat">Total Power <span class="val">${powerStr}</span></span>
        ${vmCount ? `<span class="fleet-stat"><span class="val" style="color:#a78bfa">${vmCount}</span> VMs</span>` : ''}
    </div>`;
}

// -----------------------------------------------------------------------
// Utilization bar
// -----------------------------------------------------------------------

function renderUtilBar(label, pct) {
    const color = utilColor(pct);
    const w = pct != null ? Math.min(pct, 100) : 0;
    return `<div class="util-row">
        <span class="util-label">${label}</span>
        <div class="util-track">
            <div class="util-fill" style="width:${w}%;background:${color}"></div>
        </div>
        <span class="util-pct" style="color:${color}">${fmtPct(pct)}</span>
    </div>`;
}

// -----------------------------------------------------------------------
// Device card
// -----------------------------------------------------------------------

function renderDeviceCard(dev) {
    const subtitle = `${dev.label}${dev.ip ? ` · ${dev.ip}` : ''}`;
    if (!dev.online || !dev.current) {
        return `<div class="dev-card offline">
            <div class="dev-header">
                <span class="online-dot off"></span>
                <span class="dev-alias" style="color:${dev.color}">${dev.alias || dev.label}</span>
            </div>
            <div class="dev-type">${subtitle}</div>
            <div style="color:#444;font-size:0.85em;margin-top:16px;text-align:center">OFFLINE</div>
        </div>`;
    }

    const c = dev.current;
    const chartId = 'chart-' + dev.id;

    // Services from processes
    const procs = dev.processes || [];
    const dockerProcs = procs.filter(p => p.kind === 'docker');
    const topProcs = procs.filter(p => p.kind === 'process').slice(0, 6);

    let servicesHtml = '';
    if (dockerProcs.length || topProcs.length) {
        const pills = [];
        for (const d of dockerProcs) {
            pills.push(`<span class="svc-pill docker"><span class="svc-dot"></span>${d.name}</span>`);
        }
        for (const p of topProcs.slice(0, 4)) {
            pills.push(`<span class="svc-pill"><span class="svc-dot"></span>${p.name}</span>`);
        }
        servicesHtml = `<div class="dev-services">${pills.join('')}</div>`;
    }

    const logBtn = dev.ip ? `<span class="log-btn-icon" title="View logs" onclick="openLogViewer('${dev.id}','${dev.alias||dev.label}','${dev.color}')">&#9776;</span>` : '';

    return `<div class="dev-card">
        <div class="dev-header">
            <span class="online-dot on"></span>
            <span class="dev-alias" style="color:${dev.color}">${dev.alias || dev.label}</span>
            ${logBtn}
        </div>
        <div class="dev-type">${subtitle}</div>

        <div class="util-bars">
            ${renderUtilBar('CPU', c.cpu_pct)}
            ${renderUtilBar('GPU', c.gpu_pct)}
            ${renderUtilBar('RAM', c.ram_pct)}
        </div>

        <div class="dev-chart-wrap"><canvas id="${chartId}"></canvas></div>

        <div class="dev-stats">
            ${c.cpu_temp != null ? `<span>${c.cpu_temp.toFixed(0)}\u00B0C cpu</span>` : ''}
            ${c.gpu_temp != null ? `<span>${c.gpu_temp.toFixed(0)}\u00B0C gpu</span>` : ''}
            ${c.power_mw != null ? `<span>${(c.power_mw/1000).toFixed(1)}W</span>` : ''}
            <span>up ${fmtUptime(c.uptime_secs)}</span>
        </div>

        ${(dev.disks && dev.disks.length) ? dev.disks.map(dk => `<div class="disk-row">
            <span>${dk.label}</span>
            <div class="disk-track"><div class="disk-fill" style="width:${dk.pct || 0}%;background:${dk.pct > 80 ? '#f87171' : dk.pct > 60 ? '#fbbf24' : '#5c6bc0'}"></div></div>
            <span>${dk.used_gb}/${dk.total_gb} GB</span>
        </div>`).join('') : (c.disk_used_gb != null ? `<div class="disk-row">
            <span>Disk</span>
            <div class="disk-track"><div class="disk-fill" style="width:${c.disk_pct || 0}%"></div></div>
            <span>${c.disk_used_gb}/${c.disk_total_gb} GB</span>
        </div>` : '')}

        ${servicesHtml}
    </div>`;
}

// -----------------------------------------------------------------------
// VM card
// -----------------------------------------------------------------------

function renderVmCard(gcsVms) {
    const vms = (gcsVms || []);
    const running = vms.filter(v => v.status === 'RUNNING');

    let vmList = '';
    if (running.length) {
        vmList = running.map(v =>
            `<div style="display:flex;align-items:center;gap:6px;font-size:0.78em;margin:3px 0">
                <span class="online-dot on" style="width:6px;height:6px"></span>
                <span style="color:#e0e0e0">${v.name}</span>
                <span style="color:#555;font-size:0.85em">(${v.type})</span>
            </div>`
        ).join('');
    }

    const stopped = vms.filter(v => v.status !== 'RUNNING');
    if (stopped.length) {
        vmList += stopped.map(v =>
            `<div style="display:flex;align-items:center;gap:6px;font-size:0.78em;margin:3px 0;opacity:0.4">
                <span class="online-dot off" style="width:6px;height:6px"></span>
                <span>${v.name}</span>
            </div>`
        ).join('');
    }

    return `<div class="dev-card vm-card">
        <div class="dev-header">
            <span style="color:#a78bfa;font-size:1.1em">&#9729;</span>
            <span class="dev-alias" style="color:#a78bfa">Cloud VMs</span>
        </div>
        <div class="dev-type">GCP Compute Engine</div>
        ${running.length ? `<div style="color:#a78bfa;font-weight:700;font-size:0.85em;margin:8px 0">${running.length} running</div>` :
            `<div style="color:#444;font-size:0.85em;margin:8px 0">No VMs running</div>`}
        ${vmList}
    </div>`;
}

// -----------------------------------------------------------------------
// Device chart (24h CPU/GPU/RAM)
// -----------------------------------------------------------------------

function renderDeviceChart(canvasId, history, dev) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !history || history.length < 2) return;

    const labels = history.map(h => {
        const d = new Date(h.ts * 1000);
        return d.getHours() + ':' + String(d.getMinutes()).padStart(2, '0');
    });

    const datasets = [{
        label: 'CPU',
        data: history.map(h => h.cpu),
        borderColor: dev.color,
        backgroundColor: dev.color + '18',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
        tension: 0.3,
    }];

    if (history.some(h => h.gpu != null)) {
        datasets.push({
            label: 'GPU',
            data: history.map(h => h.gpu),
            borderColor: '#f87171',
            backgroundColor: '#f8717112',
            borderWidth: 1.5,
            pointRadius: 0,
            fill: true,
            tension: 0.3,
        });
    }
    if (history.some(h => h.ram != null)) {
        datasets.push({
            label: 'RAM',
            data: history.map(h => h.ram),
            borderColor: '#a78bfa',
            backgroundColor: '#a78bfa12',
            borderWidth: 1,
            pointRadius: 0,
            fill: false,
            tension: 0.3,
            borderDash: [4, 3],
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
                legend: {
                    display: true,
                    position: 'top',
                    labels: { color: '#666', font: { size: 9 }, boxWidth: 12, padding: 6 },
                },
                tooltip: {
                    backgroundColor: '#1a1a3a',
                    titleColor: '#e0e0e0',
                    bodyColor: '#bbb',
                    borderColor: '#2a2a4a',
                    borderWidth: 1,
                    titleFont: { size: 10 },
                    bodyFont: { size: 10 },
                },
            },
            scales: {
                x: {
                    display: true,
                    ticks: { color: '#444', font: { size: 8 }, maxTicksLimit: 6 },
                    grid: { display: false },
                },
                y: { display: false, min: 0, max: 100 },
            },
            animation: false,
        },
    });
}

// -----------------------------------------------------------------------
// Fleet render
// -----------------------------------------------------------------------

function renderComputeFleet(computeData, apiData) {
    const devices = computeData?.devices || [];
    const gcsVms = apiData?.gcs_pipeline?.all_vms || [];

    // Fleet summary
    renderFleetSummary(devices, gcsVms);

    // Build cards
    const cards = devices.map(dev => renderDeviceCard(dev));
    cards.push(renderVmCard(gcsVms));

    // Destroy old chart instances before replacing DOM
    Object.keys(computeCharts).forEach(k => {
        computeCharts[k].destroy();
        delete computeCharts[k];
    });

    document.getElementById('fleet-grid').innerHTML =
        `<div class="fleet-grid">${cards.join('')}</div>`;

    // Render charts after DOM insert
    requestAnimationFrame(() => {
        devices.forEach(dev => {
            if (dev.online && dev.history && dev.history.length > 1) {
                renderDeviceChart('chart-' + dev.id, dev.history, dev);
            }
        });
    });
}

// -----------------------------------------------------------------------
// API / Acceleration section (collapsible)
// -----------------------------------------------------------------------

function renderAccelCard(name, label, data, isCost) {
    if (!data || !data.configured) {
        return `<div class="accel-card not-configured">
            <div class="provider"><span class="dot" style="background:${providerColor(name)}"></span>${label}</div>
            <div class="config-hint">Not configured</div>
        </div>`;
    }
    if (data.error && !data.token_metrics) {
        return `<div class="accel-card">
            <div class="provider"><span class="dot" style="background:${providerColor(name)}"></span>${label}</div>
            <div class="error-msg">${data.error}</div>
        </div>`;
    }
    const tm = data.token_metrics || {position:0, velocity:0, acceleration:0};
    const cm = data.cost_metrics || {position:0, velocity:0, acceleration:0};
    const m = isCost ? cm : tm;
    const fmt = isCost ? fmtDollar : fmtNum;
    return `<div class="accel-card">
        <div class="provider"><span class="dot" style="background:${providerColor(name)}"></span>${label}</div>
        <div class="metrics-row">
            <div class="metric-box"><div class="label">Position</div><div class="value neutral">${fmt(m.position)}</div><div class="unit">${isCost ? 'total spend' : 'total tokens'}</div></div>
            <div class="metric-box"><div class="label">Velocity</div><div class="value ${signClass(m.velocity)}">${signPrefix(m.velocity)}${fmt(m.velocity)}</div><div class="unit">7d avg/day</div></div>
            <div class="metric-box accel-emphasis"><div class="label">Accel</div><div class="value ${signClass(m.acceleration)}">${signPrefix(m.acceleration)}${isCost ? fmtDollar(m.acceleration) : fmtNum(m.acceleration)}</div><div class="unit">5d &Delta;v</div></div>
        </div>
    </div>`;
}

function renderGcsCard(gcs) {
    if (!gcs || !gcs.metrics) {
        return `<div class="accel-card not-configured">
            <div class="provider"><span class="dot" style="background:${providerColor('gcs')}"></span>Training Pipeline</div>
            <div class="config-hint">No data</div>
        </div>`;
    }
    const m = gcs.metrics;
    return `<div class="accel-card">
        <div class="provider"><span class="dot" style="background:${providerColor('gcs')}"></span>Training Pipeline</div>
        <div class="metrics-row">
            <div class="metric-box"><div class="label">Position</div><div class="value neutral">${fmtNum(m.position)}</div><div class="unit">sessions</div></div>
            <div class="metric-box"><div class="label">Velocity</div><div class="value ${signClass(m.velocity)}">${signPrefix(m.velocity)}${fmtNum(m.velocity)}</div><div class="unit">7d avg/day</div></div>
            <div class="metric-box accel-emphasis"><div class="label">Accel</div><div class="value ${signClass(m.acceleration)}">${signPrefix(m.acceleration)}${fmtNum(m.acceleration)}</div><div class="unit">5d &Delta;v</div></div>
        </div>
    </div>`;
}

function renderAccelTable(data) {
    const rows = [];
    for (const [name, label] of [['anthropic','Anthropic'],['openai','OpenAI'],['xai','xAI/Grok']]) {
        const p = data.providers[name];
        if (!p || !p.configured || !p.token_metrics) continue;
        rows.push({name: label, color: providerColor(name), tm: p.token_metrics, cm: p.cost_metrics || {position:0,velocity:0,acceleration:0}});
    }
    if (data.gcs_pipeline?.metrics) {
        rows.push({name:'GCS Pipeline', color: providerColor('gcs'), tm: data.gcs_pipeline.metrics, cm:{position:0,velocity:0,acceleration:0}});
    }
    if (!rows.length) return '';

    let dailyHtml = '';
    for (const r of rows) {
        const last5 = (r.tm.daily || []).slice(-5);
        if (!last5.length) continue;
        dailyHtml += `<tr style="border-top:2px solid #2a2a4a"><td colspan="5" style="color:${r.color};font-weight:bold;padding-top:8px">${r.name}</td></tr>`;
        for (const d of last5) {
            dailyHtml += `<tr><td>${d.date}</td><td class="num">${fmtNum(d.value)}</td><td class="num">${fmtNum(d.cumulative)}</td><td class="num ${signClass(d.velocity)}">${signPrefix(d.velocity)}${fmtNum(d.velocity)}</td><td class="num ${signClass(d.acceleration)}">${signPrefix(d.acceleration)}${fmtNum(d.acceleration)}</td></tr>`;
        }
    }
    return `<table class="accel-table"><thead><tr><th>Date</th><th class="num">Value</th><th class="num">Cumul</th><th class="num">Vel</th><th class="num">Accel</th></tr></thead><tbody>${dailyHtml}</tbody></table>`;
}

function renderChart(data) {
    const colors = {anthropic:'#d4a574', openai:'#10b981', xai:'#60a5fa', gcs:'#a78bfa'};
    let allDates = new Set();
    for (const [name, p] of Object.entries(data.providers)) {
        if (p.token_metrics?.daily) p.token_metrics.daily.forEach(d => allDates.add(d.date));
    }
    if (data.gcs_pipeline?.metrics?.daily) data.gcs_pipeline.metrics.daily.forEach(d => allDates.add(d.date));
    allDates = [...allDates].sort();
    if (!allDates.length) return '';
    return `<div class="chart-section">
        <div style="font-size:0.8em;color:#5c6bc0;font-weight:600;margin-bottom:10px">ACCELERATION OVER TIME</div>
        <div class="chart-container"><canvas id="accelChart"></canvas></div>
    </div>`;
}

function updateAccelChart(data) {
    const canvas = document.getElementById('accelChart');
    if (!canvas) return;
    const colors = {anthropic:'#d4a574', openai:'#10b981', xai:'#60a5fa', gcs:'#a78bfa'};
    let allDates = new Set();
    const sources = [];
    for (const [name, label] of [['anthropic','Anthropic'],['openai','OpenAI'],['xai','xAI']]) {
        const p = data.providers[name];
        if (!p?.token_metrics?.daily) continue;
        p.token_metrics.daily.forEach(d => allDates.add(d.date));
        sources.push({name, label, daily: p.token_metrics.daily, color: colors[name]});
    }
    if (data.gcs_pipeline?.metrics?.daily) {
        data.gcs_pipeline.metrics.daily.forEach(d => allDates.add(d.date));
        sources.push({name:'gcs', label:'GCS', daily: data.gcs_pipeline.metrics.daily, color: colors.gcs});
    }
    allDates = [...allDates].sort();
    if (!allDates.length) return;

    const datasets = sources.map(s => {
        const dateMap = {};
        s.daily.forEach(d => dateMap[d.date] = d.acceleration);
        return {
            label: s.label, data: allDates.map(d => dateMap[d] || 0),
            borderColor: s.color, backgroundColor: s.color + '22',
            borderWidth: 2, pointRadius: 3, pointBackgroundColor: s.color,
            fill: true, tension: 0.3,
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
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { labels: { color: '#888', font: { family: "'SF Mono', monospace", size: 10 } } },
                tooltip: { backgroundColor: '#1a1a3a', titleColor: '#e0e0e0', bodyColor: '#bbb', borderColor: '#2a2a4a', borderWidth: 1 },
            },
            scales: {
                x: { ticks: { color: '#555', font: { size: 9 } }, grid: { color: 'rgba(42,42,74,0.3)' } },
                y: { title: { display: true, text: 'Accel (\u0394v)', color: '#5c6bc0', font: { size: 10 } }, ticks: { color: '#666', font: { size: 9 } }, grid: { color: 'rgba(42,42,74,0.3)' } },
            },
        },
    });
}

function renderStatusBar(data) {
    const pills = [];
    for (const [name, label] of [['anthropic','Anthropic'],['openai','OpenAI'],['xai','xAI/Grok']]) {
        const p = data.providers[name];
        const dotClass = !p || !p.configured ? 'dot-gray' : p.error ? 'dot-red' : 'dot-green';
        pills.push(`<div class="status-pill"><span class="dot ${dotClass}"></span>${label}</div>`);
    }
    const gcsOk = data.gcs_pipeline?.metrics?.position > 0;
    pills.push(`<div class="status-pill"><span class="dot ${gcsOk ? 'dot-green' : 'dot-yellow'}"></span>GCS</div>`);
    return `<div class="status-bar">${pills.join('')}</div>`;
}

function renderApiSection(data) {
    const el = document.getElementById('api-section');
    const collapsed = _apiSectionCollapsed;

    el.innerHTML = `
        <div class="collapsible-header ${collapsed ? 'collapsed' : ''}" onclick="toggleApiSection()">
            <span class="arrow">\u25BE</span> API Usage & Acceleration
        </div>
        <div class="collapsible-body ${collapsed ? 'hidden' : ''}" id="api-body">
            ${renderStatusBar(data)}

            <div style="margin-top:12px;font-size:0.85em;color:#5c6bc0;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">Tokens</div>
            <div class="accel-grid">
                ${renderAccelCard('anthropic', 'Anthropic', data.providers.anthropic, false)}
                ${renderAccelCard('openai', 'OpenAI', data.providers.openai, false)}
                ${renderAccelCard('xai', 'xAI', data.providers.xai, false)}
                ${renderGcsCard(data.gcs_pipeline)}
            </div>

            <div style="margin-top:12px;font-size:0.85em;color:#5c6bc0;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">Spend</div>
            <div class="accel-grid">
                ${renderAccelCard('anthropic', 'Anthropic', data.providers.anthropic, true)}
                ${renderAccelCard('openai', 'OpenAI', data.providers.openai, true)}
                ${renderAccelCard('xai', 'xAI', data.providers.xai, true)}
            </div>

            ${renderChart(data)}
            ${renderAccelTable(data)}
        </div>
    `;

    if (!collapsed) {
        requestAnimationFrame(() => updateAccelChart(data));
    }
}

function toggleApiSection() {
    _apiSectionCollapsed = !_apiSectionCollapsed;
    if (_lastApiData) renderApiSection(_lastApiData);
}

// -----------------------------------------------------------------------
// Network traffic section
// -----------------------------------------------------------------------

function renderNetworkCard(dev) {
    const chartId = 'net-chart-' + dev.id;
    const todayTotal = dev.today_rx + dev.today_tx;
    const totalCls = dev.is_total ? ' net-total' : '';
    return `<div class="net-card${totalCls}">
        <div class="net-header">
            <span class="net-alias" style="color:${dev.color}">${dev.alias}</span>
        </div>
        <div class="net-stat">
            <span class="net-label">\u2193 Received</span>
            <span class="net-val" style="color:${dev.color}">${fmtBytes(dev.today_rx)}</span>
        </div>
        <div class="net-stat">
            <span class="net-label">\u2191 Sent</span>
            <span class="net-val" style="color:#f87171">${fmtBytes(dev.today_tx)}</span>
        </div>
        <div class="net-stat">
            <span class="net-label">Total</span>
            <span class="net-val">${fmtBytes(todayTotal)}</span>
        </div>
        <div class="net-rate">\u2193${fmtRate(dev.rate_rx)} / \u2191${fmtRate(dev.rate_tx)}</div>
        ${dev.daily && dev.daily.length > 1 ? `<div class="net-chart-wrap"><canvas id="${chartId}"></canvas></div>` : ''}
    </div>`;
}

function renderNetworkChart(canvasId, daily, dev) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !daily || daily.length < 2) return;

    const labels = daily.map(d => d.date.slice(5)); // MM-DD
    const rxData = daily.map(d => d.rx / 1e9); // GB
    const txData = daily.map(d => d.tx / 1e9);

    const datasets = [
        {
            label: 'RX',
            data: rxData,
            backgroundColor: dev.color + 'aa',
            borderColor: dev.color,
            borderWidth: 1,
        },
        {
            label: 'TX',
            data: txData,
            backgroundColor: '#f87171aa',
            borderColor: '#f87171',
            borderWidth: 1,
        },
    ];

    if (networkCharts[canvasId]) {
        networkCharts[canvasId].data.labels = labels;
        networkCharts[canvasId].data.datasets = datasets;
        networkCharts[canvasId].update('none');
        return;
    }

    networkCharts[canvasId] = new Chart(canvas, {
        type: 'bar',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: { color: '#666', font: { size: 9 }, boxWidth: 10, padding: 4 },
                },
                tooltip: {
                    backgroundColor: '#1a1a3a',
                    titleColor: '#e0e0e0',
                    bodyColor: '#bbb',
                    borderColor: '#2a2a4a',
                    borderWidth: 1,
                    callbacks: {
                        label: function(ctx) {
                            return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(2) + ' GB';
                        }
                    }
                },
            },
            scales: {
                x: {
                    stacked: true,
                    ticks: { color: '#444', font: { size: 8 }, maxRotation: 0 },
                    grid: { display: false },
                },
                y: {
                    stacked: true,
                    ticks: {
                        color: '#555',
                        font: { size: 8 },
                        callback: function(v) { return v.toFixed(1) + ' GB'; }
                    },
                    grid: { color: 'rgba(42,42,74,0.2)' },
                },
            },
            animation: false,
        },
    });
}

function renderNetworkSection(data) {
    const el = document.getElementById('network-section');
    const devices = data?.devices || [];
    if (!devices.length) {
        el.innerHTML = '';
        return;
    }

    const collapsed = _networkSectionCollapsed;

    // Fleet-wide today total
    const todayTotal = devices.reduce((s, d) => s + d.today_rx + d.today_tx, 0);

    // Destroy old charts before DOM replace
    Object.keys(networkCharts).forEach(k => {
        networkCharts[k].destroy();
        delete networkCharts[k];
    });

    const cards = devices.map(d => renderNetworkCard(d)).join('');

    el.innerHTML = `
        <div class="collapsible-header ${collapsed ? 'collapsed' : ''}" onclick="toggleNetworkSection()">
            <span class="arrow">\u25BE</span> Network Traffic
            <span class="net-today-badge">Today: <span class="val">${fmtBytes(todayTotal)}</span></span>
        </div>
        <div class="collapsible-body ${collapsed ? 'hidden' : ''}" id="network-body">
            <div class="net-grid">${cards}</div>
        </div>
    `;

    if (!collapsed) {
        requestAnimationFrame(() => {
            devices.forEach(dev => {
                if (dev.daily && dev.daily.length > 1) {
                    renderNetworkChart('net-chart-' + dev.id, dev.daily, dev);
                }
            });
        });
    }
}

function toggleNetworkSection() {
    _networkSectionCollapsed = !_networkSectionCollapsed;
    if (_lastNetworkData) renderNetworkSection(_lastNetworkData);
}

// -----------------------------------------------------------------------
// Log viewer
// -----------------------------------------------------------------------

let _logDeviceId = null;
let _logAlias = '';
let _logColor = '';
let _logInterval = null;
let _logPaused = false;
let _logLastTs = '';
let _logAutoScroll = true;

function escapeHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function openLogViewer(deviceId, alias, color) {
    _logDeviceId = deviceId;
    _logAlias = alias;
    _logColor = color;
    _logPaused = false;
    _logLastTs = '';
    _logAutoScroll = true;

    document.getElementById('log-title').innerHTML = `<span style="color:${color}">${alias}</span> logs`;
    document.getElementById('log-body').innerHTML = '';
    document.getElementById('log-pause-btn').textContent = 'Pause';
    document.getElementById('log-pause-btn').classList.remove('active');
    document.getElementById('log-status-left').textContent = 'Connecting...';
    document.getElementById('log-status-right').textContent = '';
    document.getElementById('log-service').innerHTML = '<option value="">All services</option>';

    const overlay = document.getElementById('log-overlay');
    overlay.classList.add('open');

    fetchLogServices(deviceId);
    fetchLogs(true);

    if (_logInterval) clearInterval(_logInterval);
    _logInterval = setInterval(() => {
        if (!_logPaused) fetchLogs(false);
    }, 3000);

    // Auto-scroll detection
    const body = document.getElementById('log-body');
    body.onscroll = () => {
        const atBottom = body.scrollHeight - body.scrollTop - body.clientHeight < 40;
        _logAutoScroll = atBottom;
    };
}

function closeLogViewer() {
    document.getElementById('log-overlay').classList.remove('open');
    if (_logInterval) { clearInterval(_logInterval); _logInterval = null; }
    _logDeviceId = null;
}

async function fetchLogs(isInitial) {
    if (!_logDeviceId) return;
    const service = document.getElementById('log-service').value;
    const lines = document.getElementById('log-lines').value;
    let url = `/api/logs/${_logDeviceId}?lines=${lines}`;
    if (service) url += `&service=${encodeURIComponent(service)}`;
    if (!isInitial && _logLastTs) url += `&since=${encodeURIComponent(_logLastTs)}`;

    try {
        const resp = await fetch(url);
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            document.getElementById('log-status-left').textContent = err.error || 'Error';
            return;
        }
        const data = await resp.json();
        const body = document.getElementById('log-body');

        if (isInitial) {
            body.innerHTML = data.lines.map(l => `<div class="log-line">${escapeHtml(l)}</div>`).join('');
        } else {
            // Append new lines (skip first if it overlaps with last)
            let newLines = data.lines;
            if (newLines.length === 0) {
                document.getElementById('log-status-left').textContent = `${body.children.length} lines · live`;
                return;
            }
            const html = newLines.map(l => `<div class="log-line">${escapeHtml(l)}</div>`).join('');
            body.insertAdjacentHTML('beforeend', html);
        }

        // Trim to 2000 lines
        while (body.children.length > 2000) body.removeChild(body.firstChild);

        if (data.last_timestamp) _logLastTs = data.last_timestamp;
        document.getElementById('log-status-left').textContent = `${body.children.length} lines · live`;
        document.getElementById('log-status-right').textContent = new Date().toLocaleTimeString();

        if (_logAutoScroll) body.scrollTop = body.scrollHeight;
    } catch (e) {
        document.getElementById('log-status-left').textContent = 'Fetch error';
    }
}

async function fetchLogServices(deviceId) {
    try {
        const resp = await fetch(`/api/logs/${deviceId}/services`);
        if (!resp.ok) return;
        const data = await resp.json();
        const sel = document.getElementById('log-service');
        for (const svc of data.services) {
            const opt = document.createElement('option');
            opt.value = svc;
            opt.textContent = svc;
            sel.appendChild(opt);
        }
    } catch (e) {}
}

function toggleLogPause() {
    _logPaused = !_logPaused;
    const btn = document.getElementById('log-pause-btn');
    btn.textContent = _logPaused ? 'Resume' : 'Pause';
    btn.classList.toggle('active', _logPaused);
    const status = document.getElementById('log-status-left');
    if (_logPaused) {
        status.textContent = status.textContent.replace('· live', '· paused');
    } else {
        status.textContent = status.textContent.replace('· paused', '· live');
    }
}

function onLogServiceChange() {
    _logLastTs = '';
    fetchLogs(true);
}

function onLogLinesChange() {
    _logLastTs = '';
    fetchLogs(true);
}

// Close on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && _logDeviceId) closeLogViewer();
});

// -----------------------------------------------------------------------
// Refresh loops
// -----------------------------------------------------------------------

async function fetchCompute() {
    try {
        const resp = await fetch('/api/compute');
        _lastComputeData = await resp.json();
        renderComputeFleet(_lastComputeData, _lastApiData);
        document.getElementById('refresh-info').textContent =
            'Fleet: ' + new Date().toLocaleTimeString() + ' (15s)';
    } catch (e) {
        console.error('Compute fetch error:', e);
    }
}

async function fetchApi() {
    try {
        const resp = await fetch('/api/data');
        _lastApiData = await resp.json();
        renderApiSection(_lastApiData);
        // Re-render fleet too (for VMs)
        if (_lastComputeData) renderComputeFleet(_lastComputeData, _lastApiData);
    } catch (e) {
        console.error('API fetch error:', e);
    }
}

async function fetchNetwork() {
    try {
        const resp = await fetch('/api/network');
        _lastNetworkData = await resp.json();
        renderNetworkSection(_lastNetworkData);
    } catch (e) {
        console.error('Network fetch error:', e);
    }
}

// Initial load
fetchCompute();
fetchApi();
fetchNetwork();

// Compute fleet: 15s refresh, API data: 60s refresh, Network: 30s refresh
setInterval(fetchCompute, 15000);
setInterval(fetchApi, 60000);
setInterval(fetchNetwork, 30000);
</script>
<div id="dsl-version" style="position:fixed;bottom:8px;right:8px;font-family:'SF Mono',Consolas,monospace;font-size:11px;color:rgba(255,255,255,0.35);background:rgba(0,0,0,0.25);padding:2px 8px;border-radius:4px;pointer-events:none;z-index:9998;letter-spacing:0.5px">__VERSION__</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ramp — Acceleration Dashboard")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    # Start remote network polling thread
    threading.Thread(target=_poll_remote_network, daemon=True).start()

    print(f"Ramp Dashboard: http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
