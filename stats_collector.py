#!/usr/bin/env python3
"""Ramp Stats Collector — collects hardware metrics and writes to SQLite or pushes to remote.

Supports:
  - Jetson (tegrastats) — auto-detected
  - Desktop NVIDIA GPU (nvidia-smi) — auto-detected
  - CPU-only systems — fallback

Environment variables:
  RAMP_DEVICE_ID   — device identifier (default: jetson-orin-nano)
  RAMP_PUSH_URL    — if set, POST data to this URL instead of writing local SQLite
  RAMP_DB_PATH     — SQLite path (default: /ssd/ramp/stats.db)
  RAMP_INTERVAL    — collection interval in seconds (default: 60)

Database: /ssd/ramp/stats.db (or RAMP_DB_PATH)
"""

import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import psutil

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEVICE_ID = os.environ.get("RAMP_DEVICE_ID", "jetson-orin-nano")
PUSH_URL = os.environ.get("RAMP_PUSH_URL", "")
DB_PATH = Path(os.environ.get("RAMP_DB_PATH", "/ssd/ramp/stats.db"))
INTERVAL_SECS = int(os.environ.get("RAMP_INTERVAL", "60"))
PRUNE_DAYS = 30

# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

def init_db(conn):
    conn.execute("PRAGMA journal_mode=WAL")
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
    """)
    conn.commit()

# ---------------------------------------------------------------------------
# GPU platform detection
# ---------------------------------------------------------------------------

_gpu_platform = None  # "tegrastats", "nvidia-smi", or None

def detect_gpu_platform():
    global _gpu_platform
    if _gpu_platform is not None:
        return _gpu_platform

    # Check for tegrastats (Jetson)
    try:
        result = subprocess.run(
            ["which", "tegrastats"], capture_output=True, timeout=5)
        if result.returncode == 0:
            _gpu_platform = "tegrastats"
            return _gpu_platform
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Check for nvidia-smi (desktop NVIDIA)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            _gpu_platform = "nvidia-smi"
            return _gpu_platform
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    _gpu_platform = "none"
    return _gpu_platform

# ---------------------------------------------------------------------------
# tegrastats parsing
# ---------------------------------------------------------------------------

def parse_tegrastats() -> dict:
    """Run tegrastats once and parse the output line. Returns partial dict."""
    result = {}
    try:
        proc = subprocess.run(
            ["tegrastats", "--interval", "100"],
            capture_output=True, text=True, timeout=1,
        )
        line = (proc.stdout or "").split("\n")[0]
        if not line:
            return result

        # GPU utilization: GR3D_FREQ 45%
        m = re.search(r"GR3D_FREQ\s+(\d+)%", line)
        if m:
            result["gpu_pct"] = float(m.group(1))

        # CPU temp: CPU@45.5C or CPU_TEMP 45C
        m = re.search(r"CPU(?:_TEMP)?\s*[@:]?\s*([\d.]+)C", line)
        if m:
            result["cpu_temp"] = float(m.group(1))

        # GPU temp: GPU@50C or GPU_TEMP 50C
        m = re.search(r"GPU(?:_TEMP)?\s*[@:]?\s*([\d.]+)C", line)
        if m:
            result["gpu_temp"] = float(m.group(1))

        # Power: VIN_SYS_5V0 5000mW
        m = re.search(r"VIN_SYS_5V0\s+(\d+)mW", line)
        if m:
            result["power_mw"] = int(m.group(1))

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return result

# ---------------------------------------------------------------------------
# nvidia-smi parsing
# ---------------------------------------------------------------------------

def parse_nvidia_smi() -> dict:
    """Query nvidia-smi for GPU metrics. Returns partial dict."""
    result = {}
    try:
        proc = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            parts = proc.stdout.strip().split(",")
            if len(parts) >= 1:
                try:
                    result["gpu_pct"] = float(parts[0].strip())
                except ValueError:
                    pass
            if len(parts) >= 2:
                try:
                    result["gpu_temp"] = float(parts[1].strip())
                except ValueError:
                    pass
            if len(parts) >= 3:
                try:
                    # nvidia-smi reports power in watts, convert to mW
                    result["power_mw"] = int(float(parts[2].strip()) * 1000)
                except ValueError:
                    pass
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # CPU temp via lm-sensors or thermal zone
    if "cpu_temp" not in result:
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                result["cpu_temp"] = float(f.read().strip()) / 1000.0
        except (FileNotFoundError, ValueError, PermissionError):
            pass

    return result

# ---------------------------------------------------------------------------
# Process collection
# ---------------------------------------------------------------------------

def collect_processes() -> list:
    """Collect top 10 processes by CPU usage + running docker containers."""
    procs = []

    # Top processes by CPU
    try:
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                info = p.info
                if info["cpu_percent"] is not None and info["cpu_percent"] > 0:
                    procs.append({
                        "name": info["name"] or "?",
                        "cpu": round(info["cpu_percent"], 1),
                        "mem": round(info["memory_percent"] or 0, 1),
                        "kind": "process",
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass

    # Sort by CPU descending, take top 10
    procs.sort(key=lambda x: x["cpu"], reverse=True)
    procs = procs[:10]

    # Docker containers
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                parts = line.split("\t", 1)
                if parts:
                    procs.append({
                        "name": parts[0],
                        "cpu": 0,
                        "mem": 0,
                        "kind": "docker",
                    })
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return procs

# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------

def collect() -> dict:
    ts = int(time.time())

    # CPU
    cpu_pct = psutil.cpu_percent(interval=0)

    # RAM
    mem = psutil.virtual_memory()
    ram_pct = mem.percent
    ram_used_mb = mem.used // (1024 * 1024)
    ram_total_mb = mem.total // (1024 * 1024)

    # Disk — collect all mounted physical partitions
    disks = []
    DISK_LABELS = {
        "/": "eMMC",
        "/ssd": "NVMe",
        "/mnt/pirate-data": "2TB External",
    }
    seen_devices = set()
    for part in psutil.disk_partitions(all=False):
        if part.device in seen_devices:
            continue
        if part.fstype in ("squashfs", "tmpfs", "devtmpfs", "overlay", "vfat"):
            continue
        seen_devices.add(part.device)
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except (PermissionError, FileNotFoundError):
            continue
        disks.append({
            "mount": part.mountpoint,
            "label": DISK_LABELS.get(part.mountpoint, part.mountpoint),
            "used_gb": round(usage.used / (1024 ** 3), 2),
            "total_gb": round(usage.total / (1024 ** 3), 2),
            "pct": usage.percent,
        })

    # Primary disk for backward compat (prefer /ssd, fallback /)
    primary = next((d for d in disks if d["mount"] == "/ssd"),
                   next((d for d in disks if d["mount"] == "/"), None))
    if primary:
        disk_pct = primary["pct"]
        disk_used_gb = primary["used_gb"]
        disk_total_gb = primary["total_gb"]
    else:
        disk_pct = None
        disk_used_gb = None
        disk_total_gb = None

    # Uptime
    uptime_secs = int(time.time() - psutil.boot_time())

    row = {
        "timestamp": ts,
        "device_id": DEVICE_ID,
        "cpu_pct": cpu_pct,
        "gpu_pct": None,
        "ram_pct": ram_pct,
        "ram_used_mb": ram_used_mb,
        "ram_total_mb": ram_total_mb,
        "disk_pct": disk_pct,
        "disk_used_gb": disk_used_gb,
        "disk_total_gb": disk_total_gb,
        "cpu_temp": None,
        "gpu_temp": None,
        "uptime_secs": uptime_secs,
        "power_mw": None,
        "disks": disks,
    }

    # Merge GPU data based on platform
    platform = detect_gpu_platform()
    if platform == "tegrastats":
        row.update(parse_tegrastats())
    elif platform == "nvidia-smi":
        row.update(parse_nvidia_smi())

    return row

# ---------------------------------------------------------------------------
# Push mode
# ---------------------------------------------------------------------------

def push_data(row: dict, processes: list):
    """POST collected data to remote Ramp server."""
    import urllib.request
    import urllib.error

    payload = dict(row)
    payload["processes"] = processes

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        PUSH_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"Push failed: {e}", file=sys.stderr, flush=True)
        return False

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

_running = True

def _handle_sigterm(signum, frame):
    global _running
    print("SIGTERM received, shutting down.", flush=True)
    _running = False

def main():
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    mode = "push" if PUSH_URL else "local"
    print(f"Stats collector started. Device: {DEVICE_ID}, Mode: {mode}, "
          f"GPU: {detect_gpu_platform()}", flush=True)

    conn = None
    if not PUSH_URL:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH))
        init_db(conn)
        print(f"DB: {DB_PATH}", flush=True)

    while _running:
        try:
            row = collect()
            processes = collect_processes()

            if PUSH_URL:
                ok = push_data(row, processes)
                status = "pushed" if ok else "push-failed"
            else:
                # Build stats row without disks array (stored separately)
                stats_row = {k: v for k, v in row.items() if k != "disks"}
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
                    stats_row,
                )
                # Update processes
                ts = row["timestamp"]
                conn.execute(
                    "DELETE FROM device_processes WHERE device_id=?",
                    (DEVICE_ID,))
                for proc in processes[:10]:
                    conn.execute(
                        """INSERT INTO device_processes
                           (snapshot_ts, device_id, name, cpu_pct, mem_pct, kind)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (ts, DEVICE_ID, proc.get("name", "?"),
                         proc.get("cpu", 0), proc.get("mem", 0),
                         proc.get("kind", "process")),
                    )
                # Update disks
                disks = row.get("disks", [])
                if disks:
                    conn.execute(
                        "DELETE FROM device_disks WHERE device_id=?",
                        (DEVICE_ID,))
                    for dk in disks:
                        conn.execute(
                            """INSERT INTO device_disks
                               (snapshot_ts, device_id, mount, label, used_gb, total_gb, pct)
                               VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (ts, DEVICE_ID, dk["mount"], dk.get("label", dk["mount"]),
                             dk["used_gb"], dk["total_gb"], dk["pct"]),
                        )
                conn.commit()

                # Prune rows older than PRUNE_DAYS
                cutoff = int(time.time()) - (PRUNE_DAYS * 86400)
                conn.execute("DELETE FROM device_stats WHERE timestamp < ?", (cutoff,))
                conn.execute("DELETE FROM device_processes WHERE snapshot_ts < ?", (cutoff,))
                conn.execute("DELETE FROM device_disks WHERE snapshot_ts < ?", (cutoff,))
                conn.commit()
                status = "written"

            print(f"[{row['timestamp']}] {status} cpu={row['cpu_pct']}% "
                  f"gpu={row['gpu_pct']}% ram={row['ram_pct']}% "
                  f"disk={row['disk_pct']}% procs={len(processes)}", flush=True)
        except Exception as e:
            print(f"Collection error: {e}", file=sys.stderr, flush=True)

        # Sleep in small increments to respect SIGTERM
        for _ in range(INTERVAL_SECS):
            if not _running:
                break
            time.sleep(1)

    if conn:
        conn.close()
    print("Stats collector stopped.", flush=True)

if __name__ == "__main__":
    main()
