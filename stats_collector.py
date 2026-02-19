#!/usr/bin/env python3
"""Ramp Stats Collector — collects hardware metrics from the Jetson and writes to SQLite.

Runs as a daemon every 60 seconds. No FastAPI dependency — pure stdlib + psutil.
Database: /ssd/ramp/stats.db
"""

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

DB_PATH = Path("/ssd/ramp/stats.db")
DEVICE_ID = "jetson-orin-nano"
INTERVAL_SECS = 60
PRUNE_DAYS = 30

# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

def init_db(conn):
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
    """)
    conn.commit()

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
# Metric collection
# ---------------------------------------------------------------------------

def collect() -> dict:
    ts = int(time.time())

    # CPU
    cpu_pct = psutil.cpu_percent(interval=1)

    # RAM
    mem = psutil.virtual_memory()
    ram_pct = mem.percent
    ram_used_mb = mem.used // (1024 * 1024)
    ram_total_mb = mem.total // (1024 * 1024)

    # Disk
    try:
        disk = psutil.disk_usage("/ssd")
        disk_pct = disk.percent
        disk_used_gb = round(disk.used / (1024 ** 3), 2)
        disk_total_gb = round(disk.total / (1024 ** 3), 2)
    except FileNotFoundError:
        disk = psutil.disk_usage("/")
        disk_pct = disk.percent
        disk_used_gb = round(disk.used / (1024 ** 3), 2)
        disk_total_gb = round(disk.total / (1024 ** 3), 2)

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
    }

    # Merge tegrastats data
    row.update(parse_tegrastats())
    return row

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

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    init_db(conn)
    print(f"Stats collector started. DB: {DB_PATH}", flush=True)

    while _running:
        try:
            row = collect()
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
            conn.commit()

            # Prune rows older than PRUNE_DAYS
            cutoff = int(time.time()) - (PRUNE_DAYS * 86400)
            conn.execute("DELETE FROM device_stats WHERE timestamp < ?", (cutoff,))
            conn.commit()

            print(f"[{row['timestamp']}] cpu={row['cpu_pct']}% "
                  f"gpu={row['gpu_pct']}% ram={row['ram_pct']}% "
                  f"disk={row['disk_pct']}%", flush=True)
        except Exception as e:
            print(f"Collection error: {e}", file=sys.stderr, flush=True)

        # Sleep in small increments to respect SIGTERM
        for _ in range(INTERVAL_SECS):
            if not _running:
                break
            time.sleep(1)

    conn.close()
    print("Stats collector stopped.", flush=True)

if __name__ == "__main__":
    main()
