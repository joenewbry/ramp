#!/usr/bin/env bash
# ramp-autodeploy.sh — runs on Jetson via systemd timer every 60s
# Checks for new commits on main, pulls and restarts services if changed.
set -euo pipefail

REPO_DIR="/ssd/ramp"
VENV="$REPO_DIR/.venv"

cd "$REPO_DIR"
git fetch origin main --quiet 2>&1

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" = "$REMOTE" ]; then
    exit 0  # Nothing to do
fi

echo "[$(date)] New commits detected, pulling..."

# Snapshot requirements.txt before pull to detect changes
REQS_BEFORE=$(git show HEAD:requirements.txt 2>/dev/null || echo "")

git pull --rebase origin main

REQS_AFTER=$(cat requirements.txt 2>/dev/null || echo "")

if [ "$REQS_BEFORE" != "$REQS_AFTER" ]; then
    echo "[$(date)] requirements.txt changed, reinstalling deps..."
    "$VENV/bin/pip" install -q -r requirements.txt
fi

echo "[$(date)] Restarting services..."
sudo systemctl restart ramp-stats ramp-dashboard
echo "[$(date)] Deploy complete ($(git rev-parse --short HEAD))"
