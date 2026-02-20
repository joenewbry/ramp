#!/usr/bin/env bash
# setup-collector.sh — Deploy Ramp stats collector to a device
#
# Usage:
#   ./setup-collector.sh <device-id> [push-url]
#
# Examples:
#   ./setup-collector.sh jetson-orin-nano                    # local SQLite mode
#   ./setup-collector.sh orin-nano-2 http://prometheus:8097  # push to central server
#   ./setup-collector.sh rtx4080-workstation http://ramp.digitalsurfacelabs.com  # push mode
#
# This script:
#   1. Creates /ssd/ramp/ directory structure
#   2. Copies stats_collector.py
#   3. Sets up Python venv with psutil
#   4. Creates stats.env with device config
#   5. Installs and starts systemd service

set -euo pipefail

DEVICE_ID="${1:?Usage: $0 <device-id> [push-url]}"
PUSH_URL="${2:-}"
INSTALL_DIR="/ssd/ramp"
LOG_DIR="/ssd/logs"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Ramp Stats Collector Setup ==="
echo "Device ID: ${DEVICE_ID}"
echo "Mode:      ${PUSH_URL:+push to ${PUSH_URL}}${PUSH_URL:-local SQLite}"
echo "Install:   ${INSTALL_DIR}"
echo ""

# Create directories
sudo mkdir -p "${INSTALL_DIR}" "${LOG_DIR}"
sudo chown "$(whoami):$(whoami)" "${INSTALL_DIR}" "${LOG_DIR}"

# Copy collector script
cp "${SCRIPT_DIR}/stats_collector.py" "${INSTALL_DIR}/stats_collector.py"
echo "Copied stats_collector.py"

# Set up venv if needed
if [ ! -d "${INSTALL_DIR}/.venv" ]; then
    echo "Creating Python venv..."
    python3 -m venv "${INSTALL_DIR}/.venv"
fi
"${INSTALL_DIR}/.venv/bin/pip" install -q psutil
echo "Python venv ready with psutil"

# Create environment file
cat > "${INSTALL_DIR}/stats.env" <<EOF
RAMP_DEVICE_ID=${DEVICE_ID}
RAMP_PUSH_URL=${PUSH_URL}
RAMP_DB_PATH=${INSTALL_DIR}/stats.db
RAMP_INTERVAL=60
EOF
echo "Created stats.env"

# Install systemd service
SERVICE_FILE="/etc/systemd/system/ramp-stats.service"
sudo cp "${SCRIPT_DIR}/ramp-stats.service" "${SERVICE_FILE}"
# Update user in service file to current user
sudo sed -i "s/User=prometheus/User=$(whoami)/" "${SERVICE_FILE}"
sudo systemctl daemon-reload
sudo systemctl enable ramp-stats
sudo systemctl restart ramp-stats
echo "Systemd service installed and started"

# Verify
sleep 2
if systemctl is-active --quiet ramp-stats; then
    echo ""
    echo "=== SUCCESS ==="
    echo "Stats collector running as device '${DEVICE_ID}'"
    echo "Check logs: tail -f ${LOG_DIR}/ramp-stats.log"
else
    echo ""
    echo "=== WARNING ==="
    echo "Service may not have started. Check:"
    echo "  sudo systemctl status ramp-stats"
    echo "  tail -f ${LOG_DIR}/ramp-stats.log"
fi
