#!/usr/bin/env bash
# setup-jetson.sh — One-time setup of the ramp stack on Prometheus.
# After this runs, just `git push` and the Jetson will auto-pull within 60s.
set -euo pipefail

REMOTE="prometheus@192.168.0.18"
REPO_DIR="/ssd/ramp"
REPO_URL="${1:-}"  # Pass the GitHub SSH URL as first arg, e.g. git@github.com:joenewbry/ramp.git

if [ -z "$REPO_URL" ]; then
    echo "Usage: $0 <github-ssh-url>"
    echo "  e.g. $0 git@github.com:joenewbry/ramp.git"
    exit 1
fi

echo "==> Cloning / updating repo on Jetson..."
ssh -o ConnectTimeout=10 "${REMOTE}" "
    set -euo pipefail
    mkdir -p /ssd/logs

    if [ -d '${REPO_DIR}/.git' ]; then
        echo 'Repo exists, pulling latest...'
        git -C '${REPO_DIR}' fetch origin main
        git -C '${REPO_DIR}' reset --hard origin/main
    else
        echo 'Cloning fresh...'
        rm -rf '${REPO_DIR}'
        git clone '${REPO_URL}' '${REPO_DIR}'
    fi
"

echo "==> Creating venv if missing..."
ssh -o ConnectTimeout=10 "${REMOTE}" "
    if [ ! -d '${REPO_DIR}/.venv' ]; then
        python3.10 -m venv '${REPO_DIR}/.venv'
    fi
    '${REPO_DIR}/.venv/bin/pip' install -q -r '${REPO_DIR}/requirements.txt'
"

echo "==> Granting prometheus passwordless sudo for service restarts..."
ssh -o ConnectTimeout=10 "${REMOTE}" "
    SUDOERS_LINE='prometheus ALL=(ALL) NOPASSWD: /bin/systemctl restart ramp-stats, /bin/systemctl restart ramp-dashboard, /bin/systemctl restart ramp-stats ramp-dashboard'
    SUDOERS_FILE='/etc/sudoers.d/prometheus-ramp'
    if ! sudo grep -q 'ramp-stats' \"\$SUDOERS_FILE\" 2>/dev/null; then
        echo \"\$SUDOERS_LINE\" | sudo tee \"\$SUDOERS_FILE\" > /dev/null
        sudo chmod 440 \"\$SUDOERS_FILE\"
        echo 'sudoers entry added'
    else
        echo 'sudoers entry already present'
    fi
"

echo "==> Installing systemd services and timer..."
ssh -o ConnectTimeout=10 "${REMOTE}" "
    set -euo pipefail
    chmod +x '${REPO_DIR}/ramp-autodeploy.sh'

    sudo cp '${REPO_DIR}/ramp-stats.service'      /etc/systemd/system/
    sudo cp '${REPO_DIR}/ramp-autodeploy.service' /etc/systemd/system/
    sudo cp '${REPO_DIR}/ramp-autodeploy.timer'   /etc/systemd/system/
    sudo systemctl daemon-reload

    sudo systemctl enable  ramp-stats
    sudo systemctl restart ramp-stats

    sudo systemctl enable  ramp-autodeploy.timer
    sudo systemctl restart ramp-autodeploy.timer

    echo 'Services installed and running.'
    sudo systemctl status ramp-autodeploy.timer --no-pager
"

echo ""
echo "==> Setup complete!"
echo "    Workflow: git push origin main  →  Jetson auto-pulls within ~60s"
echo "    Logs: ssh prometheus 'tail -f /ssd/logs/ramp-autodeploy.log'"
