#!/usr/bin/env bash
# Install user systemd service so the bot starts on login/boot (linger optional).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_DIR="${HOME}/.config/systemd/user"
SERVICE_FILE="${SERVICE_DIR}/tryzub-trade.service"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then PY="$(command -v python3)"; fi

mkdir -p "$SERVICE_DIR"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Tryzub Trade AI Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${ROOT}
ExecStart=${PY} ${ROOT}/main.py --watchdog
Restart=always
RestartSec=30
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now tryzub-trade.service
echo "OK: user service tryzub-trade.service enabled."
echo "Status: systemctl --user status tryzub-trade.service"
echo "For start even before login: sudo loginctl enable-linger \$USER"
