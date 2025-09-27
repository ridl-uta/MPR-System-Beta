#!/usr/bin/env bash
# Install the GEOPM apply script + systemd units on this node.

set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_SRC="${SRC_DIR}/geopm_apply.sh"
SERVICE_SRC="${SRC_DIR}/systemd/geopm-apply.service"
PATH_SRC="${SRC_DIR}/systemd/geopm-apply.path"
TIMER_SRC="${SRC_DIR}/systemd/geopm-apply.timer"

[[ $EUID -eq 0 ]] || { echo "Please run as root (sudo)." >&2; exit 1; }

install -Dm755 "$SCRIPT_SRC" /usr/local/sbin/geopm_apply.sh
install -Dm644 "$SERVICE_SRC" /etc/systemd/system/geopm-apply.service
install -Dm644 "$PATH_SRC" /etc/systemd/system/geopm-apply.path
install -Dm644 "$TIMER_SRC" /etc/systemd/system/geopm-apply.timer

systemctl daemon-reload

# Control enabling via env flags:
#   GEOPM_ENABLE_TIMER=1     -> enable/start timer
#   GEOPM_DISABLE_PATH=1     -> do not enable path watcher

if [[ "${GEOPM_DISABLE_PATH:-0}" != "1" ]]; then
  systemctl enable --now geopm-apply.path
fi

if [[ "${GEOPM_ENABLE_TIMER:-0}" == "1" ]]; then
  systemctl enable --now geopm-apply.timer
fi

echo "Installed. You can now:"
echo " - Use path watcher (geopm-apply.path) to react to local file changes"
echo " - Or enable periodic timer: GEOPM_ENABLE_TIMER=1 sudo bash $0"
echo "Create/update /shared/geopm/freq/$(hostname).conf and check 'systemctl status geopm-apply'"
