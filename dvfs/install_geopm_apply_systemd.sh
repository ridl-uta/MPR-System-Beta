#!/usr/bin/env bash
# Install the GEOPM apply script + systemd units on this node.

set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_SRC="${SRC_DIR}/geopm_apply.sh"
SERVICE_SRC="${SRC_DIR}/systemd/geopm-apply.service"
PATH_SRC="${SRC_DIR}/systemd/geopm-apply.path"

[[ $EUID -eq 0 ]] || { echo "Please run as root (sudo)." >&2; exit 1; }

install -Dm755 "$SCRIPT_SRC" /usr/local/sbin/geopm_apply.sh
install -Dm644 "$SERVICE_SRC" /etc/systemd/system/geopm-apply.service
install -Dm644 "$PATH_SRC" /etc/systemd/system/geopm-apply.path

systemctl daemon-reload
systemctl enable --now geopm-apply.path

echo "Installed. Create/update /shared/geopm/freq/$(hostname).conf and watch 'systemctl status geopm-apply'"

