#!/usr/bin/env bash
# Install the GEOPM apply script + systemd units on this node.

set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_SRC="${SRC_DIR}/geopm_apply.sh"
SERVICE_SRC="${SRC_DIR}/systemd/geopm-apply.service"

[[ $EUID -eq 0 ]] || { echo "Please run as root (sudo)." >&2; exit 1; }

install -Dm755 "$SCRIPT_SRC" /usr/local/sbin/geopm_apply.sh
install -Dm644 "$SERVICE_SRC" /etc/systemd/system/geopm-apply.service

systemctl daemon-reload

echo "Installed. You can now trigger applies via:"
echo " - SSH: dvfs/run_geopm_apply_ssh.sh -H <hosts>"
echo " - Locally on node: systemctl start geopm-apply.service"
echo "Ensure /shared/geopm/freq/$(hostname).conf exists on each node."
