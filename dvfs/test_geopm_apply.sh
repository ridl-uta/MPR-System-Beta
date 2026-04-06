#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: test_geopm_apply.sh [options]

Install the GEOPM apply helper on a target node, stage a host config at
/shared/geopm/freq/<hostname>.conf, trigger geopm-apply.service, print readback,
then restore max frequency. The original host config is restored on exit.

Options:
  --node HOST           Target node hostname (default: localhost)
  --ssh-user USER       Optional SSH username for remote target
  --remote-repo-dir DIR Repo path on remote node (default: same as local repo)
  --shared-conf-dir DIR Shared config dir (default: /shared/geopm/freq)
  --cores "0 1"         Space-separated core IDs to target (default: "0 1")
  --target-mhz 2200     Test frequency in MHz (default: 2200)
  --restore-mhz 2400    Restore frequency in MHz (default: 2400)
  --control-kind KIND   PERF_CTL or CORE_MAX (default: PERF_CTL)
  --cpufreq-sync        Enable CPUFreq policy sync in test config
  --no-cpufreq-sync     Disable CPUFreq policy sync in test config (default)
  --skip-install        Do not run dvfs/install_geopm_apply_systemd.sh
  --keep-temp           Keep temporary files on local/remote target
  -h, --help            Show this help and exit
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET_NODE="localhost"
SSH_USER=""
REMOTE_REPO_DIR="${REPO_DIR}"
SHARED_CONF_DIR="/shared/geopm/freq"
CORES="0 1"
TARGET_MHZ="2200"
RESTORE_MHZ="2400"
CONTROL_KIND="PERF_CTL"
CPUFREQ_SYNC=0
CPUFREQ_GOVERNOR="userspace"
CPUFREQ_MIN_KHZ="1000000"
SKIP_INSTALL=0
KEEP_TEMP=0

while (( $# > 0 )); do
  case "$1" in
    --node)
      shift
      [[ $# -gt 0 ]] || { echo "[ERR] --node requires a value" >&2; exit 1; }
      TARGET_NODE="$1"
      shift
      ;;
    --ssh-user)
      shift
      [[ $# -gt 0 ]] || { echo "[ERR] --ssh-user requires a value" >&2; exit 1; }
      SSH_USER="$1"
      shift
      ;;
    --remote-repo-dir)
      shift
      [[ $# -gt 0 ]] || { echo "[ERR] --remote-repo-dir requires a value" >&2; exit 1; }
      REMOTE_REPO_DIR="$1"
      shift
      ;;
    --shared-conf-dir)
      shift
      [[ $# -gt 0 ]] || { echo "[ERR] --shared-conf-dir requires a value" >&2; exit 1; }
      SHARED_CONF_DIR="$1"
      shift
      ;;
    --cores)
      shift
      [[ $# -gt 0 ]] || { echo "[ERR] --cores requires a value" >&2; exit 1; }
      CORES="$1"
      shift
      ;;
    --target-mhz)
      shift
      [[ $# -gt 0 ]] || { echo "[ERR] --target-mhz requires a value" >&2; exit 1; }
      TARGET_MHZ="$1"
      shift
      ;;
    --restore-mhz)
      shift
      [[ $# -gt 0 ]] || { echo "[ERR] --restore-mhz requires a value" >&2; exit 1; }
      RESTORE_MHZ="$1"
      shift
      ;;
    --control-kind)
      shift
      [[ $# -gt 0 ]] || { echo "[ERR] --control-kind requires a value" >&2; exit 1; }
      CONTROL_KIND="$1"
      shift
      ;;
    --cpufreq-sync)
      CPUFREQ_SYNC=1
      shift
      ;;
    --no-cpufreq-sync)
      CPUFREQ_SYNC=0
      shift
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    --keep-temp)
      KEEP_TEMP=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERR] unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

[[ "${CONTROL_KIND}" == "PERF_CTL" || "${CONTROL_KIND}" == "CORE_MAX" ]] || {
  echo "[ERR] --control-kind must be PERF_CTL or CORE_MAX" >&2
  exit 1
}

[[ "${TARGET_MHZ}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "[ERR] --target-mhz must be numeric" >&2
  exit 1
}
[[ "${RESTORE_MHZ}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "[ERR] --restore-mhz must be numeric" >&2
  exit 1
}

command -v sudo >/dev/null 2>&1 || { echo "[ERR] sudo not found" >&2; exit 1; }
command -v hostname >/dev/null 2>&1 || { echo "[ERR] hostname not found" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "[ERR] python3 not found" >&2; exit 1; }

LOCAL_HOST_SHORT="$(hostname -s)"
LOCAL_HOST_FQDN="$(hostname -f 2>/dev/null || true)"

is_local_target() {
  local candidate="${1}"
  [[ "${candidate}" == "localhost" || "${candidate}" == "127.0.0.1" || "${candidate}" == "${LOCAL_HOST_SHORT}" || "${candidate}" == "${LOCAL_HOST_FQDN}" ]]
}

TARGET_IS_LOCAL=0
if is_local_target "${TARGET_NODE}"; then
  TARGET_IS_LOCAL=1
fi

SSH_TARGET="${TARGET_NODE}"
if [[ -n "${SSH_USER}" ]]; then
  SSH_TARGET="${SSH_USER}@${TARGET_NODE}"
fi

LOCAL_TMPDIR="$(mktemp -d)"
LOCAL_TEST_CONF="${LOCAL_TMPDIR}/test.conf"
LOCAL_RESTORE_CONF="${LOCAL_TMPDIR}/restore.conf"
LOCAL_BACKUP_CONF="${LOCAL_TMPDIR}/original.conf"
REMOTE_TMPDIR=""
REMOTE_BACKUP_CONF=""
TARGET_HOST_SHORT=""
TARGET_CONF_PATH=""
BACKUP_WAS_PRESENT=0

cleanup() {
  if [[ -n "${TARGET_CONF_PATH}" ]]; then
    if (( TARGET_IS_LOCAL == 1 )); then
      if (( BACKUP_WAS_PRESENT == 1 )) && [[ -f "${LOCAL_BACKUP_CONF}" ]]; then
        cp "${LOCAL_BACKUP_CONF}" "${TARGET_CONF_PATH}" >/dev/null 2>&1 || true
      else
        rm -f "${TARGET_CONF_PATH}" >/dev/null 2>&1 || true
      fi
    elif [[ -n "${REMOTE_TMPDIR}" ]]; then
      if (( BACKUP_WAS_PRESENT == 1 )) && [[ -n "${REMOTE_BACKUP_CONF}" ]]; then
        ssh "${SSH_TARGET}" "cp '${REMOTE_BACKUP_CONF}' '${TARGET_CONF_PATH}'" >/dev/null 2>&1 || true
      else
        ssh "${SSH_TARGET}" "rm -f '${TARGET_CONF_PATH}'" >/dev/null 2>&1 || true
      fi
    fi
  fi

  if (( KEEP_TEMP == 0 )); then
    rm -rf "${LOCAL_TMPDIR}"
    if (( TARGET_IS_LOCAL == 0 )) && [[ -n "${REMOTE_TMPDIR}" ]]; then
      ssh "${SSH_TARGET}" "rm -rf '${REMOTE_TMPDIR}'" >/dev/null 2>&1 || true
    fi
  else
    echo "[INFO] keeping local temp files in ${LOCAL_TMPDIR}"
    if (( TARGET_IS_LOCAL == 0 )) && [[ -n "${REMOTE_TMPDIR}" ]]; then
      echo "[INFO] keeping remote temp files in ${TARGET_NODE}:${REMOTE_TMPDIR}"
    fi
  fi
}
trap cleanup EXIT

run_target_cmd() {
  local cmd="$1"
  if (( TARGET_IS_LOCAL == 1 )); then
    bash -lc "${cmd}"
  else
    ssh "${SSH_TARGET}" "${cmd}"
  fi
}

copy_to_target() {
  local src_path="$1"
  local dst_path="$2"
  if (( TARGET_IS_LOCAL == 1 )); then
    cp "${src_path}" "${dst_path}"
  else
    scp -q "${src_path}" "${SSH_TARGET}:${dst_path}"
  fi
}

make_conf() {
  local out_path="$1"
  local freq_hz="$2"
  cat >"${out_path}" <<EOF
### RULE geopm-apply-test
FREQ_HZ=${freq_hz}
CONTROL_KIND=${CONTROL_KIND}
EOF
  if (( CPUFREQ_SYNC == 1 )); then
    cat >>"${out_path}" <<EOF
CPUFREQ_SYNC=1
CPUFREQ_GOVERNOR=${CPUFREQ_GOVERNOR}
CPUFREQ_MIN_KHZ=${CPUFREQ_MIN_KHZ}
EOF
  fi
  cat >>"${out_path}" <<EOF
CORES="${CORES}"
EOF
}

print_readback() {
  local core
  echo "[INFO] GEOPM readback from ${TARGET_NODE}"
  for core in ${CORES}; do
    echo "core ${core}:"
    run_target_cmd "geopmread MSR::PERF_CTL:FREQ core ${core} || true"
    run_target_cmd "geopmread CPU_FREQUENCY_MAX_CONTROL core ${core} || true"
    run_target_cmd "geopmread CPU_FREQUENCY_STATUS core ${core} || true"
  done
  echo "[INFO] sysfs readback from ${TARGET_NODE}"
  for core in ${CORES}; do
    run_target_cmd "if [ -r /sys/devices/system/cpu/cpu${core}/cpufreq/scaling_max_freq ]; then printf 'cpu%s scaling_max_freq=%s kHz\\n' '${core}' \"\$(cat /sys/devices/system/cpu/cpu${core}/cpufreq/scaling_max_freq)\"; else printf 'cpu%s scaling_max_freq=<unavailable>\\n' '${core}'; fi"
  done
}

print_service_debug() {
  echo "[INFO] geopm-apply.service status from ${TARGET_NODE}"
  run_target_cmd "systemctl --no-pager --full status geopm-apply.service || true"
  echo "[INFO] geopm-apply.service journal from ${TARGET_NODE}"
  run_target_cmd "journalctl -u geopm-apply.service -n 40 --no-pager || true"
}

run_service_apply() {
  if (( TARGET_IS_LOCAL == 1 )); then
    sudo -n systemctl start geopm-apply.service >/dev/null 2>&1 || systemctl start geopm-apply.service >/dev/null 2>&1 || sudo systemctl start geopm-apply.service
  else
    ssh -tt "${SSH_TARGET}" "sudo -n systemctl start geopm-apply.service >/dev/null 2>&1 || systemctl start geopm-apply.service >/dev/null 2>&1 || sudo systemctl start geopm-apply.service"
  fi
}

install_helper() {
  echo "[INFO] installing helper on ${TARGET_NODE}"
  if (( TARGET_IS_LOCAL == 1 )); then
    sudo bash "${REPO_DIR}/dvfs/install_geopm_apply_systemd.sh"
  else
    ssh -tt "${SSH_TARGET}" "cd '${REMOTE_REPO_DIR}' && sudo bash dvfs/install_geopm_apply_systemd.sh"
  fi
}

if (( TARGET_IS_LOCAL == 0 )); then
  command -v ssh >/dev/null 2>&1 || { echo "[ERR] ssh not found" >&2; exit 1; }
  command -v scp >/dev/null 2>&1 || { echo "[ERR] scp not found" >&2; exit 1; }
fi

TARGET_HZ="$(python3 - <<PY
print(int(round(float("${TARGET_MHZ}") * 1e6)))
PY
)"
RESTORE_HZ="$(python3 - <<PY
print(int(round(float("${RESTORE_MHZ}") * 1e6)))
PY
)"

make_conf "${LOCAL_TEST_CONF}" "${TARGET_HZ}"
make_conf "${LOCAL_RESTORE_CONF}" "${RESTORE_HZ}"

if (( SKIP_INSTALL == 0 )); then
  install_helper
fi

TARGET_HOST_SHORT="$(run_target_cmd "hostname -s")"
TARGET_CONF_PATH="${SHARED_CONF_DIR}/${TARGET_HOST_SHORT}.conf"

if (( TARGET_IS_LOCAL == 0 )); then
  REMOTE_TMPDIR="$(run_target_cmd "mktemp -d /tmp/geopm_apply_test.XXXXXX")"
  REMOTE_BACKUP_CONF="${REMOTE_TMPDIR}/original.conf"
fi

BACKUP_WAS_PRESENT="$(run_target_cmd "if [ -f '${TARGET_CONF_PATH}' ]; then echo 1; else echo 0; fi")"
if [[ "${BACKUP_WAS_PRESENT}" == "1" ]]; then
  if (( TARGET_IS_LOCAL == 1 )); then
    cp "${TARGET_CONF_PATH}" "${LOCAL_BACKUP_CONF}"
  else
    run_target_cmd "cp '${TARGET_CONF_PATH}' '${REMOTE_BACKUP_CONF}'"
  fi
else
  BACKUP_WAS_PRESENT=0
fi

copy_to_target "${LOCAL_TEST_CONF}" "${TARGET_CONF_PATH}"

echo "[INFO] verifying installed helper on ${TARGET_NODE}"
run_target_cmd "grep -n 'total_failed > 0 || total_applied == 0' /usr/local/sbin/geopm_apply.sh"
echo "[INFO] staged config path=${TARGET_CONF_PATH}"
echo "[INFO] cpufreq_sync=${CPUFREQ_SYNC} cpufreq_governor=${CPUFREQ_GOVERNOR} cpufreq_min_khz=${CPUFREQ_MIN_KHZ}"

echo "[INFO] applying test frequency ${TARGET_MHZ} MHz on ${TARGET_NODE} cores: ${CORES}"
set +e
run_service_apply
TEST_RC=$?
set -e
echo "[INFO] test apply exit code: ${TEST_RC}"
if (( TEST_RC != 0 )); then
  print_service_debug
fi
print_readback

copy_to_target "${LOCAL_RESTORE_CONF}" "${TARGET_CONF_PATH}"

echo "[INFO] restoring ${RESTORE_MHZ} MHz on ${TARGET_NODE} cores: ${CORES}"
set +e
run_service_apply
RESTORE_RC=$?
set -e
echo "[INFO] restore exit code: ${RESTORE_RC}"
if (( RESTORE_RC != 0 )); then
  print_service_debug
fi
print_readback

echo "[INFO] done"
