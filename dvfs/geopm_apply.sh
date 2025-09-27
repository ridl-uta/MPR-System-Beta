#!/usr/bin/env bash
# Apply GEOPM CPU frequency settings from a host-specific config file.
# Intended to be run via systemd as a oneshot service.

set -euo pipefail

log() { echo "[geopm-apply] $*"; }
err() { echo "[geopm-apply][ERROR] $*" >&2; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || { err "'$1' not found in PATH"; exit 1; }; }

need_cmd geopmread
need_cmd geopmwrite
need_cmd lscpu

CONF_PATH="${1:-/shared/geopm/freq/$(hostname).conf}"
if [[ ! -r "$CONF_PATH" ]]; then
  err "Config not found or unreadable: $CONF_PATH"
  exit 1
fi

# shellcheck source=/dev/null
set -a
source "$CONF_PATH"
set +a

# Supported config variables (all optional unless noted):
# - FREQ_HZ: target frequency in Hz; if "max" or empty, use max available.
# - CORES: space-separated core IDs to target (e.g., "0 2 4 6 8").
# - CPUS: space-separated logical CPU IDs to target. Ignored if CORES is set.
# - CONTROL_KIND: AUTO | CORE_MAX | CPU  (default: AUTO)
# - RESPECT_CPUSET: 1 to limit to process cpuset (default: 0)
# - RETRIES: write attempts per target (default: 5)
# - RETRY_SLEEP: seconds between retries (default: 0.25)

CONTROL_KIND=${CONTROL_KIND:-AUTO}
RESPECT_CPUSET=${RESPECT_CPUSET:-0}
RETRIES=${RETRIES:-5}
RETRY_SLEEP=${RETRY_SLEEP:-0.25}

# Resolve frequency
resolve_max_hz() {
  local val khz mhz
  val=$(geopmread CPU_FREQUENCY_MAX_AVAIL board 0 2>/dev/null || true)
  if [[ -n "${val:-}" && "$val" != "NaN" ]]; then
    awk -v v="$val" 'BEGIN{if (v+0>0) printf "%.0f", v}'
    return 0
  fi
  if [[ -r /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq ]]; then
    khz=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq)
    echo $((khz * 1000))
    return 0
  fi
  mhz=$(LC_ALL=C lscpu 2>/dev/null | awk '/max MHz/ {print $3}')
  if [[ -n "${mhz:-}" ]]; then
    awk -v m="$mhz" 'BEGIN{printf "%.0f", m*1e6}'
    return 0
  fi
  return 1
}

if [[ -z "${FREQ_HZ:-}" || "$FREQ_HZ" == "max" ]]; then
  if ! FREQ_HZ=$(resolve_max_hz); then
    err "Unable to determine max frequency"
    exit 1
  fi
fi

log "Using FREQ_HZ=${FREQ_HZ}"

# Build allowed CPU set (optional)
expand_list() {
  local list="$1" part a b i
  IFS=',' read -ra parts <<< "$list"
  for part in "${parts[@]}"; do
    if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      a=${BASH_REMATCH[1]}; b=${BASH_REMATCH[2]}
      for ((i=a; i<=b; ++i)); do printf "%d\n" "$i"; done
    else
      printf "%d\n" "$part"
    fi
  done
}

ALLOWED_CPUS=""
if [[ "$RESPECT_CPUSET" == "1" ]]; then
  allowed_list=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status || true)
  if [[ -n "${allowed_list:-}" ]]; then
    ALLOWED_CPUS=$(expand_list "$allowed_list" | tr '\n' ' ')
    log "Restricting to cpuset: ${allowed_list} => ${ALLOWED_CPUS}"
  else
    log "RESPECT_CPUSET=1 but could not read Cpus_allowed_list; proceeding without restriction"
  fi
fi

in_allowed() {
  local cpu="$1"
  [[ -z "$ALLOWED_CPUS" ]] && return 0
  grep -q "\b${cpu}\b" <<<" ${ALLOWED_CPUS} "
}

# Build target lists
map_core_to_cpus() {
  local core_id="$1"
  lscpu -p=CPU,CORE | awk -F, -v k="$core_id" '/^[^#]/ && $2==k {print $1}'
}

TARGET_CPUS=()
TARGET_CORES=()

if [[ -n "${CORES:-}" ]]; then
  for c in $CORES; do TARGET_CORES+=("$c"); done
  # also build CPUs for fallback/CPU domain
  for c in "${TARGET_CORES[@]}"; do
    while read -r cpu; do
      [[ -z "$cpu" ]] && continue
      in_allowed "$cpu" || continue
      TARGET_CPUS+=("$cpu")
    done < <(map_core_to_cpus "$c")
  done
elif [[ -n "${CPUS:-}" ]]; then
  for cpu in $CPUS; do
    in_allowed "$cpu" || continue
    TARGET_CPUS+=("$cpu")
  done
else
  # default: all CPUs
  while read -r cpu; do
    in_allowed "$cpu" || continue
    TARGET_CPUS+=("$cpu")
  done < <(lscpu -p=CPU | awk -F, '/^[^#]/ {print $1}')
fi

write_with_retry() {
  local cmd="$1"; shift
  local attempt=1
  while true; do
    if $cmd "$@"; then
      return 0
    fi
    if (( attempt >= RETRIES )); then
      return 1
    fi
    sleep "$RETRY_SLEEP"
    attempt=$((attempt+1))
  done
}

has_access_core_max=0
if geopmaccess -l 2>/dev/null | grep -q "CPU_FREQUENCY_MAX_CONTROL.*core"; then
  has_access_core_max=1
fi

applied=0
failed=0

apply_core_max() {
  local rc=0
  if [[ ${#TARGET_CORES[@]} -eq 0 ]]; then
    # derive cores from CPUs if not provided
    mapfile -t TARGET_CORES < <(lscpu -p=CPU,CORE | awk -F, '/^[^#]/{print $2}' | sort -n | uniq)
  fi
  for core in "${TARGET_CORES[@]}"; do
    if write_with_retry geopmwrite CPU_FREQUENCY_MAX_CONTROL core "$core" "$FREQ_HZ"; then
      log "core ${core}: MAX_CONTROL=${FREQ_HZ}"
      applied=$((applied+1))
    else
      err "core ${core}: write MAX_CONTROL failed"
      rc=1
      failed=$((failed+1))
    fi
  done
  return $rc
}

apply_cpu_ctrl() {
  local rc=0
  if [[ ${#TARGET_CPUS[@]} -eq 0 ]]; then
    mapfile -t TARGET_CPUS < <(lscpu -p=CPU | awk -F, '/^[^#]/{print $1}')
  fi
  for cpu in "${TARGET_CPUS[@]}"; do
    if write_with_retry geopmwrite CPU_FREQUENCY_CONTROL cpu "$cpu" "$FREQ_HZ"; then
      log "cpu ${cpu}: CONTROL=${FREQ_HZ}"
      applied=$((applied+1))
    else
      err "cpu ${cpu}: write CONTROL failed"
      rc=1
      failed=$((failed+1))
    fi
  done
  return $rc
}

case "$CONTROL_KIND" in
  CORE_MAX)
    if ! apply_core_max; then
      err "CORE_MAX writes failed"
      exit 2
    fi
    ;;
  CPU)
    if ! apply_cpu_ctrl; then
      err "CPU CONTROL writes failed"
      exit 2
    fi
    ;;
  AUTO|*)
    if [[ "$has_access_core_max" == "1" && -n "${CORES:-}" ]]; then
      if apply_core_max; then
        :
      else
        log "Falling back to CPU domain control"
        apply_cpu_ctrl || { err "Fallback CPU CONTROL writes failed"; exit 2; }
      fi
    else
      apply_cpu_ctrl || { err "CPU CONTROL writes failed"; exit 2; }
    fi
    ;;
esac

log "Applied=${applied} Failed=${failed}"
exit 0

