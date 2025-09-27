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

DEFAULT_CONF="/shared/geopm/freq/$(hostname).conf"
DEFAULT_DIR="/shared/geopm/freq/$(hostname).d"

# Build list of configuration files to apply. Accepts multiple args; if none
# provided, use default conf and any files in <host>.d/*.conf (if present).
CONFIGS=()
if (( $# == 0 )); then
  [[ -r "$DEFAULT_CONF" ]] && CONFIGS+=("$DEFAULT_CONF")
  if [[ -d "$DEFAULT_DIR" ]]; then
    while IFS= read -r -d '' f; do CONFIGS+=("$f"); done < <(find "$DEFAULT_DIR" -maxdepth 1 -type f -name '*.conf' -print0 | sort -z)
  fi
else
  for arg in "$@"; do
    if [[ -d "$arg" ]]; then
      while IFS= read -r -d '' f; do CONFIGS+=("$f"); done < <(find "$arg" -maxdepth 1 -type f -name '*.conf' -print0 | sort -z)
    elif [[ -r "$arg" ]]; then
      CONFIGS+=("$arg")
    else
      err "Config path not found or unreadable: $arg"
      exit 1
    fi
  done
fi

if (( ${#CONFIGS[@]} == 0 )); then
  err "No configuration files found. Looked for $DEFAULT_CONF and $DEFAULT_DIR/*.conf"
  exit 1
fi

# Supported config variables (per config; optional unless noted):
# - FREQ_HZ: target frequency in Hz; if "max" or empty, use max available.
# - CORES: space-separated core IDs to target (e.g., "0 2 4 6 8").
# - CPUS: space-separated logical CPU IDs to target. Ignored if CORES is set.
# - CONTROL_KIND: AUTO | CORE_MAX | CPU  (default: AUTO)
# - RESPECT_CPUSET: 1 to limit to process cpuset (default: 0)
# - RETRIES: write attempts per target (default: 5)
# - RETRY_SLEEP: seconds between retries (default: 0.25)

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

map_core_to_cpus() {
  local core_id="$1"
  lscpu -p=CPU,CORE | awk -F, -v k="$core_id" '/^[^#]/ && $2==k {print $1}'
}

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

# Apply a single configuration file
apply_one_conf() {
  local conf="$1"
  log "Applying config: $conf"

  # Support multi-rule blocks within a single file, delimited by lines starting
  # with '### RULE'. Each block contains the same variable names as a normal
  # single-config file.
  if grep -qE '^[[:space:]]*###[[:space:]]+RULE' "$conf"; then
    local tmpdir
    tmpdir=$(mktemp -d)
    # Split into numbered blocks rule_00.conf, rule_01.conf, ...
    awk -v dir="$tmpdir" '
      BEGIN { n = -1 }
      /^[[:space:]]*###[[:space:]]+RULE/ { n++; next }
      n >= 0 { printf "%s\n", $0 >> (dir "/rule_" sprintf("%02d", n) ".conf") }
    ' "$conf"
    local blocks=("$tmpdir"/rule_*.conf)
    if ls ${blocks[*]} >/dev/null 2>&1; then
      local b
      for b in "${blocks[@]}"; do
        [[ -e "$b" ]] || continue
        log "Applying rule block: $(basename "$b") from $conf"
        apply_block_from_file "$b" || true
      done
    else
      err "Found RULE markers but no blocks parsed in $conf"
    fi
    rm -rf "$tmpdir"
    return 0
  fi

  # Single-block classic config
  apply_block_from_file "$conf"
}

total_applied=0
total_failed=0
for conf in "${CONFIGS[@]}"; do
  apply_one_conf "$conf" || true
done

log "Total Applied=${total_applied} Failed=${total_failed}"
exit 0

# Helper: apply a single block config from file (contains FREQ_HZ/CORES/CPUS/etc.)
apply_block_from_file() {
  local conf="$1"

  # Reset per-config variables/state
  unset FREQ_HZ CORES CPUS CONTROL_KIND RESPECT_CPUSET RETRIES RETRY_SLEEP
  ALLOWED_CPUS=""
  TARGET_CPUS=()
  TARGET_CORES=()
  applied=0
  failed=0

  # shellcheck source=/dev/null
  set -a
  source "$conf"
  set +a

  CONTROL_KIND=${CONTROL_KIND:-AUTO}
  RESPECT_CPUSET=${RESPECT_CPUSET:-0}
  RETRIES=${RETRIES:-5}
  RETRY_SLEEP=${RETRY_SLEEP:-0.25}

  if [[ -z "${FREQ_HZ:-}" || "$FREQ_HZ" == "max" ]]; then
    if ! FREQ_HZ=$(resolve_max_hz); then
      err "Unable to determine max frequency"
      return 1
    fi
  fi
  log "Using FREQ_HZ=${FREQ_HZ}"

  # Build allowed CPU set (optional)
  if [[ "$RESPECT_CPUSET" == "1" ]]; then
    local allowed_list
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
  if [[ -n "${CORES:-}" ]]; then
    local c
    for c in $CORES; do TARGET_CORES+=("$c"); done
    for c in "${TARGET_CORES[@]}"; do
      while read -r cpu; do
        [[ -z "$cpu" ]] && continue
        in_allowed "$cpu" || continue
        TARGET_CPUS+=("$cpu")
      done < <(map_core_to_cpus "$c")
    done
  elif [[ -n "${CPUS:-}" ]]; then
    local cpu
    for cpu in $CPUS; do
      in_allowed "$cpu" || continue
      TARGET_CPUS+=("$cpu")
    done
  else
    while read -r cpu; do
      in_allowed "$cpu" || continue
      TARGET_CPUS+=("$cpu")
    done < <(lscpu -p=CPU | awk -F, '/^[^#]/ {print $1}')
  fi

  local has_access_core_max=0
  if geopmaccess -l 2>/dev/null | grep -q "CPU_FREQUENCY_MAX_CONTROL.*core"; then
    has_access_core_max=1
  fi

  case "$CONTROL_KIND" in
    CORE_MAX)
      if ! apply_core_max; then
        err "CORE_MAX writes failed"
        return 2
      fi
      ;;
    CPU)
      if ! apply_cpu_ctrl; then
        err "CPU CONTROL writes failed"
        return 2
      fi
      ;;
    AUTO|*)
      if [[ "$has_access_core_max" == "1" && -n "${CORES:-}" ]]; then
        if apply_core_max; then :; else
          log "Falling back to CPU domain control"
          apply_cpu_ctrl || { err "Fallback CPU CONTROL writes failed"; return 2; }
        fi
      else
        apply_cpu_ctrl || { err "CPU CONTROL writes failed"; return 2; }
      fi
      ;;
  esac

  log "Applied=${applied} Failed=${failed} (block: $conf)"
  total_applied=$((total_applied + applied))
  total_failed=$((total_failed + failed))
}
