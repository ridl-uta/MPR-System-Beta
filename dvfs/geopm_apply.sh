#!/usr/bin/env bash
# Apply GEOPM core frequency settings from a host-specific config file.
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
# - CPUS: space-separated logical CPU IDs to target. For backward compatibility,
#         these are mapped to unique core IDs and ignored if CORES is set.
# - CONTROL_KIND: AUTO | PERF_CTL | CORE_MAX  (default: PERF_CTL)
# - CPUFREQ_SYNC: 1 to align kernel cpufreq policy with FREQ_HZ on targeted
#                 cores before GEOPM writes (default: 0)
# - CPUFREQ_GOVERNOR: governor used when CPUFREQ_SYNC=1 (default: userspace)
# - CPUFREQ_MIN_KHZ: minimum cpufreq policy floor when CPUFREQ_SYNC=1
#                    (default: 1000000)
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

map_cpu_to_core() {
  local cpu_id="$1"
  lscpu -p=CPU,CORE | awk -F, -v k="$cpu_id" '/^[^#]/ && $1==k {print $2; exit}'
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

sysfs_write() {
  local path="$1" value="$2"
  printf '%s\n' "$value" > "$path"
}

hz_to_khz() {
  local hz="$1"
  awk -v hz="$hz" 'BEGIN{printf "%.0f", hz/1000}'
}

read_cpu_list_file() {
  local path="$1" raw
  raw=$(tr -s '[:space:]' ',' < "$path")
  raw=${raw#,}
  raw=${raw%,}
  [[ -n "${raw:-}" ]] || return 0
  expand_list "$raw"
}

sync_cpufreq_policy() {
  local target_khz policy_min_khz policy_path policy_name
  local target_cpus=""
  local core cpu current_min current_max extra_cpus
  local selected=0 rc=0

  if [[ ${#TARGET_CORES[@]} -eq 0 ]]; then
    mapfile -t TARGET_CORES < <(lscpu -p=CORE | awk -F, '/^[^#]/{print $1}' | sort -n | uniq)
  fi

  for core in "${TARGET_CORES[@]}"; do
    while read -r cpu; do
      [[ -z "$cpu" ]] && continue
      target_cpus+="${cpu} "
    done < <(map_core_to_cpus "$core")
  done

  [[ -n "${target_cpus:-}" ]] || return 0

  target_khz=$(hz_to_khz "$FREQ_HZ")
  policy_min_khz="$CPUFREQ_MIN_KHZ"
  if (( policy_min_khz > target_khz )); then
    policy_min_khz="$target_khz"
  fi
  for policy_path in /sys/devices/system/cpu/cpufreq/policy*; do
    [[ -d "$policy_path" && -r "$policy_path/affected_cpus" ]] || continue

    selected=0
    while read -r cpu; do
      [[ -z "$cpu" ]] && continue
      if grep -q "\b${cpu}\b" <<<" ${target_cpus} "; then
        selected=1
        break
      fi
    done < <(read_cpu_list_file "$policy_path/affected_cpus")

    [[ "$selected" == "1" ]] || continue

    policy_name=$(basename "$policy_path")
    extra_cpus=""
    while read -r cpu; do
      [[ -z "$cpu" ]] && continue
      if ! grep -q "\b${cpu}\b" <<<" ${target_cpus} "; then
        extra_cpus+="${cpu} "
      fi
    done < <(read_cpu_list_file "$policy_path/affected_cpus")
    if [[ -n "${extra_cpus:-}" ]]; then
      log "policy ${policy_name}: target expands to CPUs ${extra_cpus}"
    fi

    if [[ -e "$policy_path/scaling_governor" ]]; then
      if ! write_with_retry sysfs_write "$policy_path/scaling_governor" "$CPUFREQ_GOVERNOR"; then
        err "policy ${policy_name}: failed to set governor=${CPUFREQ_GOVERNOR}"
        rc=1
        continue
      fi
    fi

    current_min=$(cat "$policy_path/scaling_min_freq")
    current_max=$(cat "$policy_path/scaling_max_freq")
    if (( target_khz >= current_max )); then
      if ! write_with_retry sysfs_write "$policy_path/scaling_max_freq" "$target_khz"; then
        err "policy ${policy_name}: failed to set scaling_max_freq=${target_khz}"
        rc=1
        continue
      fi
      if ! write_with_retry sysfs_write "$policy_path/scaling_min_freq" "$policy_min_khz"; then
        err "policy ${policy_name}: failed to set scaling_min_freq=${policy_min_khz}"
        rc=1
        continue
      fi
    else
      if ! write_with_retry sysfs_write "$policy_path/scaling_min_freq" "$policy_min_khz"; then
        err "policy ${policy_name}: failed to set scaling_min_freq=${policy_min_khz}"
        rc=1
        continue
      fi
      if ! write_with_retry sysfs_write "$policy_path/scaling_max_freq" "$target_khz"; then
        err "policy ${policy_name}: failed to set scaling_max_freq=${target_khz}"
        rc=1
        continue
      fi
    fi

    if [[ -e "$policy_path/scaling_setspeed" ]]; then
      if ! write_with_retry sysfs_write "$policy_path/scaling_setspeed" "$target_khz"; then
        err "policy ${policy_name}: failed to set scaling_setspeed=${target_khz}"
        rc=1
        continue
      fi
    fi

    log "policy ${policy_name}: governor=${CPUFREQ_GOVERNOR} min_khz=${policy_min_khz} max_khz=${target_khz}"
  done

  return $rc
}

# Helper: apply a single block config from file (contains FREQ_HZ/CORES/CPUS/etc.)
apply_block_from_file() {
  local conf="$1"

  # Reset per-config variables/state
  unset FREQ_HZ CORES CPUS CONTROL_KIND CPUFREQ_SYNC CPUFREQ_GOVERNOR CPUFREQ_MIN_KHZ RESPECT_CPUSET RETRIES RETRY_SLEEP
  ALLOWED_CPUS=""
  TARGET_CORES=()
  applied=0
  failed=0

  # shellcheck source=/dev/null
  set -a
  source "$conf"
  set +a

  CONTROL_KIND=${CONTROL_KIND:-PERF_CTL}
  if [[ "$CONTROL_KIND" == "CPU" ]]; then
    CONTROL_KIND="PERF_CTL"
  fi
  CPUFREQ_SYNC=${CPUFREQ_SYNC:-0}
  CPUFREQ_GOVERNOR=${CPUFREQ_GOVERNOR:-userspace}
  CPUFREQ_MIN_KHZ=${CPUFREQ_MIN_KHZ:-1000000}
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

  core_in_allowed() {
    local core="$1"
    local cpu
    [[ -z "$ALLOWED_CPUS" ]] && return 0
    while read -r cpu; do
      [[ -z "$cpu" ]] && continue
      if in_allowed "$cpu"; then
        return 0
      fi
    done < <(map_core_to_cpus "$core")
    return 1
  }

  # Build target lists
  if [[ -n "${CORES:-}" ]]; then
    local c
    for c in $CORES; do
      core_in_allowed "$c" || continue
      TARGET_CORES+=("$c")
    done
  elif [[ -n "${CPUS:-}" ]]; then
    local cpu
    local core
    for cpu in $CPUS; do
      in_allowed "$cpu" || continue
      core=$(map_cpu_to_core "$cpu")
      [[ -z "$core" ]] && continue
      TARGET_CORES+=("$core")
    done
  else
    while read -r core; do
      [[ -z "$core" ]] && continue
      core_in_allowed "$core" || continue
      TARGET_CORES+=("$core")
    done < <(lscpu -p=CORE | awk -F, '/^[^#]/ {print $1}' | sort -n | uniq)
  fi

  if (( ${#TARGET_CORES[@]} > 0 )); then
    mapfile -t TARGET_CORES < <(printf '%s\n' "${TARGET_CORES[@]}" | awk 'NF' | sort -n | uniq)
  fi

  if [[ "$CPUFREQ_SYNC" == "1" ]]; then
    sync_cpufreq_policy || { err "CPUFreq policy sync failed"; return 2; }
  fi

  local has_access_perf_ctl=0
  local has_access_core_max=0
  if geopmaccess -l 2>/dev/null | grep -q "MSR::PERF_CTL:FREQ.*core"; then
    has_access_perf_ctl=1
  fi
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
    PERF_CTL)
      if ! apply_perf_ctl; then
        err "PERF_CTL writes failed"
        return 2
      fi
      ;;
    AUTO|*)
      if [[ "$has_access_perf_ctl" == "1" ]]; then
        if apply_perf_ctl; then
          :
        elif [[ "$has_access_core_max" == "1" ]]; then
          log "Falling back to CORE_MAX"
          apply_core_max || { err "Fallback CORE_MAX writes failed"; return 2; }
        else
          err "PERF_CTL writes failed and no CORE_MAX fallback is available"
          return 2
        fi
      elif [[ "$has_access_core_max" == "1" ]]; then
        apply_core_max || { err "CORE_MAX writes failed"; return 2; }
      else
        err "Neither MSR::PERF_CTL:FREQ core nor CPU_FREQUENCY_MAX_CONTROL core is available"
        return 2
      fi
      ;;
  esac

  log "Applied=${applied} Failed=${failed} (block: $conf)"
  total_applied=$((total_applied + applied))
  total_failed=$((total_failed + failed))
}

apply_core_max() {
  local rc=0
  if [[ ${#TARGET_CORES[@]} -eq 0 ]]; then
    mapfile -t TARGET_CORES < <(lscpu -p=CORE | awk -F, '/^[^#]/{print $1}' | sort -n | uniq)
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

apply_perf_ctl() {
  local rc=0
  if [[ ${#TARGET_CORES[@]} -eq 0 ]]; then
    mapfile -t TARGET_CORES < <(lscpu -p=CORE | awk -F, '/^[^#]/{print $1}' | sort -n | uniq)
  fi
  for core in "${TARGET_CORES[@]}"; do
    if write_with_retry geopmwrite MSR::PERF_CTL:FREQ core "$core" "$FREQ_HZ"; then
      log "core ${core}: PERF_CTL:FREQ=${FREQ_HZ}"
      applied=$((applied+1))
    else
      err "core ${core}: write PERF_CTL:FREQ failed"
      rc=1
      failed=$((failed+1))
    fi
  done
  return $rc
}

# Apply a single configuration file
apply_one_conf() {
  local conf="$1"
  local rc=0
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
        if ! apply_block_from_file "$b"; then
          rc=1
        fi
      done
    else
      err "Found RULE markers but no blocks parsed in $conf"
      rc=1
    fi
    rm -rf "$tmpdir"
    return "$rc"
  fi

  # Single-block classic config
  apply_block_from_file "$conf"
}

total_applied=0
total_failed=0
for conf in "${CONFIGS[@]}"; do
  if ! apply_one_conf "$conf"; then
    :
  fi
done

log "Total Applied=${total_applied} Failed=${total_failed}"
if (( total_failed > 0 || total_applied == 0 )); then
  exit 2
fi
exit 0
