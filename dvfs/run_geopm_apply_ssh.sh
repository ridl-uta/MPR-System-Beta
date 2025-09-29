#!/usr/bin/env bash
# Run GEOPM frequency apply on remote nodes via SSH from a head/login node.
#
# By default this triggers the systemd oneshot service `geopm-apply.service`
# on each node if available, falling back to calling the apply script directly:
#   /usr/local/sbin/geopm_apply.sh
#
# Requirements on target nodes:
# - geopm_apply.sh installed at /usr/local/sbin (see dvfs/install_geopm_apply_systemd.sh)
# - Optional: systemd unit geopm-apply.service installed for convenience
# - A per-host config at /shared/geopm/freq/<hostname>.conf (or use --conf)

set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [host ...]

Hosts can be provided as positional args or via --hosts <file>.

Options:
  -h, --help            Show this help and exit
  -H, --hosts FILE      File with hostnames (one per line; # comments ok)
  -u, --user USER       SSH username
  -i, --identity FILE   SSH identity (private key)
  -c, --concurrency N   Max parallel SSH sessions (default: 8)
      --no-sudo         Do not attempt sudo on remote
      --direct          Call apply script directly (skip systemd service)
      --conf PATH       Config path to pass to geopm_apply.sh on remote
      --dry-run         Print actions without executing
      --insecure-ssh    Disable StrictHostKeyChecking (use with care)

Examples:
  # Apply using systemd service per node (default path /shared/geopm/freq/%H.conf)
  $(basename "$0") -H nodes.txt

  # Apply directly with a specific config path on all nodes
  $(basename "$0") --direct --conf /shared/geopm/freq/custom.conf nodeA nodeB

  # From within a Slurm job allocation
  scontrol show hostnames "$SLURM_JOB_NODELIST" | $(basename "$0") -c 16 --no-sudo -H -
EOF
}

HOST_FILES=()
HOSTS=()
SSH_USER=""
SSH_IDENTITY=""
CONCURRENCY=8
ALLOW_SUDO=1
FORCE_DIRECT=0
CONF_PATH=""
DRY_RUN=0
INSECURE_SSH=0

parse_hosts_file() {
  local f="$1"
  if [[ "$f" == "-" ]]; then
    # stdin
    while IFS= read -r line; do
      [[ -z "${line//[[:space:]]/}" ]] && continue
      [[ "$line" =~ ^[[:space:]]*# ]] && continue
      HOSTS+=("$line")
    done
  else
    [[ -r "$f" ]] || { echo "[ERR] cannot read hosts file: $f" >&2; exit 1; }
    while IFS= read -r line; do
      [[ -z "${line//[[:space:]]/}" ]] && continue
      [[ "$line" =~ ^[[:space:]]*# ]] && continue
      HOSTS+=("$line")
    done <"$f"
  fi
}

# Parse args
if (( $# == 0 )); then
  usage; exit 1
fi

while (( $# > 0 )); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -H|--hosts)
      shift; [[ $# -gt 0 ]] || { echo "[ERR] --hosts needs a file" >&2; exit 1; }
      HOST_FILES+=("$1"); shift ;;
    -u|--user)
      shift; [[ $# -gt 0 ]] || { echo "[ERR] --user needs a value" >&2; exit 1; }
      SSH_USER="$1"; shift ;;
    -i|--identity)
      shift; [[ $# -gt 0 ]] || { echo "[ERR] --identity needs a file" >&2; exit 1; }
      SSH_IDENTITY="$1"; shift ;;
    -c|--concurrency)
      shift; [[ $# -gt 0 ]] || { echo "[ERR] --concurrency needs a number" >&2; exit 1; }
      CONCURRENCY="$1"; shift ;;
    --no-sudo)
      ALLOW_SUDO=0; shift ;;
    --direct)
      FORCE_DIRECT=1; shift ;;
    --conf)
      shift; [[ $# -gt 0 ]] || { echo "[ERR] --conf needs a path" >&2; exit 1; }
      CONF_PATH="$1"; FORCE_DIRECT=1; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --insecure-ssh)
      INSECURE_SSH=1; shift ;;
    --)
      shift; break ;;
    -*)
      echo "[ERR] unknown option: $1" >&2; exit 1 ;;
    *)
      HOSTS+=("$1"); shift ;;
  esac
done

# Any remaining args as hosts
while (( $# > 0 )); do HOSTS+=("$1"); shift; done

for f in "${HOST_FILES[@]:-}"; do
  parse_hosts_file "$f"
done

if (( ${#HOSTS[@]} == 0 )); then
  echo "[ERR] no hosts provided" >&2
  exit 1
fi

command -v ssh >/dev/null 2>&1 || { echo "[ERR] ssh not found" >&2; exit 1; }

ssh_base=(ssh -o BatchMode=yes -o ConnectTimeout=5)
if (( INSECURE_SSH == 1 )); then
  ssh_base+=( -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null )
fi
if [[ -n "$SSH_IDENTITY" ]]; then
  ssh_base+=( -i "$SSH_IDENTITY" )
fi

# Build remote action template
build_remote_cmd() {
  local direct="$1" conf="$2" allow_sudo="$3"
  local conf_q=""
  if [[ -n "$conf" ]]; then
    # quote for remote shell
    conf_q=$(printf '%q' "$conf")
  fi

  if (( direct == 1 )); then
    if (( allow_sudo == 1 )); then
      echo "sudo -n /usr/local/sbin/geopm_apply.sh ${conf_q} || /usr/local/sbin/geopm_apply.sh ${conf_q}"
    else
      echo "/usr/local/sbin/geopm_apply.sh ${conf_q}"
    fi
  else
    if (( allow_sudo == 1 )); then
      echo "sudo -n systemctl start geopm-apply.service >/dev/null 2>&1 || systemctl start geopm-apply.service >/dev/null 2>&1 || sudo -n /usr/local/sbin/geopm_apply.sh ${conf_q} || /usr/local/sbin/geopm_apply.sh ${conf_q}"
    else
      echo "systemctl start geopm-apply.service >/dev/null 2>&1 || /usr/local/sbin/geopm_apply.sh ${conf_q}"
    fi
  fi
}

REMOTE_CMD_TEMPLATE=$(build_remote_cmd "$FORCE_DIRECT" "$CONF_PATH" "$ALLOW_SUDO")

run_one() {
  local host="$1"
  local target="$host"
  if [[ -n "$SSH_USER" ]]; then
    target="${SSH_USER}@${host}"
  fi
  local cmd=("${ssh_base[@]}" "$target" "$REMOTE_CMD_TEMPLATE")
  if (( DRY_RUN == 1 )); then
    printf '[DRY] %s\n' "${cmd[*]}"
    return 0
  fi
  if "${cmd[@]}"; then
    echo "[$host] OK"
  else
    echo "[$host] FAIL" >&2
    return 1
  fi
}

# Run with basic concurrency
rc=0
if command -v bash >/dev/null 2>&1 && [[ "${BASH_VERSINFO:-0}" -ge 4 ]]; then
  # Use wait -n if available
  running=0
  pids=()
  hosts_copy=("${HOSTS[@]}")
  i=0
  for host in "${hosts_copy[@]}"; do
    run_one "$host" &
    pids+=($!)
    running=$((running+1))
    if (( running >= CONCURRENCY )); then
      if wait -n; then :; else rc=1; fi
      running=$((running-1))
    fi
  done
  # Wait remaining
  for pid in "${pids[@]}"; do
    if wait "$pid"; then :; else rc=1; fi
  done
else
  # Fallback: sequential
  for host in "${HOSTS[@]}"; do
    if run_one "$host"; then :; else rc=1; fi
  done
fi

exit "$rc"

