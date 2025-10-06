# DVFS + GEOPM: On‑Demand Apply

Minimal runbook to install the one‑shot service on nodes and trigger applies from a head/login node.

## Install On Each Compute Node
- From the repo root on the node:
  - `sudo bash dvfs/install_geopm_apply_systemd.sh`
- This installs:
  - Service: `dvfs/systemd/geopm-apply.service`
  - Script: `dvfs/geopm_apply.sh`

## Create Per‑Host Configs
- Place per‑host configs on a shared path:
  - `/shared/geopm/freq/<hostname>.conf`
- Minimal multi‑rule example: `dvfs/geopm/hostname.conf.example`
  - Copy/rename per node and edit frequencies/cores.

## Trigger From Head Node
- Using the helper script (prefers service; falls back to direct):
  - Single host: `dvfs/run_geopm_apply_ssh.sh -u <user> <host>`
  - Host list: `dvfs/run_geopm_apply_ssh.sh -u <user> -H data/nodes.txt`
  - Direct (bypass service): `dvfs/run_geopm_apply_ssh.sh --direct --no-sudo -u <user> -H data/nodes.txt`
- Without the helper (robust one‑liner):
  - `while read -r h; do ssh "$h" 'sudo -n systemctl start geopm-apply.service 2>/dev/null || /usr/local/sbin/geopm_apply.sh'; done < data/nodes.txt`

## Verify On A Node
- Read back controls (CORE_MAX example):
- `for c in 0 2 4 6 8; do echo -n "core $c: "; geopmread CPU_FREQUENCY_MAX_CONTROL core $c; done`
- `for c in 1 3 5 7 9; do echo -n "core $c: "; geopmread CPU_FREQUENCY_MAX_CONTROL core $c; done`
- Service logs:
  - `sudo journalctl -u geopm-apply.service -n 50 --no-pager`

## Notes
- The service runs `geopm_apply.sh /shared/geopm/freq/%H.conf` (processes multi‑RULE blocks inside that file).
- Direct runs with no args also include any drop‑ins at `/shared/geopm/freq/<hostname>.d/*.conf`.
- To avoid sudo prompts when using the service via SSH, add a NOPASSWD sudoers rule for starting `geopm-apply.service`.

## Troubleshooting
- "sudo: a password is required": use `--direct --no-sudo` or configure NOPASSWD.
- Writes fail without sudo: ensure user has GEOPM write perms (`geopmaccess -l`) or run via service with sudo.
- No config found: confirm `/shared/geopm/freq/<hostname>.conf` exists on the node.
- GEOPM daemon not running: check `sudo systemctl status geopmd.service`.

## Record Performance Runs
To sweep job/frequency combinations and log runtime + average power:
```
python3 -m main_controller \
    --mode record_performance \
    --record-idle-baseline \
    --idle-sample-seconds 45 \
    --record-output-csv output/perf_results.csv \
    --pdu-map data/mapping.json \
    --pdu-user apc \
    --pdu-password ridl123 \
    --pdu-csv output/pdu_log.csv \
    --events-csv output/overload_events.csv
```
This collects a 45-second idle baseline, launches the sbatch variations listed in `data/slurm_scripts.txt`, applies GEOPM reductions as jobs start, and appends per-job metrics to `output/perf_results.csv`.
