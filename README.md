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
nohup python3 -m main_controller \
    --mode record_performance \
    --record-idle-baseline \
    --idle-sample-seconds 45 \
    --record-output-csv output/perf_results.csv \
    --pdu-map data/mapping.json \
    --pdu-user apc \
    --pdu-password ridl123 \
    --pdu-csv output/pdu_log.csv \
    --events-csv output/overload_events.csv \
    > main_controller.log 2>&1 &
```
This collects a 45-second idle baseline, launches the sbatch variations listed in `data/slurm_scripts.txt`, applies GEOPM reductions as jobs start, and appends per-job metrics to `output/perf_results.csv`.

## Record Performance (Dry-Run Preview)
Preview the generated Slurm submissions (including resolved nodes, ranks, size and lookups) without launching any jobs:
```
nohup python3 -m main_controller \
    --mode record_performance \
    --record-dry-run \
    --record-idle-baseline \
    --idle-sample-seconds 45 \
    --pdu-map data/mapping.json \
    --pdu-user apc \
    --pdu-password ridl123 \
    --pdu-csv output/pdu_log.csv \
    --events-csv output/overload_events.csv \
    > main_controller.log 2>&1 &
```
Notes:
- Writes “[Dry-Run] …” lines to `main_controller.log` with the exact `sbatch --wrap srun` commands that would be submitted.
- Still collects the idle baseline if `--record-idle-baseline` is present; omit it to preview instantly.
- No jobs are submitted; the controller exits after preview.

## Overload Detection Experiments
To monitor live power and trigger DVFS reductions when thresholds are exceeded:
```
nohup python3 -m main_controller \
    --mode run_experiment \
    --detect-overload \
    --threshold-w 850 \
    --hysteresis-w 40 \
    --min-over 5 \
    --cooldown 30 \
    --pdu-map data/mapping.json \
    --pdu-user apc \
    --pdu-password ridl123 \
    --pdu-csv output/pdu_log.csv \
    --events-csv output/overload_events.csv \
    > main_controller.log 2>&1 &
```
With these flags the controller streams PDU data, raises events when sustained load crosses 850 W, and calls the market + DVFS managers to reduce job frequencies until the overload is handled.

## Quick Stress Test (Single Node)
- Purpose: sweep CPU frequency on one node and observe power/runtime response.
- Requirements: `stress-ng` on compute nodes; optional PDU config for power logging.

Example:
```
nohup python3 utilities/simple_stress_record.py \
  --max-freq 2400 --min-freq 1000 --interval 200 \
  --duration 180 --threads 10 \
  --nodelist ridlserver11 --exclude ridlserver12 \
  --pdu-map data/mapping.json \
  --pdu-user apc --pdu-password ridl123 \
  --pdu-csv output/pdu_log_simple.csv \
  --output-csv output/stress_results.csv \
  > stress_record.log 2>&1 &
```
Tail `stress_record.log` for progress; results accumulate in `output/stress_results.csv`.
Use `--nodelist` / `--exclude` (supports Slurm syntax like `nodeA,nodeB` or `node[01-04]`) or export `TARGET_NODE` / `EXCLUDE_NODE` to steer node placement. Update to the latest repo revision to ensure these CLI options are available.

Quick way to inspect the collected metrics (including `avg_power_w`, `net_avg_power_w`, and `idle_power_w`):
```
column -t -s, output/stress_results.csv | sed '1,5p'
```

Example to target a specific node and exclude another:
```
# Launch the sweep and capture power metrics for ridlserver11
nohup python3 utilities/simple_stress_record.py \
  --max-freq 2400 --min-freq 1000 --interval 200 \
  --duration 180 --threads 10 \
  --nodelist ridlserver11 --exclude ridlserver12 \
  --pdu-map data/mapping.json \
  --pdu-user apc --pdu-password ridl123 \
  --pdu-csv output/pdu_log_simple.csv \
  --output-csv output/stress_results_ridlserver11.csv \
  > stress_record_ridlserver11.log 2>&1 &

# Preview the first few result rows (shows avg/net/idle power)
column -t -s, output/stress_results_ridlserver11.csv | sed '1,10p'
```
To target a group while excluding one of them:
```
nohup python3 utilities/simple_stress_record.py \
  --max-freq 2400 --min-freq 1000 --interval 200 \
  --duration 180 --threads 10 \
  --nodelist ridlserver[11-14] --exclude ridlserver13 \
  --output-csv output/stress_results_ridlserver11_14.csv \
  > stress_record_ridlserver11_14.log 2>&1 &
```
```
```
Notes:
- Uses `data/slurm_scripts/run_stressng.slurm` (matrix + cpu stressors) and binds cores.
- Provide --pdu-map / --pdu-user / --pdu-password to log power to --pdu-csv; the helper records an idle baseline for the selected nodes before the sweep so avg_power_w, net_avg_power_w, and idle_power_w are populated. Without these flags the power columns stay blank.
- Records an idle baseline for `--idle-seconds` (default 45) before the sweep.
- Generates target frequencies from max→min by `--interval` MHz; appends one row per step to `--output-csv`.
