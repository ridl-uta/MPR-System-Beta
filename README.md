# MPR System Beta

Main runtime entrypoint is `run_main.py` (renamed from `prototype_run.py`).

## Install Dependencies
From repo root:

```bash
python3 -m pip install -r requirements.txt -r mpr_int/requirements.txt -r power_monitor/requirements.txt
```

If you need real APC telnet polling in `power_monitor`, use Python 3.12 (telnetlib is removed in Python 3.13+).

## Run `run_main.py`

Show CLI help:

```bash
python3 run_main.py --help
```

### Foreground (blocking terminal)

```bash
python3 run_main.py \
  --rank xsbenchmpi=2 \
  --current-power-w 900 \
  --target-capacity-w 700 \
  --skip-dvfs-apply
```

### Background (non-blocking terminal)

```bash
mkdir -p output
nohup python3 run_main.py \
  --rank comd=1 \
  --rank minife=2 \
  --enable-power-monitor \
  --pdu-user <user> \
  --pdu-password <password> \
  --target-capacity-w 900 \
  > output/run_main.log 2>&1 < /dev/null &
echo $! > output/run_main.pid
```

Useful process commands:

```bash
tail -f output/run_main.log
ps -fp "$(cat output/run_main.pid)"
kill "$(cat output/run_main.pid)"
```

### 1) Dry-run Slurm submission only

```bash
python3 run_main.py \
  --dry-run \
  --rank xsbenchmpi=2 \
  --current-power-w 600 \
  --target-capacity-w 700 \
  --skip-dvfs-apply
```

### 1b) Dry-run in background (non-blocking)

```bash
mkdir -p output
nohup python3 run_main.py \
  --dry-run \
  --rank xsbenchmpi=2 \
  --current-power-w 600 \
  --target-capacity-w 700 \
  --skip-dvfs-apply \
  > output/run_main_dry.log 2>&1 < /dev/null &
echo $! > output/run_main_dry.pid
```

### 2) Run market path (force overload)

```bash
python3 run_main.py \
  --dry-run \
  --rank xsbenchmpi=2 \
  --current-power-w 900 \
  --target-capacity-w 700 \
  --skip-dvfs-apply
```

### 2b) Dry-run market path in background (non-blocking)

```bash
mkdir -p output
nohup python3 run_main.py \
  --dry-run \
  --rank xsbenchmpi=2 \
  --current-power-w 900 \
  --target-capacity-w 700 \
  --skip-dvfs-apply \
  > output/run_main_market_dry.log 2>&1 < /dev/null &
echo $! > output/run_main_market_dry.pid
```

### 3) Enable background power monitor

```bash
python3 run_main.py \
  --dry-run \
  --rank comd=1 \
  --rank minife=2 \
  --enable-power-monitor \
  --pdu-user <user> \
  --pdu-password <password> \
  --target-capacity-w 900
```

### 3b) Dry-run with power monitor in background (non-blocking)

```bash
mkdir -p output
nohup python3 run_main.py \
  --dry-run \
  --rank comd=1 \
  --rank minife=2 \
  --enable-power-monitor \
  --pdu-user <user> \
  --pdu-password <password> \
  --target-capacity-w 900 \
  > output/run_main_power_dry.log 2>&1 < /dev/null &
echo $! > output/run_main_power_dry.pid
```

## Job Submission Data Sources

- Slurm scripts: `job_scheduler/data/slurm_scripts/run_*.slurm`
- Per-rank benchmark args: `job_scheduler/data/slurm_scripts/*.csv`
- Script metadata (workdir/bin): `job_scheduler/data/slurm_scripts.txt`
- Performance workbook used by market: `job_scheduler/data/all_model_data.xlsx`

For `minimd` and `comd`, script args are loaded from the CSV by rank unless overridden with `--job-args`.

## Common Submission Flags

- `--rank JOB=RANK` (required, repeatable)
- `--cpus-per-rank`, `--ranks-per-node`
- `--partition`, `--time-limit`, `--nodelist`, `--exclude`
- `--submit-interval-s` for periodic spacing between submissions
- `--mpi-iface {pmi2,pmix}`
- `--slurm-output /path/job-%j.out`
- `--submit-env KEY=VALUE` (repeatable)
- `--job-args "JOB=<arg string>"` (repeatable)
- `--disable-rank-profiles` to ignore CSV rank profiles

## Notes

- `--dry-run` prints exact `sbatch` commands and does not submit.
- If overload is `<= 0`, market/DVFS actions are skipped.
- DVFS apply requires real submitted allocations (`--dry-run` + `--skip-submit` will not have allocations).
