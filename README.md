# MPR System Beta

Main runtime entrypoint is `run_main.py`.

## Install Dependencies

From repo root:

```bash
python3 -m pip install -r requirements.txt -r mpr_int/requirements.txt -r power_monitor/requirements.txt
```

If you need real APC telnet polling in `power_monitor`, use Python 3.12 (`telnetlib` is removed in Python 3.13+).

## CLI Help

```bash
python3 run_main.py --help
```

## Runtime Model

Execution is controlled by a few independent switches:

1. Submission mode:
- normal submit (default)
- `--dry-run` (preview `sbatch` commands)
- `--skip-submit` (do not submit any jobs)

2. Power source:
- `--current-power-w <watts>` (static value)
- `--enable-power-monitor` + PDU credentials (live samples)

3. Control mode:
- full control (MPR + DVFS apply + reset on `OVERLOAD_END`)
- `--skip-dvfs-apply` (MPR only, no DVFS apply/reset)
- `--detect-overload-only` (event detection only, no MPR/DVFS actions)

4. Post-job event window:
- `--post-jobs-monitor-s <seconds>` (default: `60`)
- keeps monitor/event loop alive after jobs complete so late `OVERLOAD_END` can still fire

Important behavior:
- DVFS reset to max frequency is applied on `OVERLOAD_END`.
- If `OVERLOAD_END` does not arrive before `--post-jobs-monitor-s` expires, a safety reset to max frequency is applied before exit (except `--detect-overload-only` mode).

## Command Templates (All Common Variations)

### A) Dry-run submit preview only

```bash
python3 run_main.py \
  --dry-run \
  --rank xsbenchmpi=2 \
  --rank comd=2 \
  --current-power-w 680 \
  --target-capacity-w 730
```

### B) Dry-run with forced overload path (MPR executes, DVFS skipped)

```bash
python3 run_main.py \
  --dry-run \
  --rank xsbenchmpi=2 \
  --rank comd=2 \
  --current-power-w 900 \
  --target-capacity-w 730 \
  --skip-dvfs-apply
```

### C) Real submit + static power input (no live monitor)

```bash
python3 run_main.py \
  --rank xsbenchmpi=2 \
  --rank comd=2 \
  --current-power-w 900 \
  --target-capacity-w 730
```

### D) Real submit + live PDU monitor + full control

```bash
python3 run_main.py \
  --rank xsbenchmpi=2 \
  --rank comd=2 \
  --enable-power-monitor \
  --pdu-user apc \
  --pdu-password '<PASS>' \
  --pdu-csv output/pdu_log.csv \
  --target-capacity-w 730 \
  --submit-interval-s 15 \
  --power-print-interval-s 2 \
  --post-jobs-monitor-s 60
```

### E) Real submit + live monitor + MPR only (no DVFS apply)

```bash
python3 run_main.py \
  --rank xsbenchmpi=2 \
  --rank comd=2 \
  --enable-power-monitor \
  --pdu-user apc \
  --pdu-password '<PASS>' \
  --target-capacity-w 730 \
  --skip-dvfs-apply \
  --post-jobs-monitor-s 60
```

### F) Real submit + live monitor + detect-only mode (no MPR/DVFS actions)

```bash
python3 run_main.py \
  --rank xsbenchmpi=2 \
  --rank comd=2 \
  --enable-power-monitor \
  --pdu-user apc \
  --pdu-password '<PASS>' \
  --target-capacity-w 730 \
  --detect-overload-only \
  --post-jobs-monitor-s 60
```

### G) Observe existing system only (no submit, detect-only)

```bash
python3 run_main.py \
  --skip-submit \
  --rank xsbenchmpi=2 \
  --enable-power-monitor \
  --pdu-user apc \
  --pdu-password '<PASS>' \
  --target-capacity-w 730 \
  --detect-overload-only \
  --post-jobs-monitor-s 120
```

Notes:
- `--rank` is still required by parser even with `--skip-submit`.
- With `--skip-submit`, if there are no submitted IDs in this run, event loop will not start.

### H) Detached full-control run (nohup)

```bash
mkdir -p output
nohup python3 run_main.py \
  --rank xsbenchmpi=2 \
  --rank comd=2 \
  --submit-interval-s 15 \
  --enable-power-monitor \
  --pdu-user apc \
  --pdu-password '<PASS>' \
  --pdu-csv output/pdu_log.csv \
  --target-capacity-w 730 \
  --overload-handled-high-margin-w 20 \
  --overload-handled-low-margin-w 15 \
  --power-print-interval-s 2 \
  --post-jobs-monitor-s 60 \
  --quiet \
  > output/run_main_detached.log 2>&1 < /dev/null &
echo $! > output/run_main_detached.pid
```

### I) Detached detect-only run

```bash
mkdir -p output
nohup python3 run_main.py \
  --rank xsbenchmpi=2 \
  --rank comd=2 \
  --submit-interval-s 15 \
  --enable-power-monitor \
  --pdu-user apc \
  --pdu-password '<PASS>' \
  --pdu-csv output/pdu_log.csv \
  --target-capacity-w 730 \
  --detect-overload-only \
  --post-jobs-monitor-s 60 \
  --quiet \
  > output/run_main_detect_only.log 2>&1 < /dev/null &
echo $! > output/run_main_detect_only.pid
```

Process commands:

```bash
tail -f output/run_main_detached.log
ps -fp "$(cat output/run_main_detached.pid)"
kill "$(cat output/run_main_detached.pid)"
```

## Overload Detection and Lifecycle Parameters

Use these to tune event behavior:

- `--target-capacity-w`
- `--overload-hysteresis-w`
- `--overload-min-over-s`
- `--overload-cooldown-s`
- `--overload-handled-window-s`
- `--overload-handled-high-margin-w`
- `--overload-handled-low-margin-w`
- `--post-jobs-monitor-s`

Practical tip:
- If you want `OVERLOAD_END` to appear after jobs finish, increase `--post-jobs-monitor-s` and/or reduce `--overload-cooldown-s`.

## DVFS Parameters

- `--max-freq-mhz`
- `--dvfs-step-mhz`
- `--dvfs-verify-tol-mhz`
- `--dvfs-ssh-user`
- `--skip-dvfs-apply`
- `--detect-overload-only` (disables all MPR/DVFS actions)

## Submission Parameters

- `--rank JOB=RANK` (required, repeatable)
- `--cpus-per-rank`, `--ranks-per-node`
- `--partition`, `--time-limit`
- `--submit-interval-s`
- `--nodelist`, `--exclude`
- `--mpi-iface {pmi2,pmix}`
- `--slurm-output`
- `--submit-env KEY=VALUE` (repeatable)
- `--job-args "JOB=<arg string>"` (repeatable)
- `--disable-rank-profiles`
- `--show-allocation`

## Monitoring Parameters

- `--enable-power-monitor`
- `--pdu-user`, `--pdu-password`
- `--pdu-csv`, `--pdu-map`
- `--power-interval-s`
- `--power-deadline-s`
- `--power-print-interval-s`
- `--power-startup-wait-s`
- `--job-poll-interval-s`
- `--job-status-print-interval-s`

## Job Submission Data Sources

- Slurm scripts: `job_scheduler/data/slurm_scripts/run_*.slurm`
- Per-rank benchmark args: `job_scheduler/data/slurm_scripts/*.csv`
- Script metadata (workdir/bin): `job_scheduler/data/slurm_scripts.txt`
- Market workbook by rank: `job_scheduler/data/all_model_data_by_rank.xlsx`
- Legacy workbook: `job_scheduler/data/all_model_data.xlsx`

For `minimd` and `comd`, script args are loaded from CSV by rank unless overridden with `--job-args`.

## Notes

- `--dry-run` prints exact `sbatch` commands and does not submit.
- If overload is `<= 0`, market/DVFS reduction actions are skipped.
- DVFS apply requires real submitted allocations.
- `--detect-overload-only` keeps event detection active while skipping MPR and DVFS operations.

## Overload Detection Unit Tests

Unit tests for handled-zone overload logic are in `tests/test_overload_detection.py`.
DVFS core-mapping and verification tests are in `tests/test_slurm_core_mapping.py`.

Run only this test module:

```bash
python3 -m unittest -v tests.test_overload_detection
```

Run all tests under `tests/`:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Run only the DVFS core-mapping tests:

```bash
python3 -m unittest -v tests.test_slurm_core_mapping
```

Run one specific test:

```bash
python3 -m unittest -v tests.test_overload_detection.TestOverloadDetectionHandledZone.test_handled_zone_allows_power_above_target
```

These tests validate:
- handled zone can include watts above target (`handled_high_margin_w`)
- zero high margin does not mark above-target power as handled
- explicit low-margin override is applied to handled-zone bounds
- Slurm CPU IDs are translated into unique physical core IDs before `CORE_MAX` apply
- DVFS verification does not fall back to unrelated node-wide readback values

## Live Slurm Core-Mapping Check

Use this to validate a real running Slurm job against the CPU-to-core translation path used by DVFS.

Show available options:

```bash
python3 -m dvfs.test_slurm_core_mapping --help
```

Check one running job:

```bash
python3 -m dvfs.test_slurm_core_mapping --job-id 3173
```

Check one running job with JSON output:

```bash
python3 -m dvfs.test_slurm_core_mapping --job-id 3173 --json
```

Typical flow:

```bash
squeue -u "$USER"
python3 -m dvfs.test_slurm_core_mapping --job-id <jobid> --json
```

Expected result:
- `PASS` means observed `cores_by_node` matches CPU-to-core translation from the live Slurm allocation.
- `FAIL` means the allocation path is still targeting the wrong identifiers, so DVFS verification is not trustworthy yet.
