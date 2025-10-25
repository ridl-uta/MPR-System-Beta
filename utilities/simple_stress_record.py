#!/usr/bin/env python3
"""Run a single-node stress test across a frequency sweep and record power.

This helper:
- Starts the PDU power monitor (if mapping/user/password/csv are provided)
- Records an idle baseline for --idle-seconds
- For each target frequency from max->min by --interval MHz:
  * Submits data/slurm_scripts/run_stressng.slurm via sbatch
  * Applies DVFS to the job's cores using managers.DVFSManager
  * Waits for completion and gathers Start/End times via sacct
  * Computes average and net average power over the run window from the PDU CSV
  * Appends a row to --output-csv

Requirements:
- Slurm CLI (sbatch, squeue, sacct) in PATH
- stress-ng available on the compute node(s)
- PDU config if power recording is desired: mapping JSON, user, password
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from managers import DVFSManager, PowerMonitor


def run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def parse_job_id(sbatch_stdout: str) -> Optional[str]:
    # Expected: "Submitted batch job <id>"
    parts = sbatch_stdout.strip().split()
    if len(parts) >= 4 and parts[0] == "Submitted" and parts[1] == "batch" and parts[2] == "job":
        return parts[-1]
    # Some Slurm builds return just the ID
    if sbatch_stdout.strip().isdigit():
        return sbatch_stdout.strip()
    return None


def poll_job_state(job_id: str, interval_s: float = 2.0) -> Optional[str]:
    proc = run(["squeue", "-h", "-j", job_id, "-o", "%T"])  # type: ignore[list-item]
    if proc.returncode != 0:
        return None
    state = proc.stdout.strip()
    return state or None


def wait_for_state(job_id: str, desired: Tuple[str, ...], timeout_s: float = 120.0, poll_s: float = 2.0) -> Optional[str]:
    """Poll squeue until job enters a desired state or timeout."""
    deadline = time.time() + timeout_s
    last_state: Optional[str] = None
    while time.time() < deadline:
        state = poll_job_state(job_id)
        if state:
            last_state = state
            if state in desired:
                return state
        elif last_state and last_state in desired:
            return last_state
        time.sleep(poll_s)
    return last_state


def wait_job(job_id: str, poll_s: float = 3.0) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # Return (state, start_iso, end_iso)
    # Poll until sacct reports terminal state
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}
    last_state: Optional[str] = None
    while True:
        state = poll_job_state(job_id)
        if state is None:
            break
        last_state = state
        if state in terminal:
            break
        time.sleep(poll_s)
    # Query sacct for start/end
    proc = run(["sacct", "-j", job_id, "-X", "--parsable2", "-o", "JobID,State,Start,End,NodeList%"], check=False)
    final_state = last_state
    start_iso = end_iso = None
    if proc.returncode == 0:
        lines = [ln for ln in proc.stdout.splitlines() if ln]
        if len(lines) >= 2:
            # header + rows; pick the batch or step 0 line if present, else first
            header = lines[0].split("|")
            idx_state = header.index("State") if "State" in header else -1
            idx_start = header.index("Start") if "Start" in header else -1
            idx_end = header.index("End") if "End" in header else -1
            # pick first non-wrap row with a timestamp
            for row in lines[1:]:
                cols = row.split("|")
                st = cols[idx_state] if idx_state >= 0 else ""
                s = cols[idx_start] if idx_start >= 0 else ""
                e = cols[idx_end] if idx_end >= 0 else ""
                if s:
                    final_state, start_iso, end_iso = st, s, e
                    break
    return final_state, start_iso, end_iso


def iso_to_dt(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def compute_avg_power(pdu_csv: Path, start: datetime, end: datetime) -> Optional[float]:
    if not pdu_csv.exists():
        return None
    try:
        with pdu_csv.open() as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return None
            try:
                total_idx = header.index("total_watts")
            except ValueError:
                total_idx = len(header) - 1
            cnt = 0
            s = 0.0
            for row in reader:
                if len(row) <= total_idx:
                    continue
                try:
                    ts = datetime.fromisoformat(row[0])
                except ValueError:
                    continue
                ts = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
                if start <= ts <= end:
                    try:
                        s += float(row[total_idx])
                        cnt += 1
                    except ValueError:
                        continue
            if cnt == 0:
                return None
            return s / cnt
    except Exception:
        return None


def append_result(csv_path: Path, row: dict) -> None:
    # Ensure header alignment and append
    fields = [
        "job_id",
        "freq_mhz",
        "reduction",
        "start",
        "end",
        "duration_s",
        "avg_power_w",
        "net_avg_power_w",
        "nodes",
    ]
    need_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if need_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def main() -> int:
    ap = argparse.ArgumentParser(description="Run stress-ng across a DVFS sweep and record power")
    ap.add_argument("--max-freq", type=float, default=2400.0)
    ap.add_argument("--min-freq", type=float, default=1000.0)
    ap.add_argument("--interval", type=int, default=200, help="MHz step between targets")
    ap.add_argument("--duration", type=int, default=180, help="Seconds per stress run")
    ap.add_argument("--threads", type=int, default=10, help="OMP threads / cpus-per-task")
    ap.add_argument("--partition", default="debug")
    ap.add_argument("--workdir", default="/shared")
    ap.add_argument("--nodelist", default=None, help="Nodes to target (passed to sbatch --nodelist)")
    ap.add_argument("--exclude", default=None, help="Nodes to exclude (sbatch --exclude)")
    ap.add_argument("--pdu-map", type=Path)
    ap.add_argument("--pdu-user")
    ap.add_argument("--pdu-password")
    ap.add_argument("--pdu-csv", type=Path, default=Path("output/pdu_log_simple.csv"))
    ap.add_argument("--idle-seconds", type=int, default=45)
    ap.add_argument("--output-csv", type=Path, default=Path("output/stress_results.csv"))
    args = ap.parse_args()

    print("[INFO] Starting stress sweep: max={} MHz, min={} MHz, interval={} MHz".format(
        args.max_freq, args.min_freq, args.interval
    ), flush=True)

    args.pdu_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Optional power monitor
    pm: Optional[PowerMonitor] = None
    if args.pdu_map and args.pdu_user and args.pdu_password:
        pm = PowerMonitor(
            map_path=str(args.pdu_map),
            user=args.pdu_user,
            password=args.pdu_password,
            csv_path=str(args.pdu_csv),
            interval=1.0,
            deadline=2.0,
        )
        pm.start()
        # Idle baseline
        time.sleep(max(1, args.idle_seconds))

    # Build target list
    max_mhz = int(round(args.max_freq))
    min_mhz = int(round(args.min_freq))
    if max_mhz <= 0 or min_mhz <= 0 or max_mhz < min_mhz:
        print(f"[ERR] invalid freq bounds max={max_mhz} min={min_mhz}", file=sys.stderr)
        return 2
    targets: list[int] = []
    cur = max_mhz
    while cur >= min_mhz:
        targets.append(cur)
        cur -= int(args.interval)
    if targets[-1] != min_mhz:
        targets.append(min_mhz)

    dvfs = DVFSManager(max_freq_mhz=args.max_freq, min_freq_mhz=args.min_freq)

    for freq_mhz in targets:
        reduction = max(0.0, min(1.0, 1.0 - (freq_mhz / args.max_freq)))
        # Submit stress-ng job
        print(f"[INFO] Target {freq_mhz} MHz -> reduction {reduction:.3f}", flush=True)
        sbatch_cmd = [
            "sbatch",
            "-p", args.partition,
            "-N", "1",
            "-n", "1",
            "-c", str(args.threads),
            "-t", "00:10:00",
            "--export=ALL,WORKDIR=" + args.workdir + f",DURATION={args.duration}",
            "-o", "/shared/logs/stressng-%j.out",
        ]
        if args.nodelist:
            sbatch_cmd += ["--nodelist", args.nodelist]
        if args.exclude:
            sbatch_cmd += ["--exclude", args.exclude]
        sbatch_cmd.append(str(Path("data/slurm_scripts/run_stressng.slurm")))
        sb = run(sbatch_cmd)
        if sb.returncode != 0:
            print(f"[ERR] sbatch failed: {sb.stdout} {sb.stderr}", file=sys.stderr)
            return 3
        job_id = parse_job_id(sb.stdout)
        if not job_id:
            print(f"[ERR] could not parse job id from: {sb.stdout}", file=sys.stderr)
            return 3
        print(f"[INFO] Submitted job {job_id} for {freq_mhz} MHz", flush=True)
        # Wait for RUNNING state before applying DVFS
        state_before = wait_for_state(job_id, ("RUNNING", "COMPLETED"), timeout_s=120)
        if state_before == "RUNNING":
            try:
                dvfs.submit_reduction(job_id, reduction)
            except Exception as exc:
                print(f"[WARN] DVFS apply failed for job {job_id}: {exc}")
        else:
            print(f"[WARN] Skipping DVFS apply for job {job_id}; state={state_before}")

        # Wait for completion and get times
        state, start_iso, end_iso = wait_job(job_id)
        print(f"[INFO] Job {job_id} finished with state {state or 'UNKNOWN'}", flush=True)
        start_dt = iso_to_dt(start_iso) or datetime.now(timezone.utc)
        end_dt = iso_to_dt(end_iso) or datetime.now(timezone.utc)
        duration_s = (end_dt - start_dt).total_seconds()

        avg_power = compute_avg_power(args.pdu_csv, start_dt, end_dt) if pm else None
        row = {
            "job_id": job_id,
            "freq_mhz": freq_mhz,
            "reduction": reduction,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "duration_s": f"{duration_s:.1f}",
            "avg_power_w": f"{avg_power:.3f}" if avg_power is not None else "",
            "net_avg_power_w": "",
            "nodes": "1",
        }
        append_result(args.output_csv, row)
        print(f"[INFO] Appended results for job {job_id} to {args.output_csv}", flush=True)

    if pm:
        pm.stop()

    print(f"[DONE] Results appended to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
