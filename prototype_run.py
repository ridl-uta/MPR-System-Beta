#!/usr/bin/env python3
"""Prototype runner for Slurm scheduling + MPR negotiation.

Flow:
1. Create a JobScheduler instance.
2. Submit jobs with configured ranks through the scheduler object.
3. Fetch cached performance inputs from scheduler (auto-loaded during submit).
4. Run threaded MPR-INT negotiation.
5. Convert final bids to per-job frequency targets.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from job_scheduler import JobScheduler
from mpr_int import normalize_job_perf_data, run_threaded_mpr_int

# Edit these defaults for quick runs.
DEFAULT_JOB_RANKS: dict[str, int] = {
    "comd": 1,
    "hpccg": 2,
    "hpcg": 4,
    "minife": 2,
    "minimd": 2,
    "xsbenchmpi": 1,
}

# Map script/job names to workbook sheet names when they differ.
DEFAULT_PERF_SHEET_BY_JOB: dict[str, str] = {
    "xsbenchmpi": "xsbench",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prototype Slurm + MPR flow from terminal.")
    parser.add_argument("--dry-run", action="store_true", help="Preview sbatch commands only.")
    parser.add_argument("--skip-submit", action="store_true", help="Skip Slurm submission step.")
    parser.add_argument("--show-allocation", action="store_true", help="Query resource allocation for submitted jobs.")
    parser.add_argument(
        "--rank",
        action="append",
        default=[],
        metavar="JOB=RANK",
        help="Override one job rank, e.g. --rank hpcg=6 (repeatable).",
    )
    parser.add_argument(
        "--sheet-map",
        action="append",
        default=[],
        metavar="JOB=SHEET",
        help="Override workbook sheet mapping, e.g. --sheet-map xsbenchmpi=xsbench.",
    )
    parser.add_argument(
        "--jobs",
        default="",
        help="Optional comma-separated subset of jobs to run (defaults to all configured).",
    )
    parser.add_argument("--cpus-per-rank", type=int, default=10)
    parser.add_argument("--ranks-per-node", type=int, default=2)
    parser.add_argument("--partition", default="debug")
    parser.add_argument("--time-limit", default="00:30:00")
    parser.add_argument("--nodelist", default=None)
    parser.add_argument("--exclude", default=None)
    parser.add_argument("--target-reduction-w", type=float, default=700.0)
    parser.add_argument("--max-freq-mhz", type=float, default=2400.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--quiet", action="store_true", help="Disable verbose MPR logs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    scheduler = JobScheduler()

    # Start from defaults and then apply CLI overrides.
    job_ranks = dict(DEFAULT_JOB_RANKS)
    sheet_map = dict(DEFAULT_PERF_SHEET_BY_JOB)
    for item in args.rank:
        if "=" not in item:
            raise ValueError(f"Expected JOB=RANK, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Missing job name in rank override: {item}")
        job_ranks[key] = int(value.strip())
    for item in args.sheet_map:
        if "=" not in item:
            raise ValueError(f"Expected JOB=SHEET, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Missing job name in sheet-map override: {item}")
        sheet_map[key] = value.strip()
    scheduler.set_perf_sheet_map(sheet_map)

    if args.jobs.strip():
        selected = [token.strip() for token in args.jobs.split(",") if token.strip()]
        job_ranks = {name: job_ranks[name] for name in selected if name in job_ranks}
        if not job_ranks:
            raise RuntimeError("No valid jobs selected via --jobs.")

    print("Source Slurm scripts:", scheduler.source_scripts_dir)
    print("Available jobs:", scheduler.available_jobs())
    print("Performance workbook:", scheduler.perf_xlsx_path)
    print("Job ranks:", job_ranks)

    if args.skip_submit:
        print("Skipping Slurm submission step (--skip-submit).")
        submit_df = pd.DataFrame()
    else:
        submit_df = scheduler.submit_jobs(
            job_ranks=job_ranks,
            cpus_per_rank=args.cpus_per_rank,
            ranks_per_node=args.ranks_per_node,
            partition=args.partition,
            time_limit=args.time_limit,
            nodelist=args.nodelist,
            exclude=args.exclude,
            dry_run=args.dry_run,
        )
        print("\nSubmission summary:")
        if submit_df.empty:
            print("No submissions attempted.")
        else:
            print(submit_df.to_string(index=False))

    if args.show_allocation and not args.dry_run and not submit_df.empty:
        submitted_ids = [
            str(job_id)
            for job_id in submit_df.get("job_id", pd.Series(dtype=str)).tolist()
            if str(job_id) not in {"", "None", "nan"}
        ]
        if submitted_ids:
            print("\nSubmitted job allocations:")
            for job_id in submitted_ids:
                allocation = scheduler.get_job_resource_allocation(job_id)
                print(json.dumps(allocation, indent=2))

    job_perf_data, perf_audit_df = scheduler.get_cached_perf_data(
        job_names=list(job_ranks.keys())
    )
    print("\nPerformance data audit:")
    if perf_audit_df.empty:
        print("No cached audit rows. Submit jobs (or disable --skip-submit) to auto-load performance data.")
    else:
        print(perf_audit_df.to_string(index=False))
    print("Loaded job models:", list(job_perf_data.keys()))

    if not job_perf_data:
        raise RuntimeError(
            "No performance data in scheduler cache. "
            "Call submit_job/submit_jobs first (auto_load_perf_data=True) before running MPR."
        )

    result = run_threaded_mpr_int(
        job_perf_data=job_perf_data,
        target_reduction_w=args.target_reduction_w,
        q_bounds=(0.1, 5.0),
        max_iters=100,
        delta_q_tol=0.05,
        residual_abs_tol=5.0,
        cycle_q_tol=0.05,
        host=args.host,
        port=args.port,
        verbose=not args.quiet,
    )

    summary = {
        "converged": result["converged"],
        "convergence_mode": result["convergence_mode"],
        "final_q": result["final_q"],
        "final_reduction_w": result["final_reduction_w"],
        "final_residual_w": result["final_residual_w"],
        "negotiation_time_s": result["negotiation_time_s"],
    }
    print("\nMPR summary:")
    print(json.dumps(summary, indent=2))

    models = normalize_job_perf_data(job_perf_data)
    delta_max_by_job = {model.client_id: float(model.delta_max) for model in models}
    q_final = float(result["final_q"])

    job_reduction: dict[str, float] = {}
    job_freq_mhz: dict[str, float] = {}
    for client_id, bid in dict(result["final_bids"]).items():
        delta = max(delta_max_by_job[client_id] - float(bid) / q_final, 0.0)
        delta = min(delta, 1.0)
        job_reduction[client_id] = delta
        job_freq_mhz[client_id] = args.max_freq_mhz * (1.0 - delta)

    freq_df = (
        pd.DataFrame(
            {
                "job": list(job_freq_mhz.keys()),
                "reduction_fraction": [job_reduction[key] for key in job_freq_mhz.keys()],
                "target_freq_mhz": [job_freq_mhz[key] for key in job_freq_mhz.keys()],
                "target_freq_ghz": [job_freq_mhz[key] / 1000.0 for key in job_freq_mhz.keys()],
            }
        )
        .sort_values("job")
        .reset_index(drop=True)
    )
    print("\nFrequency targets:")
    print(freq_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
