#!/usr/bin/env python3
"""Prototype runner for Slurm scheduling + MPR negotiation."""

from __future__ import annotations

import argparse
import json
import threading
from typing import Any

import pandas as pd

from dvfs import DVFSController
from job_scheduler import JobScheduler
from mpr_int import normalize_job_perf_data, run_threaded_mpr_int
from power_monitor import PowerMonitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prototype Slurm + MPR flow from terminal.")
    parser.add_argument("--dry-run", action="store_true", help="Preview sbatch commands only.")
    parser.add_argument("--skip-submit", action="store_true", help="Skip Slurm submission step.")
    parser.add_argument("--show-allocation", action="store_true", help="Query resource allocation for submitted jobs.")
    parser.add_argument(
        "--rank",
        action="append",
        required=True,
        metavar="JOB=RANK",
        help="Job rank spec (repeatable), e.g. --rank comd=1 --rank minife=2",
    )
    parser.add_argument("--cpus-per-rank", type=int, default=10)
    parser.add_argument("--ranks-per-node", type=int, default=2)
    parser.add_argument("--partition", default="debug")
    parser.add_argument("--time-limit", default="00:30:00")
    parser.add_argument(
        "--submit-interval-s",
        type=float,
        default=0.0,
        help="Delay in seconds between periodic job submissions (default: 0).",
    )
    parser.add_argument("--nodelist", default=None)
    parser.add_argument("--exclude", default=None)
    parser.add_argument("--target-reduction-w", type=float, default=700.0)
    parser.add_argument("--max-freq-mhz", type=float, default=2400.0)
    parser.add_argument(
        "--enable-power-monitor",
        action="store_true",
        help="Run background power monitor and stream power/event updates.",
    )
    parser.add_argument(
        "--pdu-user",
        default=None,
        help="Username for PDU access (required when --enable-power-monitor is set).",
    )
    parser.add_argument(
        "--pdu-password",
        default=None,
        help="Password for PDU access (required when --enable-power-monitor is set).",
    )
    parser.add_argument(
        "--pdu-csv",
        default="output/pdu_log.csv",
        help="CSV path for power monitor samples.",
    )
    parser.add_argument(
        "--pdu-map",
        default=None,
        help="Optional mapping.json override; defaults to module-local mapping.",
    )
    parser.add_argument(
        "--power-interval-s",
        type=float,
        default=1.0,
        help="Power sampling interval in seconds.",
    )
    parser.add_argument(
        "--power-deadline-s",
        type=float,
        default=15.0,
        help="Per-PDU poll timeout in seconds.",
    )
    parser.add_argument(
        "--power-print-interval-s",
        type=float,
        default=1.0,
        help="Interval for printing power monitor status in terminal.",
    )
    parser.add_argument(
        "--skip-dvfs-apply",
        action="store_true",
        help="Skip applying computed per-job frequencies through DVFS controller.",
    )
    parser.add_argument(
        "--dvfs-ssh-user",
        default=None,
        help="Optional SSH user for remote DVFS apply (e.g. ridl).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--quiet", action="store_true", help="Disable verbose MPR logs.")
    args = parser.parse_args()
    if args.enable_power_monitor and (not args.pdu_user or not args.pdu_password):
        parser.error("--enable-power-monitor requires --pdu-user and --pdu-password.")
    args.job_ranks = parse_job_ranks(args.rank)
    return args


def parse_job_ranks(rank_items: list[str]) -> dict[str, int]:
    job_ranks: dict[str, int] = {}
    for item in rank_items:
        if "=" not in item:
            raise ValueError(f"Expected JOB=RANK, got: {item}")
        job, rank_text = item.split("=", 1)
        job = job.strip()
        if not job:
            raise ValueError(f"Missing job name in: {item}")
        rank = int(rank_text.strip())
        if rank <= 0:
            raise ValueError(f"Rank must be > 0 for job '{job}', got {rank}.")
        job_ranks[job] = rank
    if not job_ranks:
        raise ValueError("At least one --rank JOB=RANK must be provided.")
    return job_ranks

def submit_jobs(scheduler: JobScheduler, args: argparse.Namespace, job_ranks: dict[str, int]) -> pd.DataFrame:
    if args.skip_submit:
        print("Skipping Slurm submission step (--skip-submit).")
        return pd.DataFrame()

    if args.submit_interval_s > 0:
        submit_df = scheduler.submit_jobs_periodic(
            job_ranks=job_ranks,
            cpus_per_rank=args.cpus_per_rank,
            ranks_per_node=args.ranks_per_node,
            partition=args.partition,
            time_limit=args.time_limit,
            nodelist=args.nodelist,
            exclude=args.exclude,
            dry_run=args.dry_run,
            submit_interval_s=args.submit_interval_s,
        )
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
    return submit_df


def print_allocations(
    scheduler: JobScheduler,
    args: argparse.Namespace,
    submit_df: pd.DataFrame,
    job_ranks: dict[str, int],
) -> None:
    if not args.show_allocation or args.dry_run or submit_df.empty:
        return
    allocations = scheduler.get_submitted_job_allocations(
        job_names=list(job_ranks.keys()),
        latest_only=True,
    )
    if allocations:
        print("\nSubmitted job allocations:")
        print(json.dumps(allocations, indent=2))


def fetch_job_perf_data(
    scheduler: JobScheduler,
    job_ranks: dict[str, int],
) -> dict[str, pd.DataFrame]:
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
    return job_perf_data


def run_market(args: argparse.Namespace, job_perf_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
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
    return result


def compute_frequency_targets(
    result: dict[str, Any],
    job_perf_data: dict[str, pd.DataFrame],
    max_freq_mhz: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    models = normalize_job_perf_data(job_perf_data)
    delta_max_by_job = {model.client_id: float(model.delta_max) for model in models}
    q_final = float(result["final_q"])

    job_reduction: dict[str, float] = {}
    job_freq_mhz: dict[str, float] = {}
    for client_id, bid in dict(result["final_bids"]).items():
        delta = max(delta_max_by_job[client_id] - float(bid) / q_final, 0.0)
        delta = min(delta, 1.0)
        job_reduction[client_id] = delta
        job_freq_mhz[client_id] = max_freq_mhz * (1.0 - delta)

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
    return freq_df, job_freq_mhz


def apply_dvfs(
    scheduler: JobScheduler,
    args: argparse.Namespace,
    job_freq_mhz: dict[str, float],
) -> None:
    if args.skip_dvfs_apply:
        print("\nDVFS apply skipped (--skip-dvfs-apply).")
        return

    allocations = scheduler.get_submitted_job_allocations(
        job_names=list(job_freq_mhz.keys()),
        latest_only=True,
    )
    if not allocations:
        print("\nNo submitted job allocations available for DVFS apply. Skipping.")
        return

    dvfs_controller = DVFSController(ssh_user=args.dvfs_ssh_user)
    dvfs_rows: list[dict[str, object]] = []
    for job_name, target_freq_mhz in job_freq_mhz.items():
        allocation = allocations.get(job_name)
        if not allocation:
            print(f"[DVFS] No allocation for job '{job_name}', skipping.")
            continue
        for row in dvfs_controller.apply_to_job_allocation(
            allocation=allocation,
            frequency_mhz=float(target_freq_mhz),
            dry_run=args.dry_run,
        ):
            row_with_job = dict(row)
            row_with_job["job"] = job_name
            dvfs_rows.append(row_with_job)

    if not dvfs_rows:
        print("\nDVFS apply produced no core-level actions.")
        return

    dvfs_df = pd.DataFrame(dvfs_rows)
    print("\nDVFS apply summary:")
    print(dvfs_df.to_string(index=False))


def build_power_monitor(args: argparse.Namespace) -> PowerMonitor | None:
    if not args.enable_power_monitor:
        return None
    return PowerMonitor(
        username=args.pdu_user,
        password=args.pdu_password,
        csv_path=args.pdu_csv,
        map_path=args.pdu_map,
        interval_s=args.power_interval_s,
        deadline_s=args.power_deadline_s,
    )


def power_monitor_worker(
    monitor: PowerMonitor,
    stop_event: threading.Event,
    print_interval_s: float,
) -> None:
    interval = max(0.1, float(print_interval_s))
    last_sample_ts: str | None = None

    while not stop_event.is_set():
        events = monitor.consume_events()
        for event in events:
            event_name = str(event.get("event", "EVENT"))
            message = str(event.get("message", ""))
            print(f"[PowerEvent] {event_name} | {message}")

        sample = monitor.get_last_sample()
        if sample:
            ts = str(sample.get("timestamp", ""))
            if ts and ts != last_sample_ts:
                total_watts = float(sample.get("total_watts", 0.0))
                node_totals = sample.get("node_totals", {})
                print(
                    f"[PowerSample] ts={ts} total_watts={total_watts:.1f} "
                    f"nodes={node_totals}"
                )
                last_sample_ts = ts

        stop_event.wait(interval)


def start_power_monitor(
    args: argparse.Namespace,
) -> tuple[PowerMonitor | None, threading.Event | None, threading.Thread | None]:
    monitor = build_power_monitor(args)
    if monitor is None:
        return None, None, None

    monitor.start()
    stop_event = threading.Event()
    worker = threading.Thread(
        target=power_monitor_worker,
        args=(monitor, stop_event, args.power_print_interval_s),
        daemon=True,
        name="PowerMonitorEvents",
    )
    worker.start()

    print(
        "Power monitor enabled:",
        f"mapping={monitor.resolved_mapping_path}",
        f"csv={args.pdu_csv}",
    )
    return monitor, stop_event, worker


def stop_power_monitor(
    monitor: PowerMonitor | None,
    stop_event: threading.Event | None,
    worker: threading.Thread | None,
) -> None:
    if stop_event is not None:
        stop_event.set()
    if worker is not None and worker.is_alive():
        worker.join(timeout=2.0)
    if monitor is not None:
        monitor.stop()


def main() -> int:
    args = parse_args()
    scheduler = JobScheduler()
    job_ranks = args.job_ranks
    monitor, monitor_stop_event, monitor_worker = start_power_monitor(args)

    try:
        print("Source Slurm scripts:", scheduler.source_scripts_dir)
        print("Available jobs:", scheduler.available_jobs())
        print("Performance workbook:", scheduler.perf_xlsx_path)
        print("Job ranks:", job_ranks)

        submit_df = submit_jobs(scheduler, args, job_ranks)
        print_allocations(scheduler, args, submit_df, job_ranks)

        job_perf_data = fetch_job_perf_data(scheduler, job_ranks)
        market_result = run_market(args, job_perf_data)
        _, job_freq_mhz = compute_frequency_targets(
            market_result,
            job_perf_data,
            max_freq_mhz=args.max_freq_mhz,
        )
        apply_dvfs(scheduler, args, job_freq_mhz)
    finally:
        stop_power_monitor(monitor, monitor_stop_event, monitor_worker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
