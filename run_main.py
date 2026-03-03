#!/usr/bin/env python3
"""Main runner for Slurm scheduling, power monitoring, MPR negotiation, and DVFS apply."""

from __future__ import annotations

import argparse
import json
import shlex
import threading
import time
from typing import Any

import pandas as pd

from dvfs import DVFSController
from job_scheduler import JobScheduler
from mpr_int import normalize_job_perf_data, run_threaded_mpr_int
from power_monitor import PowerMonitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Slurm + power + MPR + DVFS flow from terminal.")
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
    parser.add_argument(
        "--mpi-iface",
        default=None,
        choices=["pmi2", "pmix"],
        help="Optional MPI interface export passed to Slurm jobs.",
    )
    parser.add_argument(
        "--slurm-output",
        default=None,
        help="Optional sbatch output path (-o), e.g. /shared/logs/job-%%j.out.",
    )
    parser.add_argument(
        "--submit-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra env export passed to all submitted jobs (repeatable).",
    )
    parser.add_argument(
        "--job-args",
        action="append",
        default=[],
        metavar="JOB=ARG STRING",
        help=(
            "Override script args for a specific job (repeatable), "
            "e.g. --job-args \"minimd=-i in.lj.miniMD -n 8000 -t 10\""
        ),
    )
    parser.add_argument(
        "--disable-rank-profiles",
        action="store_true",
        help="Disable module CSV rank profiles when building job env/script args.",
    )
    parser.add_argument(
        "--target-capacity-w",
        type=float,
        default=700.0,
        help="Power capacity threshold. Overload watts = max(current_power - target_capacity, 0).",
    )
    parser.add_argument(
        "--current-power-w",
        type=float,
        default=None,
        help="Optional direct current total power reading in watts. If omitted, uses latest power-monitor sample.",
    )
    parser.add_argument(
        "--power-startup-wait-s",
        type=float,
        default=5.0,
        help="Seconds to wait for first power sample when --current-power-w is not provided.",
    )
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
    try:
        args.job_ranks = parse_job_ranks(args.rank)
        args.submit_env_map = parse_key_value_overrides(args.submit_env)
        args.job_args_map = parse_job_args_overrides(args.job_args)
    except ValueError as exc:
        parser.error(str(exc))
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


def parse_key_value_overrides(items: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Expected KEY=VALUE in --submit-env, got: {raw}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Missing env key in --submit-env value: {raw}")
        overrides[key] = value
    return overrides


def parse_job_args_overrides(items: list[str]) -> dict[str, list[str]]:
    overrides: dict[str, list[str]] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Expected JOB=ARG STRING in --job-args, got: {raw}")
        job_name, args_text = raw.split("=", 1)
        job_name = job_name.strip()
        if not job_name:
            raise ValueError(f"Missing job name in --job-args value: {raw}")
        args_tokens = shlex.split(args_text.strip())
        overrides[job_name] = args_tokens
    return overrides

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
            output_path=args.slurm_output,
            mpi_iface=args.mpi_iface,
            env_overrides=args.submit_env_map,
            script_args_by_job=args.job_args_map,
            use_rank_profiles=not args.disable_rank_profiles,
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
            output_path=args.slurm_output,
            mpi_iface=args.mpi_iface,
            env_overrides=args.submit_env_map,
            script_args_by_job=args.job_args_map,
            use_rank_profiles=not args.disable_rank_profiles,
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


def run_market(
    args: argparse.Namespace,
    job_perf_data: dict[str, pd.DataFrame],
    target_reduction_w: float,
) -> dict[str, Any]:
    result = run_threaded_mpr_int(
        job_perf_data=job_perf_data,
        target_reduction_w=target_reduction_w,
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


def resolve_target_reduction_w(
    args: argparse.Namespace,
    monitor: PowerMonitor | None,
) -> tuple[float, float]:
    if args.current_power_w is not None:
        current_power_w = float(args.current_power_w)
    else:
        if monitor is None:
            raise RuntimeError(
                "Need current power to compute overload. Provide --current-power-w "
                "or enable monitor with --enable-power-monitor."
            )

        sample: dict[str, object] | None = monitor.get_last_sample()
        waited = 0.0
        while sample is None and waited < float(args.power_startup_wait_s):
            step = min(0.2, float(args.power_startup_wait_s) - waited)
            if step <= 0:
                break
            time.sleep(step)
            waited += step
            sample = monitor.get_last_sample()

        if sample is None:
            raise RuntimeError(
                "Power monitor did not produce a sample in time. "
                "Increase --power-startup-wait-s or pass --current-power-w."
            )

        current_power_w = float(sample.get("total_watts", 0.0))

    target_reduction_w = max(0.0, current_power_w - float(args.target_capacity_w))
    return target_reduction_w, current_power_w


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
        target_reduction_w, current_power_w = resolve_target_reduction_w(args, monitor)
        print(
            "\nCapacity check:",
            f"current_power_w={current_power_w:.3f}",
            f"target_capacity_w={args.target_capacity_w:.3f}",
            f"overload_w={target_reduction_w:.3f}",
        )
        if target_reduction_w <= 0.0:
            print("No overload detected (current power is within target capacity).")
            return 0

        market_result = run_market(args, job_perf_data, target_reduction_w)
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
