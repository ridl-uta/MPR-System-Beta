#!/usr/bin/env python3
"""Main runner for Slurm scheduling, power monitoring, MPR negotiation, and DVFS apply."""

from __future__ import annotations

import argparse
import json
import queue
import shlex
import threading
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from dvfs import DVFSController
from job_scheduler import JobScheduler
from mpr_int import normalize_job_perf_data, run_threaded_mpr_int
from power_monitor import PowerMonitor, make_simple_overload_ctx, simple_overload_update

ACTIVE_JOB_STATES = {
    "PENDING",
    "CONFIGURING",
    "RUNNING",
    "COMPLETING",
    "STAGE_OUT",
    "SUSPENDED",
    "RESIZING",
    "SIGNALING",
    "REQUEUED",
}


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
    parser.add_argument(
        "--job-poll-interval-s",
        type=float,
        default=5.0,
        help="Polling interval (seconds) for submitted Slurm job states.",
    )
    parser.add_argument(
        "--job-status-print-interval-s",
        type=float,
        default=15.0,
        help="Interval (seconds) for printing job-watch status lines.",
    )
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
    raw_residual = float(result.get("final_residual_w", 0.0))
    market_failed = raw_residual < 0.0
    result["final_residual_w_raw"] = raw_residual
    result["final_residual_w"] = max(0.0, raw_residual)
    result["market_failed"] = bool(market_failed)
    result["market_status"] = "FAILED" if market_failed else "OK"

    summary = {
        "market_status": result["market_status"],
        "converged": result["converged"],
        "convergence_mode": result["convergence_mode"],
        "final_q": result["final_q"],
        "final_reduction_w": result["final_reduction_w"],
        "final_residual_w": result["final_residual_w"],
        "negotiation_time_s": result["negotiation_time_s"],
    }
    if market_failed:
        summary["failure_reason"] = "negative_residual"
        summary["residual_shortfall_w"] = abs(raw_residual)
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


def wait_for_initial_power_sample(
    monitor: PowerMonitor,
    startup_wait_s: float,
) -> dict[str, object]:
    sample: dict[str, object] | None = monitor.get_last_sample()
    waited = 0.0
    timeout_s = max(0.0, float(startup_wait_s))
    while sample is None and waited < timeout_s:
        step = min(0.2, timeout_s - waited)
        if step <= 0:
            break
        time.sleep(step)
        waited += step
        sample = monitor.get_last_sample()

    if sample is None:
        raise RuntimeError(
            "Power monitor did not produce a sample in time. "
            "Increase --power-startup-wait-s."
        )
    return sample


def collect_submitted_job_ids(submit_df: pd.DataFrame) -> list[str]:
    if submit_df.empty or "status" not in submit_df.columns or "job_id" not in submit_df.columns:
        return []
    rows = submit_df[submit_df["status"].astype(str) == "SUBMITTED"]
    job_ids: list[str] = []
    for value in rows["job_id"].tolist():
        text = str(value).strip()
        if not text or text.lower() in {"none", "nan"}:
            continue
        if text not in job_ids:
            job_ids.append(text)
    return job_ids


def query_submitted_job_states(
    scheduler: JobScheduler,
    job_ids: list[str],
) -> dict[str, str]:
    if not job_ids:
        return {}

    cmd = [
        scheduler.helper.squeue_cmd,
        "-h",
        "-j",
        ",".join(job_ids),
        "-o",
        "%i|%T",
    ]
    proc = scheduler.helper.run_command(cmd)
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or "no details"
        print(f"[JobWatch] squeue query failed: {stderr}")
        return {job_id: "UNKNOWN" for job_id in job_ids}

    states: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        row = line.strip()
        if not row:
            continue
        parts = row.split("|", 1)
        if len(parts) != 2:
            continue
        job_id = parts[0].strip()
        state = parts[1].strip().upper()
        if job_id:
            states[job_id] = state

    # If a submitted job no longer appears in squeue, treat it as terminal.
    for job_id in job_ids:
        states.setdefault(job_id, "COMPLETED")
    return states


def has_active_jobs(job_states: dict[str, str]) -> bool:
    return any(str(state).upper() in ACTIVE_JOB_STATES for state in job_states.values())


def drain_overload_events(
    overload_event_queue: queue.Queue[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if overload_event_queue is None:
        return []
    events: list[dict[str, Any]] = []
    while True:
        try:
            events.append(overload_event_queue.get_nowait())
        except queue.Empty:
            break
    return events


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
) -> dict[str, dict[str, Any]]:
    return apply_dvfs_with_allocations(
        scheduler=scheduler,
        args=args,
        job_freq_mhz=job_freq_mhz,
        allocations_hint=None,
    )


def apply_dvfs_with_allocations(
    *,
    scheduler: JobScheduler,
    args: argparse.Namespace,
    job_freq_mhz: dict[str, float],
    allocations_hint: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if args.skip_dvfs_apply:
        print("\nDVFS apply skipped (--skip-dvfs-apply).")
        return {}

    allocations: dict[str, dict[str, Any]] = dict(allocations_hint or {})
    missing_jobs = [
        name
        for name in job_freq_mhz.keys()
        if name not in allocations or not allocations[name].get("cores_by_node")
    ]
    if missing_jobs:
        fetched = scheduler.get_submitted_job_allocations(
            job_names=missing_jobs,
            latest_only=True,
        )
        allocations.update(fetched)

    if not allocations:
        print("\nNo submitted job allocations available for DVFS apply. Skipping.")
        return {}

    dvfs_controller = DVFSController(ssh_user=args.dvfs_ssh_user)
    dvfs_rows: list[dict[str, object]] = []
    used_allocations: dict[str, dict[str, Any]] = {}
    for job_name, target_freq_mhz in job_freq_mhz.items():
        allocation = allocations.get(job_name)
        if not allocation:
            print(f"[DVFS] No allocation for job '{job_name}', skipping.")
            continue
        cores_by_node = allocation.get("cores_by_node", {})
        if not isinstance(cores_by_node, dict) or not cores_by_node:
            print(f"[DVFS] No core mapping for job '{job_name}', skipping.")
            continue
        before_count = len(dvfs_rows)
        for row in dvfs_controller.apply_to_job_allocation(
            allocation=allocation,
            frequency_mhz=float(target_freq_mhz),
            dry_run=args.dry_run,
        ):
            row_with_job = dict(row)
            row_with_job["job"] = job_name
            dvfs_rows.append(row_with_job)
        if len(dvfs_rows) > before_count:
            used_allocations[job_name] = allocation

    if not dvfs_rows:
        print("\nDVFS apply produced no core-level actions.")
        return used_allocations

    dvfs_df = pd.DataFrame(dvfs_rows)
    print("\nDVFS apply summary:")
    print(dvfs_df.to_string(index=False))
    return used_allocations


def parse_sample_timestamp(ts_text: str) -> datetime:
    try:
        return datetime.fromisoformat(str(ts_text))
    except ValueError:
        return datetime.now(timezone.utc)


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
    overload_event_queue: queue.Queue[dict[str, Any]],
    target_capacity_w: float,
    sample_period_s: float,
) -> None:
    print_interval = max(0.1, float(print_interval_s))
    poll_interval = max(0.1, min(0.5, float(sample_period_s) * 0.5))
    next_print_at = 0.0
    last_sample_ts: str | None = None
    overload_ctx = make_simple_overload_ctx(
        sample_period_s=max(0.1, float(sample_period_s)),
        threshold_w=float(target_capacity_w),
    )

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
                ts_dt = parse_sample_timestamp(ts)
                overload_ctx["last_raw_sample"] = total_watts
                event, payload = simple_overload_update(overload_ctx, ts_dt, total_watts)
                if event:
                    event_name = str(event[0]) if isinstance(event, tuple) else str(event)
                    reduction_w = float(
                        overload_ctx.get(
                            "required_reduction_w",
                            max(0.0, total_watts - float(target_capacity_w)),
                        )
                    )
                    overload_record: dict[str, Any] = {
                        "timestamp": ts,
                        "event": event_name,
                        "payload": payload if isinstance(payload, dict) else {},
                        "total_watts": total_watts,
                        "target_capacity_w": float(target_capacity_w),
                        "required_reduction_w": max(0.0, reduction_w),
                        "state": str(overload_ctx.get("state", "")),
                    }
                    overload_event_queue.put(overload_record)
                    print(
                        f"[OverloadEvent] {event_name} "
                        f"total_watts={total_watts:.1f} "
                        f"required_reduction_w={overload_record['required_reduction_w']:.3f}"
                    )
                now = time.monotonic()
                if now >= next_print_at:
                    print(
                        f"[PowerSample] ts={ts} total_watts={total_watts:.1f} "
                        f"nodes={node_totals}"
                    )
                    next_print_at = now + print_interval
                last_sample_ts = ts

        stop_event.wait(poll_interval)


def start_power_monitor(
    args: argparse.Namespace,
) -> tuple[
    PowerMonitor | None,
    threading.Event | None,
    threading.Thread | None,
    queue.Queue[dict[str, Any]] | None,
]:
    monitor = build_power_monitor(args)
    if monitor is None:
        return None, None, None, None

    monitor.start()
    stop_event = threading.Event()
    overload_event_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    worker = threading.Thread(
        target=power_monitor_worker,
        args=(
            monitor,
            stop_event,
            args.power_print_interval_s,
            overload_event_queue,
            args.target_capacity_w,
            args.power_interval_s,
        ),
        daemon=True,
        name="PowerMonitorEvents",
    )
    worker.start()

    print(
        "Power monitor enabled:",
        f"mapping={monitor.resolved_mapping_path}",
        f"csv={args.pdu_csv}",
    )
    return monitor, stop_event, worker, overload_event_queue


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


def apply_overload_reduction(
    *,
    args: argparse.Namespace,
    scheduler: JobScheduler,
    job_perf_data: dict[str, pd.DataFrame],
    current_power_w: float,
    allocations_cache: dict[str, dict[str, Any]],
) -> tuple[bool, dict[str, dict[str, Any]]]:
    target_reduction_w = max(0.0, float(current_power_w) - float(args.target_capacity_w))
    print(
        "\nCapacity check:",
        f"current_power_w={current_power_w:.3f}",
        f"target_capacity_w={args.target_capacity_w:.3f}",
        f"overload_w={target_reduction_w:.3f}",
    )
    if target_reduction_w <= 0.0:
        print("No overload detected (current power is within target capacity).")
        return False, allocations_cache

    market_result = run_market(args, job_perf_data, target_reduction_w)
    if bool(market_result.get("market_failed")):
        print(
            "[Control] Market failed: negative residual indicates target reduction "
            "was not met. Skipping DVFS reduction apply."
        )
        return False, allocations_cache
    _, job_freq_mhz = compute_frequency_targets(
        market_result,
        job_perf_data,
        max_freq_mhz=args.max_freq_mhz,
    )
    used = apply_dvfs_with_allocations(
        scheduler=scheduler,
        args=args,
        job_freq_mhz=job_freq_mhz,
        allocations_hint=allocations_cache,
    )
    allocations_cache.update(used)
    return True, allocations_cache


def apply_reset_to_max_frequency(
    *,
    scheduler: JobScheduler,
    args: argparse.Namespace,
    job_names: list[str],
    allocations_cache: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    reset_map = {name: float(args.max_freq_mhz) for name in job_names}
    print(f"\nResetting DVFS to max frequency ({args.max_freq_mhz:.3f} MHz) after overload end.")
    used = apply_dvfs_with_allocations(
        scheduler=scheduler,
        args=args,
        job_freq_mhz=reset_map,
        allocations_hint=allocations_cache,
    )
    allocations_cache.update(used)
    return allocations_cache


def run_event_driven_control_loop(
    *,
    scheduler: JobScheduler,
    args: argparse.Namespace,
    monitor: PowerMonitor,
    overload_event_queue: queue.Queue[dict[str, Any]],
    submit_df: pd.DataFrame,
    job_ranks: dict[str, int],
    job_perf_data: dict[str, pd.DataFrame],
) -> None:
    submitted_job_ids = collect_submitted_job_ids(submit_df)
    if not submitted_job_ids:
        print("No submitted job IDs found. Event-driven loop not started.")
        return

    startup_sample = wait_for_initial_power_sample(monitor, float(args.power_startup_wait_s))
    startup_power = float(startup_sample.get("total_watts", 0.0))
    startup_overload = max(0.0, startup_power - float(args.target_capacity_w))
    print(
        "\nCapacity check:",
        f"current_power_w={startup_power:.3f}",
        f"target_capacity_w={args.target_capacity_w:.3f}",
        f"overload_w={startup_overload:.3f}",
    )
    if startup_overload <= 0.0:
        print("No overload detected at startup; waiting for overload events.")

    loop_interval_s = max(0.2, min(1.0, float(args.power_interval_s)))
    job_poll_interval_s = max(1.0, float(args.job_poll_interval_s))
    job_print_interval_s = max(1.0, float(args.job_status_print_interval_s))
    next_job_poll = 0.0
    next_job_print = 0.0
    job_states: dict[str, str] = {}

    reduction_active = False
    reset_applied = True
    allocations_cache: dict[str, dict[str, Any]] = {}

    while True:
        now = time.monotonic()
        if now >= next_job_poll:
            job_states = query_submitted_job_states(scheduler, submitted_job_ids)
            next_job_poll = now + job_poll_interval_s

        if now >= next_job_print:
            state_parts = [f"{job_id}:{job_states.get(job_id, 'UNKNOWN')}" for job_id in submitted_job_ids]
            print(f"[JobWatch] active={has_active_jobs(job_states)} states={state_parts}")
            next_job_print = now + job_print_interval_s

        for event in drain_overload_events(overload_event_queue):
            event_name = str(event.get("event", ""))
            ts = str(event.get("timestamp", ""))
            total_watts = float(event.get("total_watts", 0.0))
            required_w = float(
                event.get(
                    "required_reduction_w",
                    max(0.0, total_watts - float(args.target_capacity_w)),
                )
            )
            print(
                f"[Control] event={event_name} ts={ts} "
                f"total_watts={total_watts:.3f} required_reduction_w={required_w:.3f}"
            )

            if event_name == "OVERLOAD_START":
                reduction_applied, allocations_cache = apply_overload_reduction(
                    args=args,
                    scheduler=scheduler,
                    job_perf_data=job_perf_data,
                    current_power_w=total_watts,
                    allocations_cache=allocations_cache,
                )
                if reduction_applied:
                    reduction_active = True
                    reset_applied = False
            elif event_name == "OVERLOAD_END":
                if reduction_active and not reset_applied:
                    allocations_cache = apply_reset_to_max_frequency(
                        scheduler=scheduler,
                        args=args,
                        job_names=list(job_ranks.keys()),
                        allocations_cache=allocations_cache,
                    )
                    reduction_active = False
                    reset_applied = True
            elif event_name == "OVERLOAD_HANDLED":
                print("[Control] overload handled band reached.")

        jobs_active = has_active_jobs(job_states)
        if not jobs_active:
            if reduction_active and not reset_applied:
                print("[Control] Jobs ended while reduction active; applying safety DVFS reset.")
                allocations_cache = apply_reset_to_max_frequency(
                    scheduler=scheduler,
                    args=args,
                    job_names=list(job_ranks.keys()),
                    allocations_cache=allocations_cache,
                )
                reduction_active = False
                reset_applied = True
            print("All submitted jobs completed. Exiting event-driven control loop.")
            break

        time.sleep(loop_interval_s)


def main() -> int:
    args = parse_args()
    scheduler = JobScheduler()
    job_ranks = args.job_ranks
    monitor, monitor_stop_event, monitor_worker, overload_event_queue = start_power_monitor(args)

    try:
        print("Source Slurm scripts:", scheduler.source_scripts_dir)
        print("Available jobs:", scheduler.available_jobs())
        print("Performance workbook:", scheduler.perf_xlsx_path)
        print("Job ranks:", job_ranks)

        submit_df = submit_jobs(scheduler, args, job_ranks)
        print_allocations(scheduler, args, submit_df, job_ranks)

        job_perf_data = fetch_job_perf_data(scheduler, job_ranks)
        if monitor is not None and overload_event_queue is not None:
            run_event_driven_control_loop(
                scheduler=scheduler,
                args=args,
                monitor=monitor,
                overload_event_queue=overload_event_queue,
                submit_df=submit_df,
                job_ranks=job_ranks,
                job_perf_data=job_perf_data,
            )
        else:
            target_reduction_w, current_power_w = resolve_target_reduction_w(args, monitor)
            print(
                "\nCapacity check:",
                f"current_power_w={current_power_w:.3f}",
                f"target_capacity_w={args.target_capacity_w:.3f}",
                f"overload_w={target_reduction_w:.3f}",
            )
            if target_reduction_w > 0.0:
                market_result = run_market(args, job_perf_data, target_reduction_w)
                if bool(market_result.get("market_failed")):
                    print(
                        "Market failed: negative residual indicates target reduction "
                        "was not met. Skipping DVFS reduction apply."
                    )
                else:
                    _, job_freq_mhz = compute_frequency_targets(
                        market_result,
                        job_perf_data,
                        max_freq_mhz=args.max_freq_mhz,
                    )
                    apply_dvfs(scheduler, args, job_freq_mhz)
            else:
                print("No overload detected (current power is within target capacity).")
    finally:
        stop_power_monitor(monitor, monitor_stop_event, monitor_worker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
