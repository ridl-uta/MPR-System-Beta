#!/usr/bin/env python3
"""Main runner for Slurm scheduling, power monitoring, MPR negotiation, and DVFS apply."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Iterable
import json
import math
import queue
import shlex
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from dvfs import DVFSController
from job_scheduler import JobScheduler
from mpr_int import normalize_job_perf_data, run_threaded_mpr_int
from power_monitor import PowerMonitor, make_simple_overload_ctx, simple_overload_update
from power_monitor.mapping import load_mapping

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

CONTROL_ELIGIBLE_JOB_STATES = ACTIVE_JOB_STATES - {"PENDING", "REQUEUED"}
REAPPLY_TRIGGER_EVENT_NAMES = {"SPIKE_WARNING"}


def parse_lscpu_core_ids(topology_text: str) -> list[int]:
    core_ids: set[int] = set()
    for line in str(topology_text).splitlines():
        row = line.strip()
        if not row or row.startswith("#"):
            continue
        parts = row.split(",", 1)
        if len(parts) != 2:
            continue
        try:
            core_id = int(parts[1].strip())
        except ValueError:
            continue
        core_ids.add(core_id)
    return sorted(core_ids)


def discover_node_core_ids(
    node_name: str,
    *,
    ssh_user: str | None,
) -> list[int]:
    topology_cmd = "lscpu -p=CPU,CORE"
    if DVFSController._is_local_host(node_name):
        cmd = ["bash", "-lc", topology_cmd]
    else:
        ssh_target = f"{ssh_user}@{node_name}" if ssh_user else str(node_name)
        cmd = ["ssh", "-o", "BatchMode=yes", ssh_target, topology_cmd]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or "no details"
        raise RuntimeError(
            f"failed to discover cores for node '{node_name}' (exit {proc.returncode}): {stderr}"
        )

    core_ids = parse_lscpu_core_ids(proc.stdout)
    if not core_ids:
        raise RuntimeError(f"no cores discovered for node '{node_name}' from lscpu output")
    return core_ids


def get_mapped_reset_nodes(args: argparse.Namespace) -> list[str]:
    mapping_cfg = load_mapping(args.pdu_map)
    nodes: list[str] = []
    for pdu in mapping_cfg.pdus:
        for node_name in pdu.outlet_map.values():
            normalized = str(node_name).strip()
            if normalized and normalized not in nodes:
                nodes.append(normalized)
    return sorted(nodes)


def discover_node_core_ids_by_node(
    *,
    node_names: list[str],
    ssh_user: str | None,
    max_workers: int = 8,
) -> dict[str, list[int]]:
    if not node_names:
        return {}

    core_ids_by_node: dict[str, list[int]] = {}
    worker_count = min(max(1, int(max_workers)), len(node_names))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_node = {
            executor.submit(discover_node_core_ids, node_name, ssh_user=ssh_user): node_name
            for node_name in node_names
        }
        for future in as_completed(future_to_node):
            node_name = future_to_node[future]
            core_ids_by_node[node_name] = future.result()
    return core_ids_by_node


def build_dvfs_apply_plan(
    *,
    args: argparse.Namespace,
    job_freq_mhz: dict[str, float],
    allocations: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, int], dict[str, Any]], dict[str, dict[str, Any]]]:
    node_core_targets: dict[str, dict[int, dict[str, Any]]] = {}
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

        requested_target_mhz = float(target_freq_mhz)
        snapped_target_mhz = snap_floor_frequency_mhz(
            requested_mhz=requested_target_mhz,
            step_mhz=float(args.dvfs_step_mhz),
        )
        if abs(snapped_target_mhz - requested_target_mhz) > 1e-9:
            print(
                f"[DVFS] {job_name}: target snapped "
                f"{requested_target_mhz:.3f} -> {snapped_target_mhz:.3f} MHz "
                f"(step={float(args.dvfs_step_mhz):.3f})"
            )

        added = False
        for node_name, cores in cores_by_node.items():
            if not isinstance(cores, Iterable) or isinstance(cores, (str, bytes)):
                continue
            node_targets = node_core_targets.setdefault(str(node_name), {})
            for core in cores:
                core_number = int(core)
                meta = {
                    "job": job_name,
                    "requested_target_mhz": requested_target_mhz,
                    "applied_target_mhz": snapped_target_mhz,
                }
                existing = node_targets.get(core_number)
                if existing is not None:
                    if abs(float(existing["applied_target_mhz"]) - snapped_target_mhz) > 1e-9:
                        raise RuntimeError(
                            "conflicting DVFS targets for "
                            f"node={node_name} core={core_number}: "
                            f"{existing['job']}->{existing['applied_target_mhz']:.3f} MHz vs "
                            f"{job_name}->{snapped_target_mhz:.3f} MHz"
                        )
                    continue
                node_targets[core_number] = meta
                added = True

        if added:
            used_allocations[job_name] = allocation

    node_rules: dict[str, list[dict[str, Any]]] = {}
    core_metadata: dict[tuple[str, int], dict[str, Any]] = {}
    for node_name, core_targets in node_core_targets.items():
        freq_to_cores: dict[float, list[int]] = {}
        for core_number, meta in sorted(core_targets.items()):
            freq_to_cores.setdefault(float(meta["applied_target_mhz"]), []).append(int(core_number))
            core_metadata[(str(node_name), int(core_number))] = dict(meta)
        node_rules[str(node_name)] = [
            {"core_numbers": sorted(cores), "frequency_mhz": float(target_mhz)}
            for target_mhz, cores in sorted(freq_to_cores.items())
        ]

    return node_rules, core_metadata, used_allocations


def apply_full_frequency_reset_to_all_nodes(
    args: argparse.Namespace,
    *,
    action_label: str,
    summary_job_label: str,
) -> None:
    if args.skip_dvfs_apply:
        print(f"\n{action_label} skipped (--skip-dvfs-apply).")
        return

    node_names = get_mapped_reset_nodes(args)
    if not node_names:
        print(f"\n{action_label} skipped (no mapped nodes found).")
        return

    dvfs_controller = DVFSController(
        ssh_user=args.dvfs_ssh_user,
        verify_tolerance_mhz=float(args.dvfs_verify_tol_mhz),
        cpufreq_sync=bool(args.cpufreq_sync),
        cpufreq_governor="userspace",
        cpufreq_min_khz=1_000_000,
    )

    print(
        f"\n{action_label}:",
        f"target_freq_mhz={float(args.max_freq_mhz):.3f}",
        f"nodes={node_names}",
    )
    core_ids_by_node = discover_node_core_ids_by_node(
        node_names=node_names,
        ssh_user=args.dvfs_ssh_user,
        max_workers=dvfs_controller.concurrency,
    )
    node_rules = {
        node_name: [
            {
                "core_numbers": core_ids,
                "frequency_mhz": float(args.max_freq_mhz),
            }
        ]
        for node_name, core_ids in core_ids_by_node.items()
        if core_ids
    }
    dvfs_rows = []
    for row in dvfs_controller.apply_plan_by_node(
        node_rules=node_rules,
        dry_run=bool(args.dry_run),
    ):
        row_with_job = dict(row)
        row_with_job["job"] = summary_job_label
        row_with_job["requested_target_mhz"] = float(args.max_freq_mhz)
        row_with_job["applied_target_mhz"] = float(args.max_freq_mhz)
        dvfs_rows.append(row_with_job)

    if not dvfs_rows:
        print(f"{action_label} produced no core-level actions.")
        return

    print_dvfs_apply_summary(pd.DataFrame(dvfs_rows))


def apply_startup_reset_to_max_frequency(args: argparse.Namespace) -> None:
    apply_full_frequency_reset_to_all_nodes(
        args,
        action_label="Startup DVFS reset",
        summary_job_label="startup_reset",
    )


def apply_overload_end_reset_to_max_frequency(args: argparse.Namespace) -> None:
    apply_full_frequency_reset_to_all_nodes(
        args,
        action_label="Overload-end DVFS reset",
        summary_job_label="overload_end_reset",
    )


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
        help=(
            "Job rank spec (repeatable), e.g. --rank comd=1 --rank minife=2. "
            "Repeating the same job name submits multiple instances, e.g. "
            "--rank xsbenchmpi=2 --rank xsbenchmpi=4"
        ),
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
    parser.add_argument(
        "--pre-submit-wait-s",
        type=float,
        default=0.0,
        help="One-time delay in seconds before the first job submission (default: 0).",
    )
    parser.add_argument(
        "--post-overload-end-rank",
        action="append",
        default=[],
        metavar="JOB=RANK",
        help=(
            "Additional job rank spec (repeatable) submitted once after the first "
            "OVERLOAD_END and --post-overload-end-wait-s delay."
        ),
    )
    parser.add_argument(
        "--post-overload-end-wait-s",
        type=float,
        default=0.0,
        help=(
            "One-time delay in seconds after OVERLOAD_END before submitting "
            "--post-overload-end-rank jobs (default: 0)."
        ),
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
            "e.g. --job-args \"minimd=-i in.lj.miniMD -n 8000 -t 10\". "
            "For duplicated jobs, a base-name override applies to all instances."
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
        "--target-offset-w",
        "--mpr-reduction-offset-w",
        dest="target_offset_w",
        type=float,
        default=0.0,
        help=(
            "Extra watts added to the overload reduction passed to MPR only. "
            "Example: current=920, target=840, offset=20 -> MPR target reduction = 100 W."
        ),
    )
    parser.add_argument(
        "--overload-hysteresis-w",
        type=float,
        default=80.0,
        help="Hysteresis width (watts) below target used by overload detector.",
    )
    parser.add_argument(
        "--overload-min-over-s",
        type=float,
        default=10.0,
        help="Seconds above threshold required to enter OVERLOAD state.",
    )
    parser.add_argument(
        "--wait-before-mpr-s",
        type=float,
        default=0.0,
        help=(
            "Seconds to wait after OVERLOAD_START before running MPR. "
            "During this window, spike events and power samples can raise "
            "the reduction target to the highest observed value."
        ),
    )
    parser.add_argument(
        "--overload-cooldown-s",
        type=float,
        default=30.0,
        help="Seconds below low threshold required to emit OVERLOAD_END.",
    )
    parser.add_argument(
        "--overload-handled-window-s",
        type=float,
        default=10.0,
        help="Seconds inside handled zone required to emit OVERLOAD_HANDLED.",
    )
    parser.add_argument(
        "--overload-handled-high-margin-w",
        type=float,
        default=0.0,
        help="Allowed watts above target for handled-zone classification.",
    )
    parser.add_argument(
        "--overload-handled-low-margin-w",
        type=float,
        default=None,
        help=(
            "Deprecated compatibility option. Ignored; handled detection now "
            "uses only target capacity plus --overload-handled-high-margin-w."
        ),
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
    parser.add_argument(
        "--idle-sample-s",
        type=float,
        default=0.0,
        help=(
            "Optional pre-submit idle-baseline sampling window in seconds. "
            "When > 0, the first market target is computed from idle power plus "
            "the active jobs' modeled full-power baseline."
        ),
    )
    parser.add_argument(
        "--idle-warmup-s",
        type=float,
        default=5.0,
        help="Warmup time before collecting idle-baseline power samples.",
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
        "--post-jobs-monitor-s",
        type=float,
        default=60.0,
        help=(
            "Keep monitoring power/events for this many seconds after jobs complete "
            "(default: 60)."
        ),
    )
    parser.add_argument(
        "--skip-dvfs-apply",
        action="store_true",
        help="Skip applying computed per-job frequencies through DVFS controller.",
    )
    parser.add_argument(
        "--detect-overload-only",
        action="store_true",
        help=(
            "Detect and log overload/handled/end events only. "
            "Skips MPR negotiation and DVFS apply/reset actions."
        ),
    )
    parser.add_argument(
        "--dvfs-ssh-user",
        default=None,
        help="Optional SSH user for remote DVFS apply (e.g. ridl).",
    )
    parser.add_argument(
        "--dvfs-step-mhz",
        type=float,
        default=100.0,
        help="Snap DVFS targets down to this hardware step in MHz (default: 100).",
    )
    parser.add_argument(
        "--dvfs-verify-tol-mhz",
        type=float,
        default=25.0,
        help="Allowed absolute target-vs-readback error (MHz) for PASS/FAIL verification.",
    )
    cpufreq_sync_group = parser.add_mutually_exclusive_group()
    cpufreq_sync_group.add_argument(
        "--cpufreq-sync",
        dest="cpufreq_sync",
        action="store_true",
        help="Enable kernel cpufreq policy sync alongside GEOPM DVFS apply (default).",
    )
    cpufreq_sync_group.add_argument(
        "--no-cpufreq-sync",
        dest="cpufreq_sync",
        action="store_false",
        help="Disable kernel cpufreq policy sync and use GEOPM DVFS apply only.",
    )
    parser.set_defaults(cpufreq_sync=True)
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
    if args.dvfs_step_mhz <= 0:
        parser.error("--dvfs-step-mhz must be > 0.")
    if args.dvfs_verify_tol_mhz < 0:
        parser.error("--dvfs-verify-tol-mhz must be >= 0.")
    if args.idle_sample_s < 0:
        parser.error("--idle-sample-s must be >= 0.")
    if args.idle_warmup_s < 0:
        parser.error("--idle-warmup-s must be >= 0.")
    if args.post_jobs_monitor_s < 0:
        parser.error("--post-jobs-monitor-s must be >= 0.")
    if args.pre_submit_wait_s < 0:
        parser.error("--pre-submit-wait-s must be >= 0.")
    if args.post_overload_end_wait_s < 0:
        parser.error("--post-overload-end-wait-s must be >= 0.")
    if args.overload_hysteresis_w < 0:
        parser.error("--overload-hysteresis-w must be >= 0.")
    if args.overload_min_over_s < 0:
        parser.error("--overload-min-over-s must be >= 0.")
    if args.wait_before_mpr_s < 0:
        parser.error("--wait-before-mpr-s must be >= 0.")
    if args.overload_cooldown_s < 0:
        parser.error("--overload-cooldown-s must be >= 0.")
    if args.overload_handled_window_s < 0:
        parser.error("--overload-handled-window-s must be >= 0.")
    if args.overload_handled_high_margin_w < 0:
        parser.error("--overload-handled-high-margin-w must be >= 0.")
    if (
        args.overload_handled_low_margin_w is not None
        and args.overload_handled_low_margin_w < 0
    ):
        parser.error("--overload-handled-low-margin-w must be >= 0.")
    if args.idle_sample_s > 0.0 and not args.enable_power_monitor:
        parser.error("--idle-sample-s requires --enable-power-monitor.")
    try:
        args.job_ranks, args.job_base_names, args.job_display_names = parse_job_specs(args.rank)
        if args.post_overload_end_rank:
            (
                args.post_overload_end_job_ranks,
                args.post_overload_end_job_base_names,
                args.post_overload_end_job_display_names,
            ) = parse_job_specs(args.post_overload_end_rank)
        else:
            args.post_overload_end_job_ranks = {}
            args.post_overload_end_job_base_names = {}
            args.post_overload_end_job_display_names = {}
        args.submit_env_map = parse_key_value_overrides(args.submit_env)
        args.job_args_map = parse_job_args_overrides(args.job_args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def normalize_job_name(job_text: str) -> tuple[str, str]:
    name = str(job_text).strip()
    if not name:
        raise ValueError("Missing job name.")

    if "@" not in name:
        return name, name

    base_name, alias = name.split("@", 1)
    base_name = base_name.strip()
    alias = alias.strip()
    if not base_name or not alias:
        raise ValueError(
            f"Expected JOB@ALIAS format with non-empty base and alias, got: {job_text}"
        )
    return f"{base_name}@{alias}", base_name


def parse_job_specs(rank_items: list[str]) -> tuple[dict[str, int], dict[str, str], dict[str, str]]:
    parsed_items: list[tuple[str, str, int, bool]] = []
    base_counts: dict[str, int] = {}

    for item in rank_items:
        if "=" not in item:
            raise ValueError(f"Expected JOB=RANK, got: {item}")
        job_text, rank_text = item.split("=", 1)
        job_name, base_name = normalize_job_name(job_text)
        rank = int(rank_text.strip())
        if rank <= 0:
            raise ValueError(f"Rank must be > 0 for job '{job_name}', got {rank}.")
        explicit_alias = "@" in str(job_text)
        parsed_items.append((job_name, base_name, rank, explicit_alias))
        base_counts[base_name] = base_counts.get(base_name, 0) + 1

    job_ranks: dict[str, int] = {}
    job_base_names: dict[str, str] = {}
    job_display_names: dict[str, str] = {}
    seen_counts: dict[str, int] = {}
    for normalized_name, base_name, rank, explicit_alias in parsed_items:
        if explicit_alias:
            job_key = normalized_name
            display_name = normalized_name
        else:
            occurrence = seen_counts.get(base_name, 0) + 1
            seen_counts[base_name] = occurrence
            if base_counts[base_name] > 1:
                job_key = f"{base_name}#{occurrence}"
            else:
                job_key = base_name
            display_name = base_name
        if job_key in job_ranks:
            raise ValueError(f"Duplicate job key '{job_key}' in --rank entries.")
        job_ranks[job_key] = rank
        job_base_names[job_key] = base_name
        job_display_names[job_key] = display_name
    if not job_ranks:
        raise ValueError("At least one --rank JOB=RANK must be provided.")
    return job_ranks, job_base_names, job_display_names


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
        job_name, _ = normalize_job_name(job_name)
        args_tokens = shlex.split(args_text.strip())
        overrides[job_name] = args_tokens
    return overrides

def submit_jobs(
    scheduler: JobScheduler,
    args: argparse.Namespace,
    job_ranks: dict[str, int],
    job_base_names: dict[str, str],
    job_display_names: dict[str, str],
) -> pd.DataFrame:
    if args.skip_submit:
        print("Skipping Slurm submission step (--skip-submit).")
        return pd.DataFrame()

    if args.pre_submit_wait_s > 0 and not args.dry_run:
        print(
            f"Waiting {float(args.pre_submit_wait_s):.3f} seconds before first job submission "
            "(--pre-submit-wait-s)."
        )
        time.sleep(float(args.pre_submit_wait_s))

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
            job_base_names=job_base_names,
            job_display_names=job_display_names,
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
            job_base_names=job_base_names,
            job_display_names=job_display_names,
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


def uniquify_job_specs(
    job_ranks: dict[str, int],
    job_base_names: dict[str, str],
    job_display_names: dict[str, str],
    *,
    existing_keys: set[str],
    suffix_tag: str,
) -> tuple[dict[str, int], dict[str, str], dict[str, str]]:
    unique_job_ranks: dict[str, int] = {}
    unique_job_base_names: dict[str, str] = {}
    unique_job_display_names: dict[str, str] = {}
    reserved_keys = set(existing_keys)

    for job_key, rank_count in job_ranks.items():
        candidate_key = job_key
        suffix_index = 1
        while candidate_key in reserved_keys or candidate_key in unique_job_ranks:
            candidate_key = f"{job_key}__{suffix_tag}{suffix_index}"
            suffix_index += 1

        base_name = job_base_names.get(job_key, job_key)
        display_name = job_display_names.get(job_key, base_name)
        if candidate_key != job_key:
            display_name = candidate_key

        unique_job_ranks[candidate_key] = rank_count
        unique_job_base_names[candidate_key] = base_name
        unique_job_display_names[candidate_key] = display_name
        reserved_keys.add(candidate_key)

    return unique_job_ranks, unique_job_base_names, unique_job_display_names


def submit_post_overload_end_jobs(
    scheduler: JobScheduler,
    args: argparse.Namespace,
    *,
    existing_job_keys: set[str],
) -> tuple[pd.DataFrame, dict[str, int], dict[str, str], dict[str, str]]:
    post_job_ranks = getattr(args, "post_overload_end_job_ranks", {})
    if not post_job_ranks:
        return pd.DataFrame(), {}, {}, {}

    (
        unique_job_ranks,
        unique_job_base_names,
        unique_job_display_names,
    ) = uniquify_job_specs(
        post_job_ranks,
        getattr(args, "post_overload_end_job_base_names", {}),
        getattr(args, "post_overload_end_job_display_names", {}),
        existing_keys=existing_job_keys,
        suffix_tag="post_overload_end",
    )

    submit_df = scheduler.submit_jobs(
        job_ranks=unique_job_ranks,
        cpus_per_rank=args.cpus_per_rank,
        ranks_per_node=args.ranks_per_node,
        partition=args.partition,
        time_limit=args.time_limit,
        nodelist=args.nodelist,
        exclude=args.exclude,
        output_path=args.slurm_output,
        mpi_iface=args.mpi_iface,
        env_overrides=args.submit_env_map,
        job_base_names=unique_job_base_names,
        job_display_names=unique_job_display_names,
        script_args_by_job=args.job_args_map,
        use_rank_profiles=not args.disable_rank_profiles,
        dry_run=args.dry_run,
    )

    print("\nPost-OVERLOAD_END submission summary:")
    if submit_df.empty:
        print("No follow-up submissions attempted.")
    else:
        print(submit_df.to_string(index=False))

    return submit_df, unique_job_ranks, unique_job_base_names, unique_job_display_names


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
    *,
    strict: bool = True,
    audit_heading: str = "Performance data audit",
    loaded_heading: str = "Loaded job models",
) -> dict[str, pd.DataFrame]:
    job_perf_data, perf_audit_df = scheduler.get_cached_perf_data(
        job_names=list(job_ranks.keys())
    )
    print(f"\n{audit_heading}:")
    if perf_audit_df.empty:
        print("No cached audit rows. Submit jobs (or disable --skip-submit) to auto-load performance data.")
    else:
        print(perf_audit_df.to_string(index=False))
    print(f"{loaded_heading}:", list(job_perf_data.keys()))

    if strict and not job_perf_data:
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
        recompute_final_bids=False,
        enable_feasible_fallback=False,
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
) -> tuple[float, float, float]:
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

    overload_w = max(0.0, current_power_w - float(args.target_capacity_w))
    target_reduction_w = compute_mpr_target_reduction_w(
        overload_w,
        float(args.target_offset_w),
    )
    return target_reduction_w, current_power_w, overload_w


def compute_mpr_target_reduction_w(
    base_reduction_w: float,
    target_offset_w: float,
) -> float:
    base = max(0.0, float(base_reduction_w))
    if base <= 0.0:
        return 0.0
    return max(0.0, base + float(target_offset_w))


def compute_baseline_power_max(job_perf_data: dict[str, pd.DataFrame]) -> float:
    models = normalize_job_perf_data(job_perf_data)
    return float(sum(float(model.power_max) for model in models))


def estimate_fixed_overhead_w(
    *,
    current_power_w: float,
    active_jobs_power_max_w: float,
    current_saved_power_w: float,
) -> float:
    estimated_cluster_baseline_w = max(0.0, float(current_power_w)) + max(0.0, float(current_saved_power_w))
    return max(0.0, estimated_cluster_baseline_w - max(0.0, float(active_jobs_power_max_w)))


def resolve_reapply_fixed_overhead_w(
    *,
    idle_power_w: float | None,
    current_power_w: float,
    active_jobs_power_max_w: float,
    current_saved_power_w: float,
) -> tuple[float, str]:
    if idle_power_w is not None:
        return max(0.0, float(idle_power_w)), "idle_baseline"
    return (
        estimate_fixed_overhead_w(
            current_power_w=current_power_w,
            active_jobs_power_max_w=active_jobs_power_max_w,
            current_saved_power_w=current_saved_power_w,
        ),
        "dynamic_estimate",
    )


def compute_reapply_target_reduction_w_from_baseline(
    *,
    active_jobs_power_max_w: float,
    fixed_overhead_w: float,
    target_capacity_w: float,
    target_offset_w: float,
) -> float:
    cluster_baseline_w = max(0.0, float(active_jobs_power_max_w)) + max(0.0, float(fixed_overhead_w))
    base_reduction_w = max(0.0, cluster_baseline_w - float(target_capacity_w))
    return compute_mpr_target_reduction_w(base_reduction_w, float(target_offset_w))


def measure_idle_power_baseline(
    *,
    monitor: PowerMonitor,
    sample_window_s: float,
    warmup_s: float,
    poll_interval_s: float,
    startup_wait_s: float,
) -> float:
    window_s = max(0.0, float(sample_window_s))
    if window_s <= 0.0:
        raise ValueError("sample_window_s must be > 0")

    _ = wait_for_initial_power_sample(monitor, startup_wait_s)

    warmup = max(0.0, float(warmup_s))
    if warmup > 0.0:
        print("[Control] Idle baseline warmup:", f"warmup_s={warmup:.3f}")
        time.sleep(warmup)

    deadline = time.monotonic() + window_s
    seen_timestamps: set[str] = set()
    watts: list[float] = []

    while True:
        sample = monitor.get_last_sample()
        if sample is not None:
            ts = str(sample.get("timestamp", ""))
            if ts and ts not in seen_timestamps:
                seen_timestamps.add(ts)
                watts.append(float(sample.get("total_watts", 0.0)))

        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            break
        time.sleep(min(max(0.05, float(poll_interval_s)), remaining_s))

    if not watts:
        raise RuntimeError(
            "Idle baseline sampling collected no power samples. "
            "Increase --idle-sample-s or --power-startup-wait-s."
        )

    avg_idle_power_w = float(sum(watts) / len(watts))
    print(
        "[Control] Idle power baseline:",
        f"samples={len(watts)}",
        f"avg_idle_power_w={avg_idle_power_w:.3f}",
        f"min_idle_power_w={min(watts):.3f}",
        f"max_idle_power_w={max(watts):.3f}",
    )
    return avg_idle_power_w


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


def get_latest_submitted_job_rows(
    scheduler: JobScheduler,
    job_names: list[str],
) -> pd.DataFrame:
    if not job_names:
        return pd.DataFrame()

    history = scheduler.get_submission_history(job_names=job_names)
    if history.empty:
        return history

    if "status" in history.columns:
        history = history[history["status"].astype(str) == "SUBMITTED"]
    if history.empty:
        return history

    key_column = "job_key" if "job_key" in history.columns else "job"
    if key_column not in history.columns or "job_id" not in history.columns:
        return pd.DataFrame()

    return history.groupby(key_column, as_index=False).tail(1).reset_index(drop=True)


def resolve_active_control_context(
    *,
    scheduler: JobScheduler,
    tracked_job_keys: list[str],
    job_perf_data: dict[str, pd.DataFrame],
    job_states: dict[str, str],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    latest_rows = get_latest_submitted_job_rows(scheduler, tracked_job_keys)
    if latest_rows.empty:
        return {}, {}

    key_column = "job_key" if "job_key" in latest_rows.columns else "job"
    candidate_job_names: list[str] = []
    for _, row in latest_rows.iterrows():
        job_name = str(row[key_column]).strip()
        job_id = str(row["job_id"]).strip()
        if not job_name or not job_id:
            continue
        state = str(job_states.get(job_id, "COMPLETED")).upper()
        if state in CONTROL_ELIGIBLE_JOB_STATES:
            candidate_job_names.append(job_name)

    if not candidate_job_names:
        return {}, {}

    cached_lookup = None
    if hasattr(type(scheduler), "get_cached_submitted_job_allocations"):
        cached_lookup = getattr(scheduler, "get_cached_submitted_job_allocations", None)
    elif "get_cached_submitted_job_allocations" in getattr(scheduler, "__dict__", {}):
        cached_lookup = getattr(scheduler, "get_cached_submitted_job_allocations", None)
    if callable(cached_lookup):
        allocations = cached_lookup(
            job_names=candidate_job_names,
            latest_only=True,
        )
        missing_job_names = [
            job_name
            for job_name in candidate_job_names
            if job_name not in allocations
            or not isinstance(allocations[job_name].get("cores_by_node", {}), dict)
            or not allocations[job_name].get("cores_by_node")
        ]
        if missing_job_names:
            print(
                "[Control] Allocation cache miss for active jobs:",
                missing_job_names,
            )
            fetched_allocations = scheduler.get_submitted_job_allocations(
                job_names=missing_job_names,
                latest_only=True,
            )
            allocations = dict(allocations)
            allocations.update(fetched_allocations)
    else:
        allocations = scheduler.get_submitted_job_allocations(
            job_names=candidate_job_names,
            latest_only=True,
        )
    active_allocations: dict[str, dict[str, Any]] = {}
    for job_name in candidate_job_names:
        allocation = allocations.get(job_name)
        if not allocation:
            continue
        cores_by_node = allocation.get("cores_by_node", {})
        if not isinstance(cores_by_node, dict) or not cores_by_node:
            continue
        active_allocations[job_name] = allocation

    active_job_perf_data = {
        job_name: job_perf_data[job_name]
        for job_name in candidate_job_names
        if job_name in active_allocations and job_name in job_perf_data
    }
    active_allocations = {
        job_name: active_allocations[job_name]
        for job_name in candidate_job_names
        if job_name in active_allocations
    }
    return active_job_perf_data, active_allocations


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


def wait_before_market_start(
    *,
    monitor: PowerMonitor,
    overload_event_queue: queue.Queue[dict[str, Any]],
    target_capacity_w: float,
    initial_total_watts: float,
    initial_required_reduction_w: float,
    initial_timestamp: str,
    wait_before_mpr_s: float,
    poll_interval_s: float,
) -> tuple[float, float, str, str | None]:
    wait_s = max(0.0, float(wait_before_mpr_s))
    peak_total_watts = max(0.0, float(initial_total_watts))
    peak_required_reduction_w = max(0.0, float(initial_required_reduction_w))
    peak_source = "OVERLOAD_START"
    peak_timestamp = str(initial_timestamp)

    if wait_s <= 0.0:
        return peak_total_watts, peak_required_reduction_w, peak_source, None

    print(
        "[Control] Waiting before MPR:",
        f"wait_before_mpr_s={wait_s:.3f}",
        "tracking peak reduction from overload/spike events.",
    )
    deadline = time.monotonic() + wait_s
    peak_event_names = {"OVERLOAD_START", "SPIKE_WARNING"}
    terminal_event_names = {"OVERLOAD_END", "OVERLOAD_HANDLED"}

    while True:
        sample = monitor.get_last_sample()
        if sample is not None:
            sample_total_watts = float(sample.get("total_watts", 0.0))
            sample_required_reduction_w = max(0.0, sample_total_watts - float(target_capacity_w))
            if sample_required_reduction_w > peak_required_reduction_w:
                peak_total_watts = sample_total_watts
                peak_required_reduction_w = sample_required_reduction_w
                peak_source = "POWER_SAMPLE"
                peak_timestamp = str(sample.get("timestamp", peak_timestamp))

        for event in drain_overload_events(overload_event_queue):
            event_name = str(event.get("event", ""))
            ts = str(event.get("timestamp", ""))
            total_watts = float(event.get("total_watts", 0.0))
            required_w = float(
                event.get(
                    "required_reduction_w",
                    max(0.0, total_watts - float(target_capacity_w)),
                )
            )
            print(
                f"[Control] pre-market event={event_name} ts={ts} "
                f"total_watts={total_watts:.3f} required_reduction_w={required_w:.3f}"
            )
            if event_name in peak_event_names and required_w > peak_required_reduction_w:
                peak_total_watts = total_watts
                peak_required_reduction_w = required_w
                peak_source = event_name
                peak_timestamp = ts
            if event_name in terminal_event_names:
                print(
                    f"[Control] {event_name} observed during wait-before-MPR; "
                    "skipping market start because overload no longer needs a new action."
                )
                return peak_total_watts, peak_required_reduction_w, peak_source, event_name

        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            break
        time.sleep(min(max(0.05, float(poll_interval_s)), remaining_s))

    print(
        "[Control] Pre-market peak selected:",
        f"source={peak_source}",
        f"ts={peak_timestamp}",
        f"peak_total_watts={peak_total_watts:.3f}",
        f"peak_required_reduction_w={peak_required_reduction_w:.3f}",
    )
    return peak_total_watts, peak_required_reduction_w, peak_source, None


def compute_frequency_targets(
    result: dict[str, Any],
    job_perf_data: dict[str, pd.DataFrame],
    max_freq_mhz: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    models = normalize_job_perf_data(job_perf_data)
    delta_max_by_job = {model.client_id: float(model.delta_max) for model in models}
    model_by_job = {model.client_id: model for model in models}
    q_final = float(result["final_q"])

    job_reduction: dict[str, float] = {}
    job_freq_mhz: dict[str, float] = {}
    job_power_saved_w: dict[str, float] = {}
    for client_id, bid in dict(result["final_bids"]).items():
        delta = max(delta_max_by_job[client_id] - float(bid) / q_final, 0.0)
        delta = min(delta, 1.0)
        job_reduction[client_id] = delta
        # Prefer workbook-provided frequency model when available.
        freq_mhz: float | None = None
        raw_df = job_perf_data.get(client_id)
        if raw_df is not None and "Resource Reduction" in raw_df.columns:
            freq_col: str | None = None
            for candidate in raw_df.columns:
                name = str(candidate).strip().lower()
                if name in {"frequencies", "frequency", "freq", "frequency_mhz"}:
                    freq_col = str(candidate)
                    break
            if freq_col is not None:
                rr_vals = pd.to_numeric(raw_df["Resource Reduction"], errors="coerce").to_numpy(dtype=float)
                freq_vals = pd.to_numeric(raw_df[freq_col], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(rr_vals) & np.isfinite(freq_vals)
                rr_vals = rr_vals[valid]
                freq_vals = freq_vals[valid]
                if rr_vals.size > 0:
                    order = np.argsort(rr_vals)
                    rr_vals = rr_vals[order]
                    freq_vals = freq_vals[order]
                    freq_mhz = float(np.interp(delta, rr_vals, freq_vals))

        if freq_mhz is None:
            freq_mhz = max_freq_mhz * (1.0 - delta)

        job_freq_mhz[client_id] = freq_mhz
        model = model_by_job[client_id]
        power_at_delta = float(np.interp(delta, model.rr, model.pw))
        job_power_saved_w[client_id] = float(model.power_max - power_at_delta)

    freq_df = (
        pd.DataFrame(
            {
                "job": list(job_freq_mhz.keys()),
                "reduction_fraction": [job_reduction[key] for key in job_freq_mhz.keys()],
                "target_freq_mhz": [job_freq_mhz[key] for key in job_freq_mhz.keys()],
                "target_freq_ghz": [job_freq_mhz[key] / 1000.0 for key in job_freq_mhz.keys()],
                "power_saved_w": [job_power_saved_w[key] for key in job_freq_mhz.keys()],
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

    dvfs_controller = DVFSController(
        ssh_user=args.dvfs_ssh_user,
        verify_tolerance_mhz=float(args.dvfs_verify_tol_mhz),
        cpufreq_sync=bool(args.cpufreq_sync),
        cpufreq_governor="userspace",
        cpufreq_min_khz=1_000_000,
    )
    node_rules, core_metadata, used_allocations = build_dvfs_apply_plan(
        args=args,
        job_freq_mhz=job_freq_mhz,
        allocations=allocations,
    )

    dvfs_rows: list[dict[str, object]] = []
    for row in dvfs_controller.apply_plan_by_node(
        node_rules=node_rules,
        dry_run=bool(args.dry_run),
    ):
        row_with_job = dict(row)
        meta = core_metadata.get((str(row_with_job["node_name"]), int(row_with_job["core_number"])))
        if meta is not None:
            row_with_job["job"] = meta["job"]
            row_with_job["requested_target_mhz"] = float(meta["requested_target_mhz"])
            row_with_job["applied_target_mhz"] = float(meta["applied_target_mhz"])
        dvfs_rows.append(row_with_job)

    if not dvfs_rows:
        print("\nDVFS apply produced no core-level actions.")
        return used_allocations

    dvfs_df = pd.DataFrame(dvfs_rows)
    print_dvfs_apply_summary(dvfs_df)
    return used_allocations


def snap_floor_frequency_mhz(*, requested_mhz: float, step_mhz: float) -> float:
    requested = float(requested_mhz)
    step = float(step_mhz)
    if step <= 0.0:
        raise ValueError("step_mhz must be > 0")
    snapped = math.floor(requested / step) * step
    # Keep strictly positive request for controller/config paths.
    if snapped <= 0.0:
        snapped = step
    return float(snapped)


def print_dvfs_apply_summary(dvfs_df: pd.DataFrame) -> None:
    host_summary = (
        dvfs_df.groupby(["job", "node_name"], as_index=False)
        .agg(
            cores=("core_number", "nunique"),
            requested_target_mhz=("requested_target_mhz", "first"),
            applied_target_mhz=("applied_target_mhz", "first"),
            readback_avg_mhz=("readback_avg_mhz", "first"),
            readback_min_mhz=("readback_min_mhz", "first"),
            readback_max_mhz=("readback_max_mhz", "first"),
            readback_abs_error_mhz=("readback_abs_error_mhz", "first"),
            verify_tolerance_mhz=("verify_tolerance_mhz", "first"),
            verify_status=("verify_status", "first"),
            verify_reason=("verify_reason", "first"),
            status=("status", "first"),
        )
        .sort_values(["job", "node_name"])
        .reset_index(drop=True)
    )
    print("\nDVFS apply summary:")
    print(host_summary.to_string(index=False))

    failure_rows = host_summary[~host_summary["verify_status"].isin(["PASS", "SKIP_DRY_RUN"])]
    if not failure_rows.empty:
        print("\nDVFS verification details (non-pass):")
        detail_cols = ["job", "node_name", "status", "verify_status", "verify_reason"]
        print(failure_rows[detail_cols].to_string(index=False))
        failure_keys = set(zip(failure_rows["job"], failure_rows["node_name"]))
        failure_detail = (
            dvfs_df[
                dvfs_df.apply(
                    lambda row: (row["job"], row["node_name"]) in failure_keys,
                    axis=1,
                )
            ][["job", "node_name", "run_target", "returncode", "command", "stdout", "stderr"]]
            .drop_duplicates(subset=["job", "node_name"])
            .reset_index(drop=True)
        )
        for _, row in failure_detail.iterrows():
            print(
                "\nDVFS failure payload:",
                f"job={row['job']}",
                f"node={row['node_name']}",
                f"run_target={row['run_target']}",
                f"returncode={row['returncode']}",
            )
            print(f"command: {row['command']}")
            stdout_text = str(row["stdout"]).strip()
            stderr_text = str(row["stderr"]).strip()
            if stdout_text:
                print("stdout:")
                print(stdout_text)
            if stderr_text:
                print("stderr:")
                print(stderr_text)


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
    overload_hysteresis_w: float,
    overload_min_over_s: float,
    overload_cooldown_s: float,
    overload_handled_window_s: float,
    overload_handled_high_margin_w: float,
    overload_handled_low_margin_w: float | None,
) -> None:
    print_interval = max(0.1, float(print_interval_s))
    poll_interval = max(0.1, min(0.5, float(sample_period_s) * 0.5))
    next_print_at = 0.0
    last_sample_ts: str | None = None
    overload_ctx = make_simple_overload_ctx(
        sample_period_s=max(0.1, float(sample_period_s)),
        threshold_w=float(target_capacity_w),
        hysteresis_w=float(overload_hysteresis_w),
        min_over_s=float(overload_min_over_s),
        cooldown_s=float(overload_cooldown_s),
        handled_window_s=float(overload_handled_window_s),
        handled_high_margin_w=float(overload_handled_high_margin_w),
        handled_low_margin_w=(
            None
            if overload_handled_low_margin_w is None
            else float(overload_handled_low_margin_w)
        ),
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
            args.overload_hysteresis_w,
            args.overload_min_over_s,
            args.overload_cooldown_s,
            args.overload_handled_window_s,
            args.overload_handled_high_margin_w,
            args.overload_handled_low_margin_w,
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
    base_required_reduction_w: float | None,
    allocations_cache: dict[str, dict[str, Any]],
    target_reduction_w_override: float | None = None,
) -> tuple[bool, dict[str, dict[str, Any]], float, dict[str, float]]:
    if not job_perf_data:
        print("\n[Control] No running jobs with live allocations are eligible for MPR. Skipping.")
        return False, allocations_cache, 0.0, {}

    overload_w = max(0.0, float(current_power_w) - float(args.target_capacity_w))
    base_reduction_w = (
        overload_w
        if base_required_reduction_w is None
        else max(0.0, float(base_required_reduction_w))
    )
    if target_reduction_w_override is None:
        target_reduction_w = compute_mpr_target_reduction_w(
            base_reduction_w,
            float(args.target_offset_w),
        )
        target_offset_w = float(args.target_offset_w)
    else:
        target_reduction_w = max(0.0, float(target_reduction_w_override))
        target_offset_w = math.nan
    print(
        "\nCapacity check:",
        f"current_power_w={current_power_w:.3f}",
        f"target_capacity_w={args.target_capacity_w:.3f}",
        f"overload_w={overload_w:.3f}",
        f"required_reduction_w={base_reduction_w:.3f}",
        (
            f"target_offset_w={target_offset_w:.3f}"
            if math.isfinite(target_offset_w)
            else "target_offset_w=OVERRIDE"
        ),
        f"mpr_target_reduction_w={target_reduction_w:.3f}",
    )
    if target_reduction_w <= 0.0:
        print("No overload detected (current power is within target capacity).")
        return False, allocations_cache, 0.0, {}

    market_result = run_market(args, job_perf_data, target_reduction_w)
    if bool(market_result.get("market_failed")):
        print(
            "[Control] Market failed: negative residual indicates target reduction "
            "was not met. Skipping DVFS reduction apply."
        )
        return False, allocations_cache, 0.0, {}
    freq_df, job_freq_mhz = compute_frequency_targets(
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
    saved_power_by_job = {
        str(row["job"]): float(row["power_saved_w"])
        for _, row in freq_df.iterrows()
        if pd.notna(row.get("job")) and pd.notna(row.get("power_saved_w"))
    }
    applied_reduction_w = float(sum(saved_power_by_job.values()))
    if applied_reduction_w <= 0.0:
        applied_reduction_w = float(market_result.get("final_reduction_w", target_reduction_w))
    return True, allocations_cache, applied_reduction_w, saved_power_by_job


def run_event_driven_control_loop(
    *,
    scheduler: JobScheduler,
    args: argparse.Namespace,
    monitor: PowerMonitor,
    idle_power_w: float | None,
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

    control_actions_enabled = not bool(args.detect_overload_only)
    post_jobs_monitor_s = max(0.0, float(args.post_jobs_monitor_s))
    reduction_active = False
    reset_applied = True
    jobs_completed_since: float | None = None
    market_allocations_cache: dict[str, dict[str, Any]] = {}
    tracked_job_ranks = dict(job_ranks)
    post_overload_end_submit_at: float | None = None
    post_overload_end_submitted = False
    current_job_power_saved_w_by_job: dict[str, float] = {}
    last_reduction_apply_at: float | None = None
    reapply_min_interval_s = max(5.0, float(args.power_interval_s) * 3.0)

    if not control_actions_enabled:
        print(
            "[Control] detect-overload-only mode enabled: "
            "MPR/DVFS actions are disabled."
        )

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
                if not control_actions_enabled:
                    print("[Control] OVERLOAD_START observed (no control action in detect-overload-only mode).")
                else:
                    active_job_perf_data, active_allocations = resolve_active_control_context(
                        scheduler=scheduler,
                        tracked_job_keys=list(tracked_job_ranks.keys()),
                        job_perf_data=job_perf_data,
                        job_states=job_states,
                    )
                    market_allocations_cache = active_allocations
                    selected_total_watts = total_watts
                    selected_required_w = required_w
                    terminal_event_name: str | None = None
                    if float(args.wait_before_mpr_s) > 0.0:
                        (
                            selected_total_watts,
                            selected_required_w,
                            _peak_source,
                            terminal_event_name,
                        ) = wait_before_market_start(
                            monitor=monitor,
                            overload_event_queue=overload_event_queue,
                            target_capacity_w=float(args.target_capacity_w),
                            initial_total_watts=total_watts,
                            initial_required_reduction_w=required_w,
                            initial_timestamp=ts,
                            wait_before_mpr_s=float(args.wait_before_mpr_s),
                            poll_interval_s=loop_interval_s,
                        )
                    if terminal_event_name is not None:
                        continue
                    target_reduction_override: float | None = None
                    if idle_power_w is not None and active_job_perf_data:
                        active_jobs_power_max_w = compute_baseline_power_max(active_job_perf_data)
                        target_reduction_override = compute_reapply_target_reduction_w_from_baseline(
                            active_jobs_power_max_w=active_jobs_power_max_w,
                            fixed_overhead_w=float(idle_power_w),
                            target_capacity_w=float(args.target_capacity_w),
                            target_offset_w=float(args.target_offset_w),
                        )
                        print(
                            "[Control] First-market idle baseline target:",
                            f"idle_power_w={float(idle_power_w):.3f}",
                            f"active_jobs_power_max_w={active_jobs_power_max_w:.3f}",
                            f"target_reduction_w={target_reduction_override:.3f}",
                        )
                    reduction_applied, market_allocations_cache, _applied_reduction_w, saved_power_by_job = apply_overload_reduction(
                        args=args,
                        scheduler=scheduler,
                        job_perf_data=active_job_perf_data,
                        current_power_w=selected_total_watts,
                        base_required_reduction_w=selected_required_w,
                        allocations_cache=market_allocations_cache,
                        target_reduction_w_override=target_reduction_override,
                    )
                    if reduction_applied:
                        reduction_active = True
                        reset_applied = False
                        current_job_power_saved_w_by_job = dict(saved_power_by_job)
                        last_reduction_apply_at = time.monotonic()
            elif (
                control_actions_enabled
                and reduction_active
                and event_name in REAPPLY_TRIGGER_EVENT_NAMES
            ):
                if required_w <= 0.0:
                    continue
                now_reapply = time.monotonic()
                if (
                    last_reduction_apply_at is not None
                    and (now_reapply - last_reduction_apply_at) < reapply_min_interval_s
                ):
                    continue
                active_job_perf_data, active_allocations = resolve_active_control_context(
                    scheduler=scheduler,
                    tracked_job_keys=list(tracked_job_ranks.keys()),
                    job_perf_data=job_perf_data,
                    job_states=job_states,
                )
                market_allocations_cache = active_allocations
                selected_total_watts = total_watts
                selected_required_w = required_w
                terminal_event_name: str | None = None
                if float(args.wait_before_mpr_s) > 0.0:
                    (
                        selected_total_watts,
                        selected_required_w,
                        _peak_source,
                        terminal_event_name,
                    ) = wait_before_market_start(
                        monitor=monitor,
                        overload_event_queue=overload_event_queue,
                        target_capacity_w=float(args.target_capacity_w),
                        initial_total_watts=total_watts,
                        initial_required_reduction_w=required_w,
                        initial_timestamp=ts,
                        wait_before_mpr_s=float(args.wait_before_mpr_s),
                        poll_interval_s=loop_interval_s,
                    )
                if terminal_event_name is not None:
                    continue
                active_jobs_power_max_w = compute_baseline_power_max(active_job_perf_data)
                current_saved_power_w = sum(
                    float(current_job_power_saved_w_by_job.get(job_name, 0.0))
                    for job_name in active_job_perf_data.keys()
                )
                fixed_overhead_w, fixed_overhead_source = resolve_reapply_fixed_overhead_w(
                    idle_power_w=idle_power_w,
                    current_power_w=selected_total_watts,
                    active_jobs_power_max_w=active_jobs_power_max_w,
                    current_saved_power_w=current_saved_power_w,
                )
                target_reduction_w = compute_reapply_target_reduction_w_from_baseline(
                    active_jobs_power_max_w=active_jobs_power_max_w,
                    fixed_overhead_w=fixed_overhead_w,
                    target_capacity_w=float(args.target_capacity_w),
                    target_offset_w=float(args.target_offset_w),
                )
                if target_reduction_w <= current_saved_power_w + 1e-9:
                    continue
                print(
                    "[Control] Reapplying MPR during active reduction:",
                    f"trigger={event_name}",
                    f"active_jobs_power_max_w={active_jobs_power_max_w:.3f}",
                    f"current_saved_power_w={current_saved_power_w:.3f}",
                    f"fixed_overhead_source={fixed_overhead_source}",
                    f"fixed_overhead_w={fixed_overhead_w:.3f}",
                    f"target_reduction_w={target_reduction_w:.3f}",
                )
                reduction_applied, market_allocations_cache, _applied_reduction_w, saved_power_by_job = apply_overload_reduction(
                    args=args,
                    scheduler=scheduler,
                    job_perf_data=active_job_perf_data,
                    current_power_w=selected_total_watts,
                    base_required_reduction_w=selected_required_w,
                    allocations_cache=market_allocations_cache,
                    target_reduction_w_override=target_reduction_w,
                )
                if reduction_applied:
                    reduction_active = True
                    reset_applied = False
                    current_job_power_saved_w_by_job = dict(saved_power_by_job)
                    last_reduction_apply_at = time.monotonic()
            elif event_name == "OVERLOAD_END":
                if not control_actions_enabled:
                    print("[Control] OVERLOAD_END observed (no DVFS reset in detect-overload-only mode).")
                else:
                    apply_overload_end_reset_to_max_frequency(args)
                    market_allocations_cache = {}
                    reduction_active = False
                    reset_applied = True
                    current_job_power_saved_w_by_job = {}
                    last_reduction_apply_at = None
                if (
                    args.post_overload_end_job_ranks
                    and not post_overload_end_submitted
                    and post_overload_end_submit_at is None
                ):
                    post_overload_end_submit_at = time.monotonic() + float(args.post_overload_end_wait_s)
                    print(
                        "[Control] Scheduling post-OVERLOAD_END jobs:",
                        f"wait_s={float(args.post_overload_end_wait_s):.3f}",
                        f"job_ranks={args.post_overload_end_job_ranks}",
                    )
            elif event_name == "OVERLOAD_HANDLED":
                payload = event.get("payload", {})
                if isinstance(payload, dict):
                    zone_low = payload.get("zone_low_w")
                    zone_high = payload.get("zone_high_w")
                else:
                    zone_low = None
                    zone_high = None
                if zone_low is not None and zone_high is not None:
                    print(
                        "[Control] overload handled zone reached: "
                        f"{float(zone_low):.3f}W..{float(zone_high):.3f}W"
                    )
                elif zone_high is not None:
                    print(
                        "[Control] overload handled threshold reached: "
                        f"<= {float(zone_high):.3f}W"
                    )
                else:
                    print("[Control] overload handled zone reached.")

        if (
            post_overload_end_submit_at is not None
            and not post_overload_end_submitted
            and time.monotonic() >= post_overload_end_submit_at
        ):
            (
                post_submit_df,
                new_job_ranks,
                _new_job_base_names,
                _new_job_display_names,
            ) = submit_post_overload_end_jobs(
                scheduler,
                args,
                existing_job_keys=set(tracked_job_ranks.keys()),
            )
            post_overload_end_submit_at = None
            post_overload_end_submitted = True
            if not post_submit_df.empty:
                new_job_ids = collect_submitted_job_ids(post_submit_df)
                submitted_job_ids.extend(job_id for job_id in new_job_ids if job_id not in submitted_job_ids)
                tracked_job_ranks.update(new_job_ranks)
                job_perf_data.update(
                    fetch_job_perf_data(
                        scheduler,
                        new_job_ranks,
                        strict=not args.dry_run,
                        audit_heading="Post-OVERLOAD_END performance data audit",
                        loaded_heading="Loaded post-OVERLOAD_END job models",
                    )
                )
                jobs_completed_since = None
                for job_id in new_job_ids:
                    job_states.setdefault(job_id, "PENDING")
                next_job_poll = 0.0
                next_job_print = 0.0
                print_allocations(scheduler, args, post_submit_df, new_job_ranks)

        jobs_active = has_active_jobs(job_states)
        if jobs_active:
            jobs_completed_since = None
        else:
            now_done = time.monotonic()
            if post_overload_end_submit_at is not None and not post_overload_end_submitted:
                remaining_s = max(0.0, post_overload_end_submit_at - now_done)
                if jobs_completed_since is None:
                    jobs_completed_since = now_done
                    print(
                        "[Control] All currently submitted jobs completed. "
                        f"Waiting {remaining_s:.1f}s for post-OVERLOAD_END submission."
                    )
                time.sleep(loop_interval_s)
                continue
            if jobs_completed_since is None:
                jobs_completed_since = now_done
                print(
                    "[Control] All submitted jobs completed. "
                    f"Continuing monitor/event loop for {post_jobs_monitor_s:.1f}s."
                )
            elapsed_after_jobs = now_done - jobs_completed_since
            if elapsed_after_jobs >= post_jobs_monitor_s:
                if reduction_active and not reset_applied:
                    if not control_actions_enabled:
                        print(
                            "[Control] Post-job monitor window ended before OVERLOAD_END; "
                            "detect-overload-only mode keeps frequencies unchanged."
                        )
                    else:
                        print(
                            "[Control] Post-job monitor window ended before OVERLOAD_END; "
                            "applying safety DVFS reset to max frequency."
                        )
                        apply_overload_end_reset_to_max_frequency(args)
                        market_allocations_cache = {}
                        reduction_active = False
                        reset_applied = True
                        current_job_power_saved_w_by_job = {}
                        last_reduction_apply_at = None
                print("Post-job monitoring window ended. Exiting event-driven control loop.")
                break

        time.sleep(loop_interval_s)


def main() -> int:
    args = parse_args()
    scheduler = JobScheduler()
    job_ranks = args.job_ranks
    monitor = None
    monitor_stop_event = None
    monitor_worker = None
    overload_event_queue = None
    idle_power_w: float | None = None

    try:
        apply_startup_reset_to_max_frequency(args)
        monitor, monitor_stop_event, monitor_worker, overload_event_queue = start_power_monitor(args)
        if monitor is not None and float(args.idle_sample_s) > 0.0:
            idle_power_w = measure_idle_power_baseline(
                monitor=monitor,
                sample_window_s=float(args.idle_sample_s),
                warmup_s=float(args.idle_warmup_s),
                poll_interval_s=float(args.power_interval_s),
                startup_wait_s=float(args.power_startup_wait_s),
            )
            if overload_event_queue is not None:
                _ = drain_overload_events(overload_event_queue)
        print("Source Slurm scripts:", scheduler.source_scripts_dir)
        print("Available jobs:", scheduler.available_jobs())
        print("Performance workbook:", scheduler.perf_xlsx_path)
        print("Job ranks:", job_ranks)
        if any(name != display for name, display in args.job_display_names.items()):
            print("Job display names:", args.job_display_names)
        if any(name != base for name, base in args.job_base_names.items()):
            print("Job base names:", args.job_base_names)
        if args.post_overload_end_job_ranks:
            print("Post-OVERLOAD_END job ranks:", args.post_overload_end_job_ranks)
            print(f"Post-OVERLOAD_END wait_s: {float(args.post_overload_end_wait_s):.3f}")
            if any(
                name != display
                for name, display in args.post_overload_end_job_display_names.items()
            ):
                print("Post-OVERLOAD_END job display names:", args.post_overload_end_job_display_names)
            if any(
                name != base
                for name, base in args.post_overload_end_job_base_names.items()
            ):
                print("Post-OVERLOAD_END job base names:", args.post_overload_end_job_base_names)

        submit_df = submit_jobs(
            scheduler,
            args,
            job_ranks,
            args.job_base_names,
            args.job_display_names,
        )
        print_allocations(scheduler, args, submit_df, job_ranks)

        job_perf_data = fetch_job_perf_data(
            scheduler,
            job_ranks,
            strict=not args.dry_run,
        )
        if args.dry_run:
            return 0
        if monitor is not None and overload_event_queue is not None:
            run_event_driven_control_loop(
                scheduler=scheduler,
                args=args,
                monitor=monitor,
                idle_power_w=idle_power_w,
                overload_event_queue=overload_event_queue,
                submit_df=submit_df,
                job_ranks=job_ranks,
                job_perf_data=job_perf_data,
            )
        else:
            target_reduction_w, current_power_w, overload_w = resolve_target_reduction_w(args, monitor)
            print(
                "\nCapacity check:",
                f"current_power_w={current_power_w:.3f}",
                f"target_capacity_w={args.target_capacity_w:.3f}",
                f"overload_w={overload_w:.3f}",
                f"target_offset_w={float(args.target_offset_w):.3f}",
                f"mpr_target_reduction_w={target_reduction_w:.3f}",
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
