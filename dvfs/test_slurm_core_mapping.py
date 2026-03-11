#!/usr/bin/env python3
"""Validate Slurm CPU allocation to physical core mapping for a running job.

Examples:
    python3 -m dvfs.test_slurm_core_mapping --job-id 3173
    python3 -m dvfs.test_slurm_core_mapping --job-id 3173 --json
    python3 -m dvfs.test_slurm_core_mapping --submit-job hpccg --ranks 2 --cancel-after
"""

from __future__ import annotations

import argparse
import json
import shlex
import time
from typing import Any, Dict, List

from job_scheduler import JobScheduler
from job_scheduler.helper_utility import SlurmHelperUtility


def _probe_job_topology(helper: SlurmHelperUtility, job_id: str, node_count: int) -> Dict[str, Dict[str, Any]]:
    probe_cmd = [
        helper.srun_cmd,
        f"--jobid={job_id}",
        "-N",
        str(node_count),
        "--ntasks-per-node=1",
        "--overlap",
        "--immediate=10",
        "bash",
        "-lc",
        (
            'host=$(hostname -s); '
            'cpus=$(awk \'/^Cpus_allowed_list:/ {print $2; exit}\' /proc/self/status); '
            'topo=$(lscpu -p=CPU,CORE | awk -F, \'/^[^#]/ {printf "%s:%s,", $1, $2}\' | sed \'s/,$//\'); '
            'printf "%s\\t%s\\t%s\\n" "$host" "$cpus" "$topo"'
        ),
    ]
    proc = helper.run_command(probe_cmd)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"srun probe failed with exit {proc.returncode}")

    result: Dict[str, Dict[str, Any]] = {}
    for line in proc.stdout.splitlines():
        parts = line.strip().split("\t", 2)
        if len(parts) < 3:
            continue
        host, cpulist, topo_pairs = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if not host or not cpulist or not topo_pairs:
            continue
        cpu_ids = helper.expand_cpu_list(cpulist)
        cpu_to_core = helper.parse_cpu_core_map(topo_pairs.replace(",", "\n").replace(":", ","))
        core_ids = helper.map_cpu_ids_to_core_ids(cpu_ids, cpu_to_core)
        result[host] = {
            "cpulist": cpulist,
            "cpu_ids": cpu_ids,
            "core_ids": core_ids,
        }
    return result


def _parse_key_value_overrides(items: List[str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Expected KEY=VALUE, got: {raw}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Missing key in override: {raw}")
        overrides[key] = value
    return overrides


def _submit_job_for_test(args: argparse.Namespace) -> str:
    scheduler = JobScheduler()
    row = scheduler.submit_job(
        name=str(args.submit_job),
        ranks=int(args.ranks),
        cpus_per_rank=int(args.cpus_per_rank),
        ranks_per_node=int(args.ranks_per_node),
        partition=str(args.partition),
        time_limit=str(args.time_limit),
        exclude=str(args.exclude) if args.exclude else None,
        mpi_iface=str(args.mpi_iface) if args.mpi_iface else None,
        env_overrides=_parse_key_value_overrides(list(args.submit_env or [])),
        script_args=shlex.split(str(args.job_args)) if args.job_args else None,
        dry_run=False,
        auto_load_perf_data=False,
    )
    if str(row.get("status")) != "SUBMITTED":
        raise RuntimeError(f"submission failed: {row}")
    job_id = str(row.get("job_id") or "").strip()
    if not job_id:
        raise RuntimeError(f"submission did not return a valid job_id: {row}")
    print(f"Submitted {args.submit_job} as job {job_id}")
    return job_id


def _wait_for_running(helper: SlurmHelperUtility, job_id: str, timeout_s: float, poll_s: float) -> tuple[str, List[str]]:
    deadline = time.time() + float(timeout_s)
    last_state = ""
    last_hosts: List[str] = []
    terminal_states = {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
    }
    while time.time() < deadline:
        compact_nodelist, state = helper.get_job_nodelist_and_state(job_id)
        hosts = helper.expand_nodelist(compact_nodelist)
        if state:
            last_state = state
            last_hosts = hosts
        if state.upper() == "RUNNING":
            return state, hosts
        if state.upper() in terminal_states:
            raise RuntimeError(f"job {job_id} reached terminal state before RUNNING: {state}")
        time.sleep(float(poll_s))
    raise RuntimeError(
        f"job {job_id} did not reach RUNNING within {timeout_s:.1f}s "
        f"(last_state={last_state or 'unknown'}, hosts={last_hosts})"
    )


def _cancel_job(helper: SlurmHelperUtility, job_id: str) -> None:
    proc = helper.run_command(["scancel", str(job_id)])
    if proc.returncode == 0:
        print(f"Cancelled job {job_id}")
        return
    detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
    print(f"[WARN] Failed to cancel job {job_id}: {detail}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check Slurm CPU allocation maps to expected physical cores.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--job-id", help="Running Slurm job ID to inspect.")
    source.add_argument("--submit-job", help="Submit a job by name and validate its core mapping once running.")
    parser.add_argument("--ranks", type=int, default=2, help="Ranks for --submit-job (default: 2).")
    parser.add_argument("--cpus-per-rank", type=int, default=10, help="CPUs per rank for --submit-job (default: 10).")
    parser.add_argument("--ranks-per-node", type=int, default=2, help="Ranks per node for --submit-job (default: 2).")
    parser.add_argument("--partition", default="debug", help="Slurm partition for --submit-job (default: debug).")
    parser.add_argument("--time-limit", default="00:30:00", help="Slurm time limit for --submit-job.")
    parser.add_argument("--exclude", default=None, help="Optional Slurm --exclude list for --submit-job.")
    parser.add_argument("--mpi-iface", default=None, help="Optional MPI iface for --submit-job (pmi2/pmix).")
    parser.add_argument("--submit-env", action="append", default=[], help="Extra KEY=VALUE env override for --submit-job.")
    parser.add_argument("--job-args", default=None, help="Override script args for --submit-job, quoted as one string.")
    parser.add_argument("--wait-running-s", type=float, default=120.0, help="How long to wait for a submitted job to reach RUNNING.")
    parser.add_argument("--poll-s", type=float, default=2.0, help="Polling interval while waiting for RUNNING.")
    parser.add_argument("--cancel-after", action="store_true", help="Cancel a job submitted via --submit-job after validation.")
    parser.add_argument("--json", action="store_true", help="Print full result as JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    helper = SlurmHelperUtility()
    submitted_job = False
    job_id = str(args.job_id or "").strip()
    if args.submit_job:
        job_id = _submit_job_for_test(args)
        submitted_job = True

    try:
        compact_nodelist, state = helper.get_job_nodelist_and_state(job_id)
        hosts = helper.expand_nodelist(compact_nodelist)
        if args.submit_job:
            state, hosts = _wait_for_running(helper, job_id, args.wait_running_s, args.poll_s)
        elif state.upper() != "RUNNING":
            raise SystemExit(f"Job {job_id} is not RUNNING (state={state or 'unknown'}).")
        if not hosts:
            raise SystemExit(f"No hosts found for job {job_id}. state={state or 'unknown'}")

        _, observed_core_map = helper.collect_job_cores(job_id)
        expected = _probe_job_topology(helper, job_id, len(hosts))

        mismatches: List[Dict[str, Any]] = []
        for host in hosts:
            expected_entry = expected.get(host, {})
            expected_cores = list(expected_entry.get("core_ids", []))
            observed_cores = list(observed_core_map.get(host, []))
            if expected_cores != observed_cores:
                mismatches.append(
                    {
                        "host": host,
                        "expected_cores": expected_cores,
                        "observed_cores": observed_cores,
                        "cpulist": expected_entry.get("cpulist", ""),
                    }
                )

        payload = {
            "job_id": job_id,
            "state": state,
            "nodes": hosts,
            "observed_cores_by_node": observed_core_map,
            "expected": expected,
            "mismatches": mismatches,
            "ok": not mismatches,
            "submitted_by_test": submitted_job,
        }

        print(f"Job {job_id} state={state} nodes={','.join(hosts)}")
        for host in hosts:
            entry = expected.get(host, {})
            print(
                f"{host}: cpus={entry.get('cpulist', '')} "
                f"expected_cores={entry.get('core_ids', [])} "
                f"observed_cores={observed_core_map.get(host, [])}"
            )

        if args.json:
            print("\nJSON:")
            print(json.dumps(payload, indent=2, sort_keys=True))

        if mismatches:
            print("\nFAIL: observed core mapping does not match CPU-to-core translation.")
            return 1

        print("\nPASS: observed core mapping matches CPU-to-core translation.")
        return 0
    finally:
        if submitted_job and args.cancel_after and job_id:
            _cancel_job(helper, job_id)


if __name__ == "__main__":
    raise SystemExit(main())
