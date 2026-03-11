#!/usr/bin/env python3
"""Validate Slurm CPU allocation to physical core mapping for a running job.

Examples:
    python3 -m dvfs.test_slurm_core_mapping --job-id 3173
    python3 -m dvfs.test_slurm_core_mapping --job-id 3173 --json
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check Slurm CPU allocation maps to expected physical cores.")
    parser.add_argument("--job-id", required=True, help="Running Slurm job ID to inspect.")
    parser.add_argument("--json", action="store_true", help="Print full result as JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    helper = SlurmHelperUtility()

    compact_nodelist, state = helper.get_job_nodelist_and_state(args.job_id)
    hosts = helper.expand_nodelist(compact_nodelist)
    if not hosts:
        raise SystemExit(f"No hosts found for job {args.job_id}. state={state or 'unknown'}")

    _, observed_core_map = helper.collect_job_cores(args.job_id)
    expected = _probe_job_topology(helper, args.job_id, len(hosts))

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
        "job_id": str(args.job_id),
        "state": state,
        "nodes": hosts,
        "observed_cores_by_node": observed_core_map,
        "expected": expected,
        "mismatches": mismatches,
        "ok": not mismatches,
    }

    print(f"Job {args.job_id} state={state} nodes={','.join(hosts)}")
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


if __name__ == "__main__":
    raise SystemExit(main())
