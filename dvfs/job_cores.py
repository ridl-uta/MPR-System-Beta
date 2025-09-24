#!/usr/bin/env python3
"""Print CPU cores per node for a Slurm job.

This is a Python port of the helper originally implemented as job_cores.sh.

Usage examples:
  ./job_cores.py <jobid>
  ./job_cores.py <jobid> --bash
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from typing import Dict, List


class JobCoresError(Exception):
    """Raised when job_cores cannot discover CPU bindings."""

    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print exact CPU cores per node for a Slurm job",
    )
    parser.add_argument("jobid", help="Slurm job ID to inspect")
    parser.add_argument(
        "--bash",
        action="store_true",
        help="Emit Bash arrays (CORES_<host>=(...)) instead of key/value output",
    )
    return parser.parse_args()


def run_command(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    """Run command and return CompletedProcess without raising on failure."""
    try:
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print(f"[ERR] command not found: {cmd[0]}", file=sys.stderr)
        sys.exit(127)


def expand_list(cpulist: str) -> str:
    """Expand compact CPU list (e.g. 0-3,8,10-11) into space-separated numbers."""
    numbers: List[int] = []
    for chunk in cpulist.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            start_str, end_str = chunk.split('-', 1)
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError:
                continue
            step = 1 if end >= start else -1
            numbers.extend(range(start, end + step, step))
        else:
            try:
                numbers.append(int(chunk))
            except ValueError:
                continue
    return ' '.join(str(n) for n in numbers)


def sanitize_host(host: str) -> str:
    """Sanitize hostname for use in Bash variable names."""
    return re.sub(r"[^A-Za-z0-9_]", "_", host)


def extract_nodelist(jobid: str) -> tuple[str, str]:
    """Return (compact_nodelist, job_state)."""
    squeue_nodes = run_command(["squeue", "-h", "-j", jobid, "-o", "%N"])
    compact = squeue_nodes.stdout.strip()

    squeue_state = run_command(["squeue", "-h", "-j", jobid, "-o", "%T"])
    state = squeue_state.stdout.strip()

    if not compact:
        job_info = run_command(["scontrol", "show", "job", "-o", jobid])
        match = re.search(r"\bNodeList=([^ ]+)", job_info.stdout)
        if match:
            candidate = match.group(1).strip()
            if candidate and candidate != "(null)":
                compact = candidate
        if not state:
            match_state = re.search(r"\bJobState=([^ ]+)", job_info.stdout)
            if match_state:
                state = match_state.group(1).strip()

    return compact, state


def collect_host_cores(jobid: str) -> tuple[str, Dict[str, List[int]]]:
    """Return (state, cores_per_host) for the provided job ID."""
    compact_nodelist, state = extract_nodelist(jobid)

    if not compact_nodelist:
        raise JobCoresError("no NodeList (job pending/not found)", exit_code=2)

    hostnames = run_command(["scontrol", "show", "hostnames", compact_nodelist])
    hosts = [line.strip() for line in hostnames.stdout.splitlines() if line.strip()]
    if not hosts:
        raise JobCoresError("no NodeList (job pending/not found)", exit_code=2)

    node_count = len(hosts)

    probe_cmd = [
        "srun",
        f"--jobid={jobid}",
        "-N",
        str(node_count),
        "--ntasks-per-node=1",
        "--overlap",
        "--immediate=10",
        "bash",
        "-lc",
        'printf "%s " "$(hostname -s)"; awk \'/^Cpus_allowed_list:/ {print $2; exit}\' /proc/self/status',
    ]

    probe = run_command(probe_cmd)
    output = probe.stdout.strip()

    if not output:
        messages = []
        stderr_text = probe.stderr.strip()
        if stderr_text:
            messages.append(stderr_text)
        messages.append("no output from srun probe (job fully busy or overlap disabled)")
        raise JobCoresError('\n'.join(messages), exit_code=3)

    host_map: Dict[str, List[int]] = {}
    for line in output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        host, cpulist = parts
        expanded = expand_list(cpulist)
        if not expanded:
            continue
        try:
            host_map[host] = [int(token) for token in expanded.split()]
        except ValueError:
            continue

    if not host_map:
        raise JobCoresError("no cores discovered in probe output", exit_code=3)

    return state, host_map


def main() -> None:
    args = parse_args()
    jobid = args.jobid

    try:
        state, host_map = collect_host_cores(jobid)
    except JobCoresError as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        sys.exit(exc.exit_code)

    if state and state.upper() != "RUNNING":
        print(f"[WARN] job state: {state}")

    for host, cores in host_map.items():
        expanded = ' '.join(str(core) for core in cores)
        if args.bash:
            var = sanitize_host(host)
            print(f"CORES_{var}=({expanded})")
        else:
            print(f"{host}: {expanded}")


if __name__ == "__main__":
    main()
