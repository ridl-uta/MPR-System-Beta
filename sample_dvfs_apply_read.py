#!/usr/bin/env python3
"""Apply a DVFS frequency sweep to a Slurm job and read it back.

This utility reuses the same DVFS path as main_controller:
  managers.dvfs_manager.DVFSManager -> dvfs.apply_reduction -> geopm_apply.sh

If you do not have a running job, this utility can submit a basic sleep job,
wait for RUNNING, apply/read the sweep, then cancel it automatically.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

from dvfs import collect_host_cores

try:
    from managers.dvfs_manager import DVFSManager
except Exception:
    dvfs_mgr_path = Path(__file__).resolve().parent / "managers" / "dvfs_manager.py"
    spec = importlib.util.spec_from_file_location("sample_dvfs_manager", dvfs_mgr_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load DVFS manager module: {dvfs_mgr_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    DVFSManager = module.DVFSManager


def _target_to_reduction(
    target_mhz: float,
    *,
    max_freq_mhz: float,
    min_freq_mhz: float,
) -> tuple[float, float]:
    """Convert a target MHz value into a normalized reduction fraction."""
    bounded = min(max(target_mhz, min_freq_mhz), max_freq_mhz)
    reduction = max(0.0, min(1.0, 1.0 - (bounded / max_freq_mhz)))
    return bounded, reduction


def _is_local_host(host: str) -> bool:
    local = {
        "localhost",
        "127.0.0.1",
        socket.gethostname(),
        socket.getfqdn(),
        socket.gethostname().split(".", 1)[0],
    }
    return host in local


def _read_geopm_control_for_host(
    host: str,
    cores: Iterable[int],
    *,
    signal: str,
    domain: str,
    ssh_user: str | None,
) -> subprocess.CompletedProcess[str]:
    core_list = " ".join(str(c) for c in sorted(set(cores)))
    read_cmd = (
        f"for c in {core_list}; do "
        f"printf 'core %s: ' \"$c\"; "
        f"geopmread {shlex.quote(signal)} {shlex.quote(domain)} \"$c\" "
        "2>/dev/null || echo read-failed; "
        "done"
    )
    if _is_local_host(host):
        return subprocess.run(
            ["bash", "-lc", read_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    target = f"{ssh_user}@{host}" if ssh_user else host
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", target, "bash", "-lc", read_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def _print_readback(host_map: Dict[str, List[int]], *, signal: str, domain: str, ssh_user: str | None) -> int:
    rc = 0
    for host, cores in host_map.items():
        print(f"\n[{host}] reading {signal}/{domain} for {len(cores)} core(s)")
        result = _read_geopm_control_for_host(
            host,
            cores,
            signal=signal,
            domain=domain,
            ssh_user=ssh_user,
        )
        if result.stdout:
            print(result.stdout.rstrip())
        if result.returncode != 0:
            rc = result.returncode
            print(f"[WARN] readback failed on {host}: exit={result.returncode}", file=sys.stderr)
            if result.stderr:
                print(result.stderr.rstrip(), file=sys.stderr)
    return rc


def _parse_job_id(text: str) -> str | None:
    output = (text or "").strip()
    if not output:
        return None
    if output.isdigit():
        return output
    if ";" in output:
        first = output.split(";", 1)[0].strip()
        if first.isdigit():
            return first
    match = re.search(r"Submitted batch job\s+(\d+)", output)
    if match:
        return match.group(1)
    return None


def _submit_basic_job(args: argparse.Namespace) -> str:
    wrap_cmd = (
        "echo '[sample-dvfs] basic job started on $(hostname)'; "
        f"sleep {int(args.basic_sleep_seconds)}"
    )
    cmd = [
        "sbatch",
        "--parsable",
        "-p",
        args.basic_partition,
        "-N",
        str(args.basic_nodes),
        "-n",
        str(args.basic_ranks),
        "-c",
        str(args.basic_cpus_per_task),
        "-t",
        args.basic_time_limit,
        "--job-name",
        args.basic_job_name,
        "--wrap",
        wrap_cmd,
    ]
    if args.basic_nodelist:
        cmd += ["--nodelist", args.basic_nodelist]
    if args.basic_exclude:
        cmd += ["--exclude", args.basic_exclude]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch failed (exit {result.returncode}): "
            f"{(result.stdout or '').strip()} {(result.stderr or '').strip()}"
        )
    job_id = _parse_job_id(result.stdout) or _parse_job_id(result.stderr)
    if not job_id:
        raise RuntimeError(
            f"Unable to parse job id from sbatch output: "
            f"{(result.stdout or '').strip()} {(result.stderr or '').strip()}"
        )
    return job_id


def _wait_for_running(job_id: str, timeout_s: int) -> str | None:
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}
    deadline = time.time() + max(1, int(timeout_s))
    last_state: str | None = None
    while time.time() < deadline:
        proc = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            state = proc.stdout.strip() or None
            if state:
                last_state = state
                if state == "RUNNING":
                    return state
                if state in terminal:
                    return state
        time.sleep(2.0)
    return last_state


def _cancel_job(job_id: str) -> None:
    proc = subprocess.run(
        ["scancel", job_id],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        print(f"[INFO] canceled submitted job {job_id}")
        return
    details = (proc.stderr or proc.stdout or "").strip()
    print(f"[WARN] failed to cancel submitted job {job_id}: {details}", file=sys.stderr)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Apply a max->min DVFS sweep for a Slurm job and read back GEOPM controls."
    )
    p.add_argument("job_id", nargs="?", help="Slurm job id currently running")
    p.add_argument(
        "--max-freq-mhz",
        type=float,
        default=2400.0,
        help="Maximum frequency bound in MHz (default: 2400)",
    )
    p.add_argument(
        "--min-freq-mhz",
        type=float,
        default=1000.0,
        help="Minimum frequency bound in MHz (default: 1000)",
    )
    p.add_argument(
        "--interval-mhz",
        type=int,
        default=200,
        help="Step size for the sweep in MHz (default: 200)",
    )
    p.add_argument(
        "--conf-dir",
        default="/shared/geopm/freq",
        help="Directory for per-host GEOPM config files",
    )
    p.add_argument(
        "--nodes-file",
        type=Path,
        default=Path("data/nodes.txt"),
        help="Nodes file passed to run_geopm_apply_ssh.sh",
    )
    p.add_argument(
        "--read-signal",
        default="CPU_FREQUENCY_MAX_CONTROL",
        help="GEOPM signal/control to read back",
    )
    p.add_argument(
        "--read-domain",
        default="core",
        help="GEOPM domain to read back (default: core)",
    )
    p.add_argument(
        "--ssh-user",
        default=None,
        help="Optional SSH user for remote readback (default: current user)",
    )
    p.add_argument(
        "--wait-seconds",
        type=float,
        default=2.0,
        help="Seconds to wait after apply before readback",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not apply changes; only show what would be done",
    )
    p.add_argument(
        "--submit-basic-job",
        action="store_true",
        help="Submit a basic sleep job, wait for RUNNING, then apply/read the sweep",
    )
    p.add_argument(
        "--basic-job-name",
        default="sample-dvfs-basic",
        help="Job name for --submit-basic-job",
    )
    p.add_argument(
        "--basic-partition",
        default="debug",
        help="Partition for --submit-basic-job",
    )
    p.add_argument(
        "--basic-time-limit",
        default="00:20:00",
        help="Time limit for --submit-basic-job",
    )
    p.add_argument(
        "--basic-nodes",
        type=int,
        default=1,
        help="Node count for --submit-basic-job",
    )
    p.add_argument(
        "--basic-ranks",
        type=int,
        default=1,
        help="MPI ranks for --submit-basic-job",
    )
    p.add_argument(
        "--basic-cpus-per-task",
        type=int,
        default=10,
        help="CPUs per task for --submit-basic-job",
    )
    p.add_argument(
        "--basic-sleep-seconds",
        type=int,
        default=1200,
        help="Sleep duration inside --submit-basic-job",
    )
    p.add_argument(
        "--basic-nodelist",
        default=None,
        help="Optional --nodelist for --submit-basic-job",
    )
    p.add_argument(
        "--basic-exclude",
        default=None,
        help="Optional --exclude for --submit-basic-job",
    )
    p.add_argument(
        "--wait-running-timeout",
        type=int,
        default=180,
        help="Seconds to wait for job to reach RUNNING before aborting",
    )
    p.add_argument(
        "--keep-submitted-job",
        action="store_true",
        help="Do not cancel auto-submitted job after sweep",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.max_freq_mhz <= 0 or args.min_freq_mhz <= 0:
        print("[ERR] max/min frequencies must be positive", file=sys.stderr)
        return 2
    if args.min_freq_mhz > args.max_freq_mhz:
        print("[ERR] min frequency cannot be greater than max frequency", file=sys.stderr)
        return 2
    if args.interval_mhz <= 0:
        print("[ERR] interval must be positive", file=sys.stderr)
        return 2
    if args.submit_basic_job and args.job_id:
        print(
            "[ERR] Provide either <job_id> or --submit-basic-job, not both",
            file=sys.stderr,
        )
        return 2
    if not args.submit_basic_job and not args.job_id:
        print(
            "[ERR] Missing <job_id>. Provide a running job id or use --submit-basic-job",
            file=sys.stderr,
        )
        return 2

    submitted_job_id: str | None = None
    job_id = args.job_id
    if args.submit_basic_job:
        try:
            submitted_job_id = _submit_basic_job(args)
        except Exception as exc:
            print(f"[ERR] Failed to submit basic job: {exc}", file=sys.stderr)
            return 3
        job_id = submitted_job_id
        print(f"[INFO] submitted basic job: {job_id}")

    assert job_id is not None
    wait_state = _wait_for_running(job_id, args.wait_running_timeout)
    if wait_state != "RUNNING":
        print(
            f"[ERR] Job {job_id} did not reach RUNNING within {args.wait_running_timeout}s "
            f"(last_state={wait_state or 'UNKNOWN'})",
            file=sys.stderr,
        )
        if submitted_job_id and not args.keep_submitted_job:
            _cancel_job(submitted_job_id)
        return 3

    try:
        state, host_map = collect_host_cores(job_id)
    except Exception as exc:
        print(f"[ERR] Could not discover job cores for {job_id}: {exc}", file=sys.stderr)
        if submitted_job_id and not args.keep_submitted_job:
            _cancel_job(submitted_job_id)
        return 3

    print(f"[INFO] job={job_id} state={state or 'UNKNOWN'} hosts={len(host_map)}")
    print(
        f"[INFO] sweep max={args.max_freq_mhz:.3f}MHz min={args.min_freq_mhz:.3f}MHz "
        f"interval={args.interval_mhz}MHz"
    )

    mgr = DVFSManager(
        max_freq_mhz=args.max_freq_mhz,
        min_freq_mhz=args.min_freq_mhz,
        conf_dir=args.conf_dir,
        nodes_file=args.nodes_file,
    )

    max_mhz = int(round(args.max_freq_mhz))
    min_mhz = int(round(args.min_freq_mhz))
    targets: List[int] = []
    current = max_mhz
    while current >= min_mhz:
        targets.append(current)
        current -= args.interval_mhz
    if not targets or targets[-1] != min_mhz:
        targets.append(min_mhz)

    overall_rc = 0
    try:
        for target in targets:
            bounded_target, reduction = _target_to_reduction(
                float(target),
                max_freq_mhz=args.max_freq_mhz,
                min_freq_mhz=args.min_freq_mhz,
            )
            print(
                f"\n[STEP] target={target}MHz bounded={bounded_target:.3f}MHz "
                f"reduction={reduction:.6f}"
            )
            try:
                mgr.submit_reduction(job_id, reduction, dry_run=args.dry_run)
            except Exception as exc:
                print(f"[ERR] DVFS apply failed at {target}MHz: {exc}", file=sys.stderr)
                return 4

            if args.dry_run:
                continue

            if args.wait_seconds > 0:
                time.sleep(args.wait_seconds)
            rc = _print_readback(
                host_map,
                signal=args.read_signal,
                domain=args.read_domain,
                ssh_user=args.ssh_user,
            )
            if rc != 0:
                overall_rc = rc
    finally:
        if submitted_job_id and not args.keep_submitted_job and not args.dry_run:
            _cancel_job(submitted_job_id)

    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
