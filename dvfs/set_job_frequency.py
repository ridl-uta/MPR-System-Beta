#!/usr/bin/env python3
"""Apply a fixed CPU frequency to the cores of a running Slurm job via GEOPM.

This tool reuses job_cores.py to discover which logical CPUs belong to each
allocated node, and then invokes geopmwrite to push a frequency control value
onto every CPU in that allocation.

Example:
  ./set_job_frequency.py <jobid> 2.2GHz --dry-run
  ./set_job_frequency.py <jobid> 2100 --unit MHz
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from typing import List, Tuple

import job_cores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set GEOPM frequency controls for all CPUs held by a Slurm job",
    )
    parser.add_argument("jobid", help="Slurm job ID to modify")
    parser.add_argument(
        "frequency",
        help="Target CPU frequency (accepts values like 2100, 2.2GHz, 1.8e9Hz)",
    )
    parser.add_argument(
        "--unit",
        choices=["Hz", "KHz", "MHz", "GHz"],
        default="MHz",
        help="Unit to assume when no suffix is present (default: MHz)",
    )
    parser.add_argument(
        "--geopmwrite-cmd",
        default="geopmwrite",
        help="Command used to write GEOPM signals (default: geopmwrite)",
    )
    parser.add_argument(
        "--signal",
        default="MSR::PERF_CTL:FREQ",
        help="GEOPM signal name to write (default: MSR::PERF_CTL:FREQ)",
    )
    parser.add_argument(
        "--domain",
        default="core",
        help="GEOPM domain name to target (default: core)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without running geopmwrite",
    )
    return parser.parse_args()


def require_command(cmd: str) -> None:
    if shutil.which(cmd) is None:
        print(f"[ERR] required command not found in PATH: {cmd}", file=sys.stderr)
        sys.exit(127)


def _format_frequency_token(hz: float) -> str:
    """Return a human-friendly exponential representation (e.g. 1.2e9)."""
    token = f"{hz:.6e}"
    mantissa, exp = token.split("e")
    # Trim trailing zeros in mantissa for cleaner output
    mantissa = mantissa.rstrip("0").rstrip(".")
    sign = ""
    if exp.startswith(('+', '-')):
        sign = exp[0] if exp[0] == '-' else ''
        exp = exp[1:]
    exp = exp.lstrip('0') or '0'
    return f"{mantissa}e{sign}{exp}"


def parse_frequency(value: str, default_unit: str) -> Tuple[float, str]:
    text = value.strip()
    if not text:
        raise ValueError("empty frequency")

    units = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
    unit = default_unit.lower()

    lower = text.lower()
    for suffix in ("ghz", "mhz", "khz", "hz"):
        if lower.endswith(suffix):
            unit = suffix
            text = text[: -len(suffix)]
            break

    try:
        magnitude = float(text)
    except ValueError as exc:
        raise ValueError(f"cannot parse frequency magnitude from '{value}'") from exc

    if unit not in units:
        raise ValueError(f"unsupported unit '{unit}'")

    hz = magnitude * units[unit]
    if hz <= 0:
        raise ValueError("frequency must be positive")

    hz = float(hz)
    token = _format_frequency_token(hz)
    return hz, token


def format_remote_loop(
    cores: List[int],
    geopm_cmd: str,
    signal: str,
    domain: str,
    value_token: str,
) -> str:
    core_list = ' '.join(str(c) for c in cores)
    signal_q = shlex.quote(signal)
    domain_q = shlex.quote(domain)
    value_q = shlex.quote(value_token)
    loop = (
        "set -euo pipefail; "
        f"for core in {core_list}; do "
        f"{geopm_cmd} {signal_q} {domain_q} $core {value_q}; "
        "done"
    )
    return loop


def main() -> None:
    args = parse_args()

    try:
        freq_hz, freq_token = parse_frequency(args.frequency, args.unit)
    except ValueError as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        sys.exit(1)

    srun_cmd = shutil.which("srun")
    if srun_cmd is None:
        print("[ERR] srun command not found in PATH", file=sys.stderr)
        sys.exit(127)

    geopm_tokens = shlex.split(args.geopmwrite_cmd)
    if not geopm_tokens:
        print("[ERR] invalid geopmwrite command", file=sys.stderr)
        sys.exit(1)
    require_command(geopm_tokens[0])
    geopm_cmd = ' '.join(shlex.quote(tok) for tok in geopm_tokens)

    try:
        state, host_map = job_cores.collect_host_cores(args.jobid)
    except job_cores.JobCoresError as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        sys.exit(exc.exit_code)

    if state and state.upper() != "RUNNING":
        print(f"[WARN] job state: {state}")

    overall_ok = True
    for host, cores in host_map.items():
        print(
            f"[INFO] Setting {len(cores)} cores on {host} to {freq_hz/1e6:.3f} MHz "
            f"using {args.signal}/{args.domain}"
        )
        remote = format_remote_loop(cores, geopm_cmd, args.signal, args.domain, freq_token)
        cmd = [
            srun_cmd,
            f"--jobid={args.jobid}",
            "-w",
            host,
            "-N",
            "1",
            "-n",
            "1",
            "--ntasks-per-node=1",
            "--overlap",
            "--immediate=10",
            "bash",
            "-lc",
            remote,
        ]

        if args.dry_run:
            print("[DRY-RUN]", ' '.join(shlex.quote(part) for part in cmd))
            continue

        result = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        if result.returncode != 0:
            overall_ok = False
            print(
                f"[ERR] srun failed for {host} (exit {result.returncode})",
                file=sys.stderr,
            )
            if "immediate" in result.stderr.lower():
                print(
                    "[HINT] Slurm could not start the helper step immediately. "
                    "The job may be fully busy or overlap is disabled.",
                    file=sys.stderr,
                )
            elif "overlap" in result.stderr.lower():
                print(
                    "[HINT] Slurm rejected --overlap; your site may disable overlapping steps.",
                    file=sys.stderr,
                )
            else:
                print(
                    "[HINT] Check geopmwrite permissions, job state, and Slurm logs for details.",
                    file=sys.stderr,
                )

    if not args.dry_run:
        if overall_ok:
            print("[DONE] Applied GEOPM frequency controls to all nodes")
        else:
            sys.exit(4)


if __name__ == "__main__":
    main()
