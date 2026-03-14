#!/usr/bin/env python3
"""Apply DVFS to a specific node from the control node using DVFSController.

This helper writes the per-node config via the dvfs module, triggers the
existing SSH/systemd apply path, and can optionally run the node-side verifier
over SSH.

Examples:
    python3 tests/set_dvfs_from_control_node.py \
      --node ridlserver04 --cores 2,6 --frequency-mhz 1600 --ssh-user ridl

    python3 tests/set_dvfs_from_control_node.py \
      --node ridlserver04 --cores 2,6 --frequency-mhz 1600 \
      --ssh-user ridl \
      --remote-check-path /shared/temp/MPR-System-Beta
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dvfs import DVFSController


def parse_core_list(text: str) -> List[int]:
    cores: set[int] = set()
    for token in str(text).split(","):
        part = token.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"Invalid core range '{part}': end < start.")
            cores.update(range(start, end + 1))
            continue
        cores.add(int(part))
    if not cores:
        raise ValueError("No valid cores parsed from --cores.")
    return sorted(cores)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply DVFS to one node from the control node using DVFSController."
    )
    parser.add_argument("--node", required=True, help="Target node hostname, e.g. ridlserver04.")
    parser.add_argument(
        "--cores",
        required=True,
        help="Target cores/ranges, e.g. 2,6 or 0-3.",
    )
    parser.add_argument(
        "--frequency-mhz",
        type=float,
        required=True,
        help="Target frequency in MHz.",
    )
    parser.add_argument(
        "--control-kind",
        choices=("PERF_CTL", "CORE_MAX", "AUTO"),
        default="PERF_CTL",
        help="DVFS control path to write (default: PERF_CTL).",
    )
    parser.add_argument(
        "--ssh-user",
        default=None,
        help="SSH username for remote apply.",
    )
    parser.add_argument(
        "--ssh-identity",
        default=None,
        help="Optional SSH identity file.",
    )
    parser.add_argument(
        "--conf-dir",
        default=None,
        help="Optional config directory. Default uses /shared/geopm/freq.",
    )
    parser.add_argument(
        "--verify-tol-mhz",
        type=float,
        default=25.0,
        help="Controller readback tolerance in MHz (default: 25).",
    )
    parser.add_argument(
        "--cpufreq-sync",
        action="store_true",
        help="Also align cpufreq policy on the matched node cores.",
    )
    parser.add_argument(
        "--cpufreq-governor",
        default="userspace",
        help="Governor to use when --cpufreq-sync is enabled (default: userspace).",
    )
    parser.add_argument(
        "--cpufreq-min-khz",
        type=int,
        default=1_000_000,
        help="Min cpufreq floor in kHz when --cpufreq-sync is enabled (default: 1000000).",
    )
    parser.add_argument(
        "--remote-check-path",
        default=None,
        help=(
            "Optional repo path on the compute node. When provided, run "
            "tests/check_geopm_apply_state.py over SSH after apply."
        ),
    )
    parser.add_argument(
        "--remote-samples",
        type=int,
        default=5,
        help="Samples for the remote node-side verifier (default: 5).",
    )
    parser.add_argument(
        "--remote-interval-s",
        type=float,
        default=1.0,
        help="Interval for the remote node-side verifier (default: 1.0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not apply; preview the controller command only.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the first DVFS result row as JSON.",
    )
    return parser


def run_remote_check(
    *,
    node: str,
    ssh_user: str | None,
    ssh_identity: str | None,
    remote_check_path: str,
    cores_text: str,
    frequency_mhz: float,
    control_kind: str,
    cpufreq_governor: str,
    cpufreq_min_khz: int,
    samples: int,
    interval_s: float,
) -> int:
    target = f"{ssh_user}@{node}" if ssh_user else node
    remote_cmd = [
        "cd",
        remote_check_path,
        "&&",
        "sudo",
        "python3",
        "tests/check_geopm_apply_state.py",
        "--cores",
        cores_text,
        "--freq-hz",
        f"{frequency_mhz * 1e6:.0f}",
        "--control-kind",
        control_kind,
        "--samples",
        str(samples),
        "--interval-s",
        str(interval_s),
        "--expect-governor",
        cpufreq_governor,
        "--expect-min-khz",
        str(cpufreq_min_khz),
    ]

    ssh_cmd: list[str] = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]
    if ssh_identity:
        ssh_cmd.extend(["-i", ssh_identity])
    ssh_cmd.extend([target, " ".join(shlex.quote(arg) for arg in remote_cmd)])

    proc = subprocess.run(ssh_cmd, text=True, capture_output=True, check=False)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    return int(proc.returncode)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.frequency_mhz <= 0:
        parser.error("--frequency-mhz must be > 0.")
    if args.verify_tol_mhz < 0:
        parser.error("--verify-tol-mhz must be >= 0.")
    if args.cpufreq_min_khz <= 0:
        parser.error("--cpufreq-min-khz must be > 0.")
    if args.remote_samples <= 0:
        parser.error("--remote-samples must be > 0.")
    if args.remote_interval_s <= 0:
        parser.error("--remote-interval-s must be > 0.")

    try:
        cores = parse_core_list(args.cores)
    except ValueError as exc:
        parser.error(str(exc))

    controller_kwargs = {
        "control_kind": args.control_kind,
        "ssh_user": args.ssh_user,
        "ssh_identity": args.ssh_identity,
        "verify_tolerance_mhz": float(args.verify_tol_mhz),
        "cpufreq_sync": bool(args.cpufreq_sync),
        "cpufreq_governor": args.cpufreq_governor,
        "cpufreq_min_khz": int(args.cpufreq_min_khz),
    }
    if args.conf_dir:
        controller_kwargs["conf_dir"] = args.conf_dir

    controller = DVFSController(**controller_kwargs)
    rows = controller.apply_to_cores(
        node_name=str(args.node),
        core_numbers=cores,
        frequency_mhz=float(args.frequency_mhz),
        dry_run=bool(args.dry_run),
    )
    if not rows:
        print("No DVFS rows returned.")
        return 1

    first = rows[0]
    print(
        "DVFS control-node apply:",
        f"node={args.node}",
        f"cores={cores}",
        f"requested_mhz={float(args.frequency_mhz):.3f}",
        f"control_kind={args.control_kind}",
        f"cpufreq_sync={args.cpufreq_sync}",
    )
    print(
        "Result:",
        f"status={first.get('status')}",
        f"verify_status={first.get('verify_status')}",
        f"verify_reason={first.get('verify_reason')}",
        f"readback_avg_mhz={first.get('readback_avg_mhz')}",
        f"readback_min_mhz={first.get('readback_min_mhz')}",
        f"readback_max_mhz={first.get('readback_max_mhz')}",
        f"abs_error_mhz={first.get('readback_abs_error_mhz')}",
        f"tol_mhz={first.get('verify_tolerance_mhz')}",
    )
    print("Command:", first.get("command"))

    if args.json:
        print("\nFirst row JSON:")
        print(json.dumps(first, indent=2, default=str))

    if args.remote_check_path and not args.dry_run:
        rc = run_remote_check(
            node=str(args.node),
            ssh_user=args.ssh_user,
            ssh_identity=args.ssh_identity,
            remote_check_path=str(args.remote_check_path),
            cores_text=args.cores,
            frequency_mhz=float(args.frequency_mhz),
            control_kind=args.control_kind,
            cpufreq_governor=args.cpufreq_governor,
            cpufreq_min_khz=int(args.cpufreq_min_khz),
            samples=int(args.remote_samples),
            interval_s=float(args.remote_interval_s),
        )
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
