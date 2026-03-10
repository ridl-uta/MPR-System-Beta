#!/usr/bin/env python3
"""Simple DVFS apply/readback test helper.

Examples:
    python3 -m dvfs.test_apply_readback --frequency-mhz 1800 --node localhost --cores 0-3
    python3 -m dvfs.test_apply_readback --frequency-mhz 1800 --node ridlserver02 --cores 0,1,2,3 --ssh-user ridl
    python3 -m dvfs.test_apply_readback --frequency-mhz 1800 --node localhost --cores 0-3 --dry-run
"""

from __future__ import annotations

import argparse
import json
from typing import List

from dvfs import DVFSController


def parse_core_list(text: str) -> List[int]:
    """Parse comma-separated cores and ranges, e.g. 0,1,4-7."""
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
            for core in range(start, end + 1):
                if core < 0:
                    raise ValueError(f"Core index must be >= 0, got {core}.")
                cores.add(core)
            continue
        core = int(part)
        if core < 0:
            raise ValueError(f"Core index must be >= 0, got {core}.")
        cores.add(core)
    if not cores:
        raise ValueError("No valid cores parsed from --cores.")
    return sorted(cores)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test DVFS apply + readback verification.")
    parser.add_argument(
        "--frequency-mhz",
        type=float,
        required=True,
        help="Target frequency in MHz to apply.",
    )
    parser.add_argument(
        "--node",
        default="localhost",
        help="Node hostname for DVFS apply (default: localhost).",
    )
    parser.add_argument(
        "--cores",
        default="0",
        help="Core list/ranges (default: 0), e.g. 0,1,2 or 0-7.",
    )
    parser.add_argument(
        "--ssh-user",
        default=None,
        help="Optional SSH username for remote apply.",
    )
    parser.add_argument(
        "--conf-dir",
        default=None,
        help=(
            "Optional controller config directory. "
            "Use this when /shared/geopm/freq is unavailable in your environment."
        ),
    )
    parser.add_argument(
        "--verify-tol-mhz",
        type=float,
        default=25.0,
        help="PASS/FAIL tolerance for readback average error (default: 25).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not apply frequency; preview command only.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full first-row result as JSON.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.frequency_mhz <= 0:
        parser.error("--frequency-mhz must be > 0.")
    if args.verify_tol_mhz < 0:
        parser.error("--verify-tol-mhz must be >= 0.")

    try:
        cores = parse_core_list(args.cores)
    except ValueError as exc:
        parser.error(str(exc))

    allocation = {"cores_by_node": {str(args.node): cores}}
    controller_kwargs = {
        "ssh_user": args.ssh_user,
        "verify_tolerance_mhz": float(args.verify_tol_mhz),
    }
    if args.conf_dir:
        controller_kwargs["conf_dir"] = args.conf_dir

    controller = DVFSController(**controller_kwargs)
    rows = controller.apply_to_job_allocation(
        allocation=allocation,
        frequency_mhz=float(args.frequency_mhz),
        dry_run=bool(args.dry_run),
    )
    if not rows:
        print("No DVFS rows returned.")
        return 1

    first = rows[0]
    print(
        "DVFS test:",
        f"node={args.node}",
        f"cores={cores}",
        f"requested_mhz={float(args.frequency_mhz):.3f}",
        f"applied_mhz={float(first['frequency_mhz']):.3f}",
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
