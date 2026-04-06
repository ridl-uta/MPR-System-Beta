#!/usr/bin/env python3

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


@dataclass
class NodeResult:
    node: str
    returncode: int
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run dvfs/test_geopm_apply.sh across multiple nodes in parallel and "
            "store per-node logs."
        )
    )
    parser.add_argument(
        "--nodes",
        nargs="+",
        required=True,
        help="Target hostnames to test.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum concurrent node tests (default: 4).",
    )
    parser.add_argument(
        "--output-dir",
        default="output/geopm_apply_tests",
        help="Directory for per-node logs (default: output/geopm_apply_tests).",
    )
    parser.add_argument(
        "--script-path",
        default=None,
        help="Path to test_geopm_apply.sh (default: dvfs/test_geopm_apply.sh).",
    )
    parser.add_argument(
        "--ssh-user",
        default=None,
        help="Optional SSH username passed to test_geopm_apply.sh.",
    )
    parser.add_argument(
        "--remote-repo-dir",
        default=None,
        help="Optional remote repo dir passed to test_geopm_apply.sh.",
    )
    parser.add_argument(
        "--shared-conf-dir",
        default=None,
        help="Optional shared config dir passed to test_geopm_apply.sh.",
    )
    parser.add_argument(
        "--cores",
        default=None,
        help='Optional core list passed to test_geopm_apply.sh, for example "0 1".',
    )
    parser.add_argument(
        "--target-mhz",
        type=float,
        default=None,
        help="Optional test frequency passed to test_geopm_apply.sh.",
    )
    parser.add_argument(
        "--restore-mhz",
        type=float,
        default=None,
        help="Optional restore frequency passed to test_geopm_apply.sh.",
    )
    parser.add_argument(
        "--control-kind",
        choices=("PERF_CTL", "CORE_MAX"),
        default=None,
        help="Optional control kind passed to test_geopm_apply.sh.",
    )
    cpufreq_group = parser.add_mutually_exclusive_group()
    cpufreq_group.add_argument(
        "--cpufreq-sync",
        dest="cpufreq_sync",
        action="store_true",
        help="Enable CPUFreq policy sync in the node tests.",
    )
    cpufreq_group.add_argument(
        "--no-cpufreq-sync",
        dest="cpufreq_sync",
        action="store_false",
        help="Disable CPUFreq policy sync in the node tests.",
    )
    parser.set_defaults(cpufreq_sync=None)
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip dvfs/install_geopm_apply_systemd.sh in each node test.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep per-node temporary files created by the node test.",
    )
    return parser.parse_args()


def build_base_command(args: argparse.Namespace) -> list[str]:
    repo_root = Path(__file__).resolve().parent.parent
    script_path = (
        Path(args.script_path).resolve()
        if args.script_path is not None
        else repo_root / "dvfs" / "test_geopm_apply.sh"
    )
    if not script_path.exists():
        raise FileNotFoundError(f"missing node test script: {script_path}")

    cmd = ["bash", str(script_path)]
    if args.ssh_user:
        cmd.extend(["--ssh-user", str(args.ssh_user)])
    if args.remote_repo_dir:
        cmd.extend(["--remote-repo-dir", str(args.remote_repo_dir)])
    if args.shared_conf_dir:
        cmd.extend(["--shared-conf-dir", str(args.shared_conf_dir)])
    if args.cores:
        cmd.extend(["--cores", str(args.cores)])
    if args.target_mhz is not None:
        cmd.extend(["--target-mhz", str(args.target_mhz)])
    if args.restore_mhz is not None:
        cmd.extend(["--restore-mhz", str(args.restore_mhz)])
    if args.control_kind:
        cmd.extend(["--control-kind", str(args.control_kind)])
    if args.cpufreq_sync is True:
        cmd.append("--cpufreq-sync")
    elif args.cpufreq_sync is False:
        cmd.append("--no-cpufreq-sync")
    if args.skip_install:
        cmd.append("--skip-install")
    if args.keep_temp:
        cmd.append("--keep-temp")
    return cmd


def run_for_node(*, node: str, base_cmd: list[str], output_dir: Path) -> NodeResult:
    log_path = output_dir / f"{node}.log"
    cmd = [*base_cmd, "--node", node]
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return NodeResult(node=node, returncode=int(proc.returncode), log_path=log_path)


def main() -> int:
    args = parse_args()
    if int(args.max_workers) <= 0:
        print("[ERR] --max-workers must be > 0", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_cmd = build_base_command(args)

    print(
        "[INFO] starting node tests:",
        f"nodes={args.nodes}",
        f"max_workers={int(args.max_workers)}",
        f"output_dir={output_dir}",
    )

    results: list[NodeResult] = []
    max_workers = min(int(args.max_workers), len(args.nodes))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_node = {
            executor.submit(run_for_node, node=node, base_cmd=base_cmd, output_dir=output_dir): node
            for node in args.nodes
        }
        for future in as_completed(future_to_node):
            node = future_to_node[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"[FAIL] {node}: runner exception: {exc}")
                continue
            results.append(result)
            status = "PASS" if result.returncode == 0 else "FAIL"
            print(f"[{status}] {node}: rc={result.returncode} log={result.log_path}")

    results.sort(key=lambda item: item.node)
    failures = [result for result in results if result.returncode != 0]
    print("\nSummary:")
    for result in results:
        status = "PASS" if result.returncode == 0 else "FAIL"
        print(f"{status:4s} {result.node:12s} rc={result.returncode} log={result.log_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
