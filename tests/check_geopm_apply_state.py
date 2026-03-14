#!/usr/bin/env python3
"""Validate GEOPM apply state on a Linux compute node.

This is an integration check for live node state. It verifies:
  1. systemd/journal output for the most recent apply
  2. cpufreq policy files under /sys/devices/system/cpu/cpufreq/policy*
  3. GEOPM readback for the requested control signal and status signal

Example:
  sudo python3 tests/check_geopm_apply_state.py \
    --apply \
    --cores 2,6 \
    --freq-hz 1.4e9 \
    --control-kind PERF_CTL
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def fail(message: str) -> None:
    raise RuntimeError(message)


def run_command(cmd: List[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            cmd,
            check=check,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        fail(f"Required command not found: {cmd[0]}")
    except subprocess.CalledProcessError as exc:
        stdout = exc.stdout.strip()
        stderr = exc.stderr.strip()
        fail(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout: {stdout or '<empty>'}\n"
            f"stderr: {stderr or '<empty>'}"
        )
    return proc


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
                fail(f"Invalid core range: {part}")
            cores.update(range(start, end + 1))
            continue
        cores.add(int(part))
    if not cores:
        fail("No cores parsed from --cores")
    return sorted(cores)


def expand_cpu_list(text: str) -> List[int]:
    values: List[int] = []
    raw = text.replace("\n", ",").replace(" ", ",").strip(",")
    for token in raw.split(","):
        part = token.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                fail(f"Invalid CPU range: {part}")
            values.extend(range(start, end + 1))
            continue
        values.append(int(part))
    return sorted(dict.fromkeys(values))


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="ascii").strip()
    except OSError as exc:
        fail(f"Failed to read {path}: {exc}")


def get_core_to_cpus() -> Dict[int, List[int]]:
    proc = run_command(["bash", "-lc", "lscpu -p=CPU,CORE"])
    result: Dict[int, List[int]] = {}
    for line in proc.stdout.splitlines():
        row = line.strip()
        if not row or row.startswith("#"):
            continue
        cpu_text, core_text = row.split(",", 1)
        result.setdefault(int(core_text), []).append(int(cpu_text))
    for cpu_list in result.values():
        cpu_list.sort()
    if not result:
        fail("No CPU/core mapping found from lscpu")
    return result


def geopmread_mhz(signal: str, core: int) -> float:
    proc = run_command(["geopmread", signal, "core", str(core)])
    text = proc.stdout.strip()
    if not text:
        fail(f"Empty geopmread output for {signal} core {core}")
    value = float(text)
    if value >= 1e8:
        return value / 1e6
    if value >= 1e5:
        return value / 1e3
    return value


def get_target_policies(target_cpus: set[int]) -> Dict[str, List[int]]:
    policies: Dict[str, List[int]] = {}
    base = Path("/sys/devices/system/cpu/cpufreq")
    for policy_path in sorted(base.glob("policy*")):
        affected = expand_cpu_list(read_text(policy_path / "affected_cpus"))
        if any(cpu in target_cpus for cpu in affected):
            policies[policy_path.name] = affected
    if not policies:
        fail("No cpufreq policies matched the requested cores")
    return policies


def read_policy_state(policy_name: str) -> Dict[str, str]:
    policy_path = Path("/sys/devices/system/cpu/cpufreq") / policy_name
    state = {
        "affected_cpus": read_text(policy_path / "affected_cpus"),
        "scaling_governor": read_text(policy_path / "scaling_governor"),
        "scaling_min_freq": read_text(policy_path / "scaling_min_freq"),
        "scaling_max_freq": read_text(policy_path / "scaling_max_freq"),
    }
    setspeed_path = policy_path / "scaling_setspeed"
    state["scaling_setspeed"] = read_text(setspeed_path) if setspeed_path.exists() else ""
    return state


def read_journal(service: str, *, since: str | None, lines: int) -> str:
    cmd = ["journalctl", "-u", service, "--no-pager"]
    if since is not None:
        cmd.extend(["--since", since])
    else:
        cmd.extend(["-n", str(lines)])
    proc = run_command(cmd)
    return proc.stdout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify GEOPM apply state on a live compute node.")
    parser.add_argument("--cores", required=True, help="Target cores, e.g. 2,6 or 0-3.")
    parser.add_argument("--freq-hz", required=True, type=float, help="Target frequency in Hz.")
    parser.add_argument(
        "--control-kind",
        choices=("PERF_CTL", "CORE_MAX"),
        default="PERF_CTL",
        help="Expected GEOPM control path.",
    )
    parser.add_argument(
        "--status-signal",
        default="CPU_FREQUENCY_STATUS",
        help="GEOPM signal used for actual frequency status checks.",
    )
    parser.add_argument(
        "--service",
        default="geopm-apply.service",
        help="systemd service name to inspect.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Start the service before running checks.",
    )
    parser.add_argument(
        "--journal-lines",
        type=int,
        default=200,
        help="Fallback number of journal lines to inspect when --apply is not used.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of GEOPM samples to collect (default: 5).",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=1.0,
        help="Seconds between GEOPM samples (default: 1.0).",
    )
    parser.add_argument(
        "--requested-tolerance-mhz",
        type=float,
        default=25.0,
        help="Allowed deviation for requested control readback (default: 25 MHz).",
    )
    parser.add_argument(
        "--status-tolerance-mhz",
        type=float,
        default=25.0,
        help="Allowed amount above target for status signal (default: 25 MHz).",
    )
    parser.add_argument(
        "--expect-governor",
        default="userspace",
        help="Expected cpufreq governor for matched policies (default: userspace).",
    )
    parser.add_argument(
        "--expect-min-khz",
        type=int,
        default=1000000,
        help="Expected scaling_min_freq for matched policies (default: 1000000).",
    )
    parser.add_argument(
        "--allow-policy-expansion",
        action="store_true",
        help="Allow a matched policy to include CPUs outside the requested cores.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.samples <= 0:
        fail("--samples must be > 0")
    if args.interval_s <= 0:
        fail("--interval-s must be > 0")

    cores = parse_core_list(args.cores)
    core_to_cpus = get_core_to_cpus()
    target_cpus: set[int] = set()
    for core in cores:
        if core not in core_to_cpus:
            fail(f"Core {core} is not present on this node")
        target_cpus.update(core_to_cpus[core])

    target_khz = int(round(args.freq_hz / 1000.0))
    target_mhz = args.freq_hz / 1e6
    control_signal = "MSR::PERF_CTL:FREQ" if args.control_kind == "PERF_CTL" else "CPU_FREQUENCY_MAX_CONTROL"
    log_token = "PERF_CTL:FREQ" if args.control_kind == "PERF_CTL" else "MAX_CONTROL"

    since: str | None = None
    if args.apply:
        since = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_command(["systemctl", "start", args.service])
        time.sleep(1.0)

    journal = read_journal(args.service, since=since, lines=args.journal_lines)
    policies = get_target_policies(target_cpus)

    print("Target cores:", " ".join(str(core) for core in cores))
    print("Target policies:", " ".join(sorted(policies)))

    failures: List[str] = []

    for policy_name, affected_cpus in sorted(policies.items()):
        extra_cpus = [cpu for cpu in affected_cpus if cpu not in target_cpus]
        if extra_cpus and not args.allow_policy_expansion:
            failures.append(
                f"{policy_name}: affects extra CPUs {extra_cpus}; use --allow-policy-expansion to accept this"
            )

        state = read_policy_state(policy_name)
        print(
            f"{policy_name}: affected={state['affected_cpus']} "
            f"governor={state['scaling_governor']} "
            f"min={state['scaling_min_freq']} max={state['scaling_max_freq']} "
            f"setspeed={state['scaling_setspeed'] or '<missing>'}"
        )
        if state["scaling_governor"] != args.expect_governor:
            failures.append(
                f"{policy_name}: expected scaling_governor={args.expect_governor}, got {state['scaling_governor']}"
            )
        if state["scaling_min_freq"] != str(args.expect_min_khz):
            failures.append(
                f"{policy_name}: expected scaling_min_freq={args.expect_min_khz}, got {state['scaling_min_freq']}"
            )
        if state["scaling_max_freq"] != str(target_khz):
            failures.append(
                f"{policy_name}: expected scaling_max_freq={target_khz}, got {state['scaling_max_freq']}"
            )
        if state["scaling_setspeed"] and state["scaling_setspeed"] != str(target_khz):
            failures.append(
                f"{policy_name}: expected scaling_setspeed={target_khz}, got {state['scaling_setspeed']}"
            )

        expected_policy_log = (
            f"policy {policy_name}: governor={args.expect_governor} "
            f"min_khz={args.expect_min_khz} max_khz={target_khz}"
        )
        if expected_policy_log not in journal:
            failures.append(f"journal missing: {expected_policy_log}")

    expected_freq_hz_text = f"{int(round(args.freq_hz))}"
    for core in cores:
        expected_core_log = f"core {core}: {log_token}={expected_freq_hz_text}"
        if expected_core_log not in journal:
            failures.append(f"journal missing: {expected_core_log}")

    for sample_idx in range(args.samples):
        print(f"Sample {sample_idx + 1}/{args.samples}")
        for core in cores:
            requested_mhz = geopmread_mhz(control_signal, core)
            status_mhz = geopmread_mhz(args.status_signal, core)
            print(
                f"  core {core}: requested={requested_mhz:.3f} MHz "
                f"status={status_mhz:.3f} MHz"
            )
            if abs(requested_mhz - target_mhz) > args.requested_tolerance_mhz:
                failures.append(
                    f"core {core}: requested {requested_mhz:.3f} MHz != target {target_mhz:.3f} MHz"
                )
            if status_mhz > target_mhz + args.status_tolerance_mhz:
                failures.append(
                    f"core {core}: status {status_mhz:.3f} MHz exceeds target {target_mhz:.3f} MHz"
                )
        if sample_idx + 1 < args.samples:
            time.sleep(args.interval_s)

    if failures:
        print("FAIL")
        for item in failures:
            print(f" - {item}")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
