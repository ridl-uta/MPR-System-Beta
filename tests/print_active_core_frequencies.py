#!/usr/bin/env python3
"""Print active frequency for a set of physical cores.

Examples:
    python3 tests/print_active_core_frequencies.py
    python3 tests/print_active_core_frequencies.py --cores 0-19
    python3 tests/print_active_core_frequencies.py --cores 0,1,2,8-11 --json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List


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


def normalize_freq_to_mhz(value: float) -> float:
    if value >= 1e8:
        return value / 1e6
    if value >= 1e5:
        return value / 1e3
    return value


def read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def get_cpu_to_core_map() -> Dict[int, int]:
    try:
        proc = subprocess.run(
            ["bash", "-lc", "lscpu -p=CPU,CORE"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Failed to run 'lscpu -p=CPU,CORE'. This helper is intended to run on a Linux node "
            "where lscpu and CPU frequency sysfs/GEOPM are available."
        ) from exc
    cpu_to_core: Dict[int, int] = {}
    for line in proc.stdout.splitlines():
        row = line.strip()
        if not row or row.startswith("#"):
            continue
        cpu_text, core_text = row.split(",", 1)
        cpu_to_core[int(cpu_text)] = int(core_text)
    return cpu_to_core


def get_core_to_cpus() -> Dict[int, List[int]]:
    core_to_cpus: Dict[int, List[int]] = {}
    for cpu_idx, core_idx in get_cpu_to_core_map().items():
        core_to_cpus.setdefault(core_idx, []).append(cpu_idx)
    for cpu_list in core_to_cpus.values():
        cpu_list.sort()
    return core_to_cpus


def try_geopmread(core_idx: int, signal: str) -> float | None:
    try:
        proc = subprocess.run(
            ["geopmread", signal, "core", str(core_idx)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    text = proc.stdout.strip()
    if not text:
        return None
    try:
        return normalize_freq_to_mhz(float(text))
    except ValueError:
        return None


def try_sysfs(core_idx: int, core_to_cpus: Dict[int, List[int]]) -> tuple[float | None, str]:
    cpu_ids = core_to_cpus.get(core_idx, [])
    values_mhz: List[float] = []
    source = ""
    for cpu_idx in cpu_ids:
        for rel_path in (
            f"/sys/devices/system/cpu/cpu{cpu_idx}/cpufreq/scaling_cur_freq",
            f"/sys/devices/system/cpu/cpu{cpu_idx}/cpufreq/cpuinfo_cur_freq",
        ):
            text = read_text(Path(rel_path))
            if text is None:
                continue
            try:
                values_mhz.append(normalize_freq_to_mhz(float(text)))
                source = Path(rel_path).name
                break
            except ValueError:
                continue
    if not values_mhz:
        return None, "unavailable"
    return sum(values_mhz) / float(len(values_mhz)), source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print active frequency for physical cores in a loop.")
    parser.add_argument(
        "--cores",
        default="0-19",
        help="Core list/ranges (default: 0-19), e.g. 0,1,2 or 0-7.",
    )
    parser.add_argument(
        "--signal",
        default="CPU_FREQUENCY_STATUS",
        help="GEOPM signal to use for core-domain readback (default: CPU_FREQUENCY_STATUS).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON snapshots instead of the tabular text format.",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=1.0,
        help="Seconds between prints (default: 1.0).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one snapshot and exit instead of looping until canceled.",
    )
    return parser


def collect_rows(cores: List[int], signal: str, core_to_cpus: Dict[int, List[int]]) -> List[dict[str, object]]:
    rows: List[dict[str, object]] = []
    for core_idx in cores:
        geopm_mhz = try_geopmread(core_idx, str(signal))
        if geopm_mhz is not None:
            rows.append(
                {
                    "core": core_idx,
                    "cpus": core_to_cpus.get(core_idx, []),
                    "active_mhz": round(geopm_mhz, 3),
                    "source": f"geopmread:{signal}",
                }
            )
            continue

        sysfs_mhz, sysfs_source = try_sysfs(core_idx, core_to_cpus)
        rows.append(
            {
                "core": core_idx,
                "cpus": core_to_cpus.get(core_idx, []),
                "active_mhz": None if sysfs_mhz is None else round(sysfs_mhz, 3),
                "source": f"sysfs:{sysfs_source}",
            }
        )
    return rows


def print_rows(rows: List[dict[str, object]], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(rows, indent=2), flush=True)
        return

    print(f"{'core':>4}  {'cpus':<12}  {'active_mhz':>10}  source")
    for row in rows:
        cpu_text = ",".join(str(cpu) for cpu in row["cpus"])
        freq_text = "NA" if row["active_mhz"] is None else f"{float(row['active_mhz']):.3f}"
        print(f"{int(row['core']):>4}  {cpu_text:<12}  {freq_text:>10}  {row['source']}")
    print(flush=True)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        cores = parse_core_list(args.cores)
    except ValueError as exc:
        parser.error(str(exc))
    if args.interval_s <= 0:
        parser.error("--interval-s must be > 0.")

    try:
        core_to_cpus = get_core_to_cpus()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 2

    try:
        while True:
            rows = collect_rows(cores, str(args.signal), core_to_cpus)
            print(time.strftime("[%Y-%m-%d %H:%M:%S]"))
            print_rows(rows, as_json=bool(args.json))
            if args.once:
                break
            time.sleep(float(args.interval_s))
    except KeyboardInterrupt:
        print("Stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
