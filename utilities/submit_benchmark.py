#!/usr/bin/env python3
"""Submit an MPI+threads benchmark via Slurm using a rank-config CSV.

Reads a CSV (default: data/xsbench_args.csv) with rows of the form:
    ranks,size,lookups_per_rank
and dispatches an sbatch that launches the given benchmark binary with
the appropriate per-rank lookups and size. The launcher constructs an
sbatch --wrap srun so your Slurm script does not need to parse params.

Examples (assumes the benchmark understands -s <size>, -l <lookups>, -t <threads>):
  - 1 node, 1 rank x 10 threads (small, 10000 lookups per rank):
      ./utilities/submit_benchmark.py -N 1 -n 1 -c 10 --bin ./XSBenchMPI
  - 1 node, 2 ranks x 10 threads per rank (per-rank lookups from CSV):
      ./utilities/submit_benchmark.py -N 1 -n 2 -c 10 --bin ./XSBenchMPI
  - 2 nodes, 4 ranks (2 per node) x 10 threads per rank, pin to nodes:
      ./utilities/submit_benchmark.py -N 2 -n 4 --ntasks-per-node 2 -c 10 \
         --nodelist ridlserver[01,04] --bin ./XSBenchMPI

Override CSV values with --size / --lookups if needed.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
from pathlib import Path


def read_args_row(table: Path, ranks: int) -> tuple[str | None, int | None]:
    if not table.exists():
        return None, None
    with table.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                r = int((row.get("ranks") or "").strip())
            except ValueError:
                continue
            if r != ranks:
                continue
            size = (row.get("size") or "").strip() or None
            lp = row.get("lookups_per_rank") or row.get("lookups") or ""
            try:
                lookups = int(lp) if lp else None
            except ValueError:
                lookups = None
            return size, lookups
    return None, None


def build_sbatch_wrap(
    *,
    nodes: int,
    ranks: int,
    cpus_per_task: int,
    partition: str,
    time_limit: str,
    output: str,
    nodelist: str | None,
    exclude: str | None,
    ntasks_per_node: int | None,
    mpi_iface: str,
    bench_bin: str,
    workdir: str | None,
    size: str,
    lookups: int,
) -> list[str]:
    env_exports = {
        "OMP_NUM_THREADS": str(cpus_per_task),
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
    }
    export_str = ",".join(f"{k}={v}" for k, v in env_exports.items())

    # If a workdir is provided, use it; otherwise default to an env override
    # with a sane fallback to /shared/bin so the wrap is robust when WORKDIR
    # isn't exported by the caller.
    cd_prefix = (
        f"cd {shlex.quote(workdir)}; " if workdir else "cd ${WORKDIR:-/shared/bin}; "
    )
    srun_cmd = (
        f"{cd_prefix}"
        f"srun --mpi={shlex.quote(mpi_iface)} --cpu-bind=cores "
        f"--hint=nomultithread --distribution=block:block "
        f"{shlex.quote(bench_bin)} -s {shlex.quote(size)} -l {lookups} -t $OMP_NUM_THREADS || true"
    )

    cmd = [
        "sbatch",
        "-p",
        partition,
        "-N",
        str(nodes),
        "-n",
        str(ranks),
        "-c",
        str(cpus_per_task),
        "-t",
        time_limit,
        "--export=ALL," + export_str,
        "-o",
        output,
    ]
    if ntasks_per_node:
        cmd += ["--ntasks-per-node", str(ntasks_per_node)]
    if nodelist:
        cmd += ["--nodelist", nodelist]
    if exclude:
        cmd += ["--exclude", exclude]
    cmd += ["--wrap", srun_cmd]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description="Submit MPI+threads benchmark using CSV args")
    ap.add_argument("-N", "--nodes", type=int, default=1, help="Slurm nodes (-N)")
    ap.add_argument("-n", "--ranks", type=int, default=1, help="Total MPI ranks (-n)")
    ap.add_argument("-c", "--cpus-per-task", type=int, default=10, help="Threads per rank (-c)")
    ap.add_argument("--ntasks-per-node", type=int, default=None, help="Ranks per node")
    ap.add_argument("-p", "--partition", default="debug")
    ap.add_argument("-t", "--time", dest="time_limit", default="00:30:00")
    ap.add_argument("-o", "--output", default="/shared/logs/job-%j.out")
    ap.add_argument("--nodelist", default=None)
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--mpi", dest="mpi_iface", default="pmi2", choices=["pmi2", "pmix"])
    ap.add_argument("--table", default=str(Path("data/xsbench_args.csv")), help="CSV with ranks,size,lookups_per_rank")
    ap.add_argument("--size", default=None, help="Override size from CSV")
    ap.add_argument("--lookups", type=int, default=None, help="Override lookups per rank from CSV")
    ap.add_argument("--bin", dest="bench_bin", default="./XSBenchMPI", help="Path to benchmark binary (default ./XSBenchMPI)")
    ap.add_argument("--workdir", default=None, help="Working directory to cd before running (optional)")
    ap.add_argument("--dry-run", action="store_true", help="Print sbatch command without executing")
    args = ap.parse_args()

    table = Path(args.table)
    size_csv, lookups_csv = read_args_row(table, args.ranks)
    size = args.size or size_csv or "small"
    lookups = args.lookups or lookups_csv or 10000

    cmd = build_sbatch_wrap(
        nodes=args.nodes,
        ranks=args.ranks,
        cpus_per_task=args.cpus_per_task,
        partition=args.partition,
        time_limit=args.time_limit,
        output=args.output,
        nodelist=args.nodelist,
        exclude=args.exclude,
        ntasks_per_node=args.ntasks_per_node,
        mpi_iface=args.mpi_iface,
        bench_bin=args.bench_bin,
        workdir=args.workdir,
        size=size,
        lookups=lookups,
    )

    print("Submitting:")
    print(" "+" ".join(shlex.quote(c) for c in cmd))
    if args.dry_run:
        return 0
    res = subprocess.run(cmd, text=True, capture_output=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        return res.returncode
    print(res.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
