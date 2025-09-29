#!/usr/bin/env python3
"""Dynamic voltage/frequency scheduling (DVFS) manager.

This module converts market bids into CPU frequency requests and pushes them to
GEOPM via ``set_job_frequency.py``. The work is carried out asynchronously so
callers can enqueue bids without blocking.
"""

from __future__ import annotations

import queue
import random
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

_DEFAULT_Q = 1.0
_DEFAULT_DELTA = 0.4
_DEFAULT_MAX_FREQ_MHZ = 2200.0
_DEFAULT_MIN_FREQ_MHZ = 800.0


def supply(bid: float, q: float = _DEFAULT_Q, delta: float = _DEFAULT_DELTA) -> float:
    """Return the fractional reduction requested for a given bid."""
    return max(delta - bid / q, 0.0)


@dataclass
class BidRequest:
    job_id: str
    bid: float
    dry_run: bool
    extra_args: Optional[Sequence[str]]
    demo_nodes: Optional[Sequence[str]]
    demo_cores: Optional[Sequence[int]]


class DVFSManager:
    """Manage bid-driven DVFS adjustments in a background thread."""

    def __init__(
        self,
        *,
        q: float = _DEFAULT_Q,
        delta: float = _DEFAULT_DELTA,
        max_freq_mhz: float = _DEFAULT_MAX_FREQ_MHZ,
        min_freq_mhz: float = _DEFAULT_MIN_FREQ_MHZ,
        python_executable: Optional[str] = None,
    ) -> None:
        self.q = q
        self.delta = delta
        self.max_freq_mhz = max_freq_mhz
        self.min_freq_mhz = min_freq_mhz
        self.python_executable = python_executable or sys.executable

        self._queue: queue.Queue[BidRequest] = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, name="DVFSManager", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def submit_bid(
        self,
        job_id: str,
        bid: float,
        *,
        dry_run: bool = False,
        extra_args: Optional[Sequence[str]] = None,
        demo_nodes: Optional[Sequence[str]] = None,
        demo_cores: Optional[Sequence[int]] = None,
    ) -> None:
        """Queue a bid for asynchronous processing."""
        self._queue.put(BidRequest(job_id, bid, dry_run, extra_args, demo_nodes, demo_cores))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                request = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process_request(request)
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"[DVFS] ERROR processing bid for {request.job_id}: {exc}", file=sys.stderr)
            finally:
                self._queue.task_done()

    def _process_request(self, request: BidRequest) -> None:
        freq = self._compute_frequency(request.bid)
        argv = [
            self.python_executable,
            str(Path(__file__).parent / "set_job_frequency.py"),
            str(request.job_id),
            f"{freq:.3f}",
            "--unit",
            "MHz",
        ]
        if request.extra_args:
            argv.extend(request.extra_args)

        if request.dry_run:
            nodes, cores = self._demo_selections(request.demo_nodes, request.demo_cores)
            print("[DRY-RUN] set_job_frequency.py would be invoked as:")
            print("           ", " ".join(argv))
            print(f"[DRY-RUN] Demo nodes: [{', '.join(nodes)}]")
            print(f"[DRY-RUN] Demo cores: [{', '.join(map(str, cores))}]")
            print(f"[DRY-RUN] Frequency to apply: {freq:.3f} MHz")
            return

        result = subprocess.run(argv, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"set_job_frequency.py exited with status {result.returncode}"
            )

    def _compute_frequency(self, bid: float) -> float:
        reduction = supply(bid, q=self.q, delta=self.delta)
        reduction = min(max(reduction, 0.0), 1.0)
        freq = self.max_freq_mhz * (1.0 - reduction)
        return max(self.min_freq_mhz, min(freq, self.max_freq_mhz))

    def _demo_selections(
        self,
        nodes_override: Optional[Sequence[str]],
        cores_override: Optional[Sequence[int]],
    ) -> tuple[list[str], list[int]]:
        default_nodes = [f"ridlserver0{i}" for i in range(1, 7)]
        default_cores = list(range(21))

        if nodes_override is not None:
            nodes = list(nodes_override)
        else:
            max_nodes = len(default_nodes)
            count = random.randint(2, max_nodes) if max_nodes >= 2 else max_nodes
            nodes = sorted(random.sample(default_nodes, count))

        if cores_override is not None:
            cores = list(cores_override)
        else:
            count = random.randint(1, len(default_cores))
            cores = sorted(random.sample(default_cores, count))

        return nodes, cores


# ----------------------------------------------------------------------
# CLI entry-point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal CLI: print nodes and cores for a Slurm job ID.
    import argparse
    from dvfs import job_cores

    parser = argparse.ArgumentParser(description="Show nodes and cores for a Slurm job")
    parser.add_argument("job_id", help="Slurm job identifier")
    args = parser.parse_args()

    try:
        state, host_map = job_cores.collect_host_cores(args.job_id)
    except job_cores.JobCoresError as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        sys.exit(exc.exit_code)

    if state and state.upper() != "RUNNING":
        print(f"[WARN] job state: {state}")

    for host, cores in host_map.items():
        expanded = " ".join(str(c) for c in cores)
        print(f"{host}: {expanded}")
