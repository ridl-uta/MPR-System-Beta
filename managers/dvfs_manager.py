#!/usr/bin/env python3
"""Dynamic voltage/frequency scheduling (DVFS) manager.

This module converts market bids into CPU frequency reductions and pushes them
to GEOPM via the helpers in ``dvfs.job_utilities``. The work is carried out
asynchronously so callers can enqueue bids without blocking.
"""

from __future__ import annotations

import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dvfs import (
    CONFIG_DIR_DEFAULT,
    NODES_FILE_DEFAULT,
    SetJobFrequencyError,
    apply_reduction,
    compute_frequency_from_reduction,
)

_DEFAULT_Q = 1.0
_DEFAULT_DELTA = 0.4
_DEFAULT_MAX_FREQ_MHZ = 2200.0
_DEFAULT_MIN_FREQ_MHZ = 1600.0


def supply(bid: float, q: float = _DEFAULT_Q, delta: float = _DEFAULT_DELTA) -> float:
    """Return the fractional reduction requested for a given bid."""
    return max(delta - bid / q, 0.0)


@dataclass
class BidRequest:
    job_id: str
    bid: float
    dry_run: bool
    direct: bool = False


class DVFSManager:
    """Manage bid-driven DVFS adjustments in a background thread."""

    def __init__(
        self,
        *,
        q: float = _DEFAULT_Q,
        delta: float = _DEFAULT_DELTA,
        max_freq_mhz: float = _DEFAULT_MAX_FREQ_MHZ,
        min_freq_mhz: float = _DEFAULT_MIN_FREQ_MHZ,
        conf_dir: str = CONFIG_DIR_DEFAULT,
        nodes_file: Path | str = NODES_FILE_DEFAULT,
    ) -> None:
        self.q = q
        self.delta = delta
        self.max_freq_mhz = max_freq_mhz
        self.min_freq_mhz = min_freq_mhz
        self.conf_dir = conf_dir
        self.nodes_file = Path(nodes_file)

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
    ) -> None:
        """Queue a bid for asynchronous processing."""
        self._queue.put(BidRequest(job_id, bid, dry_run, False))

    def submit_reduction(
        self,
        job_id: str,
        reduction: float,
        *,
        dry_run: bool = False,
    ) -> None:
        """Queue a direct reduction (fraction between 0 and 1)."""
        self._queue.put(BidRequest(job_id, reduction, dry_run, True))

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
        if request.direct:
            reduction = min(max(request.bid, 0.0), 1.0)
        else:
            reduction = self._compute_reduction(request.bid)

        freq_hz, freq_mhz, _ = compute_frequency_from_reduction(
            reduction,
            max_freq_mhz=self.max_freq_mhz,
            min_freq_mhz=self.min_freq_mhz,
        )

        print(
            f"[DVFS] job {request.job_id}: reduction {reduction:.3f} -> {freq_mhz:.3f} MHz"
        )

        try:
            apply_reduction(
                request.job_id,
                reduction,
                dry_run=request.dry_run,
                max_freq_mhz=self.max_freq_mhz,
                min_freq_mhz=self.min_freq_mhz,
                conf_dir=self.conf_dir,
                nodes_file=self.nodes_file,
            )
        except SetJobFrequencyError as exc:
            raise RuntimeError(str(exc)) from exc
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

    def _compute_reduction(self, bid: float) -> float:
        reduction = supply(bid, q=self.q, delta=self.delta)
        reduction = min(max(reduction, 0.0), 1.0)
        return reduction


# ----------------------------------------------------------------------
# CLI entry-point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal CLI: print nodes and cores for a Slurm job ID.
    import argparse
    from dvfs import SetJobFrequencyError, apply_reduction, compute_frequency_from_reduction

    parser = argparse.ArgumentParser(
        description="Apply a GEOPM frequency reduction to a Slurm job"
    )
    parser.add_argument("job_id", help="Slurm job identifier")
    parser.add_argument(
        "reduction",
        type=float,
        help="Fractional reduction between 0 and 1 (0=no change, 1=max reduction)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions only")
    parser.add_argument(
        "--max-freq",
        type=float,
        default=_DEFAULT_MAX_FREQ_MHZ,
        help="Maximum frequency in MHz for reduction computation",
    )
    parser.add_argument(
        "--min-freq",
        type=float,
        default=_DEFAULT_MIN_FREQ_MHZ,
        help="Minimum frequency in MHz for reduction computation",
    )
    parser.add_argument(
        "--config-dir",
        default=CONFIG_DIR_DEFAULT,
        help="Directory to write per-host GEOPM configs",
    )
    parser.add_argument(
        "--nodes-file",
        default=str(NODES_FILE_DEFAULT),
        help="File listing nodes for SSH apply (one per line)",
    )
    parser.add_argument(
        "--geopmwrite-cmd",
        default="geopmwrite",
        help="Command used to write GEOPM signals",
    )
    parser.add_argument(
        "--signal",
        default="CPU_FREQUENCY_MAX_CONTROL",
        help="GEOPM signal to write",
    )
    parser.add_argument(
        "--domain",
        default="core",
        help="GEOPM domain to target",
    )
    args = parser.parse_args()

    try:
        freq_hz, freq_mhz, _ = compute_frequency_from_reduction(
            args.reduction,
            max_freq_mhz=args.max_freq,
            min_freq_mhz=args.min_freq,
        )
    except ValueError as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"[INFO] Reduction {args.reduction:.3f} => target frequency {freq_mhz:.3f} MHz"
    )

    try:
        apply_reduction(
            args.job_id,
            args.reduction,
            dry_run=args.dry_run,
            max_freq_mhz=args.max_freq,
            min_freq_mhz=args.min_freq,
            geopmwrite_cmd=args.geopmwrite_cmd,
            signal=args.signal,
            domain=args.domain,
            conf_dir=args.config_dir,
            nodes_file=args.nodes_file,
        )
    except Exception as exc:
        exit_code = getattr(exc, "exit_code", 1)
        print(f"[ERR] {exc}", file=sys.stderr)
        sys.exit(exit_code)
