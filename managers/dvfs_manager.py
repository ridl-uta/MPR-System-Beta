#!/usr/bin/env python3
"""DVFS manager facade for applying GEOPM reductions."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from dvfs import (
    CONFIG_DIR_DEFAULT,
    NODES_FILE_DEFAULT,
    SetJobFrequencyError,
    apply_reduction,
    compute_frequency_from_reduction,
)

_DEFAULT_MAX_FREQ_MHZ = 2200.0
_DEFAULT_MIN_FREQ_MHZ = 1600.0


class DVFSManager:
    """Lightweight helper to push GEOPM frequency reductions."""

    def __init__(
        self,
        *,
        max_freq_mhz: float = _DEFAULT_MAX_FREQ_MHZ,
        min_freq_mhz: float = _DEFAULT_MIN_FREQ_MHZ,
        conf_dir: str = CONFIG_DIR_DEFAULT,
        nodes_file: Path | str = NODES_FILE_DEFAULT,
    ) -> None:
        self.max_freq_mhz = max_freq_mhz
        self.min_freq_mhz = min_freq_mhz
        self.conf_dir = conf_dir
        self.nodes_file = Path(nodes_file)
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle hooks (no-op today, kept for interface parity)
    # ------------------------------------------------------------------
    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def submit_reduction(
        self,
        job_id: str,
        reduction: float,
        *,
        dry_run: bool = False,
    ) -> None:
        """Apply a direct reduction immediately (fraction between 0 and 1)."""

        normalized = self._normalize_reduction(reduction)
        self._execute_reduction(job_id, normalized, dry_run)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize_reduction(self, reduction: float) -> float:
        return min(max(reduction, 0.0), 1.0)

    def _execute_reduction(self, job_id: str, reduction: float, dry_run: bool) -> None:
        freq_hz, freq_mhz, _ = compute_frequency_from_reduction(
            reduction,
            max_freq_mhz=self.max_freq_mhz,
            min_freq_mhz=self.min_freq_mhz,
        )

        print(
            f"[DVFS] job {job_id}: reduction {reduction:.3f} -> {freq_mhz:.3f} MHz"
        )

        try:
            apply_reduction(
                job_id,
                reduction,
                dry_run=dry_run,
                max_freq_mhz=self.max_freq_mhz,
                min_freq_mhz=self.min_freq_mhz,
                conf_dir=self.conf_dir,
                nodes_file=self.nodes_file,
            )
        except SetJobFrequencyError as exc:
            raise RuntimeError(str(exc)) from exc
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc


# ----------------------------------------------------------------------
# CLI entry-point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

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

    mgr = DVFSManager(
        max_freq_mhz=args.max_freq,
        min_freq_mhz=args.min_freq,
        conf_dir=args.config_dir,
        nodes_file=args.nodes_file,
    )

    try:
        mgr.submit_reduction(
            args.job_id,
            args.reduction,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        exit_code = getattr(exc, "exit_code", 1)
        print(f"[ERR] {exc}", file=sys.stderr)
        sys.exit(exit_code)
