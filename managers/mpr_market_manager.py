#!/usr/bin/env python3
"""Minimal MPR market manager with a background thread.

Provides start/stop hooks used by the main controller and runs a
lightweight background loop in a daemon thread. The loop currently
performs a periodic no-op tick, which you can extend later.
"""

from __future__ import annotations

import random
import threading
from typing import Dict, Iterable, List, Tuple


VALID_MPR_MODES = {"mpr_stat", "mpr_int"}

# Empirical resource reduction (fraction) vs power savings (watts)
# to be shared across supported job names.
BASE_RESOURCE_POWER_CURVE = [
    (0.311549244, 90.26631158),
    (0.337285721, 97.72303595),
    (0.363022198, 105.1797603),
    (0.381405395, 110.5059920),
    (0.413575992, 119.8268975),
    (0.448504067, 129.9467377),
    (0.496300381, 143.7949401),
    (0.519279379, 150.4527297),
    (0.555126614, 160.8388815),
    (0.592812170, 171.7576565),
    (0.624063606, 180.8122503),
    (0.665425801, 192.7962716),
    (0.723332874, 209.5739015),
    (0.776644147, 225.0199734),
    (0.824440461, 238.8681758),
    (0.863964337, 250.3195739),
    (0.903488212, 261.7709720),
    (0.947607886, 274.5539281),
    (0.977021003, 283.0758988),
    (1.0,          289.7336884),
]

SUPPORTED_JOBS = {
    "minife",
    "comd",
    "xsbench",
    "swfft",
    "hpccg",
    "simplemoc",
}


def _build_job_curve(job_name: str) -> list[tuple[float, float]]:
    """Derive a slightly perturbed resource/power curve per job.

    Uses a deterministic PRNG seeded by the job name so results are
    reproducible run-to-run.
    """

    rng = random.Random(job_name)
    curve: list[tuple[float, float]] = []
    prev_res = 0.0
    last_index = len(BASE_RESOURCE_POWER_CURVE) - 1

    for idx, (resource, power) in enumerate(BASE_RESOURCE_POWER_CURVE):
        # Apply small additive tweak while maintaining monotonic increase.
        delta = rng.uniform(-0.015, 0.015)
        adjusted_res = resource + delta
        adjusted_res = min(max(adjusted_res, prev_res + 0.002), 1.0)
        if idx == last_index:
            adjusted_res = 1.0
        prev_res = adjusted_res

        # Scale power within ±8% to simulate job-specific power response.
        power_scale = 1.0 + rng.uniform(-0.08, 0.08)
        adjusted_power = max(power * power_scale, 0.0)

        curve.append((adjusted_res, adjusted_power))

    return curve


JOB_RESOURCE_POWER = {job: _build_job_curve(job) for job in SUPPORTED_JOBS}


class MPRMarketManager:
    """Market manager with start/stop lifecycle and background thread."""

    def __init__(self, *, mpr_mode: str = "mpr_stat") -> None:
        if mpr_mode not in VALID_MPR_MODES:
            raise ValueError(f"mpr_mode must be one of {sorted(VALID_MPR_MODES)}")
        self._running = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.mpr_mode = mpr_mode

    def start(self) -> None:
        """Start background thread (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._worker,
            name="MPRMarketManager",
            daemon=True,
        )
        self._thread.start()
        self._running = True

    def stop(self) -> None:
        """Signal background thread to stop and wait briefly."""
        self._running = False
        if not self._thread:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._thread = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            finally:
                # Keep the loop lightweight; adjust sleep as needed.
                self._stop.wait(1.0)

    def _tick(self) -> None:
        """Periodic placeholder. Extend with market logic as needed."""
        # No-op by default; hook for future functionality.
        pass

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def plan_reductions(
        self,
        jobs: Iterable[Tuple[str, str]],
    ) -> Dict[str, float]:
        """Return job_id -> reduction mapping using the per-job curves.

        For ``mpr_stat`` we choose the midpoint entry of the curve. For
        ``mpr_int`` we bias toward a more aggressive reduction (75th percentile).
        Jobs that are not in ``SUPPORTED_JOBS`` are skipped.
        """

        reductions: Dict[str, float] = {}

        for job_id, job_name in jobs:
            name_key = (job_name or "").split()[0].lower()
            if name_key not in JOB_RESOURCE_POWER:
                continue

            curve = JOB_RESOURCE_POWER[name_key]
            if not curve:
                continue

            if self.mpr_mode == "mpr_int":
                index = min(len(curve) - 1, int(len(curve) * 0.75))
            else:
                index = len(curve) // 2

            reductions[job_id] = curve[index][0]

        return reductions
