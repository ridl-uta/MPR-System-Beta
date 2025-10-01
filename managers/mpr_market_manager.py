#!/usr/bin/env python3
"""Minimal MPR market manager with a background thread.

Provides start/stop hooks used by the main controller and runs a
lightweight background loop in a daemon thread. The loop currently
performs a periodic no-op tick, which you can extend later.
"""

from __future__ import annotations

import threading
import time


class MPRMarketManager:
    """Market manager with start/stop lifecycle and background thread."""

    def __init__(self) -> None:
        self._running = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

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
