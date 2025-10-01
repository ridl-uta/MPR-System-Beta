#!/usr/bin/env python3
"""Minimal MPR market manager placeholder.

Provides start/stop hooks used by the main controller without any
background worker or queue — trimmed to the essentials.
"""

from __future__ import annotations


class MPRMarketManager:
    """No-op market manager with start/stop lifecycle."""

    def __init__(self) -> None:
        self._running = False

    def start(self) -> None:
        """Mark the manager as running (idempotent)."""
        self._running = True

    def stop(self) -> None:
        """Mark the manager as stopped (idempotent)."""
        self._running = False
