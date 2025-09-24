#!/usr/bin/env python3
"""Minimal SystemController stub with background managers."""

from __future__ import annotations

import threading
import time

from dvfs_manager import DVFSManager
from managers.power_monitor import PowerMonitor
from managers.mpr_market_manager import MPRMarketManager


class SystemController:
    """Boilerplate controller that wires up power, DVFS, and market managers."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        # Lazy configuration; callers can replace these instances if needed.
        self.power_monitor = PowerMonitor()
        self.dvfs_manager = DVFSManager()
        self.market_manager = MPRMarketManager()

    def start(self) -> None:
        print("[SystemController] started")
        self.power_monitor.start()
        self.dvfs_manager.start()
        self.market_manager.start()

    def stop(self) -> None:
        if not self._stop.is_set():
            self._stop.set()
            self.market_manager.stop()
            self.dvfs_manager.stop()
            self.power_monitor.stop()
            print("[SystemController] stopped")

    def run(self) -> None:
        try:
            while not self._stop.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            print("[SystemController] interrupted")
            self.stop()


if __name__ == "__main__":
    controller = SystemController()
    controller.start()
    try:
        controller.run()
    finally:
        controller.stop()
