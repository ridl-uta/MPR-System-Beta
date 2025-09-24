#!/usr/bin/env python3
"""Minimal MainController stub with background managers."""

from __future__ import annotations

import threading
import time

from managers import DVFSManager, PowerMonitor, MPRMarketManager


class MainController:
    """Boilerplate controller"""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self.power_monitor = PowerMonitor()
        self.dvfs_manager = DVFSManager()
        self.market_manager = MPRMarketManager()

    def start(self) -> None:
        print("[MainController] started")
        self.power_monitor.start()
        self.dvfs_manager.start()
        self.market_manager.start()

    def stop(self) -> None:
        if not self._stop.is_set():
            self._stop.set()
            self.market_manager.stop()
            self.dvfs_manager.stop()
            self.power_monitor.stop()
            print("[MainController] stopped")

    def run(self) -> None:
        try:
            while not self._stop.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            print("[MainController] interrupted")
            self.stop()


if __name__ == "__main__":
    controller = MainController()
    controller.start()
    try:
        controller.run()
    finally:
        controller.stop()
