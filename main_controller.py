#!/usr/bin/env python3
"""Main controller tying together power, DVFS, and market managers.

Historically this module only instantiated the manager classes with their
defaults, leaving the power monitor permanently disabled because no mapping or
credentials were supplied.  The rewritten controller now accepts configuration
from the command line (or environment variables) and wires together sensible
defaults.  It also adds signal handling and robust start/stop bookkeeping so it
can be used as a long-running service.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Optional

from managers import DVFSManager, MPRMarketManager, PowerMonitor

# Default artefacts live alongside the repository.
_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_MAPPING = _REPO_ROOT / "data" / "mapping.json"
_DEFAULT_PDU_CSV = _REPO_ROOT / "output" / "pdu_log.csv"
_DEFAULT_EVENTS_CSV = _REPO_ROOT / "output" / "overload_events.csv"


class MainController:
    """Coordinate the background managers with clean start/stop semantics."""

    def __init__(
        self,
        *,
        power_monitor: Optional[PowerMonitor] = None,
        dvfs_manager: Optional[DVFSManager] = None,
        market_manager: Optional[MPRMarketManager] = None,
    ) -> None:
        self._stop = threading.Event()
        self.power_monitor = power_monitor or PowerMonitor()
        self.dvfs_manager = dvfs_manager or DVFSManager()
        self.market_manager = market_manager or MPRMarketManager()

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._stop.is_set():
            raise RuntimeError("cannot start a controller that has been stopped")

        logging.info("MainController starting managers")
        for name, starter in (
            ("PowerMonitor", self.power_monitor.start),
            ("DVFSManager", self.dvfs_manager.start),
            ("MPRMarketManager", self.market_manager.start),
        ):
            try:
                starter()
            except Exception:  # pragma: no cover - defensive log
                logging.exception("Failed to start %s", name)

    def stop(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        logging.info("MainController stopping managers")
        for name, stopper in (
            ("MPRMarketManager", self.market_manager.stop),
            ("DVFSManager", self.dvfs_manager.stop),
            ("PowerMonitor", self.power_monitor.stop),
        ):
            try:
                stopper()
            except Exception:  # pragma: no cover - defensive log
                logging.exception("Failed to stop %s", name)

    def run(self) -> None:
        logging.info("MainController entering run loop")
        try:
            while not self._stop.is_set():
                events = []
                if self.power_monitor:
                    try:
                        events = self.power_monitor.consume_events()
                    except AttributeError:
                        events = []
                for evt in events:
                    logging.info("[PowerEvent] %s", evt.get("message", evt))
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("MainController interrupted")
            self.stop()


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the MPR main controller")
    parser.add_argument(
        "--pdu-map",
        type=Path,
        default=_DEFAULT_MAPPING if _DEFAULT_MAPPING.exists() else None,
        help="Path to the PDU outlet mapping JSON",
    )
    parser.add_argument(
        "--pdu-user",
        default=os.getenv("PDU_USER"),
        help="Username for APC PDU access (falls back to $PDU_USER)",
    )
    parser.add_argument(
        "--pdu-password",
        default=os.getenv("PDU_PASSWORD"),
        help="Password for APC PDU access (falls back to $PDU_PASSWORD)",
    )
    parser.add_argument(
        "--pdu-csv",
        type=Path,
        default=_DEFAULT_PDU_CSV,
        help="CSV file to append PDU readings",
    )
    parser.add_argument(
        "--events-csv",
        type=Path,
        default=_DEFAULT_EVENTS_CSV,
        help="CSV file to append overload events",
    )
    parser.add_argument(
        "--power-interval",
        type=float,
        default=1.0,
        help="Sampling interval for the power monitor (seconds)",
    )
    parser.add_argument(
        "--power-deadline",
        type=float,
        default=15.0,
        help="Deadline for reading a PDU batch (seconds)",
    )
    parser.add_argument(
        "--detect-overload",
        action="store_true",
        help="Enable overload detection heuristics",
    )
    parser.add_argument(
        "--shed-watts",
        type=float,
        default=0.0,
        help="Maximum watts to shed when overload handling is enabled",
    )
    parser.add_argument(
        "--shed-margin",
        type=float,
        default=0.0,
        help="Extra watts above threshold before shedding begins",
    )
    parser.add_argument(
        "--shed-delay",
        type=float,
        default=5.0,
        help="Seconds to wait before engaging load shedding",
    )
    parser.add_argument(
        "--disable-power-monitor",
        action="store_true",
        help="Do not start the power monitor",
    )
    parser.add_argument(
        "--disable-dvfs",
        action="store_true",
        help="Do not start the DVFS manager",
    )
    parser.add_argument(
        "--disable-market",
        action="store_true",
        help="Do not start the market manager",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("MAIN_CONTROLLER_LOG", "INFO"),
        help="Logging level (DEBUG, INFO, WARNING, ...)",
    )
    return parser


def _make_power_monitor(args: argparse.Namespace) -> Optional[PowerMonitor]:
    if args.disable_power_monitor:
        logging.info("Power monitor disabled via CLI flag")
        return None

    map_path = args.pdu_map
    user = args.pdu_user
    password = args.pdu_password
    csv_path = args.pdu_csv

    if not (map_path and user and password and csv_path):
        logging.warning(
            "Power monitor not configured: map/user/password/csv are required"
        )
        return PowerMonitor()

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    events_csv = args.events_csv
    if events_csv:
        events_csv.parent.mkdir(parents=True, exist_ok=True)

    return PowerMonitor(
        map_path=str(map_path),
        user=user,
        password=password,
        csv_path=str(csv_path),
        interval=args.power_interval,
        deadline=args.power_deadline,
        detect_overload=args.detect_overload,
        shed_watts=args.shed_watts,
        shed_margin=args.shed_margin,
        shed_delay=args.shed_delay,
        events_csv=str(events_csv) if events_csv else None,
    )


def _make_dvfs_manager(args: argparse.Namespace) -> Optional[DVFSManager]:
    if args.disable_dvfs:
        logging.info("DVFS manager disabled via CLI flag")
        return None
    return DVFSManager()


def _make_market_manager(args: argparse.Namespace) -> Optional[MPRMarketManager]:
    if args.disable_market:
        logging.info("Market manager disabled via CLI flag")
        return None
    return MPRMarketManager()


def _install_signal_handlers(controller: MainController) -> None:
    def _handler(signum, _frame):  # pragma: no cover - signal handling
        logging.info("Received signal %s, stopping controller", signum)
        controller.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    controller = MainController(
        power_monitor=_make_power_monitor(args) or PowerMonitor(),
        dvfs_manager=_make_dvfs_manager(args) or DVFSManager(),
        market_manager=_make_market_manager(args) or MPRMarketManager(),
    )

    _install_signal_handlers(controller)
    controller.start()
    try:
        controller.run()
    finally:
        controller.stop()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
