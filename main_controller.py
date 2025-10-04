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
import csv
import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from datetime import datetime, timezone

from managers import DVFSManager, MPRMarketManager, PowerMonitor
from dvfs import list_running_slurm_jobs, build_sbatch_variations

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
        mode: str = "run_experiment",
    ) -> None:
        self._stop = threading.Event()
        self.power_monitor = power_monitor or PowerMonitor()
        self.dvfs_manager = dvfs_manager or DVFSManager()
        self.market_manager = market_manager or MPRMarketManager()
        self.mode = mode
        self._performance_results = []

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
        logging.info("MainController entering run loop (%s mode)", self.mode)
        try:
            if self.mode == "record_performance":
                self._run_record_performance()
            else:
                self._run_experiment()
        except KeyboardInterrupt:
            logging.info("MainController interrupted")
            self.stop()

    def _run_experiment(self) -> None:
        while not self._stop.is_set():
            events = []
            if self.power_monitor:
                try:
                    events = self.power_monitor.consume_events()
                except AttributeError:
                    events = []
            for evt in events:
                self._handle_power_event(evt)
            time.sleep(1)

    def _run_record_performance(self) -> None:
        commands = build_sbatch_variations()
        if not commands:
            logging.warning("[Record] No Slurm scripts found; exiting record mode")
            return

        if not self.dvfs_manager:
            logging.warning("[Record] DVFS manager unavailable; cannot record")
            return

        freq_targets_mhz = [2200, 2000, 1800, 1600, 1500]
        results = []

        while not self._stop.is_set():
            for freq_mhz in freq_targets_mhz:
                reduction = self._freq_to_reduction(freq_mhz)
                for cmd in commands:
                    if self._stop.is_set():
                        break
                    record = self._execute_recording_cycle(cmd, reduction, freq_mhz)
                    if record:
                        results.append(record)
            # After one full sweep, log summary and sleep before next
            if results:
                logging.info("[Record] Completed sweep; %d records", len(results))
                for rec in results:
                    logging.info(
                        "[Record] job=%s freq=%.1fMHz duration=%.1fs avg_power=%s", \
                        rec.get("job_name"), rec.get("freq_mhz"), rec.get("duration_s"), rec.get("avg_power_w")
                    )
                self._performance_results.extend(results)
                results = []
            time.sleep(5)

    def _handle_power_event(self, event: dict) -> None:
        """React to power monitor events; extend with site-specific logic."""

        name = (event.get("event") or "").upper()
        message = event.get("message") or name

        if name == "OVERLOAD_START":
            logging.warning("[PowerEvent][START] %s", message)
            self._trigger_overload_reduction(event)
        elif name == "OVERLOAD_HANDLED":
            logging.info("[PowerEvent][HANDLED] %s", message)
            # TODO: optional: confirm mitigation success or log metrics
        elif name == "OVERLOAD_END":
            logging.info("[PowerEvent][END] %s", message)
            # TODO: optional: restore state if changes were made at start
        else:
            logging.info("[PowerEvent] %s", message)

    def _trigger_overload_reduction(self, event: dict) -> None:
        """Kick off a DVFS reduction when overload starts."""

        if not self.dvfs_manager:
            logging.warning("[PowerEvent] DVFS manager unavailable; cannot reduce load")
            return

        try:
            jobs = list_running_slurm_jobs()
        except Exception as exc:
            logging.error("[PowerEvent] Failed to list Slurm jobs: %s", exc)
            return

        if not jobs:
            logging.info("[PowerEvent] No running jobs to reduce")
            return

        reductions = self.market_manager.plan_reductions(jobs)
        if not reductions:
            logging.info("[PowerEvent] No reduction plan generated")
            return

        logging.info("[PowerEvent] Applying DVFS reductions: %s", reductions)
        for job_id, reduction in reductions.items():
            try:
                self.dvfs_manager.submit_reduction(job_id, reduction)
            except Exception as exc:
                logging.error("[PowerEvent] Failed to submit reduction for job %s: %s", job_id, exc)

    def _freq_to_reduction(self, freq_mhz: float) -> float:
        if not self.dvfs_manager:
            return 0.0
        max_hz = self.dvfs_manager.max_freq_mhz * 1e6
        min_hz = self.dvfs_manager.min_freq_mhz * 1e6
        target_hz = max(freq_mhz * 1e6, min_hz)
        target_hz = min(target_hz, max_hz)
        if max_hz <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - target_hz / max_hz))

    def _execute_recording_cycle(self, cmd: list[str], reduction: float, freq_mhz: float) -> Optional[dict]:
        submit = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if submit.returncode != 0 or "Submitted batch job" not in submit.stdout:
            logging.error("[Record] sbatch failed: %s %s", submit.stdout.strip(), submit.stderr.strip())
            return None

        job_id = submit.stdout.strip().split()[-1]
        start_time = datetime.now(timezone.utc)

        try:
            self.dvfs_manager.submit_reduction(job_id, reduction)
        except Exception as exc:
            logging.error("[Record] Failed to apply reduction for job %s: %s", job_id, exc)

        self._wait_for_job_completion(job_id)

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        avg_power = self._compute_avg_power(start_time, end_time)

        record = {
            "job_id": job_id,
            "job_name": cmd[-1],
            "freq_mhz": freq_mhz,
            "reduction": reduction,
            "duration_s": duration,
            "avg_power_w": avg_power,
        }
        logging.info(
            "[Record] Completed job %s freq=%.1fMHz duration=%.1fs avg_power=%s",
            job_id,
            freq_mhz,
            duration,
            avg_power,
        )
        return record

    def _wait_for_job_completion(self, job_id: str) -> None:
        while not self._stop.is_set():
            proc = subprocess.run(
                ["squeue", "-h", "-j", job_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                break
            time.sleep(5)

    def _compute_avg_power(self, start: datetime, end: datetime) -> Optional[float]:
        if not self.power_monitor or not getattr(self.power_monitor, "csv_path", None):
            return None

        csv_path = Path(self.power_monitor.csv_path)  # type: ignore[attr-defined]
        if not csv_path.exists():
            return None

        try:
            with csv_path.open() as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    return None
                try:
                    total_idx = header.index("total_watts")
                except ValueError:
                    total_idx = len(header) - 1

                samples = []
                for row in reader:
                    if len(row) <= total_idx:
                        continue
                    ts_text = row[0]
                    try:
                        ts = datetime.fromisoformat(ts_text)
                    except ValueError:
                        continue
                    ts = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
                    if ts < start or ts > end:
                        continue
                    try:
                        samples.append(float(row[total_idx]))
                    except ValueError:
                        continue

                if not samples:
                    return None
                return sum(samples) / len(samples)
        except Exception as exc:
            logging.error("[Record] Failed to compute avg power: %s", exc)
            return None


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
        default=os.getenv("PDU_USER", "apc"),
        help="Username for APC PDU access (falls back to $PDU_USER)",
    )
    parser.add_argument(
        "--pdu-password",
        default=os.getenv("PDU_PASSWORD", "ridl123"),
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
        "--threshold-w",
        type=float,
        default=900.0,
        help="High threshold for overload detection (watts)",
    )
    parser.add_argument(
        "--hysteresis-w",
        type=float,
        default=20.0,
        help="Watts to drop below threshold before ending overload",
    )
    parser.add_argument(
        "--min-over",
        type=int,
        default=5,
        help="Seconds above threshold required to declare overload",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=30,
        help="Seconds after overload ends before re-triggering",
    )
    parser.add_argument(
        "--spike-threshold-w",
        type=float,
        default=None,
        help="Spike threshold watts (optional)",
    )
    parser.add_argument(
        "--spike-margin-w",
        type=float,
        default=50.0,
        help="Spike margin watts",
    )
    parser.add_argument(
        "--spike-persistence",
        type=float,
        default=1.0,
        help="Seconds a spike must persist before warning",
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
    parser.add_argument(
        "--mode",
        choices=["run_experiment", "record_performance"],
        default="run_experiment",
        help="Controller mode: run_experiment or record_performance",
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
        threshold_w=args.threshold_w,
        hysteresis_w=args.hysteresis_w,
        min_over_s=args.min_over,
        cooldown_s=args.cooldown,
        spike_threshold_w=args.spike_threshold_w,
        spike_margin_w=args.spike_margin_w,
        spike_persistence_s=args.spike_persistence,
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
