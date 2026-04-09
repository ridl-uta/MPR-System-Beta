#!/usr/bin/env python3
"""Regression tests for record_performance controller helpers."""

from __future__ import annotations

import types
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
import csv
import tempfile
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if "telnetlib" not in sys.modules:
    sys.modules["telnetlib"] = types.ModuleType("telnetlib")

import main_controller
from main_controller import MainController


class MainControllerRecordingTest(unittest.TestCase):
    def test_record_mode_passes_submit_env_to_builder(self) -> None:
        dummy_manager = types.SimpleNamespace(start=lambda: None, stop=lambda: None)
        controller = MainController(
            power_monitor=dummy_manager,
            dvfs_manager=dummy_manager,
            market_manager=dummy_manager,
            record_interval_mhz=100,
            record_submit_env=["JOB_REPEAT=2"],
        )

        with mock.patch("main_controller.build_sbatch_variations", return_value=[]) as mock_build:
            controller._run_record_performance()

        mock_build.assert_called_once_with(
            nodelist=None,
            cores_per_rank=10,
            exclude=None,
            submit_env=["JOB_REPEAT=2"],
        )

    def test_cores_required_from_short_flags(self) -> None:
        controller = MainController(record_interval_mhz=100)
        cmd = [
            "python3",
            "utilities/submit_benchmark.py",
            "-N",
            "2",
            "-n",
            "4",
            "-c",
            "10",
        ]
        self.assertEqual(controller._cores_required_from_cmd(cmd), 40)

    def test_cores_required_from_generated_multinode_shape(self) -> None:
        controller = MainController(record_interval_mhz=100)
        cmd = [
            "python3",
            "utilities/submit_benchmark.py",
            "-N",
            "2",
            "-n",
            "2",
            "-c",
            "10",
            "--ntasks-per-node",
            "1",
        ]
        self.assertEqual(controller._cores_required_from_cmd(cmd), 20)

    def test_main_rejects_nonpositive_record_interval(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main_controller.main(["--record-interval-mhz", "0"])
        self.assertEqual(ctx.exception.code, 2)

    def test_finalize_record_uses_applied_frequency_only(self) -> None:
        controller = MainController(record_interval_mhz=100)
        controller._get_job_nodes = lambda _job_id: None  # type: ignore[method-assign]
        controller._compute_power_stat = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

        entry = {
            "job_id": "123",
            "cmd": ["python3", "utilities/submit_benchmark.py", "--ntasks=2"],
            "freq_mhz": 1600.0,
            "requested_freq_mhz": 1600.0,
            "applied_freq_mhz": 1600.0,
            "reduction": 0.25,
            "submitted_at": datetime.now(timezone.utc),
            "start_time": datetime.now(timezone.utc),
            "dvfs_apply_status": "APPLIED",
            "dvfs_apply_error": None,
        }

        record = controller._finalize_record(entry)
        self.assertEqual(record["freq_mhz"], 1600.0)
        self.assertEqual(record["requested_freq_mhz"], 1600.0)
        self.assertEqual(record["applied_freq_mhz"], 1600.0)
        self.assertEqual(record["dvfs_apply_status"], "APPLIED")
        self.assertIsNone(record["dvfs_apply_error"])

    def test_finalize_record_keeps_failed_dvfs_frequency_blank(self) -> None:
        controller = MainController(record_interval_mhz=100)
        controller._get_job_nodes = lambda _job_id: None  # type: ignore[method-assign]
        controller._compute_power_stat = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

        entry = {
            "job_id": "123",
            "cmd": ["python3", "utilities/submit_benchmark.py", "--ntasks=2"],
            "freq_mhz": None,
            "requested_freq_mhz": 1600.0,
            "applied_freq_mhz": None,
            "reduction": 0.25,
            "submitted_at": datetime.now(timezone.utc),
            "start_time": None,
            "dvfs_apply_status": "FAILED",
            "dvfs_apply_error": "mock apply failure",
        }

        record = controller._finalize_record(entry)
        self.assertIsNone(record["freq_mhz"])
        self.assertEqual(record["requested_freq_mhz"], 1600.0)
        self.assertIsNone(record["applied_freq_mhz"])
        self.assertEqual(record["dvfs_apply_status"], "FAILED")
        self.assertEqual(record["dvfs_apply_error"], "mock apply failure")

    def test_finalize_record_uses_trimmed_median_power_stat(self) -> None:
        controller = MainController(
            record_interval_mhz=100,
            record_power_stat="median",
            record_trim_start_seconds=25.0,
            record_trim_end_seconds=10.0,
        )
        controller._get_job_nodes = lambda _job_id: ("ridlserver04", ["ridlserver04"], 2)  # type: ignore[method-assign]
        controller.idle_power_baseline = {"ridlserver04": 83.0}

        expected_start = datetime(2026, 4, 9, 6, 36, 51, tzinfo=timezone.utc)
        expected_end = datetime(2026, 4, 9, 6, 38, 29, tzinfo=timezone.utc)

        def fake_compute_power_stat(start, end, node_filter=None, **kwargs):
            self.assertEqual(start, expected_start)
            self.assertEqual(end, expected_end)
            self.assertEqual(node_filter, ["ridlserver04"])
            self.assertEqual(kwargs["stat"], "median")
            self.assertEqual(kwargs["trim_start_seconds"], 25.0)
            self.assertEqual(kwargs["trim_end_seconds"], 10.0)
            return 227.0

        controller._compute_power_stat = fake_compute_power_stat  # type: ignore[method-assign]

        entry = {
            "job_id": "4529",
            "cmd": ["python3", "utilities/submit_benchmark.py", "--ntasks=2", "/shared/src/HPCCG"],
            "freq_mhz": 2400.0,
            "requested_freq_mhz": 2400.0,
            "applied_freq_mhz": 2400.0,
            "reduction": 0.0,
            "submitted_at": expected_start,
            "start_time": expected_start,
            "dvfs_apply_status": "APPLIED",
            "dvfs_apply_error": None,
        }

        with mock.patch("main_controller.datetime") as mock_datetime:
            mock_datetime.now.return_value = expected_end
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            record = controller._finalize_record(entry)

        self.assertEqual(record["avg_power_w"], 227.0)
        self.assertEqual(record["net_avg_power_w"], 144.0)
        self.assertEqual(record["power_stat"], "median")
        self.assertEqual(record["power_trim_start_s"], 25.0)
        self.assertEqual(record["power_trim_end_s"], 10.0)

    def test_compute_power_stat_trimmed_median_matches_plateau(self) -> None:
        start = datetime(2026, 4, 9, 6, 0, 0, tzinfo=timezone.utc)
        with tempfile.NamedTemporaryFile("w", newline="", delete=False) as tmp:
            writer = csv.writer(tmp)
            writer.writerow(["timestamp", "ridlserver04", "total_watts"])
            for second in range(40):
                ts = start.replace(second=second)
                if second < 10:
                    power = 140.0
                elif second < 35:
                    power = 225.0
                else:
                    power = 100.0
                writer.writerow([ts.isoformat(), power, power])
            csv_path = Path(tmp.name)

        try:
            power_monitor = types.SimpleNamespace(csv_path=str(csv_path))
            controller = MainController(power_monitor=power_monitor, record_interval_mhz=100)
            end = start.replace(second=39)

            full_avg = controller._compute_power_stat(start, end, ["ridlserver04"], stat="avg")
            full_median = controller._compute_power_stat(start, end, ["ridlserver04"], stat="median")
            trimmed_avg = controller._compute_power_stat(
                start,
                end,
                ["ridlserver04"],
                stat="avg",
                trim_start_seconds=10.0,
                trim_end_seconds=5.0,
            )
            trimmed_median = controller._compute_power_stat(
                start,
                end,
                ["ridlserver04"],
                stat="median",
                trim_start_seconds=10.0,
                trim_end_seconds=5.0,
            )
            fallback_median = controller._compute_power_stat(
                start,
                end,
                ["ridlserver04"],
                stat="median",
                trim_start_seconds=30.0,
                trim_end_seconds=20.0,
            )

            self.assertEqual(full_avg, 188.125)
            self.assertEqual(full_median, 225.0)
            self.assertEqual(trimmed_avg, 225.0)
            self.assertEqual(trimmed_median, 225.0)
            self.assertEqual(fallback_median, 225.0)
        finally:
            csv_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
