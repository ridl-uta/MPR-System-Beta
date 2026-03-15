#!/usr/bin/env python3
"""Regression tests for record_performance controller helpers."""

from __future__ import annotations

import types
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
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
        controller._compute_avg_power = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

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
        controller._compute_avg_power = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
