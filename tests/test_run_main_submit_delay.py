from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import pandas as pd

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from run_main import parse_args, submit_jobs
from run_main import apply_dvfs_with_allocations


class TestRunMainPreSubmitWait(unittest.TestCase):
    def test_parse_args_defaults_cpufreq_sync_on(self) -> None:
        argv = [
            "run_main.py",
            "--rank",
            "hpccg=2",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertTrue(args.cpufreq_sync)

    def test_parse_args_accepts_no_cpufreq_sync(self) -> None:
        argv = [
            "run_main.py",
            "--rank",
            "hpccg=2",
            "--no-cpufreq-sync",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertFalse(args.cpufreq_sync)

    def test_parse_args_accepts_pre_submit_wait(self) -> None:
        argv = [
            "run_main.py",
            "--rank",
            "hpccg=2",
            "--pre-submit-wait-s",
            "15",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.pre_submit_wait_s, 15.0)

    def test_parse_args_rejects_negative_pre_submit_wait(self) -> None:
        argv = [
            "run_main.py",
            "--rank",
            "hpccg=2",
            "--pre-submit-wait-s",
            "-1",
        ]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as ctx:
                parse_args()
        self.assertEqual(ctx.exception.code, 2)

    def test_submit_jobs_sleeps_once_before_first_submission(self) -> None:
        scheduler = mock.Mock()
        scheduler.submit_jobs.return_value = pd.DataFrame()

        args = types.SimpleNamespace(
            skip_submit=False,
            dry_run=False,
            pre_submit_wait_s=12.5,
            submit_interval_s=0.0,
            cpus_per_rank=10,
            ranks_per_node=2,
            partition="debug",
            time_limit="00:30:00",
            nodelist=None,
            exclude=None,
            slurm_output=None,
            mpi_iface=None,
            submit_env_map={},
            job_args_map={},
            disable_rank_profiles=False,
        )

        with mock.patch("run_main.time.sleep") as mock_sleep:
            submit_jobs(
                scheduler,
                args,
                {"hpccg": 2},
                {"hpccg": "hpccg"},
                {"hpccg": "hpccg"},
            )

        mock_sleep.assert_called_once_with(12.5)
        scheduler.submit_jobs.assert_called_once()

    def test_submit_jobs_skips_wait_during_dry_run(self) -> None:
        scheduler = mock.Mock()
        scheduler.submit_jobs.return_value = pd.DataFrame()

        args = types.SimpleNamespace(
            skip_submit=False,
            dry_run=True,
            pre_submit_wait_s=12.5,
            submit_interval_s=0.0,
            cpus_per_rank=10,
            ranks_per_node=2,
            partition="debug",
            time_limit="00:30:00",
            nodelist=None,
            exclude=None,
            slurm_output=None,
            mpi_iface=None,
            submit_env_map={},
            job_args_map={},
            disable_rank_profiles=False,
        )

        with mock.patch("run_main.time.sleep") as mock_sleep:
            submit_jobs(
                scheduler,
                args,
                {"hpccg": 2},
                {"hpccg": "hpccg"},
                {"hpccg": "hpccg"},
            )

        mock_sleep.assert_not_called()
        scheduler.submit_jobs.assert_called_once()

    def test_apply_dvfs_forwards_cpufreq_sync_setting(self) -> None:
        scheduler = mock.Mock()
        controller_instance = mock.Mock()
        controller_instance.apply_to_job_allocation.return_value = [
            {
                "node_name": "ridlserver04",
                "core_number": 0,
                "readback_avg_mhz": 1800.0,
                "readback_min_mhz": 1800.0,
                "readback_max_mhz": 1800.0,
                "readback_abs_error_mhz": 0.0,
                "verify_tolerance_mhz": 25.0,
                "verify_status": "PASS",
                "verify_reason": "within_tolerance",
                "status": "APPLIED",
            }
        ]
        args = types.SimpleNamespace(
            skip_dvfs_apply=False,
            dvfs_ssh_user=None,
            dvfs_verify_tol_mhz=25.0,
            cpufreq_sync=False,
            dvfs_step_mhz=100.0,
            dry_run=False,
        )
        allocations = {
            "hpccg": {
                "cores_by_node": {"ridlserver04": [0, 1]},
            }
        }

        with mock.patch("run_main.DVFSController", return_value=controller_instance) as mock_controller:
            used = apply_dvfs_with_allocations(
                scheduler=scheduler,
                args=args,
                job_freq_mhz={"hpccg": 1892.0},
                allocations_hint=allocations,
            )

        mock_controller.assert_called_once()
        self.assertFalse(mock_controller.call_args.kwargs["cpufreq_sync"])
        controller_instance.apply_to_job_allocation.assert_called_once()
        self.assertIn("hpccg", used)


if __name__ == "__main__":
    unittest.main(verbosity=2)
