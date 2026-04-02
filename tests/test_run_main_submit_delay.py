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

from run_main import (
    apply_overload_end_reset_to_max_frequency,
    apply_startup_reset_to_max_frequency,
    parse_args,
    resolve_active_control_context,
    submit_jobs,
    submit_post_overload_end_jobs,
)
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

    def test_parse_args_accepts_post_overload_end_submission(self) -> None:
        argv = [
            "run_main.py",
            "--rank",
            "hpccg=2",
            "--post-overload-end-rank",
            "minimd=2",
            "--post-overload-end-rank",
            "comd=4",
            "--post-overload-end-wait-s",
            "30",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.post_overload_end_wait_s, 30.0)
        self.assertEqual(args.post_overload_end_job_ranks, {"minimd": 2, "comd": 4})
        self.assertEqual(
            args.post_overload_end_job_base_names,
            {"minimd": "minimd", "comd": "comd"},
        )

    def test_parse_args_rejects_negative_post_overload_end_wait(self) -> None:
        argv = [
            "run_main.py",
            "--rank",
            "hpccg=2",
            "--post-overload-end-rank",
            "minimd=2",
            "--post-overload-end-wait-s",
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

    def test_submit_post_overload_end_jobs_uses_unique_keys(self) -> None:
        scheduler = mock.Mock()
        scheduler.submit_jobs.return_value = pd.DataFrame(
            [{"job": "minimd__post_overload_end1", "job_id": "5001"}]
        )

        args = types.SimpleNamespace(
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
            dry_run=False,
            post_overload_end_job_ranks={"minimd": 2},
            post_overload_end_job_base_names={"minimd": "minimd"},
            post_overload_end_job_display_names={"minimd": "minimd"},
        )

        submit_df, job_ranks, job_base_names, job_display_names = submit_post_overload_end_jobs(
            scheduler,
            args,
            existing_job_keys={"hpccg", "minimd"},
        )

        scheduler.submit_jobs.assert_called_once()
        self.assertEqual(
            scheduler.submit_jobs.call_args.kwargs["job_ranks"],
            {"minimd__post_overload_end1": 2},
        )
        self.assertEqual(job_ranks, {"minimd__post_overload_end1": 2})
        self.assertEqual(job_base_names, {"minimd__post_overload_end1": "minimd"})
        self.assertEqual(
            job_display_names,
            {"minimd__post_overload_end1": "minimd__post_overload_end1"},
        )
        self.assertFalse(submit_df.empty)

    def test_apply_dvfs_forwards_cpufreq_sync_setting(self) -> None:
        scheduler = mock.Mock()
        controller_instance = mock.Mock()
        controller_instance.apply_plan_by_node.return_value = [
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
        controller_instance.apply_plan_by_node.assert_called_once()
        node_rules = controller_instance.apply_plan_by_node.call_args.kwargs["node_rules"]
        self.assertEqual(list(node_rules.keys()), ["ridlserver04"])
        self.assertEqual(node_rules["ridlserver04"][0]["core_numbers"], [0, 1])
        self.assertEqual(node_rules["ridlserver04"][0]["frequency_mhz"], 1800.0)
        self.assertIn("hpccg", used)

    def test_apply_dvfs_batches_same_node_targets_before_apply(self) -> None:
        scheduler = mock.Mock()
        controller_instance = mock.Mock()
        controller_instance.apply_plan_by_node.return_value = [
            {
                "node_name": "ridlserver04",
                "core_number": 0,
                "readback_avg_mhz": 1600.0,
                "readback_min_mhz": 1600.0,
                "readback_max_mhz": 1600.0,
                "readback_abs_error_mhz": 0.0,
                "verify_tolerance_mhz": 25.0,
                "verify_status": "PASS",
                "verify_reason": "within_tolerance",
                "status": "APPLIED",
            },
            {
                "node_name": "ridlserver04",
                "core_number": 2,
                "readback_avg_mhz": 1800.0,
                "readback_min_mhz": 1800.0,
                "readback_max_mhz": 1800.0,
                "readback_abs_error_mhz": 0.0,
                "verify_tolerance_mhz": 25.0,
                "verify_status": "PASS",
                "verify_reason": "within_tolerance",
                "status": "APPLIED",
            },
        ]
        args = types.SimpleNamespace(
            skip_dvfs_apply=False,
            dvfs_ssh_user=None,
            dvfs_verify_tol_mhz=25.0,
            cpufreq_sync=True,
            dvfs_step_mhz=100.0,
            dry_run=False,
        )
        allocations = {
            "hpccg": {
                "cores_by_node": {"ridlserver04": [0, 1]},
            },
            "comd": {
                "cores_by_node": {"ridlserver04": [2, 3]},
            },
        }

        with mock.patch("run_main.DVFSController", return_value=controller_instance):
            apply_dvfs_with_allocations(
                scheduler=scheduler,
                args=args,
                job_freq_mhz={"hpccg": 1600.0, "comd": 1800.0},
                allocations_hint=allocations,
            )

        controller_instance.apply_plan_by_node.assert_called_once()
        node_rules = controller_instance.apply_plan_by_node.call_args.kwargs["node_rules"]
        self.assertEqual(list(node_rules.keys()), ["ridlserver04"])
        self.assertEqual(
            node_rules["ridlserver04"],
            [
                {"core_numbers": [0, 1], "frequency_mhz": 1600.0},
                {"core_numbers": [2, 3], "frequency_mhz": 1800.0},
            ],
        )

    def test_resolve_active_control_context_excludes_completed_and_pending_jobs(self) -> None:
        scheduler = mock.Mock()
        scheduler.get_submission_history.return_value = pd.DataFrame(
            [
                {"job_key": "xsbenchmpi", "job_id": "4424", "status": "SUBMITTED"},
                {"job_key": "minimd", "job_id": "4425", "status": "SUBMITTED"},
                {"job_key": "comd", "job_id": "4426", "status": "SUBMITTED"},
                {"job_key": "xsbenchmpi__post_overload_end1", "job_id": "4428", "status": "SUBMITTED"},
                {"job_key": "comd__post_overload_end1", "job_id": "4429", "status": "SUBMITTED"},
            ]
        )
        scheduler.get_submitted_job_allocations.return_value = {
            "minimd": {
                "cores_by_node": {"ridlserver05": [0, 1]},
                "nodes": ["ridlserver05"],
            },
            "comd": {
                "cores_by_node": {},
                "nodes": ["ridlserver11"],
            },
            "xsbenchmpi__post_overload_end1": {
                "cores_by_node": {"ridlserver04": [0, 1], "ridlserver05": [0, 1]},
                "nodes": ["ridlserver04", "ridlserver05"],
            },
        }

        perf_df = pd.DataFrame(
            {
                "Resource Reduction": [0.0, 0.1],
                "Extra Execution": [0.0, 1.0],
                "Power": [100.0, 90.0],
            }
        )
        job_perf_data = {
            "xsbenchmpi": perf_df,
            "minimd": perf_df,
            "comd": perf_df,
            "xsbenchmpi__post_overload_end1": perf_df,
            "comd__post_overload_end1": perf_df,
        }
        job_states = {
            "4424": "COMPLETED",
            "4425": "RUNNING",
            "4426": "CONFIGURING",
            "4428": "RUNNING",
            "4429": "PENDING",
        }

        active_perf_data, active_allocations = resolve_active_control_context(
            scheduler=scheduler,
            tracked_job_keys=list(job_perf_data.keys()),
            job_perf_data=job_perf_data,
            job_states=job_states,
        )

        scheduler.get_submitted_job_allocations.assert_called_once_with(
            job_names=["minimd", "comd", "xsbenchmpi__post_overload_end1"],
            latest_only=True,
        )
        self.assertEqual(
            list(active_perf_data.keys()),
            ["minimd", "xsbenchmpi__post_overload_end1"],
        )
        self.assertEqual(
            list(active_allocations.keys()),
            ["minimd", "xsbenchmpi__post_overload_end1"],
        )
        self.assertNotIn("xsbenchmpi", active_perf_data)
        self.assertNotIn("comd__post_overload_end1", active_perf_data)
        self.assertNotIn("comd", active_perf_data)

    def test_resolve_active_control_context_skips_allocation_lookup_without_eligible_jobs(self) -> None:
        scheduler = mock.Mock()
        scheduler.get_submission_history.return_value = pd.DataFrame(
            [
                {"job_key": "xsbenchmpi", "job_id": "4424", "status": "SUBMITTED"},
                {"job_key": "comd__post_overload_end1", "job_id": "4429", "status": "SUBMITTED"},
            ]
        )

        active_perf_data, active_allocations = resolve_active_control_context(
            scheduler=scheduler,
            tracked_job_keys=["xsbenchmpi", "comd__post_overload_end1"],
            job_perf_data={"xsbenchmpi": pd.DataFrame(), "comd__post_overload_end1": pd.DataFrame()},
            job_states={"4424": "COMPLETED", "4429": "PENDING"},
        )

        scheduler.get_submitted_job_allocations.assert_not_called()
        self.assertEqual(active_perf_data, {})
        self.assertEqual(active_allocations, {})

    def test_reset_targets_preserve_completed_jobs_from_prior_reduction(self) -> None:
        scheduler = mock.Mock()
        scheduler.get_submission_history.return_value = pd.DataFrame(
            [
                {"job_key": "xsbenchmpi", "job_id": "4430", "status": "SUBMITTED"},
                {"job_key": "comd", "job_id": "4432", "status": "SUBMITTED"},
                {"job_key": "hpccg", "job_id": "4433", "status": "SUBMITTED"},
            ]
        )
        scheduler.get_submitted_job_allocations.return_value = {
            "comd": {
                "cores_by_node": {"ridlserver11": [0, 1]},
                "nodes": ["ridlserver11"],
            },
            "hpccg": {
                "cores_by_node": {"ridlserver12": [0, 1]},
                "nodes": ["ridlserver12"],
            },
        }

        perf_df = pd.DataFrame(
            {
                "Resource Reduction": [0.0, 0.1],
                "Extra Execution": [0.0, 1.0],
                "Power": [100.0, 90.0],
            }
        )
        active_perf_data, active_allocations = resolve_active_control_context(
            scheduler=scheduler,
            tracked_job_keys=["xsbenchmpi", "comd", "hpccg"],
            job_perf_data={"xsbenchmpi": perf_df, "comd": perf_df, "hpccg": perf_df},
            job_states={"4430": "COMPLETED", "4432": "RUNNING", "4433": "RUNNING"},
        )

        self.assertEqual(list(active_perf_data.keys()), ["comd", "hpccg"])
        self.assertEqual(list(active_allocations.keys()), ["comd", "hpccg"])

    def test_apply_startup_reset_to_max_frequency_targets_all_mapped_nodes(self) -> None:
        controller_instance = mock.Mock()
        controller_instance.concurrency = 8
        controller_instance.apply_plan_by_node.return_value = [
            {
                "node_name": "ridlserver04",
                "core_number": 0,
                "readback_avg_mhz": 2400.0,
                "readback_min_mhz": 2400.0,
                "readback_max_mhz": 2400.0,
                "readback_abs_error_mhz": 0.0,
                "verify_tolerance_mhz": 25.0,
                "verify_status": "PASS",
                "verify_reason": "within_tolerance",
                "status": "DRY_RUN",
            },
            {
                "node_name": "ridlserver05",
                "core_number": 0,
                "readback_avg_mhz": 2400.0,
                "readback_min_mhz": 2400.0,
                "readback_max_mhz": 2400.0,
                "readback_abs_error_mhz": 0.0,
                "verify_tolerance_mhz": 25.0,
                "verify_status": "PASS",
                "verify_reason": "within_tolerance",
                "status": "DRY_RUN",
            },
        ]
        args = types.SimpleNamespace(
            skip_dvfs_apply=False,
            pdu_map=None,
            dvfs_ssh_user="ridl",
            dvfs_verify_tol_mhz=25.0,
            cpufreq_sync=True,
            max_freq_mhz=2400.0,
            dry_run=True,
        )

        with (
            mock.patch("run_main.get_mapped_reset_nodes", return_value=["ridlserver04", "ridlserver05"]),
            mock.patch("run_main.discover_node_core_ids", side_effect=[[0, 1], [0, 1]]) as mock_discover,
            mock.patch("run_main.DVFSController", return_value=controller_instance) as mock_controller,
            mock.patch("run_main.print_dvfs_apply_summary") as mock_print_summary,
        ):
            apply_startup_reset_to_max_frequency(args)

        mock_controller.assert_called_once()
        self.assertEqual(mock_discover.call_count, 2)
        controller_instance.apply_plan_by_node.assert_called_once()
        node_rules = controller_instance.apply_plan_by_node.call_args.kwargs["node_rules"]
        self.assertEqual(
            node_rules,
            {
                "ridlserver04": [{"core_numbers": [0, 1], "frequency_mhz": 2400.0}],
                "ridlserver05": [{"core_numbers": [0, 1], "frequency_mhz": 2400.0}],
            },
        )
        self.assertTrue(controller_instance.apply_plan_by_node.call_args.kwargs["dry_run"])
        mock_print_summary.assert_called_once()

    def test_apply_overload_end_reset_to_max_frequency_targets_all_mapped_nodes(self) -> None:
        with mock.patch("run_main.apply_full_frequency_reset_to_all_nodes") as mock_apply:
            apply_overload_end_reset_to_max_frequency(
                types.SimpleNamespace(
                    skip_dvfs_apply=False,
                    pdu_map=None,
                    dvfs_ssh_user="ridl",
                    dvfs_verify_tol_mhz=25.0,
                    cpufreq_sync=True,
                    max_freq_mhz=2400.0,
                    dry_run=True,
                )
            )

        mock_apply.assert_called_once()
        _, kwargs = mock_apply.call_args
        self.assertEqual(kwargs["action_label"], "Overload-end DVFS reset")
        self.assertEqual(kwargs["summary_job_label"], "overload_end_reset")


if __name__ == "__main__":
    unittest.main(verbosity=2)
