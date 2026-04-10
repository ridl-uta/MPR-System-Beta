from __future__ import annotations

import queue
import types
import unittest
from unittest import mock

import pandas as pd

from run_main import (
    apply_overload_reduction,
    compute_baseline_power_max,
    compute_mpr_target_reduction_w,
    compute_reapply_target_reduction_w_from_baseline,
    estimate_fixed_overhead_w,
    measure_idle_power_baseline,
    resolve_reapply_fixed_overhead_w,
    resolve_target_reduction_w,
    run_event_driven_control_loop,
    wait_before_market_start,
)


class TestRunMainTargetOffset(unittest.TestCase):
    @staticmethod
    def _job_df(power_values: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Resource Reduction": [0.0, 0.5],
                "Extra Execution": [0.0, 1.0],
                "Power": power_values,
            }
        )

    def test_compute_target_reduction_adds_offset_when_overloaded(self) -> None:
        self.assertEqual(compute_mpr_target_reduction_w(80.0, 20.0), 100.0)

    def test_compute_target_reduction_does_not_apply_offset_without_overload(self) -> None:
        self.assertEqual(compute_mpr_target_reduction_w(0.0, 20.0), 0.0)

    def test_compute_target_reduction_allows_negative_offset_but_clamps(self) -> None:
        self.assertEqual(compute_mpr_target_reduction_w(80.0, -200.0), 0.0)

    def test_compute_baseline_power_max_sums_active_job_models(self) -> None:
        self.assertEqual(
            compute_baseline_power_max(
                {
                    "job_a": self._job_df([120.0, 90.0]),
                    "job_b": self._job_df([80.0, 60.0]),
                }
            ),
            200.0,
        )

    def test_estimate_fixed_overhead_uses_current_saved_power(self) -> None:
        self.assertEqual(
            estimate_fixed_overhead_w(
                current_power_w=794.0,
                active_jobs_power_max_w=300.0,
                current_saved_power_w=64.0,
            ),
            558.0,
        )

    def test_resolve_reapply_fixed_overhead_prefers_idle_baseline(self) -> None:
        fixed_overhead_w, source = resolve_reapply_fixed_overhead_w(
            idle_power_w=160.0,
            current_power_w=794.0,
            active_jobs_power_max_w=300.0,
            current_saved_power_w=64.0,
        )
        self.assertEqual(fixed_overhead_w, 160.0)
        self.assertEqual(source, "idle_baseline")

    def test_resolve_reapply_fixed_overhead_falls_back_to_dynamic_estimate(self) -> None:
        fixed_overhead_w, source = resolve_reapply_fixed_overhead_w(
            idle_power_w=None,
            current_power_w=794.0,
            active_jobs_power_max_w=300.0,
            current_saved_power_w=64.0,
        )
        self.assertEqual(fixed_overhead_w, 558.0)
        self.assertEqual(source, "dynamic_estimate")

    def test_compute_reapply_target_uses_active_job_baseline_formula(self) -> None:
        self.assertEqual(
            compute_reapply_target_reduction_w_from_baseline(
                active_jobs_power_max_w=320.0,
                fixed_overhead_w=558.0,
                target_capacity_w=750.0,
                target_offset_w=20.0,
            ),
            148.0,
        )

    def test_resolve_target_reduction_uses_cli_offset(self) -> None:
        args = types.SimpleNamespace(
            current_power_w=920.0,
            target_capacity_w=840.0,
            target_offset_w=20.0,
            power_startup_wait_s=0.0,
        )
        target_reduction_w, current_power_w, overload_w = resolve_target_reduction_w(args, None)
        self.assertEqual(current_power_w, 920.0)
        self.assertEqual(overload_w, 80.0)
        self.assertEqual(target_reduction_w, 100.0)

    def test_apply_overload_reduction_uses_override_target(self) -> None:
        args = types.SimpleNamespace(
            target_capacity_w=750.0,
            target_offset_w=20.0,
            max_freq_mhz=2400.0,
            skip_dvfs_apply=False,
            dvfs_ssh_user=None,
            dvfs_verify_tol_mhz=25.0,
            cpufreq_sync=True,
            dry_run=False,
        )
        scheduler = mock.Mock()
        market_result = {
            "market_failed": False,
            "final_reduction_w": 108.0,
            "final_bids": {"comd": 1.0},
            "converged": True,
            "convergence_mode": "cycle",
            "final_q": 1.0,
            "final_residual_w": 0.0,
            "negotiation_time_s": 1.0,
        }
        job_perf_data = {"comd": pd.DataFrame()}

        with (
            mock.patch("run_main.run_market", return_value=market_result) as mock_market,
            mock.patch(
                "run_main.compute_frequency_targets",
                return_value=(
                    pd.DataFrame(
                        [
                            {
                                "job": "comd",
                                "power_saved_w": 148.0,
                            }
                        ]
                    ),
                    {"comd": 2200.0},
                ),
            ),
            mock.patch(
                "run_main.apply_dvfs_with_allocations",
                return_value={"comd": {"cores_by_node": {"ridlserver11": [0, 1]}}},
            ),
        ):
            applied, allocations_cache, applied_reduction_w, saved_power_by_job = apply_overload_reduction(
                args=args,
                scheduler=scheduler,
                job_perf_data=job_perf_data,
                current_power_w=794.0,
                base_required_reduction_w=44.0,
                allocations_cache={},
                target_reduction_w_override=148.0,
            )

        mock_market.assert_called_once_with(args, job_perf_data, 148.0)
        self.assertTrue(applied)
        self.assertEqual(applied_reduction_w, 148.0)
        self.assertEqual(saved_power_by_job, {"comd": 148.0})
        self.assertIn("comd", allocations_cache)


class _StubMonitor:
    def __init__(self, sample: dict[str, object] | None) -> None:
        self._sample = sample

    def get_last_sample(self) -> dict[str, object] | None:
        return self._sample


class _SequenceMonitor:
    def __init__(self, samples: list[dict[str, object] | None]) -> None:
        self._samples = list(samples)
        self._last = self._samples[-1] if self._samples else None

    def get_last_sample(self) -> dict[str, object] | None:
        if self._samples:
            self._last = self._samples.pop(0)
        return self._last


class TestWaitBeforeMarketStart(unittest.TestCase):
    def test_ignores_ramp_reduction_during_wait_and_uses_spike_peak(self) -> None:
        overload_event_queue: queue.Queue[dict[str, object]] = queue.Queue()
        overload_event_queue.put(
            {
                "event": "SPIKE_WARNING",
                "timestamp": "2026-03-13T09:10:54+00:00",
                "total_watts": 914.0,
                "required_reduction_w": 64.0,
            }
        )
        overload_event_queue.put(
            {
                "event": "RAMP_PREDICTED",
                "timestamp": "2026-03-13T09:10:55+00:00",
                "total_watts": 930.0,
                "required_reduction_w": 80.0,
            }
        )

        monitor = _StubMonitor(
            {
                "timestamp": "2026-03-13T09:10:53+00:00",
                "total_watts": 902.0,
            }
        )

        with mock.patch("run_main.time.monotonic", side_effect=[0.0, 0.2]):
            peak_total_watts, peak_required_w, peak_source, terminal_event_name = wait_before_market_start(
                monitor=monitor,
                overload_event_queue=overload_event_queue,
                target_capacity_w=850.0,
                initial_total_watts=895.0,
                initial_required_reduction_w=45.0,
                initial_timestamp="2026-03-13T09:10:27+00:00",
                wait_before_mpr_s=0.1,
                poll_interval_s=0.1,
            )

        self.assertEqual(peak_total_watts, 914.0)
        self.assertEqual(peak_required_w, 64.0)
        self.assertEqual(peak_source, "SPIKE_WARNING")
        self.assertIsNone(terminal_event_name)

    def test_skips_market_if_overload_clears_during_wait(self) -> None:
        overload_event_queue: queue.Queue[dict[str, object]] = queue.Queue()
        overload_event_queue.put(
            {
                "event": "OVERLOAD_HANDLED",
                "timestamp": "2026-03-13T09:11:33+00:00",
                "total_watts": 875.0,
                "required_reduction_w": 64.0,
            }
        )
        monitor = _StubMonitor(None)

        with mock.patch("run_main.time.monotonic", side_effect=[0.0, 0.2]):
            peak_total_watts, peak_required_w, peak_source, terminal_event_name = wait_before_market_start(
                monitor=monitor,
                overload_event_queue=overload_event_queue,
                target_capacity_w=850.0,
                initial_total_watts=895.0,
                initial_required_reduction_w=45.0,
                initial_timestamp="2026-03-13T09:10:27+00:00",
                wait_before_mpr_s=0.1,
                poll_interval_s=0.1,
            )

        self.assertEqual(peak_total_watts, 895.0)
        self.assertEqual(peak_required_w, 45.0)
        self.assertEqual(peak_source, "OVERLOAD_START")
        self.assertEqual(terminal_event_name, "OVERLOAD_HANDLED")


class TestEventLoopSimulation(unittest.TestCase):
    @staticmethod
    def _job_df(power_values: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Resource Reduction": [0.0, 0.5],
                "Extra Execution": [0.0, 1.0],
                "Power": power_values,
            }
        )

    def test_idle_baseline_reapplies_market_multiple_times(self) -> None:
        args = types.SimpleNamespace(
            target_capacity_w=750.0,
            target_offset_w=20.0,
            power_startup_wait_s=0.0,
            power_interval_s=1.0,
            job_poll_interval_s=1.0,
            job_status_print_interval_s=1.0,
            detect_overload_only=False,
            wait_before_mpr_s=0.0,
            post_jobs_monitor_s=0.0,
            post_overload_end_job_ranks={},
            post_overload_end_wait_s=0.0,
        )
        scheduler = mock.Mock()
        monitor = mock.Mock()
        overload_event_queue: queue.Queue[dict[str, object]] = queue.Queue()
        submit_df = pd.DataFrame([{"job_id": "5001", "status": "SUBMITTED"}])
        tracked_job_ranks = {"job_a": 2, "job_b": 2, "job_c": 2}
        initial_job_perf_data = {"job_a": self._job_df([400.0, 300.0])}

        first_active = {
            "job_a": self._job_df([400.0, 300.0]),
            "job_b": self._job_df([200.0, 150.0]),
        }
        second_active = {
            "job_a": self._job_df([400.0, 300.0]),
            "job_b": self._job_df([200.0, 150.0]),
            "job_c": self._job_df([220.0, 180.0]),
        }
        third_active = {
            "job_a": self._job_df([400.0, 300.0]),
            "job_b": self._job_df([200.0, 150.0]),
            "job_c": self._job_df([300.0, 240.0]),
        }

        event_batches = [
            [
                {
                    "event": "OVERLOAD_START",
                    "timestamp": "2026-04-07T03:36:33+00:00",
                    "total_watts": 760.0,
                    "required_reduction_w": 10.0,
                }
            ],
            [
                {
                    "event": "SPIKE_WARNING",
                    "timestamp": "2026-04-07T03:37:07+00:00",
                    "total_watts": 807.0,
                    "required_reduction_w": 57.0,
                }
            ],
            [
                {
                    "event": "SPIKE_WARNING",
                    "timestamp": "2026-04-07T03:38:07+00:00",
                    "total_watts": 814.0,
                    "required_reduction_w": 64.0,
                }
            ],
            [],
        ]

        applied_results = [
            (True, {}, 30.0, {"job_a": 20.0, "job_b": 10.0}),
            (True, {}, 250.0, {"job_a": 100.0, "job_b": 70.0, "job_c": 80.0}),
            (True, {}, 330.0, {"job_a": 120.0, "job_b": 90.0, "job_c": 120.0}),
        ]

        monotonic_values = [
            0.0,
            0.1,
            6.0,
            6.1,
            6.2,
            12.0,
            12.1,
            12.2,
            20.0,
            20.1,
        ]

        with (
            mock.patch(
                "run_main.wait_for_initial_power_sample",
                return_value={"timestamp": "2026-04-07T03:36:30+00:00", "total_watts": 700.0},
            ),
            mock.patch(
                "run_main.query_submitted_job_states",
                return_value={"5001": "RUNNING"},
            ),
            mock.patch(
                "run_main.drain_overload_events",
                side_effect=event_batches,
            ),
            mock.patch(
                "run_main.resolve_active_control_context",
                side_effect=[
                    (first_active, {}),
                    (second_active, {}),
                    (third_active, {}),
                ],
            ),
            mock.patch(
                "run_main.apply_overload_reduction",
                side_effect=applied_results,
            ) as mock_apply,
            mock.patch(
                "run_main.has_active_jobs",
                side_effect=[True, True, True, True, True, True, False, False],
            ),
            mock.patch(
                "run_main.apply_overload_end_reset_to_max_frequency",
            ) as mock_reset,
            mock.patch(
                "run_main.time.monotonic",
                side_effect=monotonic_values,
            ),
            mock.patch("run_main.time.sleep", return_value=None),
        ):
            run_event_driven_control_loop(
                scheduler=scheduler,
                args=args,
                monitor=monitor,
                idle_power_w=160.0,
                overload_event_queue=overload_event_queue,
                submit_df=submit_df,
                job_ranks=tracked_job_ranks,
                job_perf_data=initial_job_perf_data,
            )

        self.assertEqual(mock_apply.call_count, 3)
        target_overrides = [
            call.kwargs["target_reduction_w_override"]
            for call in mock_apply.call_args_list
        ]
        self.assertEqual(target_overrides, [30.0, 250.0, 330.0])
        mock_reset.assert_called_once_with(args)

    def test_ramp_events_do_not_reapply_market_during_active_reduction(self) -> None:
        args = types.SimpleNamespace(
            target_capacity_w=750.0,
            target_offset_w=20.0,
            power_startup_wait_s=0.0,
            power_interval_s=1.0,
            job_poll_interval_s=1.0,
            job_status_print_interval_s=1.0,
            detect_overload_only=False,
            wait_before_mpr_s=0.0,
            post_jobs_monitor_s=0.0,
            post_overload_end_job_ranks={},
            post_overload_end_wait_s=0.0,
        )
        scheduler = mock.Mock()
        monitor = mock.Mock()
        overload_event_queue: queue.Queue[dict[str, object]] = queue.Queue()
        submit_df = pd.DataFrame([{"job_id": "5001", "status": "SUBMITTED"}])
        tracked_job_ranks = {"job_a": 2, "job_b": 2}
        initial_job_perf_data = {"job_a": self._job_df([400.0, 300.0])}

        first_active = {
            "job_a": self._job_df([400.0, 300.0]),
            "job_b": self._job_df([200.0, 150.0]),
        }

        event_batches = [
            [
                {
                    "event": "OVERLOAD_START",
                    "timestamp": "2026-04-07T03:36:33+00:00",
                    "total_watts": 760.0,
                    "required_reduction_w": 10.0,
                }
            ],
            [
                {
                    "event": "RAMP_PREDICTED",
                    "timestamp": "2026-04-07T03:37:07+00:00",
                    "total_watts": 807.0,
                    "required_reduction_w": 57.0,
                }
            ],
            [],
        ]

        with (
            mock.patch(
                "run_main.wait_for_initial_power_sample",
                return_value={"timestamp": "2026-04-07T03:36:30+00:00", "total_watts": 700.0},
            ),
            mock.patch(
                "run_main.query_submitted_job_states",
                return_value={"5001": "RUNNING"},
            ),
            mock.patch(
                "run_main.drain_overload_events",
                side_effect=event_batches,
            ),
            mock.patch(
                "run_main.resolve_active_control_context",
                return_value=(first_active, {}),
            ),
            mock.patch(
                "run_main.apply_overload_reduction",
                return_value=(True, {}, 30.0, {"job_a": 20.0, "job_b": 10.0}),
            ) as mock_apply,
            mock.patch(
                "run_main.has_active_jobs",
                side_effect=[True, True, True, False, False],
            ),
            mock.patch(
                "run_main.apply_overload_end_reset_to_max_frequency",
            ) as mock_reset,
            mock.patch(
                "run_main.time.monotonic",
                side_effect=[0.0, 0.1, 6.0, 6.1, 12.0, 12.1],
            ),
            mock.patch("run_main.time.sleep", return_value=None),
        ):
            run_event_driven_control_loop(
                scheduler=scheduler,
                args=args,
                monitor=monitor,
                idle_power_w=160.0,
                overload_event_queue=overload_event_queue,
                submit_df=submit_df,
                job_ranks=tracked_job_ranks,
                job_perf_data=initial_job_perf_data,
            )

        self.assertEqual(mock_apply.call_count, 1)
        self.assertEqual(
            mock_apply.call_args.kwargs["target_reduction_w_override"],
            30.0,
        )
        mock_reset.assert_called_once_with(args)


class TestIdlePowerBaseline(unittest.TestCase):
    def test_measure_idle_power_baseline_averages_unique_samples(self) -> None:
        monitor = _SequenceMonitor(
            [
                {"timestamp": "2026-04-07T03:00:00+00:00", "total_watts": 100.0},
                {"timestamp": "2026-04-07T03:00:00+00:00", "total_watts": 100.0},
                {"timestamp": "2026-04-07T03:00:01+00:00", "total_watts": 110.0},
                {"timestamp": "2026-04-07T03:00:02+00:00", "total_watts": 120.0},
            ]
        )

        with (
            mock.patch("run_main.time.sleep"),
            mock.patch("run_main.time.monotonic", side_effect=[0.0, 0.05, 0.10, 0.21]),
        ):
            avg_idle_power_w = measure_idle_power_baseline(
                monitor=monitor,
                sample_window_s=0.2,
                warmup_s=0.0,
                poll_interval_s=0.05,
                startup_wait_s=0.0,
            )

        self.assertEqual(avg_idle_power_w, 110.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
