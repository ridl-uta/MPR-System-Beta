from __future__ import annotations

import queue
import types
import unittest
from unittest import mock

from run_main import (
    compute_mpr_target_reduction_w,
    resolve_target_reduction_w,
    wait_before_market_start,
)


class TestRunMainTargetOffset(unittest.TestCase):
    def test_compute_target_reduction_adds_offset_when_overloaded(self) -> None:
        self.assertEqual(compute_mpr_target_reduction_w(80.0, 20.0), 100.0)

    def test_compute_target_reduction_does_not_apply_offset_without_overload(self) -> None:
        self.assertEqual(compute_mpr_target_reduction_w(0.0, 20.0), 0.0)

    def test_compute_target_reduction_allows_negative_offset_but_clamps(self) -> None:
        self.assertEqual(compute_mpr_target_reduction_w(80.0, -200.0), 0.0)

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


class _StubMonitor:
    def __init__(self, sample: dict[str, object] | None) -> None:
        self._sample = sample

    def get_last_sample(self) -> dict[str, object] | None:
        return self._sample


class TestWaitBeforeMarketStart(unittest.TestCase):
    def test_uses_peak_spike_or_ramp_reduction_seen_during_wait(self) -> None:
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
                "total_watts": 908.0,
                "required_reduction_w": 58.0,
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
