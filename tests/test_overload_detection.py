from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from typing import Any

from power_monitor.overload_detection import make_simple_overload_ctx, simple_overload_update


def _event_name(event: Any) -> str:
    if event is None:
        return ""
    if isinstance(event, tuple):
        return str(event[0])
    return str(event)


def _run_series(ctx: dict[str, Any], watts_series: list[float]) -> list[tuple[int, float, str, Any]]:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events: list[tuple[int, float, str, Any]] = []
    for i, watts in enumerate(watts_series):
        ts = base + timedelta(seconds=i)
        sample = float(watts)
        # Keep raw sample aligned with run_main usage.
        ctx["last_raw_sample"] = sample
        event, payload = simple_overload_update(ctx, ts, sample)
        name = _event_name(event)
        if name:
            events.append((i, sample, name, payload))
    return events


class TestOverloadDetectionHandledZone(unittest.TestCase):
    def test_handled_zone_allows_power_above_target(self) -> None:
        ctx = make_simple_overload_ctx(
            sample_period_s=1.0,
            threshold_w=740.0,
            hysteresis_w=20.0,
            min_over_s=3,
            cooldown_s=4,
            handled_window_s=4,
            handled_high_margin_w=12.0,  # handled upper bound = 752
        )
        watts = [
            735,
            742,
            745,
            747,  # OVERLOAD_START
            750,
            751,
            749,
            748,  # OVERLOAD_HANDLED can trigger while still above target
            747,
            749,
            748,
            751,
            760,  # leave handled zone
            759,
            758,
            734,
            733,
            732,
            731,  # may re-enter handled zone
            718,
            717,
            716,
            715,  # OVERLOAD_END (cooldown below T_low)
            714,
        ]
        events = _run_series(ctx, watts)
        names = [e[2] for e in events]
        self.assertIn("OVERLOAD_START", names)
        self.assertIn("OVERLOAD_HANDLED", names)
        self.assertIn("OVERLOAD_END", names)

        handled = next(e for e in events if e[2] == "OVERLOAD_HANDLED")
        payload = handled[3] if isinstance(handled[3], dict) else {}
        self.assertAlmostEqual(float(payload.get("zone_high_w", -1.0)), 752.0, places=6)
        self.assertNotIn("zone_low_w", payload)

    def test_zero_high_margin_does_not_handle_above_target(self) -> None:
        ctx = make_simple_overload_ctx(
            sample_period_s=1.0,
            threshold_w=740.0,
            hysteresis_w=20.0,
            min_over_s=2,
            cooldown_s=3,
            handled_window_s=3,
            handled_high_margin_w=0.0,  # handled zone high == target
        )
        watts = [742, 744, 746, 748, 747, 746, 745, 744, 743, 742, 741, 742, 745, 746]
        events = _run_series(ctx, watts)
        names = [e[2] for e in events]
        self.assertIn("OVERLOAD_START", names)
        self.assertNotIn("OVERLOAD_HANDLED", names)

    def test_low_margin_setting_is_ignored_for_handled_detection(self) -> None:
        ctx = make_simple_overload_ctx(
            sample_period_s=1.0,
            threshold_w=740.0,
            hysteresis_w=20.0,
            min_over_s=2,
            cooldown_s=5,
            handled_window_s=2,
            handled_high_margin_w=10.0,
            handled_low_margin_w=15.0,
        )
        watts = [742, 744, 710, 712, 713, 714]
        events = _run_series(ctx, watts)
        names = [e[2] for e in events]
        self.assertIn("OVERLOAD_START", names)
        self.assertIn("OVERLOAD_HANDLED", names)

        handled = next(e for e in events if e[2] == "OVERLOAD_HANDLED")
        payload = handled[3] if isinstance(handled[3], dict) else {}
        self.assertAlmostEqual(float(payload.get("zone_high_w", -1.0)), 750.0, places=6)
        self.assertNotIn("zone_low_w", payload)

    def test_overload_handled_triggers_once_per_cycle(self) -> None:
        ctx = make_simple_overload_ctx(
            sample_period_s=1.0,
            threshold_w=740.0,
            hysteresis_w=20.0,
            min_over_s=2,
            cooldown_s=3,
            handled_window_s=2,
            handled_high_margin_w=10.0,
        )
        watts = [742, 744, 748, 749, 760, 759, 748, 749, 718, 717, 716]
        events = _run_series(ctx, watts)
        names = [e[2] for e in events]

        self.assertEqual(names.count("OVERLOAD_START"), 1)
        self.assertEqual(names.count("OVERLOAD_HANDLED"), 1)
        self.assertEqual(names.count("OVERLOAD_END"), 1)

    def test_t_low_uses_min_of_dynamic_low_and_base_low(self) -> None:
        ctx = make_simple_overload_ctx(
            sample_period_s=1.0,
            threshold_w=810.0,
            hysteresis_w=80.0,
            min_over_s=2,
            cooldown_s=3,
            handled_window_s=10,
            handled_high_margin_w=25.0,
        )
        events = _run_series(
            ctx,
            [
                850.0,
                870.0,  # OVERLOAD_START, required_reduction_w becomes 60, dynamic_low=750
                725.0,
                724.0,
                723.0,  # OVERLOAD_END should fire because T_low=min(750, 730)=730
            ],
        )

        self.assertEqual(events[0][2], "OVERLOAD_START")
        self.assertEqual(events[-1][2], "OVERLOAD_END")
        self.assertAlmostEqual(float(ctx["T_low"]), 730.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
