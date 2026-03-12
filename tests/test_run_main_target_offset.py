from __future__ import annotations

import types
import unittest

from run_main import compute_mpr_target_reduction_w, resolve_target_reduction_w


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
