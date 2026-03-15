#!/usr/bin/env python3
"""Unit tests for DVFS config generation in job_utilities."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dvfs.job_utilities import apply_reduction, build_sbatch_variations


class ApplyReductionConfigTest(unittest.TestCase):
    def test_build_sbatch_variations_appends_submit_env(self) -> None:
        commands = build_sbatch_variations(submit_env=["JOB_REPEAT=2"])

        self.assertTrue(commands)
        for cmd in commands:
            self.assertIn("--submit-env", cmd)
            idx = cmd.index("--submit-env")
            self.assertEqual(cmd[idx + 1], "JOB_REPEAT=2")

    def test_apply_reduction_writes_perf_ctl_cpufreq_sync_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            conf_dir = temp_root / "conf"
            nodes_file = temp_root / "nodes.txt"
            nodes_file.write_text("ridlserver04\n", encoding="ascii")

            completed = subprocess.CompletedProcess(
                args=["run_geopm_apply_ssh.sh"],
                returncode=0,
                stdout="",
                stderr="",
            )

            with (
                mock.patch(
                    "dvfs.job_utilities.collect_host_cores",
                    return_value=("RUNNING", {"ridlserver04": [2, 6]}),
                ),
                mock.patch(
                    "dvfs.job_utilities.subprocess.run",
                    return_value=completed,
                ) as mock_run,
            ):
                freq_hz = apply_reduction(
                    "12345",
                    0.5,
                    max_freq_mhz=2400.0,
                    min_freq_mhz=1000.0,
                    conf_dir=str(conf_dir),
                    nodes_file=nodes_file,
                )

            conf_path = conf_dir / "ridlserver04.conf"
            self.assertEqual(freq_hz, 1_200_000_000.0)
            self.assertTrue(conf_path.exists())
            self.assertEqual(
                conf_path.read_text(encoding="ascii"),
                (
                    "### RULE dvfs-manager\n"
                    "FREQ_HZ=1200000000\n"
                    'CORES="2 6"\n'
                    "CONTROL_KIND=PERF_CTL\n"
                    "CPUFREQ_SYNC=1\n"
                    "CPUFREQ_GOVERNOR=userspace\n"
                    "CPUFREQ_MIN_KHZ=1000000\n"
                ),
            )

            mock_run.assert_called_once()
            cmd = mock_run.call_args.args[0]
            self.assertEqual(cmd[1:4], ["-u", "ridl", "-H"])
            self.assertEqual(Path(cmd[4]), nodes_file)


if __name__ == "__main__":
    unittest.main(verbosity=2)
