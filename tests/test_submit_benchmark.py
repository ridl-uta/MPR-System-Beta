#!/usr/bin/env python3
"""Regression tests for submit_benchmark CLI helpers."""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utilities import submit_benchmark


class SubmitBenchmarkTest(unittest.TestCase):
    def test_dry_run_exports_submit_env(self) -> None:
        stdout = io.StringIO()
        argv = [
            "submit_benchmark.py",
            "-N",
            "1",
            "-n",
            "1",
            "-c",
            "10",
            "--script",
            str((_REPO_ROOT / "data" / "slurm_scripts" / "run_hpccg.slurm").resolve()),
            "--table",
            str((_REPO_ROOT / "data" / "slurm_scripts" / "hpccg.csv").resolve()),
            "--submit-env",
            "JOB_REPEAT=2",
            "--dry-run",
        ]

        with mock.patch.object(sys, "argv", argv), redirect_stdout(stdout):
            rc = submit_benchmark.main()

        output = stdout.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("--export=ALL,", output)
        self.assertIn("JOB_REPEAT=2", output)

    def test_invalid_submit_env_is_rejected(self) -> None:
        argv = [
            "submit_benchmark.py",
            "--submit-env",
            "JOB_REPEAT",
            "--dry-run",
        ]

        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as ctx:
                submit_benchmark.main()

        self.assertIn("Invalid --submit-env", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
