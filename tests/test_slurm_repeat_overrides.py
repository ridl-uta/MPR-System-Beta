from __future__ import annotations

import re
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SLURM_SCRIPTS_DIR = _REPO_ROOT / "job_scheduler" / "data" / "slurm_scripts"


class TestSlurmRepeatOverrides(unittest.TestCase):
    def test_all_slurm_scripts_use_job_specific_then_global_repeat(self) -> None:
        expected_repeat_vars = {
            "comd": "COMD_REPEAT",
            "hpccg": "HPCCG_REPEAT",
            "hpcg": "HPCG_REPEAT",
            "minife": "MINIFE_REPEAT",
            "minimd": "MINIMD_REPEAT",
            "stressng": "STRESSNG_REPEAT",
            "xsbenchmpi": "XSBENCHMPI_REPEAT",
        }

        for job_name, repeat_var in expected_repeat_vars.items():
            script_path = _SLURM_SCRIPTS_DIR / f"run_{job_name}.slurm"
            script_text = script_path.read_text()
            pattern = re.compile(
                rf"{repeat_var}=\$\{{{repeat_var}:-\$\{{JOB_REPEAT:-1\}}\}}"
            )
            self.assertRegex(
                script_text,
                pattern,
                msg=(
                    f"{script_path.name} must use {repeat_var} first, then JOB_REPEAT, "
                    "then the script default of 1."
                ),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
