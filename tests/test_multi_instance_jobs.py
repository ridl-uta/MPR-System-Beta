from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import contextlib
import io

import pandas as pd

from job_scheduler import JobScheduler
from job_scheduler.performance_data import load_perf_data_for_jobs
from run_main import fetch_job_perf_data, parse_job_args_overrides, parse_job_specs


class TestMultiInstanceParsing(unittest.TestCase):
    def test_parse_job_specs_auto_numbers_duplicate_base_jobs(self) -> None:
        job_ranks, job_base_names, job_display_names = parse_job_specs(
            ["xsbenchmpi=2", "xsbenchmpi=4", "comd=2"]
        )

        self.assertEqual(
            job_ranks,
            {
                "xsbenchmpi#1": 2,
                "xsbenchmpi#2": 4,
                "comd": 2,
            },
        )
        self.assertEqual(
            job_base_names,
            {
                "xsbenchmpi#1": "xsbenchmpi",
                "xsbenchmpi#2": "xsbenchmpi",
                "comd": "comd",
            },
        )
        self.assertEqual(
            job_display_names,
            {
                "xsbenchmpi#1": "xsbenchmpi",
                "xsbenchmpi#2": "xsbenchmpi",
                "comd": "comd",
            },
        )

    def test_parse_job_specs_keeps_single_job_without_suffix(self) -> None:
        job_ranks, job_base_names, job_display_names = parse_job_specs(["minimd=2"])

        self.assertEqual(job_ranks, {"minimd": 2})
        self.assertEqual(job_base_names, {"minimd": "minimd"})
        self.assertEqual(job_display_names, {"minimd": "minimd"})

    def test_parse_job_args_overrides_apply_by_base_name(self) -> None:
        overrides = parse_job_args_overrides(
            [
                "xsbenchmpi=-s small -l 5000",
                "comd=-x 60 -y 60 -z 60 -N 1000",
            ]
        )
        self.assertEqual(overrides["xsbenchmpi"], ["-s", "small", "-l", "5000"])
        self.assertEqual(
            overrides["comd"],
            ["-x", "60", "-y", "60", "-z", "60", "-N", "1000"],
        )


class TestMultiInstanceSubmission(unittest.TestCase):
    def test_scheduler_dry_run_keeps_duplicate_visible_job_names(self) -> None:
        scheduler = JobScheduler()

        submit_df = scheduler.submit_jobs(
            job_ranks={"xsbenchmpi#1": 2, "xsbenchmpi#2": 4},
            job_base_names={
                "xsbenchmpi#1": "xsbenchmpi",
                "xsbenchmpi#2": "xsbenchmpi",
            },
            job_display_names={
                "xsbenchmpi#1": "xsbenchmpi",
                "xsbenchmpi#2": "xsbenchmpi",
            },
            cpus_per_rank=10,
            ranks_per_node=2,
            exclude="ridlserver02",
            dry_run=True,
            auto_load_perf_data=False,
        )

        self.assertEqual(list(submit_df["job"]), ["xsbenchmpi", "xsbenchmpi"])
        self.assertEqual(list(submit_df["job_key"]), ["xsbenchmpi#1", "xsbenchmpi#2"])
        self.assertTrue((submit_df["job_base"] == "xsbenchmpi").all())
        self.assertEqual(list(submit_df["ranks"]), [2, 4])
        self.assertTrue(
            all("run_xsbenchmpi.slurm" in command for command in submit_df["command"])
        )

    def test_scheduler_alias_perf_sheet_map_uses_base_sheet_mapping(self) -> None:
        scheduler = JobScheduler()

        scheduler.submit_jobs(
            job_ranks={"xsbenchmpi#1": 2},
            job_base_names={"xsbenchmpi#1": "xsbenchmpi"},
            job_display_names={"xsbenchmpi#1": "xsbenchmpi"},
            cpus_per_rank=10,
            ranks_per_node=2,
            dry_run=True,
            auto_load_perf_data=True,
        )

        self.assertEqual(
            scheduler.perf_sheet_map["xsbenchmpi#1"],
            scheduler.perf_sheet_map["xsbenchmpi"],
        )


class TestRankSpecificPerfSheets(unittest.TestCase):
    def test_loader_prefers_rank_specific_sheet_then_falls_back_to_base_sheet(self) -> None:
        required_df = pd.DataFrame(
            {
                "Resource Reduction": [0.1],
                "Extra Execution": [1.0],
                "Power": [10.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            xlsx_path = Path(tmpdir) / "perf.xlsx"
            with pd.ExcelWriter(xlsx_path) as writer:
                required_df.to_excel(writer, sheet_name="xsbench-rank2", index=False)
                required_df.to_excel(writer, sheet_name="comd", index=False)

            jobs, audit_df = load_perf_data_for_jobs(
                xlsx_path=xlsx_path,
                job_names=["xsbenchmpi#1", "comd#1"],
                sheet_map={
                    "xsbenchmpi#1": "xsbench",
                    "comd#1": "comd",
                },
                job_ranks={
                    "xsbenchmpi#1": 2,
                    "comd#1": 4,
                },
            )

        self.assertIn("xsbenchmpi#1", jobs)
        self.assertIn("comd#1", jobs)
        audit_rows = {str(row["job"]): dict(row) for row in audit_df.to_dict(orient="records")}
        self.assertEqual(audit_rows["xsbenchmpi#1"]["sheet"], "xsbench-rank2")
        self.assertEqual(audit_rows["comd#1"]["sheet"], "comd")

    def test_loader_accepts_underscore_rank_sheet_names(self) -> None:
        required_df = pd.DataFrame(
            {
                "Resource Reduction": [0.1],
                "Extra Execution": [1.0],
                "Power": [10.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            xlsx_path = Path(tmpdir) / "perf.xlsx"
            with pd.ExcelWriter(xlsx_path) as writer:
                required_df.to_excel(writer, sheet_name="xsbench_rank2", index=False)
                required_df.to_excel(writer, sheet_name="xsbench_rank4", index=False)

            jobs, audit_df = load_perf_data_for_jobs(
                xlsx_path=xlsx_path,
                job_names=["xsbenchmpi#1", "xsbenchmpi#2"],
                sheet_map={
                    "xsbenchmpi#1": "xsbench",
                    "xsbenchmpi#2": "xsbench",
                },
                job_ranks={
                    "xsbenchmpi#1": 2,
                    "xsbenchmpi#2": 4,
                },
            )

        self.assertIn("xsbenchmpi#1", jobs)
        self.assertIn("xsbenchmpi#2", jobs)
        audit_rows = {str(row["job"]): dict(row) for row in audit_df.to_dict(orient="records")}
        self.assertEqual(audit_rows["xsbenchmpi#1"]["sheet"], "xsbench_rank2")
        self.assertEqual(audit_rows["xsbenchmpi#2"]["sheet"], "xsbench_rank4")


class TestDryRunPerfAudit(unittest.TestCase):
    def test_fetch_job_perf_data_non_strict_allows_missing_sheets(self) -> None:
        scheduler = JobScheduler()
        scheduler._perf_audit_rows = [
            {
                "job": "xsbenchmpi#1",
                "sheet": "xsbench",
                "status": "MISSING_SHEET",
                "details": "xsbench-rank2,xsbench",
            }
        ]

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = fetch_job_perf_data(
                scheduler,
                {"xsbenchmpi#1": 2},
                strict=False,
            )

        self.assertEqual(result, {})
        self.assertIn("Performance data audit:", stdout.getvalue())
        self.assertIn("MISSING_SHEET", stdout.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
