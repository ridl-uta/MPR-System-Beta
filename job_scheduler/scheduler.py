from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .helper_utility import SlurmHelperUtility
from .performance_data import load_perf_data_for_jobs as _load_perf_data_for_jobs

MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_SCRIPTS_DIR = MODULE_ROOT / "data" / "slurm_scripts"
DEFAULT_PERF_DATA_XLSX = MODULE_ROOT / "data" / "all_model_data.xlsx"
DEFAULT_PERF_SHEET_MAP: Dict[str, str] = {
    "xsbenchmpi": "xsbench",
}


class JobScheduler:
    """High-level scheduler that submits rank-configured jobs from readonly module data."""

    def __init__(
        self,
        *,
        source_scripts_dir: Path | str | None = None,
        perf_xlsx_path: Path | str | None = None,
        perf_sheet_map: Dict[str, str] | None = None,
        helper_utility: SlurmHelperUtility | None = None,
    ) -> None:
        source = Path(source_scripts_dir) if source_scripts_dir is not None else DEFAULT_SOURCE_SCRIPTS_DIR
        self.source_scripts_dir = source.resolve()
        perf_xlsx = Path(perf_xlsx_path) if perf_xlsx_path is not None else DEFAULT_PERF_DATA_XLSX
        self.perf_xlsx_path = perf_xlsx.resolve()
        self.perf_sheet_map: Dict[str, str] = dict(DEFAULT_PERF_SHEET_MAP)
        if perf_sheet_map:
            self.perf_sheet_map.update(perf_sheet_map)

        # Cache loaded performance inputs so market modules can fetch them directly.
        self._job_perf_data_cache: Dict[str, pd.DataFrame] = {}
        self._perf_audit_rows: List[Dict[str, Any]] = []

        self.helper = helper_utility or SlurmHelperUtility()
        if not self.source_scripts_dir.exists():
            raise FileNotFoundError(f"Slurm scripts directory not found: {self.source_scripts_dir}")
        self.scripts_dir = self.source_scripts_dir

    def get_script_path(self, job_name: str) -> Path:
        return self.scripts_dir / f"run_{job_name}.slurm"

    def available_jobs(self) -> List[str]:
        jobs: List[str] = []
        for path in sorted(self.scripts_dir.glob("run_*.slurm")):
            name = path.stem
            if name.startswith("run_"):
                jobs.append(name[4:])
        return jobs

    def submit_job(
        self,
        *,
        name: str,
        cpus_per_rank: int,
        ranks_per_node: int,
        ranks: int,
        nodelist: str | None = None,
        exclude: str | None = None,
        partition: str = "debug",
        time_limit: str = "00:30:00",
        dry_run: bool = True,
        auto_load_perf_data: bool = True,
    ) -> Dict[str, Any]:
        """Submit one job by name with explicit rank/core placement settings."""
        if not dry_run and shutil.which(self.helper.sbatch_cmd) is None:
            raise RuntimeError(f"{self.helper.sbatch_cmd} not found in PATH.")

        script_path = self.get_script_path(name)
        if not script_path.exists():
            return {
                "job": name,
                "ranks": int(ranks),
                "script": str(script_path),
                "status": "MISSING_SCRIPT",
                "command": "",
                "message": "script not found in module data scripts",
            }

        cmd, nodes, ntasks_per_node = self.helper.build_submit_command(
            script_path=script_path,
            ranks=int(ranks),
            cpus_per_rank=cpus_per_rank,
            ranks_per_node=ranks_per_node,
            partition=partition,
            time_limit=time_limit,
            nodelist=nodelist,
            exclude=exclude,
        )
        cmd_str = self.helper.format_command(cmd)

        if dry_run:
            print(f"[DRY-RUN] {name}: {cmd_str}")
            row = {
                "job": name,
                "ranks": int(ranks),
                "script": str(script_path),
                "nodes": int(nodes),
                "ntasks_per_node": int(ntasks_per_node),
                "status": "DRY_RUN",
                "command": cmd_str,
            }
            if auto_load_perf_data:
                self._auto_load_perf_data_for_jobs([name])
            return row

        proc = self.helper.run_command(cmd)
        job_id = self.helper.parse_submitted_job_id(proc.stdout)
        status = "SUBMITTED" if proc.returncode == 0 else "FAILED"
        message = proc.stdout.strip() or proc.stderr.strip()
        print(f"[{status}] {name}: {message}")
        row = {
            "job": name,
            "ranks": int(ranks),
            "script": str(script_path),
            "nodes": int(nodes),
            "ntasks_per_node": int(ntasks_per_node),
            "status": status,
            "job_id": job_id,
            "command": cmd_str,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "returncode": proc.returncode,
        }
        if auto_load_perf_data:
            self._auto_load_perf_data_for_jobs([name])
        return row

    def submit_jobs(
        self,
        *,
        job_ranks: Dict[str, int] | None = None,
        name: str | None = None,
        ranks: int | None = None,
        cpus_per_rank: int,
        ranks_per_node: int,
        partition: str = "debug",
        time_limit: str = "00:30:00",
        nodelist: str | None = None,
        exclude: str | None = None,
        dry_run: bool = True,
        auto_load_perf_data: bool = True,
    ) -> pd.DataFrame:
        """Submit one or many jobs with shared placement settings.

        Usage patterns:
        - Multi-job: submit_jobs(job_ranks={"comd": 1, "hpcg": 4}, ...)
        - Single-job: submit_jobs(name="comd", ranks=1, ...)
        """
        resolved_job_ranks = dict(job_ranks or {})
        if not resolved_job_ranks:
            if name is None or ranks is None:
                raise ValueError("Provide either job_ranks or both name and ranks.")
            resolved_job_ranks[name] = int(ranks)

        rows: List[Dict[str, Any]] = []
        for job_name, rank_count in resolved_job_ranks.items():
            row = self.submit_job(
                name=job_name,
                cpus_per_rank=cpus_per_rank,
                ranks_per_node=ranks_per_node,
                ranks=int(rank_count),
                nodelist=nodelist,
                exclude=exclude,
                partition=partition,
                time_limit=time_limit,
                dry_run=dry_run,
                auto_load_perf_data=auto_load_perf_data,
            )
            rows.append(row)

        return pd.DataFrame(rows)

    def list_running_jobs(self) -> List[tuple[str, str]]:
        return self.helper.list_running_jobs()

    def get_job_resource_allocation(self, job_id: str) -> Dict[str, Any]:
        return self.helper.get_job_resource_allocation(job_id)

    def load_perf_data_for_jobs(
        self,
        *,
        job_names: List[str],
        sheet_map: Dict[str, str] | None = None,
        xlsx_path: Path | str | None = None,
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """Load performance data via scheduler-owned default workbook path."""
        chosen_xlsx = Path(xlsx_path).resolve() if xlsx_path is not None else self.perf_xlsx_path
        return _load_perf_data_for_jobs(
            xlsx_path=chosen_xlsx,
            job_names=job_names,
            sheet_map=sheet_map or {},
        )

    def set_perf_sheet_map(self, mapping: Dict[str, str]) -> None:
        """Set or override workbook sheet-name mapping used during auto-load."""
        self.perf_sheet_map.update(mapping)

    def _auto_load_perf_data_for_jobs(self, job_names: List[str]) -> None:
        loaded, audit_df = self.load_perf_data_for_jobs(
            job_names=job_names,
            sheet_map=self.perf_sheet_map,
            xlsx_path=self.perf_xlsx_path,
        )
        for job_name, df in loaded.items():
            self._job_perf_data_cache[job_name] = df
        if not audit_df.empty:
            self._perf_audit_rows.extend(audit_df.to_dict(orient="records"))

    def get_cached_perf_data(
        self,
        *,
        job_names: List[str] | None = None,
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """Return cached job performance data collected during submit calls.

        If job_names is provided, output is filtered to those jobs.
        """
        if job_names is None:
            selected = dict(self._job_perf_data_cache)
        else:
            selected = {name: self._job_perf_data_cache[name] for name in job_names if name in self._job_perf_data_cache}

        audit_df = pd.DataFrame(self._perf_audit_rows)
        if job_names is not None and not audit_df.empty and "job" in audit_df.columns:
            audit_df = audit_df[audit_df["job"].isin(job_names)].reset_index(drop=True)
        return selected, audit_df
