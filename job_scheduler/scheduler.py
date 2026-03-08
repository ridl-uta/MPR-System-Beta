from __future__ import annotations

import csv
import shlex
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from .helper_utility import SlurmHelperUtility
from .performance_data import load_perf_data_for_jobs as _load_perf_data_for_jobs

MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_SCRIPTS_DIR = MODULE_ROOT / "data" / "slurm_scripts"
DEFAULT_SOURCE_SCRIPTS_LIST = MODULE_ROOT / "data" / "slurm_scripts.txt"
DEFAULT_PERF_DATA_XLSX = MODULE_ROOT / "data" / "all_model_data_by_rank.xlsx"
DEFAULT_PERF_SHEET_MAP: Dict[str, str] = {
    "xsbenchmpi": "xsbench",
}
DEFAULT_GRID_BY_JOB: Dict[str, Tuple[int, int, int]] = {
    "hpccg": (180, 180, 180),
    "hpcg": (64, 64, 64),
    "minife": (240, 240, 240),
}
DEFAULT_BENCH_ARGS_BY_JOB: Dict[str, str] = {
    "minimd": "-i in.lj.miniMD",
    "comd": "-x 40 -y 40 -z 40 -N 1000",
}


class JobScheduler:
    """High-level scheduler that submits rank-configured jobs from readonly module data."""

    def __init__(
        self,
        *,
        source_scripts_dir: Path | str | None = None,
        source_scripts_list_path: Path | str | None = None,
        perf_xlsx_path: Path | str | None = None,
        perf_sheet_map: Dict[str, str] | None = None,
        helper_utility: SlurmHelperUtility | None = None,
    ) -> None:
        source = Path(source_scripts_dir) if source_scripts_dir is not None else DEFAULT_SOURCE_SCRIPTS_DIR
        self.source_scripts_dir = source.resolve()
        source_list = (
            Path(source_scripts_list_path)
            if source_scripts_list_path is not None
            else DEFAULT_SOURCE_SCRIPTS_LIST
        )
        self.source_scripts_list_path = source_list.resolve()
        perf_xlsx = Path(perf_xlsx_path) if perf_xlsx_path is not None else DEFAULT_PERF_DATA_XLSX
        self.perf_xlsx_path = perf_xlsx.resolve()
        self.perf_sheet_map: Dict[str, str] = dict(DEFAULT_PERF_SHEET_MAP)
        if perf_sheet_map:
            self.perf_sheet_map.update(perf_sheet_map)

        # Cache loaded performance inputs so market modules can fetch them directly.
        self._job_perf_data_cache: Dict[str, pd.DataFrame] = {}
        self._perf_audit_rows: List[Dict[str, Any]] = []
        self._submission_rows: List[Dict[str, Any]] = []

        self.helper = helper_utility or SlurmHelperUtility()
        if not self.source_scripts_dir.exists():
            raise FileNotFoundError(f"Slurm scripts directory not found: {self.source_scripts_dir}")
        self.scripts_dir = self.source_scripts_dir
        self._script_list_overrides = self._load_script_list_overrides()

    def get_script_path(self, job_name: str) -> Path:
        return self.scripts_dir / f"run_{job_name}.slurm"

    def available_jobs(self) -> List[str]:
        jobs: List[str] = []
        for path in sorted(self.scripts_dir.glob("run_*.slurm")):
            name = path.stem
            if name.startswith("run_"):
                jobs.append(name[4:])
        return jobs

    @staticmethod
    def _resolve_job_ranks(
        *,
        job_ranks: Dict[str, int] | None = None,
        name: str | None = None,
        ranks: int | None = None,
    ) -> Dict[str, int]:
        resolved_job_ranks = dict(job_ranks or {})
        if not resolved_job_ranks:
            if name is None or ranks is None:
                raise ValueError("Provide either job_ranks or both name and ranks.")
            resolved_job_ranks[name] = int(ranks)
        return resolved_job_ranks

    def _record_submission_row(self, row: Dict[str, Any]) -> None:
        self._submission_rows.append(dict(row))

    @staticmethod
    def _as_int_or_none(value: Any) -> int | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    def _table_path_for_job(self, job_name: str) -> Path:
        return self.scripts_dir / f"{job_name}.csv"

    def _read_table_row_for_ranks(self, *, job_name: str, ranks: int) -> Dict[str, str]:
        table = self._table_path_for_job(job_name)
        if not table.exists():
            return {}

        with table.open() as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row_ranks = self._as_int_or_none(row.get("ranks"))
                if row_ranks != int(ranks):
                    continue
                clean: Dict[str, str] = {}
                for key, value in row.items():
                    k = str(key or "").strip()
                    if not k:
                        continue
                    clean[k] = str(value or "").strip()
                return clean
        return {}

    def _load_script_list_overrides(self) -> Dict[str, Dict[str, str]]:
        """Parse module slurm_scripts.txt into per-job key/value overrides."""
        path = self.source_scripts_list_path
        if not path.exists():
            return {}

        by_job: Dict[str, Dict[str, str]] = {}
        with path.open() as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    tokens = shlex.split(line)
                except ValueError:
                    continue
                if not tokens:
                    continue
                script_token = tokens[0]
                stem = Path(script_token).stem
                if not stem.startswith("run_"):
                    continue
                job_name = stem[4:]
                overrides: Dict[str, str] = {}
                for token in tokens[1:]:
                    if "=" not in token:
                        continue
                    key, value = token.split("=", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key and value:
                        overrides[key] = value
                by_job[job_name] = overrides
        return by_job

    def _script_overrides_to_env(self, job_name: str) -> Dict[str, str]:
        overrides = self._script_list_overrides.get(job_name, {})
        if not overrides:
            return {}

        env: Dict[str, str] = {}
        workdir = overrides.get("workdir")
        if workdir:
            env["WORKDIR"] = workdir

        bin_path = overrides.get("bin")
        if bin_path:
            if job_name == "minimd":
                env["MINIMD_BIN"] = bin_path
            elif job_name == "comd":
                env["COMD_BIN"] = bin_path
            else:
                env["BENCH_BIN"] = bin_path

        for key, value in overrides.items():
            if key in {"workdir", "bin"}:
                continue
            env[key.upper()] = value
        return env

    def _rank_profile_for_job(self, *, job_name: str, ranks: int) -> tuple[Dict[str, str], List[str]]:
        """Mirror submit_benchmark table-driven behavior using module-local CSVs."""
        row = self._read_table_row_for_ranks(job_name=job_name, ranks=ranks)
        env: Dict[str, str] = {}
        script_args: List[str] = []

        if job_name == "xsbenchmpi":
            size = row.get("size") or "small"
            lookups_val = (
                self._as_int_or_none(row.get("lookups_per_rank"))
                or self._as_int_or_none(row.get("lookups"))
                or 10000
            )
            env["SIZE"] = str(size)
            env["LOOKUPS"] = str(int(lookups_val))
            return env, script_args

        if job_name in DEFAULT_GRID_BY_JOB:
            default_nx, default_ny, default_nz = DEFAULT_GRID_BY_JOB[job_name]
            nx = self._as_int_or_none(row.get("nx")) or default_nx
            ny = self._as_int_or_none(row.get("ny")) or default_ny
            nz = self._as_int_or_none(row.get("nz")) or default_nz
            env["NX"] = str(int(nx))
            env["NY"] = str(int(ny))
            env["NZ"] = str(int(nz))
            return env, script_args

        if job_name in DEFAULT_BENCH_ARGS_BY_JOB:
            args_raw = (
                row.get("args")
                or row.get(f"{job_name}_args")
                or row.get("bench_args")
                or DEFAULT_BENCH_ARGS_BY_JOB[job_name]
            )
            try:
                script_args = shlex.split(args_raw)
            except ValueError:
                script_args = shlex.split(DEFAULT_BENCH_ARGS_BY_JOB[job_name])
            return env, script_args

        return env, script_args

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
        output_path: str | None = None,
        mpi_iface: str | None = None,
        env_overrides: Dict[str, str] | None = None,
        script_args: List[str] | None = None,
        use_rank_profiles: bool = True,
        dry_run: bool = True,
        auto_load_perf_data: bool = True,
    ) -> Dict[str, Any]:
        """Submit one job by name with explicit rank/core placement settings."""
        if not dry_run and shutil.which(self.helper.sbatch_cmd) is None:
            raise RuntimeError(f"{self.helper.sbatch_cmd} not found in PATH.")

        script_path = self.get_script_path(name)
        if not script_path.exists():
            row = {
                "job": name,
                "ranks": int(ranks),
                "script": str(script_path),
                "status": "MISSING_SCRIPT",
                "command": "",
                "message": "script not found in module data scripts",
            }
            self._record_submission_row(row)
            return row

        profile_env: Dict[str, str] = {}
        profile_script_args: List[str] = []
        if use_rank_profiles:
            profile_env, profile_script_args = self._rank_profile_for_job(
                job_name=name,
                ranks=int(ranks),
            )

        merged_env = self._script_overrides_to_env(name)
        merged_env.update(profile_env)
        if mpi_iface:
            merged_env["MPI_IFACE"] = str(mpi_iface)
        if env_overrides:
            merged_env.update({str(k): str(v) for k, v in env_overrides.items()})

        final_script_args: List[str] = list(profile_script_args)
        if script_args is not None:
            final_script_args = [str(token) for token in script_args]

        cmd, nodes, ntasks_per_node = self.helper.build_submit_command(
            script_path=script_path,
            ranks=int(ranks),
            cpus_per_rank=cpus_per_rank,
            ranks_per_node=ranks_per_node,
            partition=partition,
            time_limit=time_limit,
            nodelist=nodelist,
            exclude=exclude,
            output_path=output_path,
            env_vars=merged_env or None,
            script_args=final_script_args or None,
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
                "env_vars": merged_env,
                "script_args": final_script_args,
            }
            if auto_load_perf_data:
                self._auto_load_perf_data_for_jobs([name])
            self._record_submission_row(row)
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
            "env_vars": merged_env,
            "script_args": final_script_args,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "returncode": proc.returncode,
        }
        if auto_load_perf_data:
            self._auto_load_perf_data_for_jobs([name])
        self._record_submission_row(row)
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
        output_path: str | None = None,
        mpi_iface: str | None = None,
        env_overrides: Dict[str, str] | None = None,
        script_args_by_job: Dict[str, List[str]] | None = None,
        use_rank_profiles: bool = True,
        dry_run: bool = True,
        auto_load_perf_data: bool = True,
    ) -> pd.DataFrame:
        """Submit one or many jobs with shared placement settings.

        Usage patterns:
        - Multi-job: submit_jobs(job_ranks={"comd": 1, "hpcg": 4}, ...)
        - Single-job: submit_jobs(name="comd", ranks=1, ...)
        """
        resolved_job_ranks = self._resolve_job_ranks(
            job_ranks=job_ranks,
            name=name,
            ranks=ranks,
        )

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
                output_path=output_path,
                mpi_iface=mpi_iface,
                env_overrides=env_overrides,
                script_args=(script_args_by_job or {}).get(job_name),
                use_rank_profiles=use_rank_profiles,
                dry_run=dry_run,
                auto_load_perf_data=auto_load_perf_data,
            )
            rows.append(row)

        return pd.DataFrame(rows)

    def submit_jobs_periodic(
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
        output_path: str | None = None,
        mpi_iface: str | None = None,
        env_overrides: Dict[str, str] | None = None,
        script_args_by_job: Dict[str, List[str]] | None = None,
        use_rank_profiles: bool = True,
        dry_run: bool = True,
        auto_load_perf_data: bool = True,
        submit_interval_s: float = 0.0,
    ) -> pd.DataFrame:
        """Submit jobs periodically with a fixed delay between submissions."""
        if submit_interval_s < 0:
            raise ValueError("submit_interval_s must be >= 0.")

        resolved_job_ranks = self._resolve_job_ranks(
            job_ranks=job_ranks,
            name=name,
            ranks=ranks,
        )

        items = list(resolved_job_ranks.items())
        rows: List[Dict[str, Any]] = []
        for index, (job_name, rank_count) in enumerate(items):
            row = self.submit_job(
                name=job_name,
                cpus_per_rank=cpus_per_rank,
                ranks_per_node=ranks_per_node,
                ranks=int(rank_count),
                nodelist=nodelist,
                exclude=exclude,
                partition=partition,
                time_limit=time_limit,
                output_path=output_path,
                mpi_iface=mpi_iface,
                env_overrides=env_overrides,
                script_args=(script_args_by_job or {}).get(job_name),
                use_rank_profiles=use_rank_profiles,
                dry_run=dry_run,
                auto_load_perf_data=auto_load_perf_data,
            )
            row = dict(row)
            row["submission_index"] = index
            rows.append(row)

            if index < len(items) - 1 and submit_interval_s > 0:
                time.sleep(submit_interval_s)

        return pd.DataFrame(rows)

    def list_running_jobs(self) -> List[tuple[str, str]]:
        return self.helper.list_running_jobs()

    def get_job_resource_allocation(self, job_id: str) -> Dict[str, Any]:
        return self.helper.get_job_resource_allocation(job_id)

    def get_submission_history(self, *, job_names: List[str] | None = None) -> pd.DataFrame:
        """Return previously attempted submissions from this scheduler instance."""
        history = pd.DataFrame(self._submission_rows)
        if history.empty or job_names is None:
            return history
        if "job" not in history.columns:
            return history
        return history[history["job"].isin(job_names)].reset_index(drop=True)

    def get_job_allocation_for_submitted(
        self,
        *,
        job_name: str,
        latest: bool = True,
    ) -> Dict[str, Any]:
        """Return node/core allocation for a submitted job name."""
        history = self.get_submission_history(job_names=[job_name])
        if history.empty:
            raise RuntimeError(f"No submission history for job '{job_name}'.")

        candidates = history[history.get("status").astype(str) == "SUBMITTED"] if "status" in history.columns else history
        if candidates.empty or "job_id" not in candidates.columns:
            raise RuntimeError(f"No submitted job_id found for job '{job_name}'.")

        row = candidates.iloc[-1] if latest else candidates.iloc[0]
        job_id = str(row["job_id"]).strip()
        if not job_id or job_id.lower() in {"none", "nan"}:
            raise RuntimeError(f"Submitted row for '{job_name}' does not contain a valid job_id.")

        return self.get_job_resource_allocation(job_id)

    def get_submitted_job_allocations(
        self,
        *,
        job_names: List[str] | None = None,
        latest_only: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Return per-job node/core allocations for submitted jobs."""
        history = self.get_submission_history(job_names=job_names)
        if history.empty:
            return {}

        if "status" in history.columns:
            history = history[history["status"].astype(str) == "SUBMITTED"]
        if history.empty or "job_id" not in history.columns or "job" not in history.columns:
            return {}

        allocations: Dict[str, Dict[str, Any]] = {}
        if latest_only:
            latest_rows = history.groupby("job", as_index=False).tail(1)
            for _, row in latest_rows.iterrows():
                job_name = str(row["job"])
                job_id = str(row["job_id"]).strip()
                if not job_id or job_id.lower() in {"none", "nan"}:
                    continue
                allocations[job_name] = self.get_job_resource_allocation(job_id)
            return allocations

        for _, row in history.iterrows():
            job_name = str(row["job"])
            job_id = str(row["job_id"]).strip()
            if not job_id or job_id.lower() in {"none", "nan"}:
                continue
            allocations[f"{job_name}:{job_id}"] = self.get_job_resource_allocation(job_id)
        return allocations

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
