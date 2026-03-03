from __future__ import annotations

import math
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


class SlurmHelperUtility:
    """Helper methods for interacting with Slurm commands."""

    def __init__(
        self,
        *,
        sbatch_cmd: str = "sbatch",
        squeue_cmd: str = "squeue",
        scontrol_cmd: str = "scontrol",
        srun_cmd: str = "srun",
    ) -> None:
        self.sbatch_cmd = sbatch_cmd
        self.squeue_cmd = squeue_cmd
        self.scontrol_cmd = scontrol_cmd
        self.srun_cmd = srun_cmd

    def run_command(self, cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                list(cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            tool = cmd[0] if cmd else "<unknown>"
            raise RuntimeError(f"Required command not found: {tool}") from exc

    def copy_slurm_scripts(self, source_dir: Path | str, target_dir: Path | str) -> List[Path]:
        """Copy Slurm scripts/assets from source into scheduler staging dir."""
        src = Path(source_dir).resolve()
        dst = Path(target_dir).resolve()

        if not src.exists():
            raise FileNotFoundError(f"Slurm source directory not found: {src}")

        dst.mkdir(parents=True, exist_ok=True)
        copied: List[Path] = []
        for path in sorted(src.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(src)
            target_path = dst / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target_path)
            copied.append(target_path)
        return copied

    def build_submit_command(
        self,
        *,
        script_path: Path,
        ranks: int,
        cpus_per_rank: int,
        ranks_per_node: int,
        partition: str = "debug",
        time_limit: str = "00:30:00",
        nodelist: str | None = None,
        exclude: str | None = None,
        output_path: str | None = None,
        env_vars: Mapping[str, str] | None = None,
        script_args: Sequence[str] | None = None,
    ) -> Tuple[List[str], int, int]:
        if ranks <= 0:
            raise ValueError(f"ranks must be > 0 for {script_path.name}")
        if ranks_per_node <= 0:
            raise ValueError("ranks_per_node must be > 0")
        if cpus_per_rank <= 0:
            raise ValueError("cpus_per_rank must be > 0")

        nodes = max(1, math.ceil(ranks / ranks_per_node))
        ntasks_per_node = min(ranks, ranks_per_node)

        cmd: List[str] = [
            self.sbatch_cmd,
            "-p",
            partition,
            "-N",
            str(nodes),
            "-n",
            str(ranks),
            "--ntasks-per-node",
            str(ntasks_per_node),
            "-c",
            str(cpus_per_rank),
            "-t",
            time_limit,
        ]
        if output_path:
            cmd += ["-o", output_path]
        if nodelist:
            cmd += ["--nodelist", nodelist]
        if exclude:
            cmd += ["--exclude", exclude]
        if env_vars:
            export_tokens = [f"{key}={value}" for key, value in env_vars.items()]
            cmd += ["--export", "ALL," + ",".join(export_tokens)]
        cmd.append(str(script_path))
        if script_args:
            cmd.extend(script_args)

        return cmd, nodes, ntasks_per_node

    @staticmethod
    def format_command(cmd: Sequence[str]) -> str:
        return " ".join(shlex.quote(token) for token in cmd)

    @staticmethod
    def parse_submitted_job_id(stdout: str) -> str | None:
        match = re.search(r"Submitted batch job\s+(\d+)", stdout or "")
        return match.group(1) if match else None

    def list_running_jobs(self) -> List[Tuple[str, str]]:
        proc = self.run_command([self.squeue_cmd, "-h", "-t", "RUNNING", "-o", "%i %j"])
        if proc.returncode != 0:
            stderr = proc.stderr.strip() or "no details"
            raise RuntimeError(f"squeue failed with exit {proc.returncode}: {stderr}")

        jobs: List[Tuple[str, str]] = []
        for line in proc.stdout.splitlines():
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split(None, 1)
            job_id = parts[0]
            job_name = parts[1] if len(parts) > 1 else ""
            jobs.append((job_id, job_name))
        return jobs

    def get_job_nodelist_and_state(self, job_id: str) -> Tuple[str, str]:
        nodes_proc = self.run_command([self.squeue_cmd, "-h", "-j", job_id, "-o", "%N"])
        state_proc = self.run_command([self.squeue_cmd, "-h", "-j", job_id, "-o", "%T"])

        compact_nodelist = nodes_proc.stdout.strip()
        state = state_proc.stdout.strip()

        if compact_nodelist and state:
            return compact_nodelist, state

        fallback = self.run_command([self.scontrol_cmd, "show", "job", "-o", job_id])
        if fallback.returncode != 0:
            return compact_nodelist, state

        if not compact_nodelist:
            match = re.search(r"\bNodeList=([^ ]+)", fallback.stdout)
            if match:
                compact_nodelist = match.group(1).strip()
        if not state:
            match = re.search(r"\bJobState=([^ ]+)", fallback.stdout)
            if match:
                state = match.group(1).strip()

        return compact_nodelist, state

    def expand_nodelist(self, compact_nodelist: str) -> List[str]:
        if not compact_nodelist or compact_nodelist == "(null)":
            return []
        proc = self.run_command([self.scontrol_cmd, "show", "hostnames", compact_nodelist])
        if proc.returncode != 0:
            return []
        return [line.strip() for line in proc.stdout.splitlines() if line.strip()]

    @staticmethod
    def expand_cpu_list(cpulist: str) -> List[int]:
        numbers: List[int] = []
        for chunk in cpulist.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "-" in chunk:
                start_s, end_s = chunk.split("-", 1)
                try:
                    start_i = int(start_s)
                    end_i = int(end_s)
                except ValueError:
                    continue
                step = 1 if end_i >= start_i else -1
                numbers.extend(range(start_i, end_i + step, step))
            else:
                try:
                    numbers.append(int(chunk))
                except ValueError:
                    continue
        return numbers

    def collect_job_cores(self, job_id: str) -> Tuple[str, Dict[str, List[int]]]:
        compact_nodelist, state = self.get_job_nodelist_and_state(job_id)
        hosts = self.expand_nodelist(compact_nodelist)
        if not hosts:
            return state, {}

        probe_cmd = [
            self.srun_cmd,
            f"--jobid={job_id}",
            "-N",
            str(len(hosts)),
            "--ntasks-per-node=1",
            "--overlap",
            "--immediate=10",
            "bash",
            "-lc",
            'printf "%s " "$(hostname -s)"; '
            "awk '/^Cpus_allowed_list:/ {print $2; exit}' /proc/self/status",
        ]
        probe = self.run_command(probe_cmd)
        if probe.returncode != 0:
            return state, {}

        host_cores: Dict[str, List[int]] = {}
        for line in probe.stdout.splitlines():
            row = line.strip()
            if not row:
                continue
            parts = row.split(None, 2)
            if len(parts) < 2:
                continue
            host = parts[0]
            cpulist = parts[1]
            expanded = self.expand_cpu_list(cpulist)
            if expanded:
                host_cores[host] = expanded
        return state, host_cores

    def get_job_resource_allocation(self, job_id: str) -> Dict[str, Any]:
        compact_nodelist, state = self.get_job_nodelist_and_state(job_id)
        hosts = self.expand_nodelist(compact_nodelist)
        _, cores_by_node = self.collect_job_cores(job_id)

        return {
            "job_id": str(job_id),
            "state": state or "",
            "compact_nodelist": compact_nodelist or "",
            "nodes": hosts,
            "node_count": len(hosts),
            "cores_by_node": cores_by_node,
            "total_cores_observed": int(sum(len(v) for v in cores_by_node.values())),
        }
