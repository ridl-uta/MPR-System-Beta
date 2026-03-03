from __future__ import annotations

import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List


class DVFSController:
    """Apply DVFS using run_geopm_apply_ssh.sh + per-host config files."""

    _DEFAULT_CONF_DIR = Path("/shared/geopm/freq")

    def __init__(
        self,
        *,
        helper_script_path: str | Path | None = None,
        conf_dir: str | Path = _DEFAULT_CONF_DIR,
        control_kind: str = "CORE_MAX",
        ssh_user: str | None = None,
        ssh_identity: str | None = None,
        no_sudo: bool = False,
        force_direct: bool = False,
        insecure_ssh: bool = False,
        concurrency: int = 8,
    ) -> None:
        self.helper_script_path = (
            Path(helper_script_path).resolve()
            if helper_script_path is not None
            else (Path(__file__).resolve().parent / "run_geopm_apply_ssh.sh")
        )
        self.conf_dir = Path(conf_dir)
        self.control_kind = str(control_kind).upper()
        self.ssh_user = ssh_user
        self.ssh_identity = ssh_identity
        self.no_sudo = bool(no_sudo)
        self.force_direct = bool(force_direct)
        self.insecure_ssh = bool(insecure_ssh)
        self.concurrency = int(concurrency)

        if self.control_kind not in {"AUTO", "CORE_MAX", "CPU"}:
            raise ValueError("control_kind must be one of: AUTO, CORE_MAX, CPU")
        if self.concurrency <= 0:
            raise ValueError("concurrency must be > 0")

    @staticmethod
    def _mhz_to_hz(freq_mhz: float) -> int:
        if freq_mhz <= 0:
            raise ValueError("frequency_mhz must be > 0")
        return int(round(freq_mhz * 1e6))

    @staticmethod
    def _tokens_to_str(tokens: Iterable[str]) -> str:
        return " ".join(shlex.quote(token) for token in tokens)

    @staticmethod
    def _normalize_cores(core_numbers: Iterable[int]) -> List[int]:
        normalized: list[int] = []
        seen: set[int] = set()
        for core in core_numbers:
            core_i = int(core)
            if core_i < 0:
                raise ValueError(f"core numbers must be >= 0, got {core_i}")
            if core_i in seen:
                continue
            seen.add(core_i)
            normalized.append(core_i)
        return sorted(normalized)

    def _host_conf_path(self, node_name: str) -> Path:
        return self.conf_dir / f"{node_name}.conf"

    def _write_host_config(self, *, node_name: str, core_numbers: List[int], frequency_hz: int) -> Path:
        self.conf_dir.mkdir(parents=True, exist_ok=True)
        conf_path = self._host_conf_path(node_name)

        lines = [
            "### RULE dvfs-controller",
            f"FREQ_HZ={int(frequency_hz)}",
            f"CONTROL_KIND={self.control_kind}",
        ]
        if core_numbers:
            lines.append(f'CORES="{" ".join(str(c) for c in core_numbers)}"')
        lines.append("")

        conf_path.write_text("\n".join(lines), encoding="ascii")
        return conf_path

    def _build_apply_command(self, *, hosts_file: str, conf_path: Path, dry_run: bool) -> list[str]:
        if not self.helper_script_path.exists():
            raise RuntimeError(f"missing helper script: {self.helper_script_path}")

        cmd: list[str] = [str(self.helper_script_path), "-H", hosts_file, "-c", str(self.concurrency)]
        if self.ssh_user:
            cmd[1:1] = ["-u", self.ssh_user]
        if self.ssh_identity:
            cmd[1:1] = ["-i", self.ssh_identity]
        if self.no_sudo:
            cmd.append("--no-sudo")
        if self.insecure_ssh:
            cmd.append("--insecure-ssh")

        # Service path is default when config is written to /shared/geopm/freq.
        # For custom conf_dir or explicit direct mode, call script directly.
        use_service_default = (self.conf_dir.resolve() == self._DEFAULT_CONF_DIR.resolve()) and not self.force_direct
        if not use_service_default:
            cmd.extend(["--direct", "--conf", str(conf_path)])

        if dry_run:
            cmd.append("--dry-run")

        return cmd

    def _run_apply_for_host(self, *, node_name: str, conf_path: Path, dry_run: bool) -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile(mode="w", encoding="ascii", delete=False) as tf:
            tf.write(f"{node_name}\n")
            hosts_file = tf.name

        try:
            cmd = self._build_apply_command(hosts_file=hosts_file, conf_path=conf_path, dry_run=dry_run)
            command_str = self._tokens_to_str(cmd)
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            status = "DRY_RUN" if dry_run else ("APPLIED" if proc.returncode == 0 else "FAILED")
            return {
                "status": status,
                "command": command_str,
                "returncode": int(proc.returncode),
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
        finally:
            Path(hosts_file).unlink(missing_ok=True)

    def apply_frequency(
        self,
        *,
        node_name: str,
        core_number: int,
        frequency_mhz: float | None = None,
        frequency_hz: int | None = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Apply one frequency value to one node/core."""
        rows = self.apply_to_cores(
            node_name=node_name,
            core_numbers=[int(core_number)],
            frequency_mhz=frequency_mhz,
            frequency_hz=frequency_hz,
            dry_run=dry_run,
        )
        if not rows:
            raise RuntimeError("no DVFS row generated")
        return rows[0]

    def apply_to_cores(
        self,
        *,
        node_name: str,
        core_numbers: Iterable[int],
        frequency_mhz: float | None = None,
        frequency_hz: int | None = None,
        dry_run: bool = False,
    ) -> List[Dict[str, Any]]:
        """Apply one frequency value to multiple cores on one node."""
        cores = self._normalize_cores(core_numbers)
        if not cores:
            return []

        if frequency_hz is None:
            if frequency_mhz is None:
                raise ValueError("Provide either frequency_mhz or frequency_hz.")
            frequency_hz = self._mhz_to_hz(float(frequency_mhz))
        if frequency_hz <= 0:
            raise ValueError("frequency_hz must be > 0")

        conf_path = self._write_host_config(
            node_name=node_name,
            core_numbers=cores,
            frequency_hz=int(frequency_hz),
        )

        apply_result = self._run_apply_for_host(
            node_name=node_name,
            conf_path=conf_path,
            dry_run=dry_run,
        )

        rows: list[dict[str, Any]] = []
        for core in cores:
            rows.append(
                {
                    "node_name": node_name,
                    "core_number": int(core),
                    "frequency_hz": int(frequency_hz),
                    "frequency_mhz": float(frequency_hz) / 1e6,
                    "control_kind": self.control_kind,
                    "config_path": str(conf_path),
                    "status": apply_result["status"],
                    "run_target": "service_script",
                    "command": apply_result["command"],
                    "returncode": apply_result["returncode"],
                    "stdout": apply_result["stdout"],
                    "stderr": apply_result["stderr"],
                }
            )
        return rows

    def apply_to_job_allocation(
        self,
        *,
        allocation: Dict[str, Any],
        frequency_mhz: float | None = None,
        frequency_hz: int | None = None,
        dry_run: bool = False,
    ) -> List[Dict[str, Any]]:
        """Apply one frequency to every node/core from scheduler allocation output."""
        node_map = allocation.get("cores_by_node", {})
        if not isinstance(node_map, dict):
            raise ValueError("allocation must contain a dict-like 'cores_by_node'.")

        results: List[Dict[str, Any]] = []
        for node_name, cores in node_map.items():
            if not isinstance(cores, Iterable):
                continue
            results.extend(
                self.apply_to_cores(
                    node_name=str(node_name),
                    core_numbers=[int(c) for c in cores],
                    frequency_mhz=frequency_mhz,
                    frequency_hz=frequency_hz,
                    dry_run=dry_run,
                )
            )
        return results
