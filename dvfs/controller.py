from __future__ import annotations

import re
import socket
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List


class DVFSController:
    """Apply DVFS using run_geopm_apply_ssh.sh + per-host config files."""

    _DEFAULT_CONF_DIR = Path("/shared/geopm/freq")
    _CPU_READBACK_RE = re.compile(
        r"^cpu([0-9]+):\s*([0-9]+(?:\.[0-9]+)?)\s*MHz$",
        flags=re.IGNORECASE,
    )
    _GEOPMREAD_RE = re.compile(
        r"^GEOPMREAD\s+([A-Z0-9_:]+)\s+(core|cpu)\s+([0-9]+)\s+([-+0-9.eE]+)\s*$",
        flags=re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        helper_script_path: str | Path | None = None,
        conf_dir: str | Path = _DEFAULT_CONF_DIR,
        control_kind: str = "PERF_CTL",
        ssh_user: str | None = None,
        ssh_identity: str | None = None,
        no_sudo: bool = False,
        force_direct: bool = False,
        insecure_ssh: bool = False,
        concurrency: int = 8,
        verify_tolerance_mhz: float = 25.0,
    ) -> None:
        self.helper_script_path = (
            Path(helper_script_path).resolve()
            if helper_script_path is not None
            else (Path(__file__).resolve().parent / "run_geopm_apply_ssh.sh")
        )
        self.conf_dir = Path(conf_dir)
        self.control_kind = str(control_kind).upper()
        if self.control_kind == "CPU":
            self.control_kind = "PERF_CTL"
        self.ssh_user = ssh_user
        self.ssh_identity = ssh_identity
        self.no_sudo = bool(no_sudo)
        self.force_direct = bool(force_direct)
        self.insecure_ssh = bool(insecure_ssh)
        self.concurrency = int(concurrency)
        self.verify_tolerance_mhz = float(verify_tolerance_mhz)

        if self.control_kind not in {"AUTO", "CORE_MAX", "PERF_CTL"}:
            raise ValueError("control_kind must be one of: AUTO, CORE_MAX, PERF_CTL")
        if self.concurrency <= 0:
            raise ValueError("concurrency must be > 0")
        if self.verify_tolerance_mhz < 0:
            raise ValueError("verify_tolerance_mhz must be >= 0")

    @staticmethod
    def _mhz_to_hz(freq_mhz: float) -> int:
        if freq_mhz <= 0:
            raise ValueError("frequency_mhz must be > 0")
        return int(round(freq_mhz * 1e6))

    @staticmethod
    def _tokens_to_str(tokens: Iterable[str]) -> str:
        return " ".join(shlex.quote(token) for token in tokens)

    @staticmethod
    def _local_host_aliases() -> set[str]:
        aliases: set[str] = {"localhost", "127.0.0.1", "::1"}
        for getter in (socket.gethostname, socket.getfqdn):
            try:
                name = getter()
            except OSError:
                continue
            if not name:
                continue
            lowered = str(name).strip().lower()
            if lowered:
                aliases.add(lowered)
                aliases.add(lowered.split(".", 1)[0])
        return aliases

    @classmethod
    def _local_ip_set(cls) -> set[str]:
        local_ips: set[str] = {"127.0.0.1", "::1"}
        for host in cls._local_host_aliases():
            try:
                infos = socket.getaddrinfo(host, None)
            except OSError:
                continue
            for info in infos:
                sockaddr = info[4]
                if sockaddr and len(sockaddr) > 0:
                    local_ips.add(str(sockaddr[0]))
        return local_ips

    @classmethod
    def _is_local_host(cls, node_name: str) -> bool:
        candidate = str(node_name).strip().lower()
        if not candidate:
            return False
        aliases = cls._local_host_aliases()
        if candidate in aliases or candidate.split(".", 1)[0] in aliases:
            return True

        try:
            target_infos = socket.getaddrinfo(candidate, None)
        except OSError:
            return False
        local_ips = cls._local_ip_set()
        for info in target_infos:
            sockaddr = info[4]
            if sockaddr and len(sockaddr) > 0 and str(sockaddr[0]) in local_ips:
                return True
        return False

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

    def _build_apply_command(
        self,
        *,
        hosts_file: str,
        conf_path: Path,
        dry_run: bool,
        force_direct: bool = False,
    ) -> tuple[list[str], str]:
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
        use_direct = force_direct or (not use_service_default)
        if use_direct:
            cmd.extend(["--direct", "--conf", str(conf_path)])

        if dry_run:
            cmd.append("--dry-run")

        if force_direct:
            run_target = "local_direct_script"
        elif use_direct:
            run_target = "direct_script"
        else:
            run_target = "service_script"
        return cmd, run_target

    def _run_apply_for_host(self, *, node_name: str, conf_path: Path, dry_run: bool) -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile(mode="w", encoding="ascii", delete=False) as tf:
            tf.write(f"{node_name}\n")
            hosts_file = tf.name

        try:
            cmd, run_target = self._build_apply_command(
                hosts_file=hosts_file,
                conf_path=conf_path,
                dry_run=dry_run,
                force_direct=False,
            )
            command_str = self._tokens_to_str(cmd)
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            status = "DRY_RUN" if dry_run else ("APPLIED" if proc.returncode == 0 else "FAILED")
            return {
                "status": status,
                "run_target": run_target,
                "command": command_str,
                "returncode": int(proc.returncode),
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
        finally:
            Path(hosts_file).unlink(missing_ok=True)

    @classmethod
    def _normalize_freq_to_mhz(cls, raw_value: float) -> float:
        # GEOPM frequency signals/controls are generally Hz-scale values.
        # Fall back to plain MHz if a small value is reported.
        val = float(raw_value)
        if abs(val) >= 100_000.0:
            return val / 1e6
        return val

    @classmethod
    def _extract_readback_values(
        cls,
        stdout: str,
        target_indices: Iterable[int] | None = None,
    ) -> tuple[List[float], str, str, str]:
        target_set: set[int] | None = None
        if target_indices is not None:
            target_set = {int(i) for i in target_indices}

        geopm_by_signal: dict[str, dict[str, list[tuple[int, float]]]] = {}
        for line in str(stdout).splitlines():
            match = cls._GEOPMREAD_RE.match(line.strip())
            if not match:
                continue
            signal = str(match.group(1)).upper()
            domain = str(match.group(2)).lower()
            index_text = str(match.group(3))
            value_text = str(match.group(4))
            try:
                index = int(index_text)
                value = float(value_text)
            except ValueError:
                continue
            if not value == value:  # NaN check
                continue
            geopm_by_signal.setdefault(signal, {}).setdefault(domain, []).append(
                (index, cls._normalize_freq_to_mhz(value))
            )

        signal_priority = (
            "MSR::PERF_CTL:FREQ",
            "CPU_FREQUENCY_CONTROL",
            "CPU_FREQUENCY_MAX_CONTROL",
            "CPU_FREQUENCY_STATUS",
        )
        for signal_name in signal_priority:
            domain_map = geopm_by_signal.get(signal_name, {})
            if not domain_map:
                continue
            for domain_name in ("core", "cpu"):
                domain_values = domain_map.get(domain_name, [])
                if not domain_values:
                    continue
                if target_set is None:
                    return [value for _, value in domain_values], "geopmread", signal_name, "ok"
                filtered_values = [value for idx, value in domain_values if idx in target_set]
                if filtered_values:
                    return filtered_values, "geopmread", signal_name, "ok"

            if target_set is not None:
                return [], "geopmread", signal_name, "target_ids_not_found_in_readback"

        sysfs_values: list[float] = []
        for line in str(stdout).splitlines():
            match = cls._CPU_READBACK_RE.match(line.strip())
            if not match:
                continue
            try:
                cpu_idx = int(match.group(1))
                cpu_mhz = float(match.group(2))
            except ValueError:
                continue
            if target_set is not None and cpu_idx not in target_set:
                continue
            sysfs_values.append(cpu_mhz)
        if sysfs_values:
            return sysfs_values, "sysfs", "scaling_max_freq", "ok"

        return [], "none", "", "no_readback_values"

    def _build_verification_fields(
        self,
        *,
        requested_mhz: float,
        status: str,
        returncode: int,
        stdout: str,
        target_core_numbers: Iterable[int] | None = None,
    ) -> Dict[str, Any]:
        fields: dict[str, Any] = {
            "verify_tolerance_mhz": float(self.verify_tolerance_mhz),
            "readback_samples": 0,
            "readback_min_mhz": None,
            "readback_max_mhz": None,
            "readback_avg_mhz": None,
            "readback_error_mhz": None,
            "readback_abs_error_mhz": None,
            "readback_source": "none",
            "readback_signal": "",
            "verify_status": "NO_READBACK",
            "verify_reason": "no_readback_values",
        }

        if status == "DRY_RUN":
            fields["verify_status"] = "SKIP_DRY_RUN"
            fields["verify_reason"] = "dry_run"
            return fields

        if int(returncode) != 0:
            fields["verify_status"] = "FAIL"
            fields["verify_reason"] = "apply_command_failed"
            return fields

        readback_values, readback_source, readback_signal, readback_reason = self._extract_readback_values(
            stdout,
            target_indices=target_core_numbers,
        )
        fields["readback_source"] = readback_source
        fields["readback_signal"] = readback_signal
        if not readback_values:
            fields["verify_status"] = "FAIL" if readback_reason == "target_ids_not_found_in_readback" else "NO_READBACK"
            fields["verify_reason"] = readback_reason if readback_reason else "no_geopmread_or_sysfs_readback"
            return fields

        readback_min = min(readback_values)
        readback_max = max(readback_values)
        readback_avg = sum(readback_values) / float(len(readback_values))
        readback_error = readback_avg - float(requested_mhz)
        readback_abs_error = abs(readback_error)
        is_pass = readback_abs_error <= float(self.verify_tolerance_mhz)
        fields.update(
            {
                "readback_samples": int(len(readback_values)),
                "readback_min_mhz": float(readback_min),
                "readback_max_mhz": float(readback_max),
                "readback_avg_mhz": float(readback_avg),
                "readback_error_mhz": float(readback_error),
                "readback_abs_error_mhz": float(readback_abs_error),
                "verify_status": "PASS" if is_pass else "FAIL",
                "verify_reason": "within_tolerance" if is_pass else "outside_tolerance",
            }
        )
        return fields

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
        requested_mhz = float(frequency_hz) / 1e6
        verification = self._build_verification_fields(
            requested_mhz=requested_mhz,
            status=str(apply_result["status"]),
            returncode=int(apply_result["returncode"]),
            stdout=str(apply_result["stdout"]),
            target_core_numbers=cores,
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
                    "run_target": apply_result["run_target"],
                    "command": apply_result["command"],
                    "returncode": apply_result["returncode"],
                    "stdout": apply_result["stdout"],
                    "stderr": apply_result["stderr"],
                    **verification,
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
