from __future__ import annotations

import shlex
import socket
import subprocess
from typing import Any, Dict, Iterable, List


class DVFSController:
    """Apply DVFS controls by node/core/frequency using geopmwrite."""

    def __init__(
        self,
        *,
        geopmwrite_cmd: str = "geopmwrite",
        signal: str = "CPU_FREQUENCY_MAX_CONTROL",
        domain: str = "core",
        ssh_user: str | None = None,
        ssh_connect_timeout_s: int = 5,
    ) -> None:
        self.geopmwrite_cmd = geopmwrite_cmd
        self.signal = signal
        self.domain = domain
        self.ssh_user = ssh_user
        self.ssh_connect_timeout_s = int(ssh_connect_timeout_s)

    @staticmethod
    def _is_local_host(host: str) -> bool:
        names = {
            "localhost",
            "127.0.0.1",
            socket.gethostname(),
            socket.gethostname().split(".", 1)[0],
            socket.getfqdn(),
        }
        return host in names

    @staticmethod
    def _mhz_to_hz(freq_mhz: float) -> int:
        if freq_mhz <= 0:
            raise ValueError("frequency_mhz must be > 0")
        return int(round(freq_mhz * 1e6))

    def _build_geopmwrite_tokens(self, *, core_number: int, frequency_hz: int) -> List[str]:
        base = shlex.split(self.geopmwrite_cmd)
        if not base:
            raise ValueError("invalid geopmwrite command")
        return base + [self.signal, self.domain, str(int(core_number)), str(int(frequency_hz))]

    @staticmethod
    def _tokens_to_str(tokens: Iterable[str]) -> str:
        return " ".join(shlex.quote(token) for token in tokens)

    def _run_local(self, tokens: List[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(tokens, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)

    def _run_remote(self, node_name: str, tokens: List[str]) -> subprocess.CompletedProcess[str]:
        target = f"{self.ssh_user}@{node_name}" if self.ssh_user else node_name
        remote_cmd = f"bash -lc {shlex.quote(self._tokens_to_str(tokens))}"
        ssh_cmd = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self.ssh_connect_timeout_s}",
            target,
            remote_cmd,
        ]
        return subprocess.run(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)

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
        if frequency_hz is None:
            if frequency_mhz is None:
                raise ValueError("Provide either frequency_mhz or frequency_hz.")
            frequency_hz = self._mhz_to_hz(float(frequency_mhz))
        if frequency_hz <= 0:
            raise ValueError("frequency_hz must be > 0")

        tokens = self._build_geopmwrite_tokens(
            core_number=int(core_number),
            frequency_hz=int(frequency_hz),
        )
        command_str = self._tokens_to_str(tokens)
        run_target = "local" if self._is_local_host(node_name) else "remote"

        if dry_run:
            return {
                "node_name": node_name,
                "core_number": int(core_number),
                "frequency_hz": int(frequency_hz),
                "frequency_mhz": float(frequency_hz) / 1e6,
                "status": "DRY_RUN",
                "run_target": run_target,
                "command": command_str,
                "returncode": 0,
                "stdout": "",
                "stderr": "",
            }

        proc = self._run_local(tokens) if run_target == "local" else self._run_remote(node_name, tokens)
        status = "APPLIED" if proc.returncode == 0 else "FAILED"
        return {
            "node_name": node_name,
            "core_number": int(core_number),
            "frequency_hz": int(frequency_hz),
            "frequency_mhz": float(frequency_hz) / 1e6,
            "status": status,
            "run_target": run_target,
            "command": command_str,
            "returncode": int(proc.returncode),
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }

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
        results: List[Dict[str, Any]] = []
        for core in core_numbers:
            results.append(
                self.apply_frequency(
                    node_name=node_name,
                    core_number=int(core),
                    frequency_mhz=frequency_mhz,
                    frequency_hz=frequency_hz,
                    dry_run=dry_run,
                )
            )
        return results

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
