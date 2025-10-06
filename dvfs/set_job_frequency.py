"""Helpers to apply GEOPM frequency reductions to Slurm jobs."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from . import job_cores

MAX_FREQ_MHZ_DEFAULT = 2200.0
MIN_FREQ_MHZ_DEFAULT = 800.0
CONFIG_DIR_DEFAULT = "/shared/geopm/freq"
NODES_FILE_DEFAULT = (Path(__file__).resolve().parent.parent / "data" / "nodes.txt").resolve()


class SetJobFrequencyError(RuntimeError):
    """Raised when applying GEOPM frequency controls fails."""


def require_command(cmd: str) -> None:
    if shutil.which(cmd) is None:
        raise FileNotFoundError(f"required command not found in PATH: {cmd}")


def _format_frequency_token(hz: float) -> str:
    """Return a scientific notation representation suitable for shell usage."""
    token = f"{hz:.6e}"
    mantissa, exponent = token.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    sign = ""
    if exponent.startswith(("+", "-")):
        sign = exponent[0] if exponent[0] == "-" else ""
        exponent = exponent[1:]
    exponent = exponent.lstrip("0") or "0"
    return f"{mantissa}e{sign}{exponent}"


def format_remote_loop(
    cores: List[int],
    geopm_cmd: str,
    signal: str,
    domain: str,
    value_token: str,
) -> str:
    core_list = " ".join(str(c) for c in cores)
    signal_q = shlex.quote(signal)
    domain_q = shlex.quote(domain)
    value_q = shlex.quote(value_token)
    loop = (
        "set -euo pipefail; "
        f"for core in {core_list}; do "
        f"{geopm_cmd} {signal_q} {domain_q} $core {value_q}; "
        "sleep 1; "
        "done"
    )
    return loop


def compute_frequency_from_reduction(
    reduction: float,
    *,
    max_freq_mhz: float = MAX_FREQ_MHZ_DEFAULT,
    min_freq_mhz: float = MIN_FREQ_MHZ_DEFAULT,
) -> Tuple[float, float, str]:
    """Return (freq_hz, freq_mhz, freq_token) for a fractional reduction."""
    if not 0.0 <= reduction <= 1.0:
        raise ValueError("reduction must be between 0 and 1 inclusive")
    if max_freq_mhz <= 0:
        raise ValueError("max frequency must be positive")
    if min_freq_mhz <= 0:
        raise ValueError("min frequency must be positive")
    if min_freq_mhz > max_freq_mhz:
        raise ValueError("min frequency must be <= max frequency")

    freq_mhz = max_freq_mhz * (1.0 - reduction)
    if freq_mhz < min_freq_mhz:
        freq_mhz = min_freq_mhz
    freq_mhz = min(freq_mhz, max_freq_mhz)

    freq_hz = freq_mhz * 1e6
    freq_token = _format_frequency_token(freq_hz)
    return freq_hz, freq_mhz, freq_token


def apply_reduction(
    jobid: str,
    reduction: float,
    *,
    geopmwrite_cmd: str = "geopmwrite",
    signal: str = "CPU_FREQUENCY_MAX_CONTROL",
    domain: str = "core",
    dry_run: bool = False,
    max_freq_mhz: float = MAX_FREQ_MHZ_DEFAULT,
    min_freq_mhz: float = MIN_FREQ_MHZ_DEFAULT,
    conf_dir: str = CONFIG_DIR_DEFAULT,
    nodes_file: Path | str = NODES_FILE_DEFAULT,
) -> float:
    """Apply a frequency reduction to all cores of a Slurm job.

    Returns the target frequency in Hz.
    """

    freq_hz, freq_mhz, freq_token = compute_frequency_from_reduction(
        reduction,
        max_freq_mhz=max_freq_mhz,
        min_freq_mhz=min_freq_mhz,
    )

    srun_cmd = shutil.which("srun")
    if srun_cmd is None:
        raise FileNotFoundError("srun command not found in PATH")

    geopm_tokens = shlex.split(geopmwrite_cmd)
    if not geopm_tokens:
        raise ValueError("invalid geopmwrite command")
    require_command(geopm_tokens[0])
    geopm_cmd = " ".join(shlex.quote(tok) for tok in geopm_tokens)

    state, host_map = job_cores.collect_host_cores(jobid)

    if state and state.upper() != "RUNNING":
        print(f"[WARN] job state: {state}")

    conf_dir_path = Path(conf_dir)
    if not dry_run:
        conf_dir_path.mkdir(parents=True, exist_ok=True)

    hosts = sorted(host_map.keys())

    overall_ok = True
    for host, cores in host_map.items():
        print(
            f"[INFO] Setting {len(cores)} cores on {host} to {freq_mhz:.3f} MHz "
            f"using {signal}/{domain}"
        )
        if not dry_run:
            conf_path = conf_dir_path / f"{host}.conf"
            core_list = " ".join(str(c) for c in cores)
            body = (
                "### RULE dvfs-manager\n"
                f"FREQ_HZ={freq_hz:.0f}\n"
                f"CORES=\"{core_list}\"\n"
                "CONTROL_KIND=CORE_MAX\n"
            )
            conf_path.write_text(body, encoding="ascii")
            print(
                f"[INFO] Wrote {conf_path} with {len(cores)} cores @ {freq_mhz:.3f} MHz"
            )

        if dry_run:
            core_list = " ".join(str(c) for c in cores)
            print(
                f"[DRY-RUN] Would write {conf_dir_path / f'{host}.conf'} "
                f"with FREQ_HZ={freq_hz:.0f} and CORES={core_list}"
            )
            continue
        
        return freq_hz

    if overall_ok:
        script_path = (Path(__file__).resolve().parent / "run_geopm_apply_ssh.sh").resolve()
        if not script_path.exists():
            raise SetJobFrequencyError(f"helper script not found: {script_path}")

        nodes_file_path = Path(nodes_file)
        if not nodes_file_path.exists():
            raise SetJobFrequencyError(f"nodes file not found: {nodes_file_path}")

        ssh_cmd = [str(script_path), "-u", "ridl", "-H", str(nodes_file_path)]
        run_result = subprocess.run(ssh_cmd, text=True, capture_output=True, check=False)
        if run_result.stdout:
            print(run_result.stdout, end="")
        if run_result.stderr:
            print(run_result.stderr, file=sys.stderr, end="")
        if run_result.returncode != 0:
            raise SetJobFrequencyError(
                f"run_geopm_apply_ssh.sh failed (exit {run_result.returncode})"
            )

        print("[DONE] Applied GEOPM frequency controls to all nodes")
        return freq_hz

    raise SetJobFrequencyError("one or more srun invocations failed")


__all__ = [
    "MAX_FREQ_MHZ_DEFAULT",
    "MIN_FREQ_MHZ_DEFAULT",
    "CONFIG_DIR_DEFAULT",
    "NODES_FILE_DEFAULT",
    "SetJobFrequencyError",
    "apply_reduction",
    "compute_frequency_from_reduction",
]
