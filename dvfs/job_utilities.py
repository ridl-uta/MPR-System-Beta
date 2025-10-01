"""Slurm job utilities for GEOPM DVFS control."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Job core discovery (port of job_cores.py)
# ---------------------------------------------------------------------------


class JobCoresError(Exception):
    """Raised when job core discovery fails."""

    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


def _run_command(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    """Run command and return CompletedProcess without raising on failure."""

    try:
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print(f"[ERR] command not found: {cmd[0]}", file=sys.stderr)
        sys.exit(127)


def expand_list(cpulist: str) -> str:
    """Expand compact CPU list (e.g. 0-3,8,10-11) into space-separated numbers."""

    numbers: List[int] = []
    for chunk in cpulist.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            start_str, end_str = chunk.split('-', 1)
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError:
                continue
            step = 1 if end >= start else -1
            numbers.extend(range(start, end + step, step))
        else:
            try:
                numbers.append(int(chunk))
            except ValueError:
                continue
    return ' '.join(str(n) for n in numbers)


def sanitize_host(host: str) -> str:
    """Sanitize hostname for use in shell variable names."""

    return re.sub(r"[^A-Za-z0-9_]", "_", host)


def _extract_nodelist(jobid: str) -> tuple[str, str]:
    """Return (compact_nodelist, job_state)."""

    squeue_nodes = _run_command(["squeue", "-h", "-j", jobid, "-o", "%N"])
    compact = squeue_nodes.stdout.strip()

    squeue_state = _run_command(["squeue", "-h", "-j", jobid, "-o", "%T"])
    state = squeue_state.stdout.strip()

    if not compact or not state:
        job_info = _run_command(["scontrol", "show", "job", "-o", jobid])
        if not compact:
            match = re.search(r"\bNodeList=([^ ]+)", job_info.stdout)
            if match:
                candidate = match.group(1).strip()
                if candidate and candidate != "(null)":
                    compact = candidate
        if not state:
            match_state = re.search(r"\bJobState=([^ ]+)", job_info.stdout)
            if match_state:
                state = match_state.group(1).strip()

    return compact, state


def collect_host_cores(jobid: str) -> tuple[str, Dict[str, List[int]]]:
    """Return (state, cores_per_host) for the provided job ID."""

    compact_nodelist, state = _extract_nodelist(jobid)

    if not compact_nodelist:
        raise JobCoresError("no NodeList (job pending/not found)", exit_code=2)

    hostnames = _run_command(["scontrol", "show", "hostnames", compact_nodelist])
    hosts = [line.strip() for line in hostnames.stdout.splitlines() if line.strip()]
    if not hosts:
        raise JobCoresError("no NodeList (job pending/not found)", exit_code=2)

    node_count = len(hosts)

    probe_cmd = [
        "srun",
        f"--jobid={jobid}",
        "-N",
        str(node_count),
        "--ntasks-per-node=1",
        "--overlap",
        "--immediate=10",
        "bash",
        "-lc",
        'printf "%s " "$(hostname -s)"; '
        "awk '/^Cpus_allowed_list:/ {print $2; exit}' /proc/self/status",
    ]

    probe = _run_command(probe_cmd)
    output = probe.stdout.strip()

    if not output:
        messages = []
        stderr_text = probe.stderr.strip()
        if stderr_text:
            messages.append(stderr_text)
        messages.append("no output from srun probe (job fully busy or overlap disabled)")
        raise JobCoresError('\n'.join(messages), exit_code=3)

    host_map: Dict[str, List[int]] = {}
    for line in output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 2:
            continue
        host, cpulist = parts[0], parts[1]
        expanded = expand_list(cpulist)
        if not expanded:
            continue
        try:
            host_map[host] = [int(token) for token in expanded.split()]
        except ValueError:
            continue

    if not host_map:
        raise JobCoresError("no cores discovered in probe output", exit_code=3)

    return state, host_map


def list_running_slurm_jobs() -> List[Tuple[str, str]]:
    """Return (job_id, job_name) tuples for all running Slurm jobs."""

    proc = _run_command(["squeue", "-h", "-t", "RUNNING", "-o", "%i %j"])
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise RuntimeError(
            f"squeue failed with exit {proc.returncode}: {stderr or 'no details'}"
        )

    jobs: List[Tuple[str, str]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        job_id = parts[0]
        job_name = parts[1] if len(parts) > 1 else ""
        jobs.append((job_id, job_name))

    return jobs


# ---------------------------------------------------------------------------
# Frequency apply helpers (formerly set_job_frequency.py)
# ---------------------------------------------------------------------------


MAX_FREQ_MHZ_DEFAULT = 2200.0
MIN_FREQ_MHZ_DEFAULT = 800.0
CONFIG_DIR_DEFAULT = "/shared/geopm/freq"
NODES_FILE_DEFAULT = (Path(__file__).resolve().parent.parent / "data" / "nodes.txt").resolve()


class SetJobFrequencyError(RuntimeError):
    """Raised when applying GEOPM frequency controls fails."""


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
    """Apply a frequency reduction to all cores of a Slurm job."""

    freq_hz, freq_mhz, freq_token = compute_frequency_from_reduction(
        reduction,
        max_freq_mhz=max_freq_mhz,
        min_freq_mhz=min_freq_mhz,
    )

    _ = geopmwrite_cmd, signal, domain  # retained for backward compatibility

    state, host_map = collect_host_cores(jobid)

    if state and state.upper() != "RUNNING":
        print(f"[WARN] job state: {state}")

    conf_dir_path = Path(conf_dir)
    if not dry_run:
        conf_dir_path.mkdir(parents=True, exist_ok=True)

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
        else:
            core_list = " ".join(str(c) for c in cores)
            print(
                f"[DRY-RUN] Would write {conf_dir_path / f'{host}.conf'} "
                f"with FREQ_HZ={freq_hz:.0f} and CORES={core_list}"
            )

    if dry_run:
        script_path = (Path(__file__).resolve().parent / "run_geopm_apply_ssh.sh").resolve()
        nodes_file_path = Path(nodes_file)
        print(
            f"[DRY-RUN] Would run: {script_path} -u ridl -H {nodes_file_path}"
        )
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
    "JobCoresError",
    "SetJobFrequencyError",
    "MAX_FREQ_MHZ_DEFAULT",
    "MIN_FREQ_MHZ_DEFAULT",
    "CONFIG_DIR_DEFAULT",
    "NODES_FILE_DEFAULT",
    "collect_host_cores",
    "list_running_slurm_jobs",
    "expand_list",
    "sanitize_host",
    "compute_frequency_from_reduction",
    "apply_reduction",
]
