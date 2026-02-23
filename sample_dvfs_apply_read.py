#!/usr/bin/env python3
"""Apply one frequency to a nodelist, sleep, then read back GEOPM controls."""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _resolve_hosts(nodelist: str) -> list[str]:
    proc = subprocess.run(
        ["scontrol", "show", "hostnames", nodelist],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        hosts = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if hosts:
            return hosts
    hosts = [token.strip() for token in nodelist.split(",") if token.strip()]
    if hosts:
        return hosts
    details = (proc.stderr or proc.stdout or "").strip()
    raise RuntimeError(f"could not resolve nodelist '{nodelist}': {details}")


def _write_host_configs(
    hosts: list[str],
    *,
    conf_dir: Path,
    freq_hz: int,
    control_kind: str,
) -> None:
    conf_dir.mkdir(parents=True, exist_ok=True)
    body = (
        "### RULE sample-dvfs\n"
        f"FREQ_HZ={freq_hz}\n"
        f"CONTROL_KIND={control_kind}\n"
    )
    for host in hosts:
        path = conf_dir / f"{host}.conf"
        path.write_text(body, encoding="ascii")
        print(f"[INFO] wrote {path} with FREQ_HZ={freq_hz}")


def _apply_configs(
    hosts: list[str],
    *,
    ssh_user: str | None,
    dry_run: bool,
) -> None:
    script_path = (Path(__file__).resolve().parent / "dvfs" / "run_geopm_apply_ssh.sh").resolve()
    if not script_path.exists():
        raise RuntimeError(f"missing helper script: {script_path}")

    with tempfile.NamedTemporaryFile(mode="w", encoding="ascii", delete=False) as tf:
        for host in hosts:
            tf.write(f"{host}\n")
        hosts_file = tf.name

    cmd = [str(script_path), "-H", hosts_file]
    if ssh_user:
        cmd[1:1] = ["-u", ssh_user]
    if dry_run:
        cmd.append("--dry-run")

    try:
        print(f"[INFO] applying via: {' '.join(cmd)}")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if proc.stdout:
            print(proc.stdout.rstrip())
        if proc.stderr:
            print(proc.stderr.rstrip(), file=sys.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"apply helper failed with exit {proc.returncode}")
    finally:
        Path(hosts_file).unlink(missing_ok=True)


def _is_local(host: str) -> bool:
    names = {
        "localhost",
        "127.0.0.1",
        socket.gethostname(),
        socket.gethostname().split(".", 1)[0],
        socket.getfqdn(),
    }
    return host in names


def _read_host(
    host: str,
    *,
    signal: str,
    domain: str,
    ssh_user: str | None,
) -> subprocess.CompletedProcess[str]:
    if domain == "core":
        id_list = "lscpu -p=CORE | awk -F, '/^[^#]/{print $1}' | sort -n -u"
    else:
        id_list = "lscpu -p=CPU | awk -F, '/^[^#]/{print $1}'"

    remote_cmd = (
        f"for i in $({id_list}); do "
        f"printf '{domain} %s: ' \"$i\"; "
        f"geopmread {signal} {domain} \"$i\" 2>/dev/null || echo read-failed; "
        "done"
    )
    if _is_local(host):
        return subprocess.run(
            ["bash", "-lc", remote_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    target = f"{ssh_user}@{host}" if ssh_user else host
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", target, "bash", "-lc", remote_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Apply a max->min frequency sweep to all nodes in a nodelist and read back.",
    )
    p.add_argument("--nodelist", required=True, help="Slurm nodelist expression or comma list")
    p.add_argument("--max-freq-mhz", type=float, required=True, help="Starting frequency in MHz")
    p.add_argument("--min-freq-mhz", type=float, required=True, help="Ending frequency in MHz")
    p.add_argument("--interval-mhz", type=int, default=200, help="Step size in MHz (default: 200)")
    p.add_argument("--sleep-seconds", type=float, default=2.0, help="Sleep before readback")
    p.add_argument(
        "--control-kind",
        default="CORE_MAX",
        choices=["AUTO", "CORE_MAX", "CPU"],
        help="Control kind used by geopm_apply.sh config",
    )
    p.add_argument("--conf-dir", type=Path, default=Path("/shared/geopm/freq"))
    p.add_argument("--read-signal", default="CPU_FREQUENCY_MAX_CONTROL")
    p.add_argument("--read-domain", default="core", choices=["core", "cpu"])
    p.add_argument("--ssh-user", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.max_freq_mhz <= 0 or args.min_freq_mhz <= 0:
        print("[ERR] --max-freq-mhz and --min-freq-mhz must be positive", file=sys.stderr)
        return 2
    if args.min_freq_mhz > args.max_freq_mhz:
        print("[ERR] --min-freq-mhz cannot be greater than --max-freq-mhz", file=sys.stderr)
        return 2
    if args.interval_mhz <= 0:
        print("[ERR] --interval-mhz must be positive", file=sys.stderr)
        return 2

    try:
        hosts = _resolve_hosts(args.nodelist)
    except Exception as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 3
    print(f"[INFO] hosts: {', '.join(hosts)}")

    max_mhz = int(round(args.max_freq_mhz))
    min_mhz = int(round(args.min_freq_mhz))
    targets: list[int] = []
    current = max_mhz
    while current >= min_mhz:
        targets.append(current)
        current -= args.interval_mhz
    if not targets or targets[-1] != min_mhz:
        targets.append(min_mhz)

    rc = 0
    for target_mhz in targets:
        freq_hz = int(round(target_mhz * 1e6))
        print(f"\n[STEP] applying {target_mhz} MHz across {len(hosts)} host(s)")
        try:
            _write_host_configs(
                hosts,
                conf_dir=args.conf_dir,
                freq_hz=freq_hz,
                control_kind=args.control_kind,
            )
            _apply_configs(hosts, ssh_user=args.ssh_user, dry_run=args.dry_run)
        except Exception as exc:
            print(f"[ERR] apply failed at {target_mhz} MHz: {exc}", file=sys.stderr)
            return 4

        if args.dry_run:
            continue

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

        for host in hosts:
            print(f"\n[{host}] readback {args.read_signal}/{args.read_domain} @ {target_mhz} MHz")
            result = _read_host(
                host,
                signal=args.read_signal,
                domain=args.read_domain,
                ssh_user=args.ssh_user,
            )
            if result.stdout:
                print(result.stdout.rstrip())
            if result.returncode != 0:
                rc = result.returncode
                print(f"[WARN] readback failed on {host}: exit={result.returncode}", file=sys.stderr)
                if result.stderr:
                    print(result.stderr.rstrip(), file=sys.stderr)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
