#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Power monitor: poll APC PDUs and log per-node totals."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import telnetlib
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Iterable, List, Optional

from ..overload_detection.overload_detection import make_simple_overload_ctx, simple_overload_update
from ..overload_detection.overload_handler import LoadShedder

# ---- Patterns ----
OUTLET_LINE_RE = re.compile(r"^\s*(\d+):.*?(\d+)\s*W\s*$")
PROMPT_RE_BYTES = re.compile(br"[>#]\s*$")
PROMPT_RE_STRING = re.compile(r"[>#]\s*$")
_APC_PROMPT_B = re.compile(br'(?:\r?\n)?apc>\s*$', re.M)
_APC_PROMPT_S = re.compile(r'(?:\r?\n)?apc>\s*$', re.M)


class TelnetAPC:
    def __init__(self, host: str, user: str, pwd: str, port: int = 23, timeout: int = 8):
        self.host, self.user, self.pwd, self.port, self.timeout = host, user, pwd, port, timeout
        self.tn: Optional[telnetlib.Telnet] = None
        self._prompts = [_APC_PROMPT_B, PROMPT_RE_BYTES]

    def connect(self):
        self.close()
        self.tn = telnetlib.Telnet(self.host, port=self.port, timeout=self.timeout)
        try:
            self.tn.expect([b"User Name :", b"Username:"], self.timeout)
            self.tn.write(self.user.encode() + b"\r\n")
            self.tn.expect([b"Password  :", b"Password:"], self.timeout)
            self.tn.write(self.pwd.encode() + b"\r\n")
            if self.tn.expect(self._prompts, self.timeout)[0] == -1:
                raise TimeoutError("login: prompt not seen")
            try:
                import socket
                self.tn.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception:
            self.close()
            raise

    def close(self):
        if self.tn:
            try:
                self.tn.write(b"exit\r\n")
            except Exception:
                pass
            try:
                self.tn.close()
            finally:
                self.tn = None

    def ensure_alive(self) -> None:
        if not self.tn:
            self.connect()
            return
        try:
            self.tn.write(b"\r\n")
            if self.tn.expect(self._prompts, 2.0)[0] == -1:
                raise TimeoutError
        except Exception:
            self.connect()

    def run(self, cmd: str, overall: float = 15.0) -> str:
        self.ensure_alive()
        try:
            return self._run_once(cmd, overall)
        except (TimeoutError, ConnectionError, EOFError):
            self.connect()
            return self._run_once(cmd, overall)

    def _run_once(self, cmd: str, overall: float) -> str:
        start = time.monotonic()
        self.tn.write(cmd.encode() + b"\r\n")  # type: ignore[union-attr]
        text = self._read_until_prompt(start + overall)
        first_token = cmd.split()[0]
        echo_re = re.compile(rf'^(?:apc>\s*)?{re.escape(first_token)}\b.*\r?\n', re.I | re.M)
        return echo_re.sub("", text, count=1).strip()

    def _read_until_prompt(self, deadline: float) -> str:
        chunks: List[bytes] = []
        while True:
            remaining = max(0.1, deadline - time.monotonic())
            try:
                idx, _, text = self.tn.expect(self._prompts, remaining)  # type: ignore[union-attr]
            except EOFError:
                raise ConnectionError("telnet connection closed")
            if text:
                chunks.append(text)
            if idx != -1:
                break
            if remaining <= 0.11:
                raise TimeoutError("prompt not seen")
        out = b"".join(chunks).decode("utf-8", errors="replace")
        out = _APC_PROMPT_S.sub("", out)
        out = PROMPT_RE_STRING.sub("", out).strip()
        return out


class PowerMonitor:
    def __init__(
        self,
        *,
        map_path: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        csv_path: Optional[str] = None,
        interval: float = 1.0,
        deadline: float = 15.0,
        detect_overload: bool = False,
        threshold_w: float = 900.0,
        hysteresis_w: float = 20.0,
        min_over_s: int = 10,
        cooldown_s: int = 30,
        spike_threshold_w: Optional[float] = None,
        spike_margin_w: float = 50.0,
        spike_persistence_s: float = 1.0,
        ramp_fast_window_s: float = 5.0,
        ramp_slow_window_s: float = 60.0,
        ramp_delta_w: float = 40.0,
        ramp_reset_w: float = 10.0,
        ramp_slope_w: float = 15.0,
        ramp_slope_reset_w: float = 5.0,
        handled_window_s: float = 10.0,
        peak_window_s: Optional[float] = None,
        shed_watts: float = 0.0,
        shed_margin: float = 0.0,
        shed_delay: float = 5.0,
        events_csv: Optional[str] = None,
    ) -> None:
        self.map_path = map_path
        self.user = user
        self.password = password
        self.csv_path = csv_path
        self.interval = float(interval)
        self.deadline = float(deadline)
        self.detect_overload = detect_overload
        self.threshold_w = float(threshold_w)
        self.hysteresis_w = float(hysteresis_w)
        self.min_over_s = int(min_over_s)
        self.cooldown_s = int(cooldown_s)
        self.spike_threshold_w = spike_threshold_w
        self.spike_margin_w = float(spike_margin_w)
        self.spike_persistence_s = float(spike_persistence_s)
        self.ramp_fast_window_s = float(ramp_fast_window_s)
        self.ramp_slow_window_s = float(ramp_slow_window_s)
        self.ramp_delta_w = float(ramp_delta_w)
        self.ramp_reset_w = float(ramp_reset_w)
        self.ramp_slope_w = float(ramp_slope_w)
        self.ramp_slope_reset_w = float(ramp_slope_reset_w)
        self.handled_window_s = float(handled_window_s)
        self.peak_window_s = peak_window_s
        self.shed_watts = float(shed_watts)
        self.shed_margin = float(shed_margin)
        self.shed_delay = float(shed_delay)
        self.events_csv = events_csv

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sessions: List[TelnetAPC] = []
        self._maps: List[Dict[int, str]] = []
        self._hosts: List[str] = []
        self._command: Optional[str] = None
        self._od_ctx = None
        self._load_shedder: Optional[LoadShedder] = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        if not self._ready():
            self._log("disabled (missing map/user/password/csv)")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="PowerMonitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._teardown()
        self._thread = None

    def run_forever(self) -> None:
        if not self._ready():
            self._log("disabled (missing map/user/password/csv)")
            return
        self._stop.clear()
        try:
            self._run_loop()
        finally:
            self._teardown()

    # ------------------------------------------------------------------
    def _ready(self) -> bool:
        return bool(self.map_path and self.user and self.password and self.csv_path)

    def _configure(self) -> None:
        with open(self.map_path, 'r') as f:  # type: ignore[arg-type]
            cfg = json.load(f)
        outlets = int(cfg.get("outlets", 24))
        self._command = f"olReading 1-{outlets} power"

        self._sessions = []
        self._maps = []
        self._hosts = []
        for p in cfg.get("pdus", []):
            host = p.get("host")
            if not host:
                continue
            port = int(p.get("port", 23))
            pmap = {int(k): str(v) for k, v in p.get("map", {}).items()}
            sess = TelnetAPC(host, self.user, self.password, port=port, timeout=8)
            try:
                sess.connect()
            except Exception as exc:
                self._log(f"ERROR: cannot connect to {host}:{port} -> {exc}")
            self._sessions.append(sess)
            self._maps.append(pmap)
            self._hosts.append(f"{host}:{port}")

        if self.detect_overload:
            self._od_ctx = make_simple_overload_ctx(
                sample_period_s=self.interval,
                threshold_w=self.threshold_w,
                hysteresis_w=self.hysteresis_w,
                min_over_s=self.min_over_s,
                cooldown_s=self.cooldown_s,
                spike_threshold_w=self.spike_threshold_w,
                spike_margin_w=self.spike_margin_w,
                spike_persistence_s=self.spike_persistence_s,
                ramp_fast_window_s=self.ramp_fast_window_s,
                ramp_slow_window_s=self.ramp_slow_window_s,
                ramp_delta_w=self.ramp_delta_w,
                ramp_reset_w=self.ramp_reset_w,
                ramp_slope_w=self.ramp_slope_w,
                ramp_slope_reset_w=self.ramp_slope_reset_w,
                handled_window_s=self.handled_window_s,
                enable_handled=False,
                peak_window_s=self.peak_window_s,
            )
            if self.shed_watts > 0 or self.shed_margin > 0:
                self._load_shedder = LoadShedder(
                    threshold_w=self.threshold_w,
                    sample_period_s=self.interval,
                    margin_w=self.shed_margin,
                    activation_delay_s=self.shed_delay,
                    max_reduction_w=self.shed_watts if self.shed_watts > 0 else None,
                )
                self._od_ctx['handled_enabled'] = True
            else:
                self._load_shedder = None
        else:
            self._od_ctx = None
            self._load_shedder = None

    def _teardown(self) -> None:
        for sess in self._sessions:
            try:
                sess.close()
            except Exception:
                pass
        self._sessions.clear()
        self._maps.clear()
        self._hosts.clear()
        self._od_ctx = None
        self._load_shedder = None

    def _run_loop(self) -> None:
        try:
            self._configure()
        except Exception as exc:
            self._log(f"failed to configure: {exc}")
            return

        if not self._sessions:
            self._log("no PDU sessions configured")
            return

        period = self.interval
        next_tick = time.monotonic()
        while not self._stop.is_set():
            next_tick += period
            self._sample_once()
            delay = next_tick - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_tick = time.monotonic()

    def _sample_once(self) -> None:
        ts = datetime.now(timezone.utc).astimezone()
        ts_iso = ts.isoformat(timespec="milliseconds")
        node_totals: Dict[str, int] = {}

        for sess, pmap, hostlabel in zip(self._sessions, self._maps, self._hosts):
            try:
                raw_outlets = sess.run(self._command, overall=self.deadline)  # type: ignore[arg-type]
                readings = parse_readings(raw_outlets)
                for outlet, node in pmap.items():
                    node_totals[node] = node_totals.get(node, 0) + readings.get(outlet, 0)
            except Exception as exc:
                self._log(f"{ts_iso} WARN: poll failed for {hostlabel} -> {exc}")

        if self.csv_path:
            append_csv(self.csv_path, ts_iso, node_totals)

        if not self._od_ctx:
            return

        raw_total = float(sum(node_totals.values()))
        adjusted_total = self._load_shedder.adjust(raw_total) if self._load_shedder else raw_total
        self._od_ctx['last_raw_sample'] = raw_total

        if self._load_shedder:
            peak_raw = self._od_ctx.get('peak_window_max_raw', raw_total)
            reduction = max(0.0, peak_raw - adjusted_total)
            if self._load_shedder.active and reduction > 0:
                base_low = self._od_ctx['T_high'] - self._od_ctx['min_hysteresis']
                new_low = max(0.0, self._od_ctx['T_high'] - reduction)
                self._od_ctx['T_low'] = min(new_low, base_low)
            else:
                self._od_ctx['T_low'] = self._od_ctx['T_high'] - self._od_ctx['min_hysteresis']

        event, info = simple_overload_update(self._od_ctx, ts, adjusted_total)

        if self._load_shedder:
            self._load_shedder.handle_event(event)

        if not event:
            return

        if isinstance(event, tuple) and event[0] == 'OVERLOAD_END':
            payload = info or {}
            if self.events_csv:
                self._write_event(payload)
            self._log(
                f"{ts_iso} OVERLOAD_END dur={payload.get('duration_s', 0):.1f}s "
                f"avg={payload.get('avg_watts', 0):.1f}W peak={payload.get('peak_watts', 0):.1f}W"
            )
        elif event == 'OVERLOAD_START':
            self._log(
                f"{ts_iso} OVERLOAD_START raw={raw_total:.1f}W adjusted={adjusted_total:.1f}W "
                f"T_high={self._od_ctx['T_high']:.1f}W T_low={self._od_ctx['T_low']:.1f}W"
            )
        elif event == 'OVERLOAD_HANDLED':
            self._log(
                f"{ts_iso} OVERLOAD_HANDLED raw={raw_total:.1f}W adjusted={adjusted_total:.1f}W"
            )
        elif event == 'RAMP_PREDICTED':
            payload = info or {}
            self._log(
                f"{ts_iso} RAMP_PREDICTED fast_avg={payload.get('fast_avg', 0):.1f}W "
                f"slow_avg={payload.get('slow_avg', 0):.1f}W slope={payload.get('slope_w_per_s', 0):.1f}W/s"
            )
        elif event == 'RAMP_WARNING':
            payload = info or {}
            self._log(
                f"{ts_iso} RAMP_WARNING fast_avg={payload.get('fast_avg', 0):.1f}W "
                f"slow_avg={payload.get('slow_avg', 0):.1f}W delta={payload.get('delta_watts', 0):.1f}W"
            )
        elif event == 'SPIKE_WARNING':
            payload = info or {}
            self._log(
                f"{ts_iso} SPIKE_WARNING raw={raw_total:.1f}W adjusted={adjusted_total:.1f}W "
                f"threshold={payload.get('threshold', 0):.1f}W duration={payload.get('duration_s', 0):.1f}s"
            )

    def _write_event(self, info: dict) -> None:
        if not self.events_csv:
            return
        need_header = not os.path.exists(self.events_csv) or os.path.getsize(self.events_csv) == 0
        with open(self.events_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if need_header:
                writer.writerow(["start", "end", "duration_s", "avg_watts", "peak_watts"])
            writer.writerow([
                info.get('start').isoformat() if info.get('start') else '',
                info.get('end').isoformat() if info.get('end') else '',
                f"{info.get('duration_s', 0):.3f}",
                f"{info.get('avg_watts', 0):.3f}",
                f"{info.get('peak_watts', 0):.3f}",
            ])

    def _log(self, message: str) -> None:
        print(f"[PowerMonitor] {message}")


def parse_readings(text: str) -> Dict[int, int]:
    res: Dict[int, int] = {}
    for ln in text.splitlines():
        m = OUTLET_LINE_RE.match(ln)
        if m:
            res[int(m.group(1))] = int(m.group(2))
    return res


def append_csv(path: str, ts_iso: str, node_totals: Dict[str, int]) -> None:
    need_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    header_nodes: List[str] = []

    if not need_header:
        with open(path, "r", newline="") as f:
            r = csv.reader(f)
            try:
                header = next(r)
            except StopIteration:
                need_header = True
            else:
                if header and header[0].lower() == "timestamp" and len(header) >= 2:
                    header_nodes = header[1:-1] if len(header) >= 3 else []
                else:
                    need_header = True

    if need_header:
        header_nodes = sorted(node_totals.keys())
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", *header_nodes, "total_watts"])
            total = sum(node_totals.get(n, 0) for n in header_nodes)
            w.writerow([ts_iso, *[node_totals.get(n, 0) for n in header_nodes], total])
        return

    total = sum(node_totals.get(n, 0) for n in header_nodes)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([ts_iso, *[node_totals.get(n, 0) for n in header_nodes], total])


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Sample APC PDUs; log per-node totals to CSV at a fixed cadence.")
    ap.add_argument("--map", help="mapping.json with pdus[].host/port/map and outlets")
    ap.add_argument("--user", help="Telnet username")
    ap.add_argument("--password", help="Telnet password")
    ap.add_argument("--csv", help="CSV file to append samples")
    ap.add_argument("--interval", type=float, default=1.0, help="Seconds between samples (default 1.0)")
    ap.add_argument("--deadline", type=float, default=15.0, help="Per-command deadline seconds (default 15)")
    ap.add_argument("--detect-overload", action="store_true")
    ap.add_argument("--threshold-w", type=float, default=900.0)
    ap.add_argument("--hysteresis-w", type=float, default=20.0)
    ap.add_argument("--min-over", type=int, default=10)
    ap.add_argument("--cooldown", type=int, default=30)
    ap.add_argument("--events-csv", type=str, default=None)
    ap.add_argument("--spike-threshold-w", type=float, default=None)
    ap.add_argument("--spike-margin-w", type=float, default=50.0)
    ap.add_argument("--spike-persistence", type=float, default=1.0)
    ap.add_argument("--ramp-fast-window", type=float, default=5.0)
    ap.add_argument("--ramp-slow-window", type=float, default=60.0)
    ap.add_argument("--ramp-delta-w", type=float, default=40.0)
    ap.add_argument("--ramp-reset-w", type=float, default=10.0)
    ap.add_argument("--ramp-slope-w", type=float, default=15.0)
    ap.add_argument("--ramp-slope-reset-w", type=float, default=5.0)
    ap.add_argument("--handled-window", type=float, default=10.0)
    ap.add_argument("--peak-window", type=float, default=None)
    ap.add_argument("--shed-watts", type=float, default=0.0)
    ap.add_argument("--shed-margin", type=float, default=0.0)
    ap.add_argument("--shed-delay", type=float, default=5.0)
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    monitor = PowerMonitor(
        map_path=args.map,
        user=args.user,
        password=args.password,
        csv_path=args.csv,
        interval=args.interval,
        deadline=args.deadline,
        detect_overload=args.detect_overload,
        threshold_w=args.threshold_w,
        hysteresis_w=args.hysteresis_w,
        min_over_s=args.min_over,
        cooldown_s=args.cooldown,
        spike_threshold_w=args.spike_threshold_w,
        spike_margin_w=args.spike_margin_w,
        spike_persistence_s=args.spike_persistence,
        ramp_fast_window_s=args.ramp_fast_window,
        ramp_slow_window_s=args.ramp_slow_window,
        ramp_delta_w=args.ramp_delta_w,
        ramp_reset_w=args.ramp_reset_w,
        ramp_slope_w=args.ramp_slope_w,
        ramp_slope_reset_w=args.ramp_slope_reset_w,
        handled_window_s=args.handled_window,
        peak_window_s=args.peak_window,
        shed_watts=args.shed_watts,
        shed_margin=args.shed_margin,
        shed_delay=args.shed_delay,
        events_csv=args.events_csv,
    )

    try:
        monitor.run_forever()
    except KeyboardInterrupt:
        monitor.stop()


if __name__ == "__main__":
    main()
