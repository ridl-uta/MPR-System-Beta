from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import threading
import time

from .apc_client import APCPDUClient
from .csv_writer import append_power_csv
from .events import EventBuffer
from .mapping import build_poll_command, default_mapping_path, load_mapping
from .parsers import parse_outlet_readings


class PowerMonitor:
    """Sample APC PDUs periodically and expose power events/snapshots."""

    def __init__(
        self,
        *,
        username: str,
        password: str,
        csv_path: str,
        map_path: str | None = None,
        interval_s: float = 1.0,
        deadline_s: float = 15.0,
        event_buffer_size: int = 256,
    ) -> None:
        self.username = username
        self.password = password
        self.csv_path = csv_path
        # Autoload module-local mapping when map_path is not provided.
        self.map_path = str(map_path) if map_path else str(default_mapping_path())
        self.interval_s = float(interval_s)
        self.deadline_s = float(deadline_s)

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._clients: list[APCPDUClient] = []
        self._outlet_maps: list[dict[int, str]] = []
        self._labels: list[str] = []
        self._poll_command: str | None = None

        self._events = EventBuffer(maxlen=event_buffer_size)
        self._sample_lock = threading.Lock()
        self._last_sample: dict[str, object] | None = None

    @property
    def resolved_mapping_path(self) -> str:
        return self.map_path

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._configure()
        self._thread = threading.Thread(target=self._run_loop, name="PowerMonitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._teardown()

    def run_forever(self) -> None:
        self._stop.clear()
        self._configure()
        try:
            self._run_loop()
        finally:
            self._teardown()

    def consume_events(self) -> list[dict[str, object]]:
        return self._events.consume()

    def peek_events(self) -> list[dict[str, object]]:
        return self._events.peek()

    def get_last_sample(self) -> dict[str, object] | None:
        with self._sample_lock:
            return dict(self._last_sample) if self._last_sample else None

    def sample_once(self) -> dict[str, int]:
        if not self._clients:
            self._configure()
        return self._sample_once()

    def _configure(self) -> None:
        mapping_cfg = load_mapping(self.map_path)
        if not mapping_cfg.pdus:
            raise RuntimeError(f"No PDUs found in mapping: {self.map_path}")

        self._teardown()
        self._poll_command = build_poll_command(mapping_cfg.outlets)

        for pdu in mapping_cfg.pdus:
            client = APCPDUClient(
                host=pdu.host,
                port=pdu.port,
                username=self.username,
                password=self.password,
            )
            client.connect()
            self._clients.append(client)
            self._outlet_maps.append(pdu.outlet_map)
            self._labels.append(f"{pdu.host}:{pdu.port}")

        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)

    def _teardown(self) -> None:
        for client in self._clients:
            try:
                client.close()
            except Exception:
                pass
        self._clients.clear()
        self._outlet_maps.clear()
        self._labels.clear()
        self._poll_command = None

    def _run_loop(self) -> None:
        interval = max(0.1, self.interval_s)
        next_tick = time.monotonic()
        while not self._stop.is_set():
            next_tick += interval
            self._sample_once()
            delay = next_tick - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_tick = time.monotonic()

    def _sample_once(self) -> dict[str, int]:
        ts_iso = datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")
        node_totals: dict[str, int] = {}

        for client, outlet_map, label in zip(self._clients, self._outlet_maps, self._labels):
            try:
                raw = client.run(self._poll_command or "", deadline_s=self.deadline_s)
                readings = parse_outlet_readings(raw)
                for outlet, node in outlet_map.items():
                    node_totals[node] = node_totals.get(node, 0) + readings.get(outlet, 0)
            except Exception as exc:
                self._events.push(
                    {
                        "timestamp": ts_iso,
                        "event": "POLL_FAILED",
                        "payload": {"pdu": label, "error": str(exc)},
                        "message": f"poll failed for {label}: {exc}",
                    }
                )

        append_power_csv(self.csv_path, ts_iso, node_totals)

        sample = {
            "timestamp": ts_iso,
            "node_totals": dict(node_totals),
            "total_watts": float(sum(node_totals.values())),
        }
        with self._sample_lock:
            self._last_sample = sample

        return node_totals
