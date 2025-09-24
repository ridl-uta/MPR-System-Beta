#!/usr/bin/env python3
"""Threaded overload monitor built on top of overload_detection."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Sequence, Tuple

from .overload_detection import make_simple_overload_ctx, simple_overload_update
from .overload_handler import LoadShedder


@dataclass
class Sample:
    timestamp: datetime
    watts: float


class OverloadMonitor:
    """Consume power samples and emit overload events in a background thread."""

    def __init__(
        self,
        *,
        ctx_kwargs: Optional[dict] = None,
        load_shedder: Optional[LoadShedder] = None,
    ) -> None:
        self.ctx = make_simple_overload_ctx(**(ctx_kwargs or {}))
        self.load_shedder = load_shedder

        if self.load_shedder is not None:
            self.ctx['handled_enabled'] = True

        self.samples: queue.Queue[Sample] = queue.Queue()
        self.events: queue.Queue[Tuple[datetime, object, Optional[dict]]] = queue.Queue()

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="OverloadMonitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def submit_sample(self, timestamp: datetime, watts: float) -> None:
        self.samples.put(Sample(timestamp, watts))

    def add_samples(self, batch: Sequence[Sample]) -> None:
        for sample in batch:
            self.samples.put(sample)

    def get_event(self, timeout: Optional[float] = None):
        try:
            return self.events.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                sample = self.samples.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process_sample(sample)
            finally:
                self.samples.task_done()

    def _process_sample(self, sample: Sample) -> None:
        raw_total = float(sample.watts)
        self.ctx['last_raw_sample'] = raw_total

        if self.load_shedder is not None:
            adjusted = self.load_shedder.adjust(raw_total)
            peak_raw = self.ctx.get('peak_window_max_raw', raw_total)
            reduction = max(0.0, peak_raw - adjusted)
            if self.load_shedder.active and reduction > 0:
                base_low = self.ctx['T_high'] - self.ctx['min_hysteresis']
                self.ctx['T_low'] = min(max(0.0, self.ctx['T_high'] - reduction), base_low)
            else:
                self.ctx['T_low'] = self.ctx['T_high'] - self.ctx['min_hysteresis']
        else:
            adjusted = raw_total

        event, info = simple_overload_update(self.ctx, sample.timestamp, adjusted)
        if self.load_shedder is not None:
            self.load_shedder.handle_event(event)

        if event:
            payload = info if isinstance(event, tuple) else info
            label = event[0] if isinstance(event, tuple) else event
            self.events.put((sample.timestamp, label, payload))


# Demo
if __name__ == "__main__":
    from datetime import datetime, timezone
    import time

    monitor = OverloadMonitor(ctx_kwargs=dict(sample_period_s=1.0, threshold_w=880, min_over_s=5, cooldown_s=20))
    monitor.start()

    now = datetime.now(timezone.utc)
    for offset, watts in enumerate([800, 820, 890, 910, 875, 860, 840, 820]):
        monitor.submit_sample(now + timedelta(seconds=offset), watts)
        time.sleep(0.05)

    time.sleep(0.5)
    while True:
        evt = monitor.get_event(timeout=0.1)
        if evt is None:
            break
        print(evt)

    monitor.stop()
