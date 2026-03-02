from __future__ import annotations

import threading
from collections import deque


class EventBuffer:
    def __init__(self, maxlen: int = 256) -> None:
        self._lock = threading.Lock()
        self._events: deque[dict[str, object]] = deque(maxlen=maxlen)

    def push(self, event: dict[str, object]) -> None:
        with self._lock:
            self._events.append(event)

    def consume(self) -> list[dict[str, object]]:
        with self._lock:
            out = list(self._events)
            self._events.clear()
        return out

    def peek(self) -> list[dict[str, object]]:
        with self._lock:
            return list(self._events)
