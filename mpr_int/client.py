from __future__ import annotations

import threading
import time
from typing import Optional

import socketio

from .data import JobModel, maximize_net_gain_with_data_brute, power_reduction_for_bid


class MPRClient(threading.Thread):
    """Socket.IO MPR-INT client.

    Client owns job data and answers two server events:
    - update_price: compute optimal bid for q
    - reduction_query: compute power reduction for (bid, q)
    """

    def __init__(
        self,
        model: JobModel,
        server_url: str,
        q_round_decimals: Optional[int] = 5,
        bid_resolution: int = 500,
        min_skip_idx: int = 5,
        connect_timeout_s: float = 10.0,
        connect_retry_interval_s: float = 0.5,
        daemon: bool = True,
    ) -> None:
        super().__init__(name=f"MPRClientSocketIO-{model.client_id}", daemon=daemon)
        self.model = model
        self.server_url = server_url
        self.q_round_decimals = q_round_decimals
        self.bid_resolution = bid_resolution
        self.min_skip_idx = min_skip_idx
        self.connect_timeout_s = float(connect_timeout_s)
        self.connect_retry_interval_s = float(max(0.05, connect_retry_interval_s))

        self._bid_cache: dict[float, float] = {}
        self._stop_event = threading.Event()
        self._registered_event = threading.Event()
        self._emit_lock = threading.Lock()
        self._connect_error: Optional[Exception] = None
        self.error: Optional[Exception] = None

        # Werkzeug WSGI backend is reliable with polling in this setup.
        self._transports = ["polling"]
        self.sio = socketio.Client(
            reconnection=True,
            logger=False,
            engineio_logger=False,
            request_timeout=max(120.0, self.connect_timeout_s),
        )
        self._configure_handlers()

    @property
    def client_id(self) -> str:
        return self.model.client_id

    def _configure_handlers(self) -> None:
        @self.sio.event
        def connect() -> None:
            self._connect_error = None
            try:
                with self._emit_lock:
                    self.sio.emit("register", {"client_id": self.client_id})
            except Exception as exc:
                self._connect_error = exc

        @self.sio.event
        def disconnect() -> None:
            self._registered_event.clear()

        @self.sio.on("register_ack")
        def register_ack(data: dict) -> None:
            if str(data.get("client_id", "")) == self.client_id:
                self._registered_event.set()

        @self.sio.on("update_price")
        def update_price(data: dict) -> None:
            try:
                req_id = int(data.get("req_id", -1))
                q = float(data["q"])
                bid = self._optimal_bid(q)
                self._safe_emit(
                    "bid_response",
                    {
                        "req_id": req_id,
                        "client_id": self.client_id,
                        "q": q,
                        "bid": bid,
                    },
                )
            except Exception as exc:
                self._safe_emit(
                    "client_error",
                    {
                        "req_id": int(data.get("req_id", -1)),
                        "client_id": self.client_id,
                        "message": f"update_price handler failed: {exc}",
                    },
                )

        @self.sio.on("reduction_query")
        def reduction_query(data: dict) -> None:
            try:
                req_id = int(data.get("req_id", -1))
                q = float(data["q"])
                bid = float(data["bid"])
                reduction = power_reduction_for_bid(self.model, bid, q)
                self._safe_emit(
                    "reduction_response",
                    {
                        "req_id": req_id,
                        "client_id": self.client_id,
                        "q": q,
                        "reduction": reduction,
                    },
                )
            except Exception as exc:
                self._safe_emit(
                    "client_error",
                    {
                        "req_id": int(data.get("req_id", -1)),
                        "client_id": self.client_id,
                        "message": f"reduction_query handler failed: {exc}",
                    },
                )

        @self.sio.on("server_error")
        def server_error(data: dict) -> None:
            self._safe_emit(
                "client_error",
                {
                    "req_id": int(data.get("req_id", -1)),
                    "client_id": self.client_id,
                    "message": str(data.get("message", "server_error")),
                },
            )

        @self.sio.event
        def connect_error(data: object) -> None:
            # Keep running to allow automatic reconnection.
            self._connect_error = RuntimeError(f"connect_error: {data}")

    def _optimal_bid(self, q: float) -> float:
        q_key = float(q) if self.q_round_decimals is None else round(float(q), self.q_round_decimals)
        bid = self._bid_cache.get(q_key)
        if bid is None:
            bid = maximize_net_gain_with_data_brute(
                self.model.rr,
                self.model.ee,
                q_key,
                self.model.delta_max,
                resolution=self.bid_resolution,
                min_skip_idx=self.min_skip_idx,
            )
            self._bid_cache[q_key] = bid
        return float(bid)

    def _safe_emit(self, event: str, payload: dict) -> bool:
        if self._stop_event.is_set():
            return False
        if not self.sio.connected:
            return False
        try:
            with self._emit_lock:
                self.sio.emit(event, payload)
            return True
        except Exception as exc:
            self._connect_error = exc
            return False

    def wait_until_registered(self, timeout_s: float = 10.0) -> bool:
        ok = self._registered_event.wait(timeout=timeout_s)
        if self.error is not None:
            return False
        return ok

    def stop(self) -> None:
        self._stop_event.set()
        try:
            if self.sio.connected:
                self.sio.disconnect()
        except Exception:
            pass

    def run(self) -> None:
        startup_deadline = time.perf_counter() + self.connect_timeout_s
        try:
            while not self._stop_event.is_set():
                try:
                    self.sio.connect(
                        self.server_url,
                        transports=self._transports,
                        wait=True,
                        wait_timeout=min(5.0, self.connect_timeout_s),
                    )
                    break
                except Exception as exc:
                    self._connect_error = exc
                    if time.perf_counter() >= startup_deadline:
                        raise TimeoutError(
                            f"Failed to connect client '{self.client_id}' to {self.server_url} "
                            f"within {self.connect_timeout_s:.1f}s."
                        ) from exc
                    self._stop_event.wait(self.connect_retry_interval_s)

            while not self._stop_event.wait(0.1):
                pass

        except Exception as exc:
            self.error = exc
            self._stop_event.set()
        finally:
            try:
                if self.sio.connected:
                    self.sio.disconnect()
            except Exception:
                pass

