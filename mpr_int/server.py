from __future__ import annotations

import queue
import threading
import time
from typing import Any, Dict, Optional
import numpy as np
import socketio
from scipy.optimize import root_scalar
from werkzeug.serving import WSGIRequestHandler, make_server


class _QuietWSGIRequestHandler(WSGIRequestHandler):
    """Suppress per-request HTTP access logs for local negotiation runs."""

    def log(self, type: str, message: str, *args: Any) -> None:
        return

class MPRServer(threading.Thread):
    """Socket.IO-based MPR-INT server.

    Clients hold performance data and compute bids/reduction responses.
    The server coordinates negotiation over Socket.IO messages.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        request_timeout_s: float = 120.0,
        reconnect_wait_s: float = 120.0,
        log_http_requests: bool = False,
        verbose: bool = False,
        daemon: bool = True,
    ) -> None:
        super().__init__(name="MPRServerSocketIO", daemon=daemon)
        self.host = host
        self.port = int(port)
        self.request_timeout_s = float(request_timeout_s)
        self.reconnect_wait_s = float(max(1.0, reconnect_wait_s))
        self.log_http_requests = bool(log_http_requests)
        self.verbose = bool(verbose)

        self.sio = socketio.Server(
            async_mode="threading",
            async_handlers=False,
            cors_allowed_origins="*",
            ping_interval=25,
            ping_timeout=120,
        )
        self.app = socketio.WSGIApp(self.sio)

        self._http_server = None
        self._registered_clients: Dict[str, str] = {}  # client_id -> sid
        self._sid_to_client: Dict[str, str] = {}
        self._reg_lock = threading.Lock()
        self._reg_cond = threading.Condition(self._reg_lock)

        self._response_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._pending_messages: list[Dict[str, Any]] = []
        self._req_id = 0
        self._ready_event = threading.Event()
        self._run_error: Optional[Exception] = None

        self._configure_handlers()

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _configure_handlers(self) -> None:
        @self.sio.event
        def connect(sid: str, environ: Dict[str, Any]) -> None:
            self._log(f"[MPRSocketIO] connect sid={sid}")

        @self.sio.event
        def disconnect(sid: str) -> None:
            with self._reg_lock:
                client_id = self._sid_to_client.pop(sid, None)
                if client_id is not None:
                    self._registered_clients.pop(client_id, None)
                self._reg_cond.notify_all()
            self._log(f"[MPRSocketIO] disconnect sid={sid} client_id={client_id}")

        @self.sio.on("register")
        def register(sid: str, data: Dict[str, Any]) -> None:
            client_id = str(data.get("client_id", "")).strip()
            if not client_id:
                self.sio.emit("server_error", {"message": "Missing client_id"}, to=sid)
                return
            with self._reg_lock:
                self._registered_clients[client_id] = sid
                self._sid_to_client[sid] = client_id
                self._reg_cond.notify_all()
            self.sio.emit("register_ack", {"client_id": client_id}, to=sid)
            self._log(f"[MPRSocketIO] registered client_id={client_id} sid={sid}")

        @self.sio.on("bid_response")
        def bid_response(sid: str, data: Dict[str, Any]) -> None:
            self._response_queue.put(
                {
                    "type": "bid",
                    "sid": sid,
                    "client_id": str(data.get("client_id", "")),
                    "req_id": int(data.get("req_id", -1)),
                    "bid": float(data.get("bid", 0.0)),
                    "q": float(data.get("q", 0.0)),
                }
            )

        @self.sio.on("reduction_response")
        def reduction_response(sid: str, data: Dict[str, Any]) -> None:
            self._response_queue.put(
                {
                    "type": "reduction",
                    "sid": sid,
                    "client_id": str(data.get("client_id", "")),
                    "req_id": int(data.get("req_id", -1)),
                    "reduction": float(data.get("reduction", 0.0)),
                    "q": float(data.get("q", 0.0)),
                }
            )

        @self.sio.on("client_error")
        def client_error(sid: str, data: Dict[str, Any]) -> None:
            self._response_queue.put(
                {
                    "type": "error",
                    "sid": sid,
                    "client_id": str(data.get("client_id", "")),
                    "req_id": int(data.get("req_id", -1)),
                    "message": str(data.get("message", "unknown client error")),
                }
            )

    def run(self) -> None:
        try:
            self._log(f"[MPRSocketIO] listening on http://{self.host}:{self.port}")
            request_handler = WSGIRequestHandler if self.log_http_requests else _QuietWSGIRequestHandler
            self._http_server = make_server(
                self.host,
                self.port,
                self.app,
                threaded=True,
                request_handler=request_handler,
            )
            self._ready_event.set()
            self._http_server.serve_forever()
        except Exception as exc:
            self._run_error = exc
            self._ready_event.set()
            raise

    def wait_until_ready(self, timeout_s: float = 10.0) -> None:
        if not self._ready_event.wait(timeout=timeout_s):
            raise TimeoutError(
                f"Socket.IO server did not become ready within {timeout_s:.1f}s."
            )
        if self._run_error is not None:
            raise RuntimeError("Socket.IO server failed during startup.") from self._run_error

    def shutdown(self) -> None:
        if self._http_server is not None:
            try:
                self._http_server.shutdown()
                self._http_server.server_close()
            except Exception:
                pass
            self._http_server = None

    def wait_for_clients(self, expected_count: int, timeout_s: float = 15.0) -> list[str]:
        deadline = time.perf_counter() + timeout_s
        with self._reg_cond:
            while len(self._registered_clients) < expected_count:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for clients. "
                        f"expected={expected_count}, connected={len(self._registered_clients)}"
                    )
                self._reg_cond.wait(timeout=remaining)
            return list(self._registered_clients.keys())

    def _next_req_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _sid_for_client(self, client_id: str, wait_timeout_s: Optional[float] = None) -> str:
        timeout_s = self.reconnect_wait_s if wait_timeout_s is None else float(wait_timeout_s)
        deadline = time.perf_counter() + timeout_s
        with self._reg_cond:
            while True:
                sid = self._registered_clients.get(client_id)
                if sid is not None:
                    return sid
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise RuntimeError(f"Client not connected: {client_id}")
                self._reg_cond.wait(timeout=remaining)

    def _pop_pending_match(self, expected_type: str, req_id: int) -> Optional[Dict[str, Any]]:
        for idx, msg in enumerate(self._pending_messages):
            if msg.get("type") == expected_type and int(msg.get("req_id", -1)) == req_id:
                return self._pending_messages.pop(idx)
        return None

    def _collect_expected(
        self,
        expected_type: str,
        req_id: int,
        expected_clients: list[str],
        allow_partial: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        received: Dict[str, Dict[str, Any]] = {}
        deadline = time.perf_counter() + self.request_timeout_s

        while len(received) < len(expected_clients):
            pending = self._pop_pending_match(expected_type, req_id)
            if pending is not None:
                received[str(pending["client_id"])] = pending
                continue

            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                if allow_partial:
                    return received
                missing = [cid for cid in expected_clients if cid not in received]
                raise TimeoutError(
                    f"Timed out waiting for {expected_type} responses from: {missing}"
                )

            try:
                msg = self._response_queue.get(timeout=remaining)
            except queue.Empty as exc:
                if allow_partial:
                    return received
                missing = [cid for cid in expected_clients if cid not in received]
                raise TimeoutError(
                    f"Timed out waiting for {expected_type} responses from: {missing}"
                ) from exc

            if msg.get("type") == "error":
                raise RuntimeError(
                    f"Client error from {msg.get('client_id')}: {msg.get('message')}"
                )

            if msg.get("type") == expected_type and int(msg.get("req_id", -1)) == req_id:
                received[str(msg["client_id"])] = msg
            else:
                self._pending_messages.append(msg)

        return received

    def request_bids(self, q_current: float, client_ids: list[str]) -> Dict[str, float]:
        remaining = list(client_ids)
        collected: Dict[str, Dict[str, Any]] = {}

        for _ in range(5):
            req_id = self._next_req_id()
            for client_id in remaining:
                sid = self._sid_for_client(client_id)
                self.sio.emit(
                    "update_price",
                    {"req_id": req_id, "q": float(q_current)},
                    to=sid,
                )
            responses = self._collect_expected(
                "bid",
                req_id,
                remaining,
                allow_partial=True,
            )
            collected.update(responses)
            remaining = [cid for cid in remaining if cid not in responses]
            if not remaining:
                break

        if remaining:
            raise TimeoutError(f"Timed out waiting for bid responses from: {remaining}")
        return {cid: float(collected[cid]["bid"]) for cid in client_ids}

    def query_total_reduction(
        self,
        q_try: float,
        bids: Dict[str, float],
        client_ids: list[str],
    ) -> float:
        remaining = list(client_ids)
        collected: Dict[str, Dict[str, Any]] = {}

        for _ in range(5):
            req_id = self._next_req_id()
            for client_id in remaining:
                sid = self._sid_for_client(client_id)
                self.sio.emit(
                    "reduction_query",
                    {"req_id": req_id, "q": float(q_try), "bid": float(bids[client_id])},
                    to=sid,
                )
            responses = self._collect_expected(
                "reduction",
                req_id,
                remaining,
                allow_partial=True,
            )
            collected.update(responses)
            remaining = [cid for cid in remaining if cid not in responses]
            if not remaining:
                break

        if remaining:
            raise TimeoutError(f"Timed out waiting for reduction responses from: {remaining}")
        return float(sum(float(collected[cid]["reduction"]) for cid in client_ids))

    def clearing_step(
        self,
        bids: Dict[str, float],
        client_ids: list[str],
        target_effective_w: float,
        q_bounds: tuple[float, float],
    ) -> tuple[float, float]:
        q_low, q_high = q_bounds
        residual_cache: Dict[float, float] = {}

        def residual_fn(q_try: float) -> float:
            q_key = round(float(q_try), 8)
            cached = residual_cache.get(q_key)
            if cached is not None:
                return cached
            total_reduction = self.query_total_reduction(float(q_try), bids, client_ids)
            residual = float(total_reduction - target_effective_w)
            residual_cache[q_key] = residual
            return residual

        res_low = residual_fn(q_low)
        res_high = residual_fn(q_high)

        if res_low > 0:
            return q_low, res_low
        if res_high < 0:
            return q_high, res_high

        res = root_scalar(
            residual_fn,
            bracket=(q_low, q_high),
            method="brentq",
            xtol=1e-3,
            rtol=1e-6,
            maxiter=20,
        )
        q_new = float(res.root)
        residual = float(residual_fn(q_new))

        step = max(1e-5, (q_high - q_low) * 1e-4)
        if residual < 0:
            q_try = min(q_high, q_new + step)
            r_try = float(residual_fn(q_try))
            if r_try >= 0:
                q_new, residual = q_try, r_try
        else:
            for _ in range(5):
                q_try = max(q_low, q_new - step)
                r_try = float(residual_fn(q_try))
                if r_try >= 0 and r_try < residual:
                    q_new, residual = q_try, r_try
                else:
                    break

        return q_new, residual
