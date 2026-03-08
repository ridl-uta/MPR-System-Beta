from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from .client import MPRClient
from .data import JobDataInput, normalize_job_perf_data
from .server import MPRServer


class ThreadedMPRIntNegotiator:
    """Runs MPR-INT using Socket.IO network protocol between server and clients."""

    def __init__(
        self,
        job_perf_data: JobDataInput,
        target_reduction_w: float,
        q_bounds: tuple[float, float] = (0.1, 5.0),
        max_iters: int = 100,
        delta_q_tol: float = 0.1,
        residual_abs_tol: float = 5.0,
        target_margin_w: float = 0.0,
        q_round_decimals: Optional[int] = 5,
        bid_resolution: int = 500,
        min_skip_idx: int = 5,
        alpha: float = 1.0,
        cycle_min_period: int = 2,
        cycle_max_period: int = 30,
        cycle_q_tol: float = 5e-2,
        cycle_pick: str = "min_abs_residual",
        request_timeout_s: float = 120.0,
        reconnect_wait_s: float = 120.0,
        startup_timeout_s: float = 20.0,
        client_start_delay_s: float = 0.05,
        recompute_final_bids: bool = True,
        enable_feasible_fallback: bool = True,
        host: str = "127.0.0.1",
        port: int = 8765,
        log_http_requests: bool = False,
        verbose: bool = False,
    ) -> None:
        if cycle_pick not in {"min_abs_residual", "min_q"}:
            raise ValueError("cycle_pick must be 'min_abs_residual' or 'min_q'.")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1].")

        self.models = normalize_job_perf_data(job_perf_data)
        self.client_ids = [m.client_id for m in self.models]

        self.target_reduction_w = float(target_reduction_w)
        self.target_effective_w = float(target_reduction_w + target_margin_w)
        self.q_bounds = (float(q_bounds[0]), float(q_bounds[1]))
        self.max_iters = int(max_iters)
        self.delta_q_tol = float(delta_q_tol)
        self.residual_abs_tol = float(residual_abs_tol)
        self.alpha = float(alpha)
        self.cycle_min_period = int(cycle_min_period)
        self.cycle_max_period = int(cycle_max_period)
        self.cycle_q_tol = float(cycle_q_tol)
        self.cycle_pick = cycle_pick
        self.startup_timeout_s = float(startup_timeout_s)
        self.client_start_delay_s = float(client_start_delay_s)
        self.recompute_final_bids = bool(recompute_final_bids)
        self.enable_feasible_fallback = bool(enable_feasible_fallback)
        self.verbose = bool(verbose)

        self.server = MPRServer(
            host=host,
            port=int(port),
            request_timeout_s=request_timeout_s,
            reconnect_wait_s=reconnect_wait_s,
            log_http_requests=log_http_requests,
            verbose=verbose,
        )

        server_url = f"http://{host}:{int(port)}"
        self.clients: Dict[str, MPRClient] = {}
        for model in self.models:
            client = MPRClient(
                model=model,
                server_url=server_url,
                q_round_decimals=q_round_decimals,
                bid_resolution=bid_resolution,
                min_skip_idx=min_skip_idx,
                connect_timeout_s=startup_timeout_s,
            )
            self.clients[model.client_id] = client

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _evaluate_self_consistent_point(self, q_value: float) -> tuple[dict[str, float], float]:
        q = float(np.clip(q_value, self.q_bounds[0], self.q_bounds[1]))
        bids = self.server.request_bids(q, self.client_ids)
        reduction = float(self.server.query_total_reduction(q, bids, self.client_ids))
        return bids, reduction

    def _find_min_feasible_self_consistent_point(
        self,
        *,
        search_iters: int = 24,
    ) -> tuple[float, dict[str, float], float] | None:
        q_low, q_high = self.q_bounds
        target = self.target_effective_w
        cache: Dict[float, tuple[dict[str, float], float]] = {}

        def evaluate(q_value: float) -> tuple[dict[str, float], float]:
            q_key = round(float(q_value), 8)
            cached = cache.get(q_key)
            if cached is not None:
                return cached
            point = self._evaluate_self_consistent_point(float(q_value))
            cache[q_key] = point
            return point

        low_bids, low_reduction = evaluate(q_low)
        if low_reduction >= target:
            return float(q_low), low_bids, low_reduction

        high_bids, high_reduction = evaluate(q_high)
        if high_reduction < target:
            return None

        lo = float(q_low)
        hi = float(q_high)
        hi_bids = high_bids
        hi_reduction = high_reduction

        for _ in range(max(1, int(search_iters))):
            mid = (lo + hi) / 2.0
            mid_bids, mid_reduction = evaluate(mid)
            if mid_reduction >= target:
                hi = mid
                hi_bids = mid_bids
                hi_reduction = mid_reduction
            else:
                lo = mid

        # Build a small candidate set around the feasible boundary and pick the
        # feasible point with minimum residual; tie-break on smaller q.
        candidate_qs = {
            round(float(hi), 8),
            round(float(q_low), 8),
            round(float(q_high), 8),
        }
        span = max(1e-6, hi - lo)
        for i in range(max(2, int(search_iters)) + 1):
            frac = i / max(2, int(search_iters))
            q_sample = lo + frac * span
            candidate_qs.add(round(float(q_sample), 8))

        best: tuple[float, dict[str, float], float] | None = None
        best_residual = float("inf")
        for q_candidate in sorted(candidate_qs):
            bids_candidate, reduction_candidate = evaluate(float(q_candidate))
            residual_candidate = float(reduction_candidate - target)
            if residual_candidate < 0.0:
                continue
            if (
                residual_candidate < best_residual
                or (
                    abs(residual_candidate - best_residual) <= 1e-9
                    and best is not None
                    and float(q_candidate) < best[0]
                )
                or best is None
            ):
                best_residual = residual_candidate
                best = (float(q_candidate), bids_candidate, float(reduction_candidate))

        return best

    def _run_negotiation(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        q_current = (self.q_bounds[0] + self.q_bounds[1]) / 2.0

        q_current_history: list[float] = []
        q_clear_history: list[float] = []
        delta_q_history: list[float] = []
        residual_history: list[float] = []
        bids_history: list[Dict[str, float]] = []

        converged = False
        convergence_mode: Optional[str] = None
        cycle_period: Optional[int] = None
        cycle_max_diff: Optional[float] = None
        selected_index: Optional[int] = None

        for iteration in range(1, self.max_iters + 1):
            bids = self.server.request_bids(q_current, self.client_ids)
            q_new, residual = self.server.clearing_step(
                bids=bids,
                client_ids=self.client_ids,
                target_effective_w=self.target_effective_w,
                q_bounds=self.q_bounds,
            )
            delta_q = abs(q_new - q_current)

            q_current_history.append(float(q_current))
            q_clear_history.append(float(q_new))
            delta_q_history.append(float(delta_q))
            residual_history.append(float(residual))
            bids_history.append(dict(bids))

            self._log(
                f"[SocketIO MPR-INT] Iter {iteration}: "
                f"q_current={q_current:.6f}, q_clear={q_new:.6f}, "
                f"delta_q={delta_q:.6f}, residual={residual:.6f}"
            )

            if delta_q < self.delta_q_tol and abs(residual) < self.residual_abs_tol:
                converged = True
                convergence_mode = "fixed_point"
                selected_index = len(q_clear_history) - 1
                break

            detected_period: Optional[int] = None
            max_period = min(self.cycle_max_period, len(q_clear_history) // 2)
            for p in range(self.cycle_min_period, max_period + 1):
                prev = np.array(q_clear_history[-2 * p : -p], dtype=float)
                curr = np.array(q_clear_history[-p:], dtype=float)
                if prev.size != p or curr.size != p:
                    continue
                max_diff = float(np.max(np.abs(curr - prev)))
                if max_diff < self.cycle_q_tol:
                    detected_period = p
                    cycle_max_diff = max_diff
                    break

            if detected_period is not None:
                cycle_start = len(q_clear_history) - detected_period
                cycle_q = np.array(q_clear_history[cycle_start:], dtype=float)
                cycle_res = np.abs(np.array(residual_history[cycle_start:], dtype=float))
                if self.cycle_pick == "min_q":
                    local_idx = int(np.argmin(cycle_q))
                else:
                    local_idx = int(np.argmin(cycle_res))
                selected_index = cycle_start + local_idx
                cycle_period = detected_period
                converged = True
                convergence_mode = "cycle"
                break

            q_current = float(
                np.clip(
                    (1.0 - self.alpha) * q_current + self.alpha * q_new,
                    self.q_bounds[0],
                    self.q_bounds[1],
                )
            )

        if selected_index is None:
            selected_index = len(q_clear_history) - 1

        final_q = float(q_clear_history[selected_index])
        selected_q_current = float(q_current_history[selected_index])
        selected_bids = dict(bids_history[selected_index])

        # By default, recompute bids at final_q so final reductions/frequencies
        # are evaluated with a self-consistent (q, bids) pair. This can be
        # disabled to preserve history-selected (q_clear, bids_at_q_current)
        # behavior used in step-by-step tracing.
        feasible_fallback_used = False
        if self.recompute_final_bids:
            final_bids = self.server.request_bids(final_q, self.client_ids)
            final_reduction = float(self.server.query_total_reduction(final_q, final_bids, self.client_ids))
            final_residual = float(final_reduction - self.target_effective_w)
            if self.enable_feasible_fallback and final_residual < 0.0:
                fallback_point = self._find_min_feasible_self_consistent_point()
                if fallback_point is not None:
                    final_q, final_bids, final_reduction = fallback_point
                    final_residual = float(final_reduction - self.target_effective_w)
                    feasible_fallback_used = True
                    convergence_mode = (
                        f"{convergence_mode}+feasible_fallback"
                        if convergence_mode
                        else "feasible_fallback"
                    )
        else:
            # Match step-by-step history selection exactly:
            # bids are those observed at q_current for selected iteration and
            # residual comes from clearing at selected q_clear with those bids.
            final_bids = dict(selected_bids)
            final_residual = float(residual_history[selected_index])
            final_reduction = float(self.target_effective_w + final_residual)
        negotiation_time_s = float(time.perf_counter() - t0)

        return {
            "final_q": final_q,
            "final_bids": final_bids,
            "final_reduction_w": final_reduction,
            "final_residual_w": final_residual,
            "target_reduction_w": self.target_reduction_w,
            "target_effective_w": self.target_effective_w,
            "converged": converged,
            "convergence_mode": convergence_mode,
            "cycle_period": cycle_period,
            "cycle_max_diff": cycle_max_diff,
            "selected_iteration": int(selected_index + 1),
            "selected_q_current": selected_q_current,
            "selected_bids": selected_bids,
            "feasible_fallback_used": feasible_fallback_used,
            "iterations_run": int(len(q_clear_history)),
            "negotiation_time_s": negotiation_time_s,
            "history": {
                "q_current": q_current_history,
                "q_clear": q_clear_history,
                "delta_q": delta_q_history,
                "residual": residual_history,
            },
        }

    def run(self) -> Dict[str, Any]:
        self.server.start()
        self.server.wait_until_ready(timeout_s=self.startup_timeout_s)

        for client in self.clients.values():
            client.start()
            if self.client_start_delay_s > 0:
                time.sleep(self.client_start_delay_s)

        try:
            self.server.wait_for_clients(len(self.clients), timeout_s=self.startup_timeout_s)
            not_acknowledged = [
                cid
                for cid, client in self.clients.items()
                if not client.wait_until_registered(timeout_s=2.0)
            ]
            if not_acknowledged:
                raise TimeoutError(
                    "Clients connected but registration ACK missing for: "
                    + ", ".join(not_acknowledged)
                )
            client_errors = {cid: c.error for cid, c in self.clients.items() if c.error is not None}
            if client_errors:
                raise RuntimeError(f"Client startup/connect errors: {client_errors}")
            result = self._run_negotiation()
        except TimeoutError as exc:
            client_errors = {cid: c.error for cid, c in self.clients.items() if c.error is not None}
            if client_errors:
                raise RuntimeError(
                    f"{exc}. Client startup/connect errors: {client_errors}"
                ) from exc
            raise
        finally:
            for client in self.clients.values():
                client.stop()
            for client in self.clients.values():
                client.join(timeout=2.0)
            self.server.shutdown()
            self.server.join(timeout=2.0)

        return result


def run_threaded_mpr_int(
    job_perf_data: JobDataInput,
    target_reduction_w: float,
    **kwargs: Any,
) -> Dict[str, Any]:
    negotiator = ThreadedMPRIntNegotiator(
        job_perf_data=job_perf_data,
        target_reduction_w=target_reduction_w,
        **kwargs,
    )
    return negotiator.run()
