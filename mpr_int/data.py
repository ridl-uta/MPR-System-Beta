from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd


JobDataInput = Union[
    Mapping[str, pd.DataFrame],
    Sequence[pd.DataFrame],
    Sequence[Tuple[str, pd.DataFrame]],
]


@dataclass(frozen=True)
class JobModel:
    client_id: str
    rr: np.ndarray
    ee: np.ndarray
    pw: np.ndarray
    delta_max: float
    power_max: float


def prepare_job_model(client_id: str, df: pd.DataFrame) -> JobModel:
    required = {"Resource Reduction", "Extra Execution", "Power"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Client '{client_id}' missing required columns: {sorted(missing)}")

    rr = pd.to_numeric(df["Resource Reduction"], errors="coerce").to_numpy()
    ee = pd.to_numeric(df["Extra Execution"], errors="coerce").to_numpy()
    pw = pd.to_numeric(df["Power"], errors="coerce").to_numpy()

    valid = np.isfinite(rr) & np.isfinite(ee) & np.isfinite(pw)
    rr, ee, pw = rr[valid], ee[valid], pw[valid]
    if rr.size == 0:
        raise ValueError(f"Client '{client_id}' has no valid numeric rows.")

    order = np.argsort(rr)
    rr, ee, pw = rr[order], ee[order], pw[order]

    return JobModel(
        client_id=client_id,
        rr=rr,
        ee=ee,
        pw=pw,
        delta_max=float(rr.max()),
        power_max=float(pw.max()),
    )


def normalize_job_perf_data(job_perf_data: JobDataInput) -> list[JobModel]:
    models: list[JobModel] = []

    if isinstance(job_perf_data, Mapping):
        for client_id, df in job_perf_data.items():
            models.append(prepare_job_model(str(client_id), df))
        return models

    for idx, item in enumerate(job_perf_data):
        if isinstance(item, tuple) and len(item) == 2:
            client_id, df = item
            models.append(prepare_job_model(str(client_id), df))
            continue
        if isinstance(item, pd.DataFrame):
            models.append(prepare_job_model(f"client_{idx}", item))
            continue
        raise TypeError(
            "job_perf_data sequence must contain pandas.DataFrame "
            "or (client_id, pandas.DataFrame) tuples."
        )

    return models


def maximize_net_gain_with_data_brute(
    rr: np.ndarray,
    ee: np.ndarray,
    q: float,
    delta_max: float,
    resolution: int = 500,
    min_skip_idx: int = 5,
) -> float:
    b_hi = max(q * delta_max, 1e-6)
    b_vals = np.linspace(1e-6, b_hi, resolution)
    x = np.maximum(delta_max - b_vals / q, 0.0)
    x_scaled = np.clip(x / delta_max, 0.0, 1.0)

    l_x = np.interp(x_scaled, rr, ee)
    cost = np.divide(l_x, x_scaled, out=np.zeros_like(l_x), where=x_scaled > 1e-6)
    gains = q * x - cost

    grad = np.gradient(gains)
    pos = np.where(grad[min_skip_idx:] > 0)[0]
    start = int(min_skip_idx + pos[0]) if pos.size else int(min_skip_idx)
    best_idx = int(start + np.argmax(gains[start:]))
    return float(b_vals[best_idx])


def power_reduction_for_bid(model: JobModel, bid: float, q: float) -> float:
    delta = max(model.delta_max - bid / q, 0.0)
    power = np.interp(delta, model.rr, model.pw)
    return float(model.power_max - power)
