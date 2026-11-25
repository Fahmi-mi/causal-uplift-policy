from collections import deque
from typing import Any

import numpy as np
import pandas as pd


def compute_psi(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    exp_s = pd.Series(expected).dropna()
    act_s = pd.Series(actual).dropna()
    if len(exp_s) == 0 or len(act_s) == 0:
        return 0.0

    if pd.api.types.is_numeric_dtype(exp_s):
        exp_arr = exp_s.to_numpy(dtype=float)
        act_arr = act_s.to_numpy(dtype=float)
        edges = np.unique(np.quantile(exp_arr, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 2:
            return 0.0
        edges[0] = -np.inf
        edges[-1] = np.inf
        exp_counts, _ = np.histogram(exp_arr, bins=edges)
        act_counts, _ = np.histogram(act_arr, bins=edges)
        exp_pct = np.clip(exp_counts / len(exp_arr), eps, None)
        act_pct = np.clip(act_counts / len(act_arr), eps, None)
    else:
        exp_freq = exp_s.astype(str).value_counts(normalize=True)
        act_freq = act_s.astype(str).value_counts(normalize=True)
        categories = exp_freq.index.union(act_freq.index)
        exp_pct = np.clip(exp_freq.reindex(categories, fill_value=0.0).to_numpy(dtype=float), eps, None)
        act_pct = np.clip(act_freq.reindex(categories, fill_value=0.0).to_numpy(dtype=float), eps, None)

    exp_pct = exp_pct / np.sum(exp_pct)
    act_pct = act_pct / np.sum(act_pct)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def classify_psi(psi: float) -> str:
    if psi < 0.1:
        return "ok"
    if psi < 0.25:
        return "moderate_drift"
    return "significant_drift"


class FeatureDriftMonitor:
    def __init__(self, baseline_df: pd.DataFrame, feature_cols: list[str], max_history: int = 5000) -> None:
        self.baseline_df = baseline_df[feature_cols].copy()
        self.feature_cols = list(feature_cols)
        self._buffer: deque[dict[str, Any]] = deque(maxlen=max_history)

    @property
    def sample_count(self) -> int:
        return len(self._buffer)

    def log_batch(self, records: list[dict[str, Any]]) -> None:
        for rec in records:
            self._buffer.append({col: rec.get(col) for col in self.feature_cols})

    def compute_report(self) -> dict[str, Any]:
        if not self._buffer:
            items = [{"feature": col, "psi": 0.0, "status": "ok"} for col in self.feature_cols]
            return {
                "incoming_samples_logged": 0,
                "max_psi": 0.0,
                "drift_detected": False,
                "features": items,
            }

        incoming_df = pd.DataFrame(list(self._buffer))
        items: list[dict[str, Any]] = []
        for col in self.feature_cols:
            psi = compute_psi(self.baseline_df[col], incoming_df[col])
            items.append({"feature": col, "psi": psi, "status": classify_psi(psi)})

        max_psi = max((item["psi"] for item in items), default=0.0)
        return {
            "incoming_samples_logged": len(self._buffer),
            "max_psi": float(max_psi),
            "drift_detected": bool(max_psi >= 0.25),
            "features": items,
        }
