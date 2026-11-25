from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklift.metrics import qini_auc_score, qini_curve, uplift_at_k, uplift_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


def _to_1d(arr: np.ndarray | pd.Series | pl.Series, dtype: Any) -> np.ndarray:
    if isinstance(arr, pl.Series):
        return arr.to_numpy().astype(dtype).ravel()
    return np.asarray(arr, dtype=dtype).ravel()


def evaluate_uplift(
    y_true: np.ndarray | pd.Series | pl.Series,
    uplift_pred: np.ndarray | pd.Series | pl.Series,
    treatment: np.ndarray | pd.Series | pl.Series,
    k_list: tuple[float, ...] = (0.1, 0.2, 0.3),
) -> dict[str, float]:
    y_arr = _to_1d(y_true, int)
    u_arr = _to_1d(uplift_pred, float)
    t_arr = _to_1d(treatment, int)

    metrics: dict[str, float] = {
        "auuc": float(uplift_auc_score(y_arr, u_arr, t_arr)),
        "qini_coeff": float(qini_auc_score(y_arr, u_arr, t_arr)),
    }
    for k in k_list:
        key = f"uplift_at_{int(round(k * 100))}"
        metrics[key] = float(uplift_at_k(y_arr, u_arr, t_arr, strategy="overall", k=k))
    return metrics


def compute_qini_curve(
    y_true: np.ndarray | pd.Series | pl.Series,
    uplift_pred: np.ndarray | pd.Series | pl.Series,
    treatment: np.ndarray | pd.Series | pl.Series,
) -> tuple[np.ndarray, np.ndarray]:
    y_arr = _to_1d(y_true, int)
    u_arr = _to_1d(uplift_pred, float)
    t_arr = _to_1d(treatment, int)
    x_vals, y_vals = qini_curve(y_arr, u_arr, t_arr)
    return np.asarray(x_vals, dtype=float), np.asarray(y_vals, dtype=float)


def benchmark_models(
    models: dict[str, Any],
    X_train: pd.DataFrame | pl.DataFrame | np.ndarray,
    t_train: np.ndarray | pd.Series | pl.Series,
    y_train: np.ndarray | pd.Series | pl.Series,
    X_test: pd.DataFrame | pl.DataFrame | np.ndarray,
    t_test: np.ndarray | pd.Series | pl.Series,
    y_test: np.ndarray | pd.Series | pl.Series,
    k_list: tuple[float, ...] = (0.1, 0.2, 0.3),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, model in models.items():
        model.fit(X_train, t_train, y_train)
        pred = model.predict_uplift(X_test)
        scores = evaluate_uplift(y_test, pred, t_test, k_list=k_list)
        rows.append({"model": name, **scores})
    return pd.DataFrame(rows).sort_values("qini_coeff", ascending=False).reset_index(drop=True)


def placebo_treatment_refutation(
    model_factory: Callable[[], Any],
    X_train: pd.DataFrame | pl.DataFrame | np.ndarray,
    t_train: np.ndarray | pd.Series | pl.Series,
    y_train: np.ndarray | pd.Series | pl.Series,
    X_test: pd.DataFrame | pl.DataFrame | np.ndarray,
    t_test: np.ndarray | pd.Series | pl.Series,
    y_test: np.ndarray | pd.Series | pl.Series,
    n_repeats: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t_train_arr = _to_1d(t_train, int)
    t_test_arr = _to_1d(t_test, int)
    rows: list[dict[str, float | int]] = []

    for run_idx in range(n_repeats):
        t_train_perm = rng.permutation(t_train_arr)
        t_test_perm = rng.permutation(t_test_arr)
        model = model_factory()
        model.fit(X_train, t_train_perm, y_train)
        pred = model.predict_uplift(X_test)
        scores = evaluate_uplift(y_test, pred, t_test_perm, k_list=(0.2,))
        rows.append({"run": run_idx, **scores})

    return pd.DataFrame(rows)


def subset_stability_check(
    model_factory: Callable[[], Any],
    X_train: pd.DataFrame | pl.DataFrame | np.ndarray,
    t_train: np.ndarray | pd.Series | pl.Series,
    y_train: np.ndarray | pd.Series | pl.Series,
    X_test: pd.DataFrame | pl.DataFrame | np.ndarray,
    t_test: np.ndarray | pd.Series | pl.Series,
    y_test: np.ndarray | pd.Series | pl.Series,
    fraction: float = 0.8,
    n_repeats: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    t_arr = _to_1d(t_train, int)
    y_arr = _to_1d(y_train, int)
    x_df = X_train.to_pandas() if isinstance(X_train, pl.DataFrame) else X_train

    splitter = StratifiedShuffleSplit(
        n_splits=n_repeats, train_size=fraction, random_state=seed
    )
    rows: list[dict[str, float | int]] = []

    for run_idx, (idx, _) in enumerate(splitter.split(np.zeros(len(t_arr)), t_arr)):
        x_sub = x_df.iloc[idx] if isinstance(x_df, pd.DataFrame) else x_df[idx]
        model = model_factory()
        model.fit(x_sub, t_arr[idx], y_arr[idx])
        pred = model.predict_uplift(X_test)
        scores = evaluate_uplift(y_test, pred, t_test, k_list=(0.2,))
        rows.append({"run": run_idx, **scores})

    return pd.DataFrame(rows)
