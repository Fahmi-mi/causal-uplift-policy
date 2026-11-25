from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix


def _validate_binary_inputs(
    uplift: np.ndarray,
    value: np.ndarray | float,
    cost: np.ndarray | float,
    budget: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.asarray(uplift, dtype=float).ravel()
    n = len(u)
    v = np.full(n, float(value), dtype=float) if np.ndim(value) == 0 else np.asarray(value, dtype=float).ravel()
    c = np.full(n, float(cost), dtype=float) if np.ndim(cost) == 0 else np.asarray(cost, dtype=float).ravel()

    if len(v) != n or len(c) != n:
        raise ValueError("uplift, value, and cost must have identical lengths.")
    if budget < 0:
        raise ValueError("budget must be non-negative.")
    if np.any(c < 0):
        raise ValueError("cost entries must be non-negative.")
    return u, v, c


def optimize_binary_policy(
    uplift: np.ndarray,
    value: np.ndarray | float,
    cost: np.ndarray | float,
    budget: float,
    segment_labels: np.ndarray | list[Any] | None = None,
    segment_caps: dict[Any, int | float] | None = None,
    allow_fractional: bool = False,
) -> dict[str, Any]:
    u, v, c = _validate_binary_inputs(uplift, value, cost, budget)
    n = len(u)
    net_profit = v * u - c

    pos_idx = np.flatnonzero(net_profit > 0.0)
    if len(pos_idx) == 0 or budget == 0:
        empty_actions = np.zeros(n, dtype=float if allow_fractional else int)
        return {
            "actions": empty_actions,
            "total_treated": 0 if not allow_fractional else 0.0,
            "total_cost": 0.0,
            "expected_net_profit": 0.0,
            "expected_incremental_conversions": 0.0,
            "status": "optimal",
        }

    c_pos = c[pos_idx]
    profit_pos = net_profit[pos_idx]
    n_pos = len(pos_idx)

    rows: list[np.ndarray] = [c_pos]
    ub_list: list[float] = [float(budget)]

    if segment_caps is not None:
        if segment_labels is None:
            raise ValueError("segment_labels must be provided when segment_caps is set.")
        seg_arr = np.asarray(segment_labels)
        if len(seg_arr) != n:
            raise ValueError("segment_labels must match uplift length.")
        seg_pos = seg_arr[pos_idx]
        for seg_name, cap in segment_caps.items():
            if cap < 0:
                raise ValueError("segment cap must be non-negative.")
            rows.append((seg_pos == seg_name).astype(float))
            ub_list.append(float(cap))

    a_matrix = csc_matrix(np.vstack(rows))
    constraints = LinearConstraint(a_matrix, lb=-np.inf, ub=np.asarray(ub_list, dtype=float))
    integrality = np.zeros(n_pos, dtype=int) if allow_fractional else np.ones(n_pos, dtype=int)
    bounds = Bounds(lb=0.0, ub=1.0)

    res = milp(
        c=-profit_pos,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
    )
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    actions = np.zeros(n, dtype=float if allow_fractional else int)
    actions[pos_idx] = res.x if allow_fractional else np.rint(res.x).astype(int)
    return {
        "actions": actions,
        "total_treated": float(np.sum(actions)) if allow_fractional else int(np.sum(actions)),
        "total_cost": float(np.dot(c, actions)),
        "expected_net_profit": float(np.dot(net_profit, actions)),
        "expected_incremental_conversions": float(np.dot(u, actions)),
        "status": "optimal",
    }


def optimize_multi_arm_policy(
    uplift_matrix: np.ndarray,
    value: np.ndarray | float,
    cost_matrix: np.ndarray | float,
    budget: float,
) -> dict[str, Any]:
    u = np.asarray(uplift_matrix, dtype=float)
    if u.ndim != 2:
        raise ValueError("uplift_matrix must be 2D of shape (n_users, n_arms).")
    if budget < 0:
        raise ValueError("budget must be non-negative.")

    n_users, n_arms = u.shape
    v = (
        np.full((n_users, 1), float(value), dtype=float)
        if np.ndim(value) == 0
        else np.asarray(value, dtype=float).reshape(n_users, 1)
    )
    c = (
        np.full((n_users, n_arms), float(cost_matrix), dtype=float)
        if np.ndim(cost_matrix) == 0
        else np.asarray(cost_matrix, dtype=float)
    )
    if c.shape != (n_users, n_arms):
        raise ValueError("cost_matrix shape must match uplift_matrix shape.")
    if np.any(c < 0):
        raise ValueError("cost_matrix entries must be non-negative.")

    net_profit = (v * u - c).ravel()
    c_flat = c.ravel()
    n_vars = n_users * n_arms

    row_idx: list[int] = []
    col_idx: list[int] = []
    data: list[float] = []
    ub_list: list[float] = []

    row_idx.extend([0] * n_vars)
    col_idx.extend(range(n_vars))
    data.extend(c_flat.tolist())
    ub_list.append(float(budget))

    for i in range(n_users):
        r = i + 1
        base = i * n_arms
        row_idx.extend([r] * n_arms)
        col_idx.extend(range(base, base + n_arms))
        data.extend([1.0] * n_arms)
        ub_list.append(1.0)

    a_matrix = csc_matrix((data, (row_idx, col_idx)), shape=(n_users + 1, n_vars))
    constraints = LinearConstraint(a_matrix, lb=-np.inf, ub=np.asarray(ub_list, dtype=float))

    res = milp(
        c=-net_profit,
        integrality=np.ones(n_vars, dtype=int),
        bounds=Bounds(lb=0.0, ub=1.0),
        constraints=constraints,
    )
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    actions = np.rint(res.x).astype(int).reshape(n_users, n_arms)
    return {
        "actions": actions,
        "total_treated": int(np.sum(actions)),
        "total_cost": float(np.sum(c * actions)),
        "expected_net_profit": float(np.sum((v * u - c) * actions)),
        "expected_incremental_conversions": float(np.sum(u * actions)),
        "status": "optimal",
    }


def budget_frontier(
    uplift: np.ndarray,
    value: np.ndarray | float,
    cost: np.ndarray | float,
    budgets: list[float] | np.ndarray,
    segment_labels: np.ndarray | list[Any] | None = None,
    segment_caps: dict[Any, int | float] | None = None,
    allow_fractional: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for b in budgets:
        res = optimize_binary_policy(
            uplift=uplift,
            value=value,
            cost=cost,
            budget=float(b),
            segment_labels=segment_labels,
            segment_caps=segment_caps,
            allow_fractional=allow_fractional,
        )
        rows.append(
            {
                "budget": float(b),
                "total_treated": res["total_treated"],
                "total_cost": res["total_cost"],
                "expected_net_profit": res["expected_net_profit"],
                "expected_incremental_conversions": res["expected_incremental_conversions"],
            }
        )
    return pd.DataFrame(rows)
