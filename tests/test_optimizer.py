import numpy as np
import pytest

from src.bandit.simulator import simulate_bandit_horizon, summarize_bandit_results
from src.models.optimizer import (
    budget_frontier,
    optimize_binary_policy,
    optimize_multi_arm_policy,
)


def test_binary_optimizer_respects_budget_and_segment_caps() -> None:
    uplift = np.array([0.30, 0.25, 0.20, 0.15, -0.10, 0.02])
    value = 10.0
    cost = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    segments = np.array(["A", "A", "B", "B", "A", "B"])
    caps = {"A": 1, "B": 2}

    res = optimize_binary_policy(
        uplift=uplift,
        value=value,
        cost=cost,
        budget=5.0,
        segment_labels=segments,
        segment_caps=caps,
    )

    actions = res["actions"]
    assert res["total_cost"] <= 5.0
    assert np.sum(actions[segments == "A"]) <= 1
    assert np.sum(actions[segments == "B"]) <= 2
    assert actions[4] == 0
    assert actions[5] == 0
    assert actions[0] == 1
    assert actions[1] == 0
    assert actions[2] == 1
    assert actions[3] == 1


def test_binary_optimizer_beats_naive_greedy_under_heterogeneous_costs() -> None:
    uplift = np.array([0.45, 0.30, 0.28])
    value = 10.0
    cost = np.array([3.5, 1.5, 1.5])
    budget = 3.5

    res = optimize_binary_policy(uplift=uplift, value=value, cost=cost, budget=budget)
    assert res["actions"].tolist() == [0, 1, 1]
    assert np.isclose(res["expected_net_profit"], 2.8)


def test_multi_arm_optimizer_assigns_at_most_one_arm_and_respects_budget() -> None:
    uplift_matrix = np.array(
        [
            [0.30, 0.10],
            [0.05, 0.25],
            [0.01, 0.01],
        ]
    )
    cost_matrix = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    res = optimize_multi_arm_policy(uplift_matrix, value=10.0, cost_matrix=cost_matrix, budget=2.0)
    actions = res["actions"]

    assert actions.shape == (3, 2)
    assert np.all(actions.sum(axis=1) <= 1)
    assert res["total_cost"] <= 2.0
    assert actions[0].tolist() == [1, 0]
    assert actions[1].tolist() == [0, 1]
    assert actions[2].tolist() == [0, 0]


def test_budget_frontier_monotonic_profit() -> None:
    uplift = np.array([0.4, 0.3, 0.2, 0.1, -0.05])
    frontier = budget_frontier(uplift, value=10.0, cost=1.0, budgets=[0.0, 1.0, 2.0, 5.0])

    profits = frontier["expected_net_profit"].to_numpy()
    assert np.all(np.diff(profits) >= -1e-9)
    assert profits[0] == pytest.approx(0.0)
    assert frontier["total_treated"].iloc[-1] == 3


def test_bandit_simulator_adapts_after_drift() -> None:
    rng = np.random.default_rng(123)
    n_steps = 800
    x = rng.normal(0.0, 1.0, size=(n_steps, 3))

    p0 = 1.0 / (1.0 + np.exp(-(0.2 * x[:, 0] - 0.8)))
    p1 = np.clip(p0 + 0.18 + 0.10 * (x[:, 1] > 0), 0.02, 0.98)
    stale_static_uplift = np.full(n_steps, 0.20)

    sim_df = simulate_bandit_horizon(
        X=x,
        probs_control=p0,
        probs_treat=p1,
        static_uplift_scores=stale_static_uplift,
        value=1.0,
        cost=0.05,
        drift_at=200,
        drift_shift=-0.30,
        seed=42,
    )
    summary = summarize_bandit_results(sim_df).set_index("policy")

    assert {"thompson_sampling", "linucb", "static_uplift", "random"}.issubset(summary.index)
    assert summary.loc["thompson_sampling", "cumulative_regret"] < summary.loc["static_uplift", "cumulative_regret"]
    assert summary.loc["linucb", "cumulative_regret"] < summary.loc["static_uplift", "cumulative_regret"]
    assert summary.loc["thompson_sampling", "cumulative_regret"] < summary.loc["random", "cumulative_regret"]
