from typing import Any

import numpy as np
import pandas as pd


class _LinearBanditBase:
    def __init__(self, n_features: int, n_arms: int = 2, reg: float = 1.0, discount: float = 0.995) -> None:
        self.n_features = n_features
        self.n_arms = n_arms
        self.reg = reg
        self.discount = discount
        self.a_mats = [reg * np.eye(n_features, dtype=float) for _ in range(n_arms)]
        self.b_vecs = [np.zeros(n_features, dtype=float) for _ in range(n_arms)]

    def _posterior(self, arm: int) -> tuple[np.ndarray, np.ndarray]:
        a_inv = np.linalg.inv(self.a_mats[arm])
        theta = a_inv @ self.b_vecs[arm]
        return theta, a_inv

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        x_vec = np.asarray(x, dtype=float).ravel()
        if self.discount < 1.0:
            self.a_mats[arm] = (
                self.discount * self.a_mats[arm]
                + (1.0 - self.discount) * self.reg * np.eye(self.n_features, dtype=float)
                + np.outer(x_vec, x_vec)
            )
            self.b_vecs[arm] = self.discount * self.b_vecs[arm] + float(reward) * x_vec
        else:
            self.a_mats[arm] += np.outer(x_vec, x_vec)
            self.b_vecs[arm] += float(reward) * x_vec


class ContextualThompsonSampling(_LinearBanditBase):
    def __init__(
        self,
        n_features: int,
        n_arms: int = 2,
        reg: float = 1.0,
        discount: float = 0.995,
        v_scale: float = 0.25,
        seed: int = 42,
    ) -> None:
        super().__init__(n_features=n_features, n_arms=n_arms, reg=reg, discount=discount)
        self.v_scale = v_scale
        self.rng = np.random.default_rng(seed)

    def select_arm(self, x: np.ndarray, value: float = 1.0, arm_costs: tuple[float, ...] = (0.0, 0.0)) -> int:
        x_vec = np.asarray(x, dtype=float).ravel()
        scores = np.empty(self.n_arms, dtype=float)
        for arm in range(self.n_arms):
            theta, a_inv = self._posterior(arm)
            sampled_theta = self.rng.multivariate_normal(theta, (self.v_scale**2) * a_inv)
            scores[arm] = float(value) * float(x_vec @ sampled_theta) - float(arm_costs[arm])
        return int(np.argmax(scores))


class LinUCBPolicy(_LinearBanditBase):
    def __init__(
        self,
        n_features: int,
        n_arms: int = 2,
        reg: float = 1.0,
        discount: float = 0.995,
        alpha: float = 0.25,
    ) -> None:
        super().__init__(n_features=n_features, n_arms=n_arms, reg=reg, discount=discount)
        self.alpha = alpha

    def select_arm(self, x: np.ndarray, value: float = 1.0, arm_costs: tuple[float, ...] = (0.0, 0.0)) -> int:
        x_vec = np.asarray(x, dtype=float).ravel()
        scores = np.empty(self.n_arms, dtype=float)
        for arm in range(self.n_arms):
            theta, a_inv = self._posterior(arm)
            mean_est = float(x_vec @ theta)
            bonus = self.alpha * float(np.sqrt(x_vec @ a_inv @ x_vec))
            scores[arm] = float(value) * (mean_est + bonus) - float(arm_costs[arm])
        return int(np.argmax(scores))


def simulate_bandit_horizon(
    X: np.ndarray,
    probs_control: np.ndarray,
    probs_treat: np.ndarray,
    static_uplift_scores: np.ndarray | None = None,
    value: float = 1.0,
    cost: float = 0.05,
    drift_at: int | None = None,
    drift_shift: float = -0.15,
    seed: int = 42,
) -> pd.DataFrame:
    x_mat = np.asarray(X, dtype=float)
    p0_base = np.asarray(probs_control, dtype=float).ravel()
    p1_base = np.asarray(probs_treat, dtype=float).ravel()
    n_steps, n_features = x_mat.shape

    x_bias = np.hstack([np.ones((n_steps, 1), dtype=float), x_mat])
    n_bandit_features = n_features + 1

    ts_policy = ContextualThompsonSampling(n_features=n_bandit_features, seed=seed)
    ucb_policy = LinUCBPolicy(n_features=n_bandit_features)
    rng = np.random.default_rng(seed)

    policies = ["thompson_sampling", "linucb", "random"]
    if static_uplift_scores is not None:
        static_arr = np.asarray(static_uplift_scores, dtype=float).ravel()
        policies.append("static_uplift")
    else:
        static_arr = None

    cum_regret = {p: 0.0 for p in policies}
    cum_revenue = {p: 0.0 for p in policies}
    arm_costs = (0.0, float(cost))
    rows: list[dict[str, Any]] = []

    for step in range(n_steps):
        shift = drift_shift if (drift_at is not None and step >= drift_at) else 0.0
        p0 = float(np.clip(p0_base[step], 0.01, 0.99))
        p1 = float(np.clip(p1_base[step] + shift, 0.01, 0.99))

        exp_net = (value * p0, value * p1 - cost)
        oracle_profit = max(exp_net)
        x_step = x_bias[step]

        for policy in policies:
            if policy == "thompson_sampling":
                arm = ts_policy.select_arm(x_step, value=value, arm_costs=arm_costs)
            elif policy == "linucb":
                arm = ucb_policy.select_arm(x_step, value=value, arm_costs=arm_costs)
            elif policy == "static_uplift":
                arm = int((value * static_arr[step] - cost) > 0.0)
            else:
                arm = int(rng.integers(0, 2))

            p_chosen = p1 if arm == 1 else p0
            y_obs = float(rng.binomial(1, p_chosen))
            net_rev = value * y_obs - cost * arm
            regret = oracle_profit - exp_net[arm]

            cum_regret[policy] += regret
            cum_revenue[policy] += net_rev

            if policy == "thompson_sampling":
                ts_policy.update(x_step, arm, y_obs)
            elif policy == "linucb":
                ucb_policy.update(x_step, arm, y_obs)

            rows.append(
                {
                    "step": step,
                    "policy": policy,
                    "arm": arm,
                    "expected_net_profit": exp_net[arm],
                    "realized_net_revenue": net_rev,
                    "regret": regret,
                    "cumulative_regret": cum_regret[policy],
                    "cumulative_net_revenue": cum_revenue[policy],
                }
            )

    return pd.DataFrame(rows)


def summarize_bandit_results(sim_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        sim_df.groupby("policy", as_index=False)
        .agg(
            total_steps=("step", "count"),
            total_treated=("arm", "sum"),
            treat_rate=("arm", "mean"),
            cumulative_regret=("cumulative_regret", "last"),
            cumulative_net_revenue=("cumulative_net_revenue", "last"),
        )
        .sort_values("cumulative_regret", ascending=True)
        .reset_index(drop=True)
    )
    return summary
