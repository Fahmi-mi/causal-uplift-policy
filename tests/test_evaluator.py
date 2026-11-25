import numpy as np
import pandas as pd
import pytest

from src.models.evaluator import (
    compute_qini_curve,
    evaluate_uplift,
    placebo_treatment_refutation,
    subset_stability_check,
)
from src.models.uplift import UpliftModelWrapper


def _make_synthetic_rct(n: int = 1200, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, size=n)
    x2 = rng.normal(0.0, 1.0, size=n)
    seg = rng.choice(["A", "B", "C"], size=n)
    t = rng.binomial(1, 0.5, size=n)

    true_tau = np.where(x1 > 0.0, 0.35, -0.15)
    base_prob = 1.0 / (1.0 + np.exp(-(0.2 * x2 - 0.5)))
    prob = np.clip(base_prob + t * true_tau, 0.01, 0.99)
    y = rng.binomial(1, prob, size=n)

    X = pd.DataFrame({"x1": x1, "x2": x2, "seg": seg})
    return X, t, y, true_tau


def test_evaluate_uplift_oracle_beats_random_and_inverse() -> None:
    rng = np.random.default_rng(7)
    _, t, y, true_tau = _make_synthetic_rct(n=2000, seed=7)

    oracle_scores = evaluate_uplift(y, true_tau, t, k_list=(0.1, 0.2, 0.3))
    random_scores = evaluate_uplift(y, rng.uniform(size=len(y)), t, k_list=(0.1, 0.2, 0.3))
    inverse_scores = evaluate_uplift(y, -true_tau, t, k_list=(0.1, 0.2, 0.3))

    assert oracle_scores["qini_coeff"] > random_scores["qini_coeff"]
    assert random_scores["qini_coeff"] > inverse_scores["qini_coeff"]
    assert oracle_scores["auuc"] > random_scores["auuc"]
    assert oracle_scores["uplift_at_20"] > inverse_scores["uplift_at_20"]

    qx, qy = compute_qini_curve(y, true_tau, t)
    assert len(qx) == len(qy)
    assert len(qx) > 2


@pytest.mark.parametrize("estimator_name", UpliftModelWrapper.SUPPORTED_ESTIMATORS)
def test_uplift_wrapper_fits_and_predicts_all_estimators(estimator_name: str) -> None:
    X, t, y, _ = _make_synthetic_rct(n=300, seed=99)
    X_train, X_test = X.iloc[:240], X.iloc[240:]
    t_train, y_train = t[:240], y[:240]

    model = UpliftModelWrapper(
        estimator_name=estimator_name,
        discrete_outcome=True,
        random_state=42,
        lgbm_params={"n_estimators": 10, "num_leaves": 7},
    )
    model.fit(X_train, t_train, y_train)
    preds = model.predict_uplift(X_test)

    assert preds.shape == (len(X_test),)
    assert np.all(np.isfinite(preds))

    if estimator_name in {"dr_learner", "causal_forest"}:
        lb, ub = model.predict_interval(X_test)
        assert lb.shape == (len(X_test),)
        assert ub.shape == (len(X_test),)
        assert np.all(lb <= ub)


def test_placebo_refutation_collapses_qini() -> None:
    X, t, y, _ = _make_synthetic_rct(n=1000, seed=11)
    X_train, X_test = X.iloc[:700], X.iloc[700:]
    t_train, t_test = t[:700], t[700:]
    y_train, y_test = y[:700], y[700:]

    factory = lambda: UpliftModelWrapper(
        "t_learner",
        random_state=42,
        lgbm_params={"n_estimators": 20, "num_leaves": 15},
    )

    real_model = factory()
    real_model.fit(X_train, t_train, y_train)
    real_qini = evaluate_uplift(y_test, real_model.predict_uplift(X_test), t_test)["qini_coeff"]

    placebo_df = placebo_treatment_refutation(
        factory, X_train, t_train, y_train, X_test, t_test, y_test, n_repeats=3, seed=42
    )
    assert len(placebo_df) == 3
    assert abs(float(placebo_df["qini_coeff"].mean())) < real_qini


def test_subset_stability_returns_expected_runs() -> None:
    X, t, y, _ = _make_synthetic_rct(n=400, seed=21)
    X_train, X_test = X.iloc[:300], X.iloc[300:]
    t_train, t_test = t[:300], t[300:]
    y_train, y_test = y[:300], y[300:]

    factory = lambda: UpliftModelWrapper(
        "s_learner",
        random_state=42,
        lgbm_params={"n_estimators": 10, "num_leaves": 7},
    )
    stability_df = subset_stability_check(
        factory, X_train, t_train, y_train, X_test, t_test, y_test, fraction=0.8, n_repeats=3, seed=42
    )
    assert len(stability_df) == 3
    assert {"auuc", "qini_coeff", "uplift_at_20"}.issubset(stability_df.columns)
    assert np.all(np.isfinite(stability_df["qini_coeff"]))
