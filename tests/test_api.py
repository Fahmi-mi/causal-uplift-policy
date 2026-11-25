import numpy as np
import polars as pl
from fastapi.testclient import TestClient

from src.api.main import DecisionEngineService, create_app


def _make_test_client() -> TestClient:
    rng = np.random.default_rng(42)
    n = 250
    x1 = rng.normal(0.0, 1.0, size=n)
    x2 = rng.normal(0.0, 1.0, size=n)
    channel = rng.choice(["Web", "Phone"], size=n)
    t = rng.binomial(1, 0.5, size=n)
    prob = np.clip(0.15 + t * np.where(x1 > 0.0, 0.25, -0.05), 0.01, 0.99)
    y = rng.binomial(1, prob, size=n)

    df = pl.DataFrame(
        {
            "recency": x1,
            "history": x2,
            "channel": channel,
            "treatment": t,
            "visit": y,
        }
    )
    engine = DecisionEngineService.from_dataframe(
        df=df,
        feature_cols=["recency", "history", "channel"],
        treatment_col="treatment",
        target_col="visit",
        estimator_name="dr_learner",
        seed=42,
        lgbm_params={"n_estimators": 10, "num_leaves": 7},
    )
    return TestClient(create_app(engine=engine))


def test_health_and_metrics_drift_detection() -> None:
    client = _make_test_client()

    health_res = client.get("/health")
    assert health_res.status_code == 200
    health_data = health_res.json()
    assert health_data["status"] == "ok"
    assert health_data["is_ready"] is True
    assert health_data["estimator"] == "dr_learner"

    for _ in range(15):
        client.post(
            "/recommend-action",
            json={
                "features": {"recency": 50.0, "history": 50.0, "channel": "Web"},
                "conversion_value": 15.0,
                "treatment_cost": 0.5,
            },
        )

    metrics_res = client.get("/metrics")
    assert metrics_res.status_code == 200
    metrics_data = metrics_res.json()
    assert metrics_data["incoming_samples_logged"] == 15
    assert metrics_data["max_psi"] >= 0.25
    assert metrics_data["drift_detected"] is True


def test_recommend_action_returns_valid_contract() -> None:
    client = _make_test_client()

    res = client.post(
        "/recommend-action",
        json={
            "features": {"recency": 1.2, "history": 0.5, "channel": "Web"},
            "conversion_value": 20.0,
            "treatment_cost": 0.5,
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["recommended_arm"] in {"treat", "holdout"}
    assert body["ci_lower"] is not None and body["ci_upper"] is not None
    assert body["ci_lower"] <= body["ci_upper"]
    assert np.isclose(body["expected_net_profit"], 20.0 * body["estimated_cate"] - 0.5)

    bad_res = client.post(
        "/recommend-action",
        json={"features": {"recency": 1.2}, "conversion_value": 10.0, "treatment_cost": 0.5},
    )
    assert bad_res.status_code == 422


def test_optimize_batch_respects_budget_and_segment_caps() -> None:
    client = _make_test_client()

    candidates = [
        {"user_id": "u1", "features": {"recency": 1.5, "history": 0.2, "channel": "Web"}, "segment": "Web"},
        {"user_id": "u2", "features": {"recency": 1.2, "history": 0.3, "channel": "Web"}, "segment": "Web"},
        {"user_id": "u3", "features": {"recency": 0.9, "history": -0.1, "channel": "Phone"}, "segment": "Phone"},
        {"user_id": "u4", "features": {"recency": -1.5, "history": -0.4, "channel": "Phone"}, "segment": "Phone"},
    ]
    res = client.post(
        "/optimize-batch",
        json={
            "candidates": candidates,
            "budget": 1.5,
            "default_value": 20.0,
            "default_cost": 0.5,
            "segment_caps": {"Web": 1, "Phone": 1},
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["total_candidates"] == 4
    assert body["total_cost"] <= 1.5
    assert body["total_treated"] <= 2
    assert len(body["allocations"]) == 4


def test_simulate_policy_endpoint_returns_offline_and_bandit_metrics() -> None:
    client = _make_test_client()

    res = client.post(
        "/simulate-policy",
        json={
            "n_steps": 40,
            "conversion_value": 15.0,
            "treatment_cost": 0.5,
            "budget": 10.0,
            "drift_at": 20,
            "drift_shift": -0.15,
            "seed": 42,
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert len(body["offline_comparison"]) == 3
    assert len(body["bandit_comparison"]) == 4
    assert {row["policy"] for row in body["bandit_comparison"]} == {
        "thompson_sampling",
        "linucb",
        "static_uplift",
        "random",
    }
