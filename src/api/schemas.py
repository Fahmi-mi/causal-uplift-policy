from typing import Any, Literal

from pydantic import BaseModel, Field


class RecommendActionRequest(BaseModel):
    features: dict[str, Any]
    conversion_value: float = Field(default=15.0, ge=0.0)
    treatment_cost: float = Field(default=0.5, ge=0.0)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)


class RecommendActionResponse(BaseModel):
    recommended_arm: Literal["treat", "holdout"]
    estimated_cate: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    expected_net_profit: float
    expected_net_roi: float
    segment_quadrant: str
    explanation: str


class BatchCandidate(BaseModel):
    user_id: str | None = None
    features: dict[str, Any]
    value: float | None = Field(default=None, ge=0.0)
    cost: float | None = Field(default=None, ge=0.0)
    segment: str | None = None


class OptimizeBatchRequest(BaseModel):
    candidates: list[BatchCandidate] = Field(min_length=1)
    budget: float = Field(ge=0.0)
    default_value: float = Field(default=15.0, ge=0.0)
    default_cost: float = Field(default=0.5, ge=0.0)
    segment_caps: dict[str, int] | None = None


class BatchAllocationItem(BaseModel):
    index: int
    user_id: str | None = None
    action: int
    recommended_arm: Literal["treat", "holdout"]
    estimated_cate: float
    expected_net_profit: float


class OptimizeBatchResponse(BaseModel):
    total_candidates: int
    total_treated: int
    total_cost: float
    expected_net_profit: float
    expected_incremental_conversions: float
    status: str
    allocations: list[BatchAllocationItem]


class SimulatePolicyRequest(BaseModel):
    n_steps: int = Field(default=500, ge=10, le=5000)
    conversion_value: float = Field(default=15.0, ge=0.0)
    treatment_cost: float = Field(default=0.5, ge=0.0)
    budget: float = Field(default=2500.0, ge=0.0)
    drift_at: int | None = Field(default=250, ge=1)
    drift_shift: float = -0.12
    seed: int = 42


class PolicyOfflineMetrics(BaseModel):
    policy: str
    treated: int
    total_cost: float
    expected_net_profit: float


class BanditPolicyMetrics(BaseModel):
    policy: str
    total_steps: int
    total_treated: int
    treat_rate: float
    cumulative_regret: float
    cumulative_net_revenue: float


class SimulatePolicyResponse(BaseModel):
    offline_comparison: list[PolicyOfflineMetrics]
    bandit_comparison: list[BanditPolicyMetrics]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    estimator: str
    is_ready: bool
    incoming_samples_logged: int


class FeatureDriftItem(BaseModel):
    feature: str
    psi: float
    status: Literal["ok", "moderate_drift", "significant_drift"]


class MetricsResponse(BaseModel):
    model_version: str
    estimator: str
    incoming_samples_logged: int
    max_psi: float
    drift_detected: bool
    features: list[FeatureDriftItem]
