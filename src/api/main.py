from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from fastapi import FastAPI, HTTPException

from src.api.drift import FeatureDriftMonitor
from src.api.schemas import (
    BatchAllocationItem,
    HealthResponse,
    MetricsResponse,
    OptimizeBatchRequest,
    OptimizeBatchResponse,
    RecommendActionRequest,
    RecommendActionResponse,
    SimulatePolicyRequest,
    SimulatePolicyResponse,
)
from src.bandit.simulator import simulate_bandit_horizon, summarize_bandit_results
from src.data.loader import ingest_hillstrom, load_config, stratified_split
from src.models.optimizer import optimize_binary_policy
from src.models.uplift import UpliftModelWrapper


def _classify_quadrant(cate: float, threshold: float = 0.02) -> str:
    if cate > threshold:
        return "Persuadable"
    if cate < -threshold:
        return "Sleeping Dog"
    return "Neutral / Sure Thing / Lost Cause"


class DecisionEngineService:
    def __init__(
        self,
        feature_cols: list[str],
        uplift_model: UpliftModelWrapper,
        naive_model: UpliftModelWrapper,
        random_model: UpliftModelWrapper,
        baseline_df: pd.DataFrame,
        test_df: pd.DataFrame,
        model_version: str = "0.1.0",
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.uplift_model = uplift_model
        self.naive_model = naive_model
        self.random_model = random_model
        self.baseline_df = baseline_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)
        self.model_version = model_version
        self.drift_monitor = FeatureDriftMonitor(self.baseline_df, self.feature_cols)

    @classmethod
    def from_dataframe(
        cls,
        df: pl.DataFrame | pd.DataFrame,
        feature_cols: list[str],
        treatment_col: str = "treatment",
        target_col: str = "visit",
        estimator_name: str = "dr_learner",
        seed: int = 42,
        lgbm_params: dict[str, Any] | None = None,
    ) -> "DecisionEngineService":
        train_df, _, test_df = stratified_split(
            df,
            stratify_cols=[treatment_col, target_col],
            train_size=0.6,
            cal_size=0.2,
            test_size=0.2,
            seed=seed,
        )
        X_train = train_df.select(feature_cols).to_pandas()
        t_train = train_df[treatment_col].to_numpy()
        y_train = train_df[target_col].to_numpy()

        uplift_model = UpliftModelWrapper(
            estimator_name=estimator_name,
            discrete_outcome=True,
            random_state=seed,
            lgbm_params=lgbm_params,
        ).fit(X_train, t_train, y_train)
        naive_model = UpliftModelWrapper(
            estimator_name="naive_propensity",
            discrete_outcome=True,
            random_state=seed,
            lgbm_params=lgbm_params,
        ).fit(X_train, t_train, y_train)
        random_model = UpliftModelWrapper(
            estimator_name="random",
            discrete_outcome=True,
            random_state=seed,
        ).fit(X_train, t_train, y_train)

        return cls(
            feature_cols=feature_cols,
            uplift_model=uplift_model,
            naive_model=naive_model,
            random_model=random_model,
            baseline_df=X_train,
            test_df=test_df.select(feature_cols).to_pandas(),
        )

    @classmethod
    def from_hillstrom(cls, config_path: str | Path = "config/hillstrom.yaml") -> "DecisionEngineService":
        cfg = load_config(config_path)
        processed_path = Path(cfg["processed_path"])
        df = pl.read_parquet(processed_path) if processed_path.exists() else ingest_hillstrom(cfg)
        return cls.from_dataframe(
            df=df,
            feature_cols=cfg["feature_cols"],
            treatment_col=cfg["treatment_col"],
            target_col="visit",
            seed=cfg["split"]["random_seed"],
        )

    def _records_to_frame(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        missing = [
            col
            for idx, rec in enumerate(records)
            for col in self.feature_cols
            if col not in rec
        ]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required feature keys: {sorted(set(missing))}",
            )
        ordered = [{col: rec[col] for col in self.feature_cols} for rec in records]
        return pd.DataFrame(ordered)

    def recommend_action(self, req: RecommendActionRequest) -> RecommendActionResponse:
        x_df = self._records_to_frame([req.features])
        self.drift_monitor.log_batch([req.features])

        cate = float(self.uplift_model.predict_uplift(x_df)[0])
        try:
            lb_arr, ub_arr = self.uplift_model.predict_interval(x_df, alpha=req.alpha)
            ci_lower, ci_upper = float(lb_arr[0]), float(ub_arr[0])
        except NotImplementedError:
            ci_lower, ci_upper = None, None

        net_profit = float(req.conversion_value * cate - req.treatment_cost)
        net_roi = float(net_profit / req.treatment_cost) if req.treatment_cost > 0 else 0.0
        arm = "treat" if net_profit > 0.0 else "holdout"
        quadrant = _classify_quadrant(cate)
        explanation = (
            f"Estimated CATE {cate:+.4f} ({quadrant}); expected net profit "
            f"${net_profit:+.2f} at ${req.treatment_cost:.2f} cost -> {arm}."
        )
        return RecommendActionResponse(
            recommended_arm=arm,
            estimated_cate=cate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            expected_net_profit=net_profit,
            expected_net_roi=net_roi,
            segment_quadrant=quadrant,
            explanation=explanation,
        )

    def optimize_batch(self, req: OptimizeBatchRequest) -> OptimizeBatchResponse:
        records = [c.features for c in req.candidates]
        x_df = self._records_to_frame(records)
        self.drift_monitor.log_batch(records)

        cates = self.uplift_model.predict_uplift(x_df)
        values = np.array(
            [c.value if c.value is not None else req.default_value for c in req.candidates],
            dtype=float,
        )
        costs = np.array(
            [c.cost if c.cost is not None else req.default_cost for c in req.candidates],
            dtype=float,
        )

        if req.segment_caps is not None:
            seg_labels = [c.segment for c in req.candidates]
            if any(s is None for s in seg_labels):
                raise HTTPException(
                    status_code=422,
                    detail="All candidates must include 'segment' when segment_caps is provided.",
                )
        else:
            seg_labels = None

        opt = optimize_binary_policy(
            uplift=cates,
            value=values,
            cost=costs,
            budget=req.budget,
            segment_labels=seg_labels,
            segment_caps=req.segment_caps,
            allow_fractional=False,
        )
        actions = opt["actions"]
        allocations = [
            BatchAllocationItem(
                index=i,
                user_id=cand.user_id,
                action=int(actions[i]),
                recommended_arm="treat" if int(actions[i]) == 1 else "holdout",
                estimated_cate=float(cates[i]),
                expected_net_profit=float((values[i] * cates[i] - costs[i]) * actions[i]),
            )
            for i, cand in enumerate(req.candidates)
        ]
        return OptimizeBatchResponse(
            total_candidates=len(req.candidates),
            total_treated=int(opt["total_treated"]),
            total_cost=float(opt["total_cost"]),
            expected_net_profit=float(opt["expected_net_profit"]),
            expected_incremental_conversions=float(opt["expected_incremental_conversions"]),
            status=str(opt["status"]),
            allocations=allocations,
        )

    def simulate_policy(self, req: SimulatePolicyRequest) -> SimulatePolicyResponse:
        n_rows = min(req.n_steps, len(self.test_df))
        x_sample = self.test_df.iloc[:n_rows].reset_index(drop=True)

        dr_pred = self.uplift_model.predict_uplift(x_sample)
        naive_pred = self.naive_model.predict_uplift(x_sample)
        rand_pred = self.random_model.predict_uplift(x_sample)

        offline_rows = []
        for name, score in [
            ("dr_learner", dr_pred),
            ("naive_propensity", naive_pred),
            ("random", rand_pred),
        ]:
            opt = optimize_binary_policy(
                uplift=score,
                value=req.conversion_value,
                cost=req.treatment_cost,
                budget=req.budget,
                allow_fractional=True,
            )
            offline_rows.append(
                {
                    "policy": name,
                    "treated": int(round(opt["total_treated"])),
                    "total_cost": float(opt["total_cost"]),
                    "expected_net_profit": float(opt["expected_net_profit"]),
                }
            )

        num_df = x_sample.select_dtypes(include="number").to_numpy(dtype=float)
        if num_df.shape[1] == 0:
            num_df = np.zeros((n_rows, 1), dtype=float)
        num_norm = (num_df - num_df.mean(axis=0)) / (num_df.std(axis=0) + 1e-8)

        p0 = np.clip(0.11 + 0.03 * np.tanh(num_norm[:, 0]), 0.02, 0.90)
        p1 = np.clip(p0 + np.clip(dr_pred, -0.05, 0.25), 0.02, 0.95)

        drift_at = req.drift_at if req.drift_at is not None and req.drift_at < n_rows else None
        sim_df = simulate_bandit_horizon(
            X=num_norm,
            probs_control=p0,
            probs_treat=p1,
            static_uplift_scores=dr_pred,
            value=req.conversion_value,
            cost=req.treatment_cost,
            drift_at=drift_at,
            drift_shift=req.drift_shift,
            seed=req.seed,
        )
        bandit_summary = summarize_bandit_results(sim_df).to_dict(orient="records")
        return SimulatePolicyResponse(
            offline_comparison=offline_rows,
            bandit_comparison=bandit_summary,
        )


@asynccontextmanager
async def lifespan(application: FastAPI):
    if not hasattr(application.state, "engine") or application.state.engine is None:
        application.state.engine = DecisionEngineService.from_hillstrom()
    yield


def create_app(engine: DecisionEngineService | None = None) -> FastAPI:
    application = FastAPI(
        title="Causal Uplift & Constrained Policy Engine",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.engine = engine

    def _get_engine() -> DecisionEngineService:
        if application.state.engine is None:
            application.state.engine = DecisionEngineService.from_hillstrom()
        return application.state.engine

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        svc = _get_engine()
        return HealthResponse(
            status="ok",
            model_version=svc.model_version,
            estimator=svc.uplift_model.estimator_name,
            is_ready=True,
            incoming_samples_logged=svc.drift_monitor.sample_count,
        )

    @application.get("/metrics", response_model=MetricsResponse)
    def metrics() -> MetricsResponse:
        svc = _get_engine()
        report = svc.drift_monitor.compute_report()
        return MetricsResponse(
            model_version=svc.model_version,
            estimator=svc.uplift_model.estimator_name,
            **report,
        )

    @application.post("/recommend-action", response_model=RecommendActionResponse)
    def recommend_action(req: RecommendActionRequest) -> RecommendActionResponse:
        return _get_engine().recommend_action(req)

    @application.post("/optimize-batch", response_model=OptimizeBatchResponse)
    def optimize_batch(req: OptimizeBatchRequest) -> OptimizeBatchResponse:
        return _get_engine().optimize_batch(req)

    @application.post("/simulate-policy", response_model=SimulatePolicyResponse)
    def simulate_policy(req: SimulatePolicyRequest) -> SimulatePolicyResponse:
        return _get_engine().simulate_policy(req)

    return application


app = create_app()
