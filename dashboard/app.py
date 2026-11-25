import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

from src.data.loader import ingest_hillstrom, load_config, stratified_split
from src.models.evaluator import compute_qini_curve, evaluate_uplift
from src.models.optimizer import budget_frontier, optimize_binary_policy
from src.models.uplift import UpliftModelWrapper


st.set_page_config(page_title="Causal Uplift Policy Simulator", layout="wide")


@st.cache_resource
def load_benchmarks():
    cfg = load_config("config/hillstrom.yaml")
    processed_path = Path(cfg["processed_path"])
    df = pl.read_parquet(processed_path) if processed_path.exists() else ingest_hillstrom(cfg)
    train_df, _, test_df = stratified_split(
        df,
        stratify_cols=cfg["split"]["stratify_cols"],
        train_size=cfg["split"]["train_size"],
        cal_size=cfg["split"]["cal_size"],
        test_size=cfg["split"]["test_size"],
        seed=cfg["split"]["random_seed"],
    )

    feature_cols = cfg["feature_cols"]
    X_train = train_df.select(feature_cols).to_pandas()
    t_train = train_df[cfg["treatment_col"]].to_numpy()
    y_train = train_df["visit"].to_numpy()

    X_test = test_df.select(feature_cols).to_pandas()
    t_test = test_df["treatment"].to_numpy()
    y_test = test_df["visit"].to_numpy()

    estimator_names = ["random", "naive_propensity", "s_learner", "t_learner", "x_learner", "dr_learner"]
    models = {}
    preds = {}
    metric_rows = []
    for name in estimator_names:
        m = UpliftModelWrapper(name, discrete_outcome=True, random_state=cfg["split"]["random_seed"])
        m.fit(X_train, t_train, y_train)
        p = m.predict_uplift(X_test)
        models[name] = m
        preds[name] = p
        scores = evaluate_uplift(y_test, p, t_test, k_list=(0.1, 0.2, 0.3))
        metric_rows.append({"model": name, **scores})

    metrics_df = pd.DataFrame(metric_rows).sort_values("qini_coeff", ascending=False).reset_index(drop=True)
    return cfg, test_df, X_test, t_test, y_test, models, preds, metrics_df


cfg, test_df, X_test, t_test, y_test, models, preds, metrics_df = load_benchmarks()

st.title("Causal Uplift & Constrained Policy Simulator")

st.sidebar.header("Campaign Economics")
budget = st.sidebar.slider("Campaign Budget ($)", min_value=0.0, max_value=6400.0, value=2500.0, step=100.0)
visit_value = st.sidebar.slider("Conversion / Visit Value ($)", min_value=5.0, max_value=50.0, value=15.0, step=1.0)
treatment_cost = st.sidebar.slider("Treatment Cost ($)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
selected_model = st.sidebar.selectbox("Policy Estimator", list(preds.keys()), index=5)

opt = optimize_binary_policy(
    uplift=preds[selected_model],
    value=visit_value,
    cost=treatment_cost,
    budget=budget,
    allow_fractional=True,
)
p_treat = float(t_test.mean())
ipw_uplift = np.where(t_test == 1, y_test / p_treat, -y_test / (1.0 - p_treat))
ipw_profit = float(np.sum(opt["actions"] * (visit_value * ipw_uplift - treatment_cost)))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Target Users", f"{int(round(opt['total_treated'])):,}")
c2.metric("Budget Spent", f"${opt['total_cost']:,.2f}")
c3.metric("Expected Net Profit", f"${opt['expected_net_profit']:,.2f}")
c4.metric("IPW Counterfactual Profit", f"${ipw_profit:,.2f}")

tab_frontier, tab_qini, tab_segments = st.tabs(["Budget & Cost Explorer", "Uplift & Qini Diagnostics", "Segment Breakdown"])

with tab_frontier:
    budgets = np.linspace(0.0, 6400.0, 17)
    frontier_df = budget_frontier(
        preds[selected_model],
        value=visit_value,
        cost=treatment_cost,
        budgets=budgets,
        allow_fractional=True,
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frontier_df["total_cost"], frontier_df["expected_net_profit"], marker="o")
    ax.axvline(opt["total_cost"], color="red", linestyle="--", label="Current Budget")
    ax.set_xlabel("Spend ($)")
    ax.set_ylabel("Expected Net Profit ($)")
    ax.legend()
    st.pyplot(fig)
    st.dataframe(frontier_df, use_container_width=True)

with tab_qini:
    st.dataframe(metrics_df, use_container_width=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, pred in preds.items():
        qx, qy = compute_qini_curve(y_test, pred, t_test)
        ax.plot(qx, qy, label=name)
    ax.set_xlabel("Targeted Users")
    ax.set_ylabel("Incremental Visits")
    ax.legend()
    st.pyplot(fig)

with tab_segments:
    cate_vals = preds["dr_learner"]
    quadrants = np.where(
        cate_vals > 0.02,
        "Persuadables",
        np.where(cate_vals < -0.02, "Sleeping Dogs", "Sure Things / Lost Causes"),
    )
    seg_df = pd.Series(quadrants).value_counts().rename_axis("segment").reset_index(name="users")
    seg_df["share"] = seg_df["users"] / len(cate_vals)
    st.dataframe(seg_df, use_container_width=True)
    st.bar_chart(seg_df.set_index("segment")["users"])
