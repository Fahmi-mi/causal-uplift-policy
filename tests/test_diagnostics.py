import numpy as np
import polars as pl

from src.data.diagnostics import check_overlap, compute_smd
from src.data.loader import stratified_split

def test_smd_balanced_vs_biased() -> None:
    rng = np.random.default_rng(42)
    n = 2000
    t = rng.binomial(1, 0.5, size=n)

    x_balanced = rng.normal(0.0, 1.0, size=n)
    x_biased = rng.normal(0.0, 1.0, size=n) + 0.8 * t

    df = pl.DataFrame(
        {
            "x_balanced": x_balanced,
            "x_biased": x_biased,
            "treatment": t
        }
    )

    smd_df = compute_smd(df, feature_cols=["x_balanced", "x_biased"], threshold=0.1)
    by_feat = smd_df.set_index("feature")

    assert by_feat.loc["x_balanced", "abs_smd"] < 0.1
    assert bool(by_feat.loc["x_balanced", "is_balanced"]) is True
    assert by_feat.loc["x_biased", "abs_smd"] > 0.5
    assert bool(by_feat.loc["x_biased", "is_balanced"]) is False

def test_overlap_detects_good_and_extreme_positivity() -> None:
    rng = np.random.default_rng(7)
    n = 1000
    t = rng.binomial(1, 0.5, size=n)

    good_df = pl.DataFrame({"x": rng.normal(0.0, 1.0, size=n), "treatment": t})
    good_summary, _ = check_overlap(good_df, feature_cols=["x"])
    assert good_summary["extreme_frac"] < 0.05

    bad_df = pl.DataFrame({"x": rng.normal(0.0, 0.2, size=n) + 4.0 * t, "treatment": t})
    bad_summary, _ = check_overlap(bad_df, feature_cols=["x"])
    assert bad_summary["extreme_frac"] > 0.9

def test_stratified_split_preserves_ratio_and_no_leakage() -> None:
    rng = np.random.default_rng(123)
    n = 1000
    df = pl.DataFrame(
        {
            "row_id": np.arange(n),
            "treatment": rng.binomial(1, 0.3, size=n),
            "conversion": rng.binomial(1, 0.1, size=n),
            "x": rng.normal(size=n)
        }
    )

    train_df, cal_df, test_df = stratified_split(
        df,
        stratify_cols=["treatment", "conversion"],
        train_size=0.6,
        cal_size=0.2,
        test_size=0.2,
        seed=42
    )

    assert len(train_df) == 600
    assert len(cal_df) == 200
    assert len(test_df) == 200

    train_ids = set(train_df["row_id"].to_list())
    cal_ids = set(cal_df["row_id"].to_list())
    test_ids = set(test_df["row_id"].to_list())

    assert train_ids.isdisjoint(cal_ids)
    assert train_ids.isdisjoint(test_ids)
    assert cal_ids.isdisjoint(test_ids)

    base_rate = df["treatment"].mean()
    assert abs(train_df["treatment"].mean() - base_rate) < 0.02
    assert abs(cal_df["treatment"].mean() - base_rate) < 0.02
    assert abs(test_df["treatment"].mean() - base_rate) < 0.02
