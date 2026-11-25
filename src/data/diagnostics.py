import numpy as np
import pandas as pd
import polars as pl

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def _to_pandas(df: pl.DataFrame | pd.DataFrame) -> pd.DataFrame:
    return df.to_pandas() if isinstance(df, pl.DataFrame) else df

def compute_smd(
    df: pl.DataFrame | pd.DataFrame,
    feature_cols: list[str],
    treatment_col: str = "treatment",
    threshold: float = 0.1
) -> pd.DataFrame:
    pdf = _to_pandas(df)
    mask = pdf[treatment_col].to_numpy().astype(bool)
    if mask.all() or (~mask).all():
        raise ValueError("Both treatment and control groups must be non-empty")

    encoded = pd.get_dummies(pdf[feature_cols], drop_first=False, dtype=float)
    treated = encoded.loc[mask]
    control = encoded.loc[~mask]
    
    mean_t = treated.mean(axis=0)
    mean_c = control.mean(axis=0)
    var_t = treated.var(axis=0, ddof=1).fillna(0.0)
    var_c = control.var(axis=0, ddof=1).fillna(0.0)

    pooled_std = np.sqrt((var_t + var_c) / 2.0)
    smd = np.where(pooled_std > 0.0, (mean_t - mean_c) / pooled_std, 0.0)

    result = pd.DataFrame(
        {
            "feature": encoded.columns,
            "mean_treated": mean_t.to_numpy(),
            "mean_control": mean_c.to_numpy(),
            "smd": smd,
            "abs_smd": np.abs(smd),
        }
    )
    result["is_balanced"] = result["abs_smd"] < threshold
    return result.sort_values("abs_smd", ascending=False).reset_index(drop=True)

def check_overlap(
    df: pl.DataFrame | pd.DataFrame,
    feature_cols: list[str],
    treatment_col: str = "treatment",
    clip_low: float = 0.05,
    clip_high: float = 0.95,
    n_splits: int = 5,
    seed: int = 42
) -> tuple[dict[str, float], np.ndarray]:
    pdf = _to_pandas(df)
    x_df = pdf[feature_cols]
    t = pdf[treatment_col].to_numpy().astype(int)

    num_cols = x_df.select_dtypes(include="number").columns.tolist()
    cat_cols = x_df.select_dtypes(exclude="number").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ],
        remainder="drop"
    )
    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed))
        ]
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    propensity = cross_val_predict(pipe, x_df, t, cv=cv, method="predict_proba")[:, 1]

    extreme_mask = (propensity < clip_low) | (propensity > clip_high)
    summary = {
        "min": float(np.min(propensity)),
        "p05": float(np.percentile(propensity, 5)),
        "median": float(np.median(propensity)),
        "p95": float(np.percentile(propensity, 95)),
        "max": float(np.max(propensity)),
        "extreme_frac": float(np.mean(extreme_mask))
    }
    return summary, propensity
