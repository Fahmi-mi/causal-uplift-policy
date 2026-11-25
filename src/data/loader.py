from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import polars as pl
import yaml

from sklift.datasets import fetch_hillstrom
from sklearn.model_selection import train_test_split

def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ingest_hillstrom(cfg: dict[str, Any]) -> pl.DataFrame:
    dataset = fetch_hillstrom(
        target_col="all",
        data_home=cfg.get("raw_dir", "data/raw"),
        download_if_missing=True
    )

    pdf = dataset.data.copy()
    pdf[cfg["treatment_arm_col"]] = dataset.treatment
    for col in cfg["outcome_cols"]:
        pdf[col] = dataset.target[col]

    arm_col = cfg["treatment_arm_col"]
    ctrl_val = cfg["control_arm_value"]
    pdf[cfg["treatment_col"]] = (pdf[arm_col] != ctrl_val).astype(int)

    df = pl.from_pandas(pdf)
    out_path = Path(cfg["processed_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    return df

def ingest_criteo(cfg: dict[str, Any]) -> pl.DataFrame:
    raw_path = Path(cfg["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Criteo raw file not found at {raw_path}. Download it into data/raw/ first."
        )

    out_path = Path(cfg["processed_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sub_cfg = cfg["subsample"]
    strat_cols = ",".join(sub_cfg["stratify_cols"])
    fraction = float(sub_cfg["fraction"])
    seed = float(sub_cfg["seed"])

    con = duckdb.connect()
    try:
        con.execute("SELECT setseed(?)", [seed])
        query = f"""
            COPY (
                WITH numbered AS (
                    SELECT *, random() AS _rand
                    FROM read_csv_auto(?)
                ),
                ranked AS (
                    SELECT *,
                        row_number() OVER (PARTITION BY {strat_cols} ORDER BY _rand) AS _rn,
                        count(*) OVER (PARTITION BY {strat_cols}) AS _cnt
                    FROM numbered
                )
                SELECT * EXCLUDE (_rand, _rn, _cnt)
                FROM ranked
                WHERE _rn <= GREATEST(1, CAST(_cnt * ? AS BIGINT))
            ) TO ? (FORMAT PARQUET)
        """
        con.execute(query, [str(raw_path), fraction, str(out_path)])
    finally:
        con.close()

    return pl.read_parquet(out_path)

def stratified_split(
    df: pl.DataFrame | pd.DataFrame,
    stratify_cols: list[str],
    train_size: float = 0.6,
    cal_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 42
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if not np.isclose(train_size + cal_size + test_size, 1.0):
        raise ValueError("train_size + cal_size + test_size must sum to 1.0.")

    pdf = df.to_pandas() if isinstance(df, pl.DataFrame) else df.copy()
    pdf = pdf.reset_index(drop=True)

    strat_key = pdf[stratify_cols].astype(str).agg("_".join, axis=1)
    train_df, temp_df = train_test_split(
        pdf, train_size=train_size, stratify=strat_key, random_state=seed
    )

    rel_cal = cal_size / (cal_size + test_size)
    cal_df, test_df = train_test_split(
        temp_df,
        train_size=rel_cal,
        stratify=strat_key.loc[temp_df.index],
        random_state=seed
    )

    return (
        pl.from_pandas(train_df.reset_index(drop=True)),
        pl.from_pandas(cal_df.reset_index(drop=True)),
        pl.from_pandas(test_df.reset_index(drop=True))
    )
