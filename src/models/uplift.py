from typing import Any, Self

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from econml.dml import CausalForestDML
from econml.dr import DRLearner
from econml.metalearners import SLearner, TLearner, XLearner
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def _to_pandas_features(X: pd.DataFrame | pl.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    if isinstance(X, pl.DataFrame):
        return X.to_pandas()
    return X


def _to_1d_array(arr: np.ndarray | pd.Series | pl.Series, dtype: Any = float) -> np.ndarray:
    if isinstance(arr, pl.Series):
        return arr.to_numpy().astype(dtype).ravel()
    return np.asarray(arr, dtype=dtype).ravel()


class UpliftModelWrapper:
    SUPPORTED_ESTIMATORS = (
        "random",
        "naive_propensity",
        "s_learner",
        "t_learner",
        "x_learner",
        "dr_learner",
        "causal_forest",
    )

    def __init__(
        self,
        estimator_name: str,
        discrete_outcome: bool = True,
        random_state: int = 42,
        lgbm_params: dict[str, Any] | None = None,
    ) -> None:
        if estimator_name not in self.SUPPORTED_ESTIMATORS:
            raise ValueError(
                f"Unsupported estimator '{estimator_name}'. Choose from {self.SUPPORTED_ESTIMATORS}."
            )
        self.estimator_name = estimator_name
        self.discrete_outcome = discrete_outcome
        self.random_state = random_state
        self.lgbm_params = lgbm_params or {}
        self.preprocessor: ColumnTransformer | None = None
        self.model: Any = None
        self._rng = np.random.default_rng(random_state)

    def _make_lgbm(self, classifier: bool) -> lgb.LGBMClassifier | lgb.LGBMRegressor:
        params: dict[str, Any] = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": self.random_state,
            "verbose": -1,
            "n_jobs": -1,
        }
        params.update(self.lgbm_params)
        return lgb.LGBMClassifier(**params) if classifier else lgb.LGBMRegressor(**params)

    def _build_estimator(self) -> Any:
        if self.estimator_name == "random":
            return None
        if self.estimator_name == "naive_propensity":
            return self._make_lgbm(classifier=self.discrete_outcome)
        if self.estimator_name == "s_learner":
            return SLearner(overall_model=self._make_lgbm(classifier=False))
        if self.estimator_name == "t_learner":
            return TLearner(models=self._make_lgbm(classifier=False))
        if self.estimator_name == "x_learner":
            return XLearner(
                models=self._make_lgbm(classifier=False),
                propensity_model=self._make_lgbm(classifier=True),
            )
        if self.estimator_name == "dr_learner":
            return DRLearner(
                model_propensity=self._make_lgbm(classifier=True),
                model_regression=self._make_lgbm(classifier=self.discrete_outcome),
                discrete_outcome=self.discrete_outcome,
                random_state=self.random_state,
            )
        return CausalForestDML(
            model_y=self._make_lgbm(classifier=self.discrete_outcome),
            model_t=self._make_lgbm(classifier=True),
            discrete_outcome=self.discrete_outcome,
            discrete_treatment=True,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def _fit_transform_features(self, X: pd.DataFrame | pl.DataFrame | np.ndarray) -> np.ndarray:
        x_in = _to_pandas_features(X)
        if isinstance(x_in, pd.DataFrame):
            num_cols = x_in.select_dtypes(include="number").columns.tolist()
            cat_cols = x_in.select_dtypes(exclude="number").columns.tolist()
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("num", "passthrough", num_cols),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                ],
                remainder="drop",
            )
            return self.preprocessor.fit_transform(x_in)
        self.preprocessor = None
        return np.asarray(x_in, dtype=float)

    def _transform_features(self, X: pd.DataFrame | pl.DataFrame | np.ndarray) -> np.ndarray:
        x_in = _to_pandas_features(X)
        if self.preprocessor is not None:
            return self.preprocessor.transform(x_in)
        return np.asarray(x_in, dtype=float)

    def fit(
        self,
        X: pd.DataFrame | pl.DataFrame | np.ndarray,
        t: np.ndarray | pd.Series | pl.Series,
        y: np.ndarray | pd.Series | pl.Series,
    ) -> Self:
        x_arr = self._fit_transform_features(X)
        t_arr = _to_1d_array(t, dtype=int)
        y_arr = _to_1d_array(y, dtype=float)
        self.model = self._build_estimator()

        if self.estimator_name == "random":
            return self
        if self.estimator_name == "naive_propensity":
            self.model.fit(x_arr, y_arr.astype(int) if self.discrete_outcome else y_arr)
            return self

        self.model.fit(y_arr, t_arr, X=x_arr)
        return self

    def predict_uplift(self, X: pd.DataFrame | pl.DataFrame | np.ndarray) -> np.ndarray:
        x_arr = self._transform_features(X)
        if self.estimator_name == "random":
            return self._rng.uniform(0.0, 1.0, size=len(x_arr))
        if self.estimator_name == "naive_propensity":
            if self.discrete_outcome:
                return np.asarray(self.model.predict_proba(x_arr)[:, 1], dtype=float).ravel()
            return np.asarray(self.model.predict(x_arr), dtype=float).ravel()
        return np.asarray(self.model.effect(x_arr), dtype=float).ravel()

    def predict_interval(
        self,
        X: pd.DataFrame | pl.DataFrame | np.ndarray,
        alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None or not hasattr(self.model, "effect_interval"):
            raise NotImplementedError(
                f"Confidence intervals are not supported for '{self.estimator_name}'."
            )
        x_arr = self._transform_features(X)
        lb, ub = self.model.effect_interval(x_arr, alpha=alpha)
        return np.asarray(lb, dtype=float).ravel(), np.asarray(ub, dtype=float).ravel()
