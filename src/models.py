"""
Baseline models and the benchmark harness for the housing-price regression problem.

A baseline answers "what accuracy do we get with almost no effort?", so that every later model has
something concrete to beat. This module provides a *ladder* of baselines from useless to sensible,
all of which take the raw feature table and predict the price in price units:

=====  ==========================================  ==========================================================
tier   baseline                                    question it answers
=====  ==========================================  ==========================================================
floor  mean price / median price                   what if we ignore every feature?
naive  mean price per ``location_type``            what does the one categorical feature give us?
naive  best single feature + straight line         how far does the strongest single predictor go?
naive  linear regression on the raw columns        what does a model with no preprocessing achieve?
ref    linear regression + preprocessing           **the reference baseline**: the project's preprocessing
                                                   (imputation, outlier capping, logs, scaling, one-hot)
                                                   and a log-price target (``make_model_pipeline``)
=====  ==========================================  ==========================================================

The reference baseline is deliberately simple and fast: it is the benchmark that feature engineering
(``src/feature_engineering.py``) and better models must beat. :func:`candidate_specs` lists ready-made
challengers (regularised linear models, tree ensembles, the engineered-feature pipeline) so the same
:func:`run_benchmark` can compare them under identical cross-validation folds and metrics.

Evaluation protocol (see ``src/evaluation.py`` for the metrics):

* every model is cross-validated on the **training set only** (repeated K-fold, identical folds),
* the **test set is used once**, to report the final hold-out numbers,
* differences to the reference are tested with the corrected resampled t-test, which is cautious
  about the overlap between cross-validation training sets.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from .evaluation import (
    HIGHER_IS_BETTER,
    PRIMARY_METRIC,
    corrected_resampled_ttest,
    cross_validate_metrics,
    metric_definitions,
    plot_diagnostics,
    regression_metrics,
    residual_report,
)
from .feature_engineering import FeatureEngineeringConfig, make_full_pipeline
from .preprocessing import PreprocessingConfig, _normalise_text, make_model_pipeline

__all__ = [
    "GroupMeanRegressor",
    "BestSingleFeatureRegressor",
    "BaselineModel",
    "BaselineSpec",
    "BenchmarkResult",
    "REFERENCE_BASELINE",
    "baseline_specs",
    "candidate_specs",
    "run_benchmark",
    "regression_metrics",
    "metric_definitions",
]

logger = logging.getLogger(__name__)

REFERENCE_BASELINE = "linear regression + preprocessing (log target)"
_FLOOR_MODEL = "mean price (ignores all features)"


# --------------------------------------------------------------------------- #
# Simple baseline estimators
# --------------------------------------------------------------------------- #
class GroupMeanRegressor(BaseEstimator, RegressorMixin):
    """Predict the mean (or median) training price of the row's group.

    A naive but informative baseline for a categorical column such as ``location_type``: it shows how much
    that column alone explains. Group labels are normalised (whitespace, capitalisation) exactly as in the
    preprocessing, and a missing or never-seen group gets the overall training statistic.
    """

    def __init__(self, column: str = "location_type", statistic: str = "mean"):
        self.column = column
        self.statistic = statistic

    def _labels(self, X: pd.DataFrame) -> pd.Series:
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in the data")
        return _normalise_text(X[self.column])

    def fit(self, X: pd.DataFrame, y) -> GroupMeanRegressor:
        if self.statistic not in ("mean", "median"):
            raise ValueError("statistic must be 'mean' or 'median'")
        target = pd.Series(np.asarray(y, dtype=float), index=X.index)
        labels = self._labels(X)
        stat = (lambda s: s.mean()) if self.statistic == "mean" else (lambda s: s.median())
        self.global_value_ = float(stat(target))
        self.group_values_ = {k: float(stat(v)) for k, v in target.groupby(labels.to_numpy(), dropna=True)}
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["group_values_"])
        mapped = self._labels(X).map(self.group_values_)
        return mapped.fillna(self.global_value_).to_numpy(dtype=float)


class BestSingleFeatureRegressor(BaseEstimator, RegressorMixin):
    """Straight-line regression on the single numeric feature most correlated with the price.

    ``fit`` picks the column (from ``features``, or every numeric column) with the largest absolute
    correlation with the training target, then fits ``price = intercept + slope * feature``. Missing values
    in that column are filled with its training median. After fitting: ``feature_``, ``coef_``,
    ``intercept_`` and ``correlations_``.
    """

    def __init__(self, features: Optional[List[str]] = None):
        self.features = features

    def fit(self, X: pd.DataFrame, y) -> BestSingleFeatureRegressor:
        target = np.asarray(y, dtype=float)
        candidates = list(self.features) if self.features else list(X.columns)
        numeric = {c: pd.to_numeric(X[c], errors="coerce") for c in candidates if c in X.columns}
        correlations = {}
        for name, col in numeric.items():
            valid = col.notna().to_numpy()
            if valid.sum() > 2 and col[valid].std() > 0:
                correlations[name] = float(np.corrcoef(col[valid], target[valid])[0, 1])
        if not correlations:
            raise ValueError("No usable numeric feature found")
        self.correlations_ = pd.Series(correlations).sort_values(key=np.abs, ascending=False)
        self.feature_ = str(self.correlations_.index[0])
        col = numeric[self.feature_]
        self.fill_ = float(col.median())
        values = col.fillna(self.fill_).to_numpy()
        slope, intercept = np.polyfit(values, target, 1)
        self.coef_, self.intercept_ = float(slope), float(intercept)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["feature_"])
        values = pd.to_numeric(X[self.feature_], errors="coerce").fillna(self.fill_).to_numpy(dtype=float)
        return self.intercept_ + self.coef_ * values


def _raw_linear_regression(config: PreprocessingConfig) -> Pipeline:
    """Linear regression on the raw columns: only what is needed to run (median fill, one-hot), no outlier
    handling, no log transforms, no scaling, and the target left on the price scale."""
    columns = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median"),
                make_column_selector(dtype_include=np.number),
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
                    ]
                ),
                list(config.categorical_features),
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("columns", columns), ("model", LinearRegression())])


# --------------------------------------------------------------------------- #
# Baseline catalogue
# --------------------------------------------------------------------------- #
@dataclass
class BaselineSpec:
    """A named model plus a description; ``build()`` returns a fresh unfitted estimator."""

    name: str
    build: Callable[[], Any]
    description: str
    tier: str  # "floor", "naive", "reference" or "challenger"


def baseline_specs(
    preprocessing_config: Optional[PreprocessingConfig] = None, group_column: str = "location_type"
) -> Dict[str, BaselineSpec]:
    """The baseline ladder, from the weakest model to the reference baseline (in that order)."""
    cfg = preprocessing_config or PreprocessingConfig()
    specs = [
        BaselineSpec(
            _FLOOR_MODEL,
            lambda: DummyRegressor(strategy="mean"),
            "Predicts the training mean for every house. R-squared about 0 by construction.",
            "floor",
        ),
        BaselineSpec(
            "median price (ignores all features)",
            lambda: DummyRegressor(strategy="median"),
            "Predicts the training median for every house.",
            "floor",
        ),
        BaselineSpec(
            f"mean price per {group_column}",
            lambda: GroupMeanRegressor(group_column),
            f"Predicts the average training price of the house's {group_column}.",
            "naive",
        ),
        BaselineSpec(
            "best single feature + straight line",
            lambda: BestSingleFeatureRegressor(),
            "Straight line on the numeric feature most correlated with price (chosen on the training data).",
            "naive",
        ),
        BaselineSpec(
            "linear regression on raw columns",
            lambda: _raw_linear_regression(cfg),
            "All raw columns (median fill, one-hot location) with no outlier handling, logs or scaling.",
            "naive",
        ),
        BaselineSpec(
            REFERENCE_BASELINE,
            lambda: make_model_pipeline(LinearRegression(), cfg),
            "Project preprocessing (imputation, outlier capping, log of skewed features, scaling, one-hot) "
            "and linear regression on log(price). THE benchmark to beat.",
            "reference",
        ),
    ]
    return {s.name: s for s in specs}


def candidate_specs(
    preprocessing_config: Optional[PreprocessingConfig] = None,
    fe_config: Optional[FeatureEngineeringConfig] = None,
    random_state: int = 42,
) -> Dict[str, BaselineSpec]:
    """Ready-made challengers for the same benchmark: regularised linear models, tree ensembles and the
    engineered-feature pipeline. All use the preprocessing and log target, like the reference baseline."""
    cfg = preprocessing_config or PreprocessingConfig()
    fe = fe_config or FeatureEngineeringConfig()
    models = {
        "ridge (alpha=10) + preprocessing": Ridge(alpha=10),
        "lasso (alpha=0.001) + preprocessing": Lasso(alpha=0.001, max_iter=20000),
        "elastic net + preprocessing": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=20000),
        "random forest + preprocessing": RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1),
        "gradient boosting + preprocessing": GradientBoostingRegressor(random_state=random_state),
    }

    def factory(model: Any) -> Callable[[], Any]:
        return lambda: make_model_pipeline(clone(model), cfg)

    specs = {
        name: BaselineSpec(
            name, factory(model), "Challenger model on the preprocessed features and log target.", "challenger"
        )
        for name, model in models.items()
    }
    specs["linear regression + engineered features (VIF)"] = BaselineSpec(
        "linear regression + engineered features (VIF)",
        lambda: make_full_pipeline(LinearRegression(), cfg, fe, "vif"),
        "Reference model plus the engineered features and VIF selection (see feature_engineering.py).",
        "challenger",
    )
    return specs


# --------------------------------------------------------------------------- #
# The baseline model as a single object
# --------------------------------------------------------------------------- #
class BaselineModel(BaseEstimator, RegressorMixin):
    """The reference baseline as one estimator: preprocessing + linear regression on log(price).

    Takes the raw feature table and predicts prices::

        baseline = BaselineModel().fit(X_train, y_train)
        baseline.predict(X_test)
        baseline.evaluate(X_test, y_test)      # all metrics on the price scale
        baseline.save("artifacts/baseline_model.joblib")
    """

    def __init__(self, preprocessing_config: Optional[PreprocessingConfig] = None, model: Optional[Any] = None):
        self.preprocessing_config = preprocessing_config
        self.model = model

    def fit(self, X: pd.DataFrame, y) -> BaselineModel:
        cfg = self.preprocessing_config or PreprocessingConfig()
        estimator = clone(self.model) if self.model is not None else LinearRegression()
        self.pipeline_ = make_model_pipeline(estimator, cfg).fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["pipeline_"])
        return np.asarray(self.pipeline_.predict(X), dtype=float)

    def evaluate(self, X: pd.DataFrame, y) -> Dict[str, float]:
        """All metrics (price scale) of this model on ``X``, ``y``."""
        return regression_metrics(y, self.predict(X), n_features=self.n_features_in_)

    def save(self, path: Union[str, Path]) -> Path:
        check_is_fitted(self, ["pipeline_"])
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Union[str, Path]) -> BaselineModel:
        return joblib.load(path)


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
@dataclass
class BenchmarkResult:
    """Everything :func:`run_benchmark` produces."""

    cv_table: pd.DataFrame  # per model: mean and std of every metric over the CV folds
    comparison: pd.DataFrame  # per model: CV metrics, gain vs the mean predictor and vs the reference, p-value
    holdout: pd.DataFrame  # per model: metrics on the held-out test set (used once)
    fold_metrics: Dict[str, pd.DataFrame]  # per model: every metric for every fold
    predictions: pd.DataFrame  # held-out true prices and every model's predictions
    residuals: Dict[str, pd.DataFrame]  # residual report of the reference model on the test set
    specs: Dict[str, BaselineSpec]
    reference: str
    protocol: Dict[str, Any] = field(default_factory=dict)
    fitted: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly summary for the run record (fold-level metrics included)."""
        return {
            "reference": self.reference,
            "protocol": self.protocol,
            "models": {n: {"tier": s.tier, "description": s.description} for n, s in self.specs.items()},
            "comparison": self.comparison,
            "holdout": self.holdout,
            "fold_metrics": {n: f.to_dict(orient="list") for n, f in self.fold_metrics.items()},
            "reference_residuals": self.residuals,
        }

    def save(self, directory: Union[str, Path], diagnostics_plot: bool = True) -> Dict[str, Path]:
        """Write the tables, predictions and (optionally) the diagnostics figure into ``directory``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written = {
            "comparison": directory / "baseline_cv_comparison.csv",
            "holdout": directory / "baseline_holdout_metrics.csv",
            "predictions": directory / "baseline_holdout_predictions.csv",
            "by_price_band": directory / "baseline_residuals_by_price_band.csv",
        }
        self.comparison.to_csv(written["comparison"])
        self.holdout.to_csv(written["holdout"])
        self.predictions.to_csv(written["predictions"], index_label="row_id")
        self.residuals["by_price_band"].to_csv(written["by_price_band"])
        if "by_group" in self.residuals:
            written["by_group"] = directory / "baseline_residuals_by_group.csv"
            self.residuals["by_group"].to_csv(written["by_group"])
        if diagnostics_plot:
            written["plot"] = plot_diagnostics(
                self.predictions["true"],
                self.predictions[self.reference],
                directory / "baseline_diagnostics.png",
                title=f"Hold-out diagnostics: {self.reference}",
            )
        return written


def _summarise_folds(folds: pd.DataFrame) -> pd.Series:
    out = {}
    for col in folds.columns:
        out[col] = folds[col].mean()
        out[f"{col}_std"] = folds[col].std(ddof=1)
    return pd.Series(out)


def run_benchmark(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    specs: Optional[Dict[str, BaselineSpec]] = None,
    reference: str = REFERENCE_BASELINE,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
    group_column: Optional[str] = "location_type",
) -> BenchmarkResult:
    """Cross-validate every model on the training set, then evaluate each once on the test set.

    All models see exactly the same folds. ``comparison`` reports, for each model, the CV mean of every
    metric, the **RMSE improvement over the mean predictor** (``skill_vs_mean_%``) and **over the reference
    baseline** (``rmse_gain_vs_reference_%``, positive = better) with the corrected resampled t-test p-value.
    ``holdout`` holds the single test-set evaluation. The residual report describes the reference model.
    """
    specs = specs if specs is not None else baseline_specs()
    if reference not in specs:
        raise ValueError(f"Reference model '{reference}' is not among the models: {list(specs)}")
    if n_splits < 2 or n_repeats < 1:
        raise ValueError("n_splits must be at least 2 and n_repeats at least 1")
    n_features = X_train.shape[1]

    fold_metrics: Dict[str, pd.DataFrame] = {}
    holdout_rows: Dict[str, Dict[str, float]] = {}
    predictions = pd.DataFrame({"true": np.asarray(y_test, dtype=float)}, index=X_test.index)
    fitted: Dict[str, Any] = {}
    for name, spec in specs.items():
        logger.info("Benchmarking '%s'", name)
        fold_metrics[name] = cross_validate_metrics(
            spec.build(), X_train, y_train, n_splits, n_repeats, random_state, n_features
        )
        model = spec.build()
        started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        pred = np.asarray(model.predict(X_test), dtype=float)
        predictions[name] = pred
        holdout_rows[name] = {**regression_metrics(y_test, pred, n_features), "fit_seconds": fit_seconds}
        fitted[name] = model

    cv_table = pd.DataFrame({n: _summarise_folds(f) for n, f in fold_metrics.items()}).T
    floor = next((n for n, s in specs.items() if s.tier == "floor"), next(iter(specs)))
    ref_scores = fold_metrics[reference][PRIMARY_METRIC].to_numpy()
    floor_rmse = cv_table.loc[floor, PRIMARY_METRIC]
    comparison = cv_table[["rmse", "rmse_std", "mae", "mape", "r2", "within_10pct", "bias", "fit_seconds"]].copy()
    comparison["skill_vs_mean_%"] = 100 * (floor_rmse - comparison["rmse"]) / floor_rmse
    gains, pvalues = [], []
    for name in comparison.index:
        scores = fold_metrics[name][PRIMARY_METRIC].to_numpy()
        _, _, p_value = corrected_resampled_ttest(ref_scores, scores, n_splits)
        gains.append(
            100
            * (cv_table.loc[reference, PRIMARY_METRIC] - cv_table.loc[name, PRIMARY_METRIC])
            / cv_table.loc[reference, PRIMARY_METRIC]
        )
        pvalues.append(np.nan if name == reference else p_value)
    comparison["rmse_gain_vs_reference_%"] = gains
    comparison["p_value_vs_reference"] = pvalues

    groups = X_test[group_column] if group_column and group_column in X_test.columns else None
    residuals = residual_report(y_test, predictions[reference], groups=groups)
    return BenchmarkResult(
        cv_table=cv_table,
        comparison=comparison,
        holdout=pd.DataFrame(holdout_rows).T,
        fold_metrics=fold_metrics,
        predictions=predictions,
        residuals=residuals,
        specs=specs,
        reference=reference,
        protocol={
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "random_state": random_state,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "primary_metric": PRIMARY_METRIC,
            "higher_is_better": list(HIGHER_IS_BETTER),
        },
        fitted=fitted,
    )
