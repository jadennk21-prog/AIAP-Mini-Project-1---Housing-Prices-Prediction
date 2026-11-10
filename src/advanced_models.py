"""
Advanced models: two architectures that differ fundamentally from the linear baseline.

=====================  ==================================================  ==========================================
model                  architecture                                        how it differs from the baseline
=====================  ==================================================  ==========================================
baseline               one linear equation on log(price)                   fixed form: price is a product of effects
gradient boosted trees hundreds of shallow decision trees built one after   piecewise-constant, no assumed form,
                       another, each correcting its predecessors' errors   handles interactions automatically
neural network (MLP)   a feed-forward network of weighted sums and          smooth, flexible curves and interactions
                       non-linear activations, averaged over several seeds  learned from data
=====================  ==================================================  ==========================================

Why these two. They were chosen by a **screening** of architecture families under identical preprocessing and
identical cross-validation folds (:func:`screening_specs`, ``python main.py --advanced-screen``), taking the best
representative of two families that work in genuinely different ways: boosted trees were the best tree-based
model (random forest, extra trees and kNN were clearly worse) and the neural network was the best non-tree model
(kernel ridge was close, SVR was worse). The screening, the tuned results and their analysis are in the README.

Fair comparison rules used throughout:

* **Same inputs.** Both models receive exactly the baseline's preprocessed features (imputation, outlier capping,
  logs, scaling, one-hot) and predict log(price), so the architecture is the only difference. (Engineered features
  were tested and *hurt* both models, see the README.)
* **Tuning without peeking.** Hyperparameters are chosen by randomized search with an inner cross-validation on the
  training rows, scored on **price-scale RMSE**, and the search runs *inside* every outer cross-validation fold
  (**nested cross-validation**), so the reported scores do not benefit from tuning on the validation folds. The
  test set is never used for tuning.
* **Same protocol as the baseline.** The models are evaluated by :func:`src.models.run_benchmark` with identical
  folds, metrics and the corrected resampled t-test, so every difference to the baseline has a p-value.
* **Reproducible.** Every random choice (search sampling, tree subsampling, network initialisation, fold
  assignment) derives from one seed.

Beyond accuracy, :func:`analyse_comparison` looks at *where* and *why* the models differ: errors by price band and
location, how often each model beats the baseline on individual homes, bootstrap confidence intervals for the
differences, error correlation and simple blends, complexity, speed and size, and permutation importance.
"""

from __future__ import annotations

import io
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils.validation import check_is_fitted

from .evaluation import regression_metrics
from .feature_engineering import FeatureEngineeringConfig, make_full_pipeline
from .models import BaselineSpec, BenchmarkResult, baseline_specs
from .preprocessing import DataPreprocessor, PreprocessingConfig, build_target_transformer

__all__ = [
    "GRADIENT_BOOSTING_NAME",
    "NEURAL_NETWORK_NAME",
    "ENGINEERED_LINEAR_NAME",
    "SeedAveragedRegressor",
    "TunedRegressor",
    "make_gradient_boosting",
    "make_neural_network",
    "advanced_specs",
    "comparison_specs",
    "screening_specs",
    "ModelAnalysis",
    "analyse_comparison",
    "error_by_group",
    "bootstrap_rmse_difference",
    "model_complexity",
    "permutation_importance_table",
    "plot_comparison",
    "tuning_summary",
]

logger = logging.getLogger(__name__)

GRADIENT_BOOSTING_NAME = "gradient boosted trees"
NEURAL_NETWORK_NAME = "neural network (MLP, seed-averaged)"
ENGINEERED_LINEAR_NAME = "linear regression + engineered features (VIF)"

#: Settings found by tuning on the training rows; used when tuning is switched off (fast runs, tests).
GRADIENT_BOOSTING_DEFAULTS: Dict[str, Any] = {
    "n_estimators": 800,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.6,
    "min_samples_leaf": 3,
}
NEURAL_NETWORK_DEFAULTS: Dict[str, Any] = {
    "hidden_layer_sizes": (32,),
    "alpha": 2.6e-4,
    "learning_rate_init": 0.01,
    "activation": "relu",
}

#: Randomized-search spaces (parameter names relative to the final estimator).
GRADIENT_BOOSTING_SPACE: Dict[str, Any] = {
    "n_estimators": [200, 400, 800],
    "learning_rate": [0.02, 0.05, 0.1],
    "max_depth": [2, 3, 4],
    "subsample": [0.6, 0.8, 1.0],
    "min_samples_leaf": [1, 3, 5, 10],
}
NEURAL_NETWORK_SPACE: Dict[str, Any] = {
    "hidden_layer_sizes": [(32,), (64,), (32, 16), (64, 32), (128, 64)],
    "alpha": loguniform(1e-4, 1e-1),
    "learning_rate_init": [1e-3, 3e-3, 1e-2],
    "activation": ["relu", "tanh"],
}

_GB_PREFIX = "regressor__model__"
_NN_PREFIX = "regressor__model__base__"


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class SeedAveragedRegressor(BaseEstimator, RegressorMixin):
    """Average the predictions of ``n_models`` copies of ``base`` that differ only in their random seed.

    A neural network's result depends on its random initialisation (different seeds gave RMSE differences of
    about 2 to 3% in this project). Averaging several seeds removes most of that luck and usually improves
    accuracy slightly; the architecture is unchanged. After fitting: ``models_`` and ``converged_fraction_``
    (share of networks that stopped because of convergence rather than the iteration limit).
    """

    def __init__(self, base: Optional[BaseEstimator] = None, n_models: int = 5, random_state: int = 0):
        self.base = base
        self.n_models = n_models
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> SeedAveragedRegressor:
        if self.base is None:
            raise ValueError("SeedAveragedRegressor needs a base estimator")
        if self.n_models < 1:
            raise ValueError("n_models must be at least 1")
        models = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)  # reported through converged_fraction_ instead
            for i in range(self.n_models):
                member = clone(self.base)
                member.set_params(random_state=self.random_state + i)
                models.append(member.fit(X, y))
        self.models_ = models
        limits = [getattr(m, "max_iter", None) for m in models]
        steps = [getattr(m, "n_iter_", None) for m in models]
        flags = [s < limit for s, limit in zip(steps, limits) if s is not None and limit is not None]
        self.converged_fraction_ = float(np.mean(flags)) if flags else float("nan")
        self.n_features_in_ = getattr(models[0], "n_features_in_", None)
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ["models_"])
        return np.mean([m.predict(X) for m in self.models_], axis=0)


class TunedRegressor(BaseEstimator, RegressorMixin):
    """Choose hyperparameters by randomized search with inner cross-validation, then refit on all rows given.

    ``estimator`` is a complete estimator (for example preprocessing + model + target transform), so every
    inner fold refits the whole chain and tuning cannot leak. The search is scored by **price-scale RMSE**
    (the project's primary metric). ``final_params`` are applied only to the final refit, which allows a cheap
    search with a heavier final model (for example one network during the search, an average of five after it).

    After fitting: ``best_params_``, ``best_score_`` (inner-CV RMSE of the best setting; optimistic by
    construction, use the outer cross-validation for honest numbers), ``cv_results_`` and ``best_estimator_``.
    Used inside an outer cross-validation this gives **nested cross-validation**.
    """

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        param_distributions: Optional[Dict[str, Any]] = None,
        n_iter: int = 12,
        inner_folds: int = 3,
        random_state: int = 42,
        n_jobs: int = 1,
        final_params: Optional[Dict[str, Any]] = None,
    ):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.inner_folds = inner_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.final_params = final_params

    def fit(self, X: Any, y: Any) -> TunedRegressor:
        if self.estimator is None or not self.param_distributions:
            raise ValueError("TunedRegressor needs an estimator and param_distributions")
        search = RandomizedSearchCV(
            clone(self.estimator),
            self.param_distributions,
            n_iter=self.n_iter,
            cv=KFold(self.inner_folds, shuffle=True, random_state=self.random_state),
            scoring="neg_root_mean_squared_error",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            refit=False,
            error_score="raise",
        )
        search.fit(X, y)
        self.cv_results_ = pd.DataFrame(search.cv_results_)
        self.best_params_ = dict(search.best_params_)
        self.best_score_ = float(-search.best_score_)
        final = clone(self.estimator).set_params(**self.best_params_)
        if self.final_params:
            final.set_params(**self.final_params)
        self.best_estimator_ = final.fit(X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ["best_estimator_"])
        return np.asarray(self.best_estimator_.predict(X), dtype=float)


def _target_transformer(config: PreprocessingConfig, standardise: bool) -> Any:
    """The configured target transform (log by default), optionally followed by standardisation.

    Neural networks train much better on a standardised target (mean 0, spread 1); trees do not care.
    """
    base = build_target_transformer(config.target_transform)
    if not standardise:
        return base
    if base is None:
        return StandardScaler()
    return Pipeline([("transform", base), ("standardise", StandardScaler())])


def _wrap(model: BaseEstimator, config: PreprocessingConfig, standardise_target: bool) -> TransformedTargetRegressor:
    """Preprocessing -> ``model``, trained on the transformed target and predicting prices."""
    pipeline = Pipeline([("preprocess", DataPreprocessor(config)), ("model", model)])
    return TransformedTargetRegressor(regressor=pipeline, transformer=_target_transformer(config, standardise_target))


def _prefixed(space: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}{name}": values for name, values in space.items()}


def make_gradient_boosting(
    config: Optional[PreprocessingConfig] = None,
    random_state: int = 42,
    tune: bool = True,
    n_iter: int = 12,
    inner_folds: int = 3,
    n_jobs: int = 1,
) -> BaseEstimator:
    """Gradient boosted trees on the preprocessed features and the log target.

    With ``tune=True`` (default) the result is a :class:`TunedRegressor` that searches
    ``GRADIENT_BOOSTING_SPACE``; with ``tune=False`` it uses ``GRADIENT_BOOSTING_DEFAULTS``.
    """
    cfg = config or PreprocessingConfig()
    model = GradientBoostingRegressor(random_state=random_state, **GRADIENT_BOOSTING_DEFAULTS)
    estimator = _wrap(model, cfg, standardise_target=False)
    if not tune:
        return estimator
    return TunedRegressor(
        estimator, _prefixed(GRADIENT_BOOSTING_SPACE, _GB_PREFIX), n_iter, inner_folds, random_state, n_jobs
    )


def make_neural_network(
    config: Optional[PreprocessingConfig] = None,
    random_state: int = 42,
    tune: bool = True,
    n_iter: int = 12,
    inner_folds: int = 3,
    n_seeds: int = 5,
    n_jobs: int = 1,
) -> BaseEstimator:
    """A feed-forward neural network (MLP) on the preprocessed features, log target (standardised).

    The final model averages ``n_seeds`` networks with different random initialisations. While tuning, a single
    network is used per candidate setting (cheaper), and the seed average is built only for the final refit.
    """
    cfg = config or PreprocessingConfig()

    def network(n_models: int) -> SeedAveragedRegressor:
        base = MLPRegressor(max_iter=3000, **NEURAL_NETWORK_DEFAULTS)
        return SeedAveragedRegressor(base, n_models=n_models, random_state=random_state)

    if not tune:
        return _wrap(network(n_seeds), cfg, standardise_target=True)
    return TunedRegressor(
        _wrap(network(1), cfg, standardise_target=True),
        _prefixed(NEURAL_NETWORK_SPACE, _NN_PREFIX),
        n_iter,
        inner_folds,
        random_state,
        n_jobs,
        final_params={"regressor__model__n_models": n_seeds},
    )


# --------------------------------------------------------------------------- #
# Catalogues of models for the benchmark
# --------------------------------------------------------------------------- #
def _clean_params(params: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}
    for key, value in params.items():
        short = key.split("__")[-1]
        cleaned[short] = value.item() if isinstance(value, np.generic) else value
    return cleaned


def _collect_tuning(model: Any) -> Dict[str, Any]:
    """What a tuned model chose (recorded for every outer fold)."""
    if not isinstance(model, TunedRegressor):
        return {}
    return {"best_params": _clean_params(model.best_params_), "inner_cv_rmse": model.best_score_}


def advanced_specs(
    config: Optional[PreprocessingConfig] = None,
    random_state: int = 42,
    tune: bool = True,
    n_iter: int = 12,
    inner_folds: int = 3,
    n_seeds: int = 5,
    n_jobs: int = 1,
) -> Dict[str, BaselineSpec]:
    """The two advanced models as benchmark entries (tier ``advanced``), ready for ``run_benchmark``."""
    cfg = config or PreprocessingConfig()
    mode = f"tuned by randomized search ({n_iter} settings, {inner_folds}-fold inner CV)" if tune else "fixed settings"
    return {
        GRADIENT_BOOSTING_NAME: BaselineSpec(
            GRADIENT_BOOSTING_NAME,
            lambda: make_gradient_boosting(cfg, random_state, tune, n_iter, inner_folds, n_jobs),
            f"Sequential ensemble of shallow regression trees on the preprocessed features and log target; {mode}.",
            "advanced",
            collect=_collect_tuning,
        ),
        NEURAL_NETWORK_NAME: BaselineSpec(
            NEURAL_NETWORK_NAME,
            lambda: make_neural_network(cfg, random_state, tune, n_iter, inner_folds, n_seeds, n_jobs),
            f"Feed-forward network (MLP) averaged over {n_seeds} seeds, on the preprocessed features and a "
            f"standardised log target; {mode}.",
            "advanced",
            collect=_collect_tuning,
        ),
    }


def comparison_specs(
    config: Optional[PreprocessingConfig] = None,
    fe_config: Optional[FeatureEngineeringConfig] = None,
    random_state: int = 42,
    tune: bool = True,
    n_iter: int = 12,
    inner_folds: int = 3,
    n_seeds: int = 5,
    include_engineered_linear: bool = True,
    n_jobs: int = 1,
) -> Dict[str, BaselineSpec]:
    """Mean predictor (floor), the reference baseline, optionally the engineered-feature linear model, and the
    two advanced models, in that order."""
    cfg = config or PreprocessingConfig()
    ladder = baseline_specs(cfg)
    specs = {name: spec for name, spec in ladder.items() if spec.tier in ("floor", "reference")}
    specs.pop("median price (ignores all features)", None)
    if include_engineered_linear:
        fe = fe_config or FeatureEngineeringConfig()
        specs[ENGINEERED_LINEAR_NAME] = BaselineSpec(
            ENGINEERED_LINEAR_NAME,
            lambda: make_full_pipeline(LinearRegression(), cfg, fe, "vif"),
            "Reference model plus the engineered features and VIF selection (previous sprint's best model).",
            "challenger",
        )
    specs.update(advanced_specs(cfg, random_state, tune, n_iter, inner_folds, n_seeds, n_jobs))
    return specs


def screening_specs(config: Optional[PreprocessingConfig] = None, random_state: int = 42) -> Dict[str, BaselineSpec]:
    """Architecture families with sensible fixed settings, for the model-selection screening.

    Everything uses the baseline's preprocessing and log target (standardised for the distance- and
    gradient-based models), so differences come from the architecture. Settings are not tuned.
    """
    cfg = config or PreprocessingConfig()

    def spec(name: str, model: BaseEstimator, standardise: bool, family: str) -> BaselineSpec:
        return BaselineSpec(
            name,
            lambda: _wrap(clone(model), cfg, standardise),
            f"{family}: preprocessing + {type(model).__name__} on the log target.",
            "screen",
        )

    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, max_iter=3000, random_state=random_state)
    mean_predictor = BaselineSpec(
        "mean price (ignores all features)",
        lambda: DummyRegressor(strategy="mean"),
        "Predicts the training mean for every house.",
        "floor",
    )
    entries = [
        mean_predictor,
        spec("linear regression", LinearRegression(), False, "linear"),
        spec("ridge (alpha=10)", Ridge(alpha=10), False, "linear"),
        spec(
            "random forest",
            RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=random_state, n_jobs=1),
            False,
            "bagged trees",
        ),
        spec(
            "extra trees",
            ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=random_state, n_jobs=1),
            False,
            "randomised trees",
        ),
        spec(
            "gradient boosting (defaults)", GradientBoostingRegressor(random_state=random_state), False, "boosted trees"
        ),
        spec(
            "gradient boosting (500 trees, lr 0.05)",
            GradientBoostingRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=random_state
            ),
            False,
            "boosted trees",
        ),
        spec(
            "hist gradient boosting", HistGradientBoostingRegressor(random_state=random_state), False, "boosted trees"
        ),
        spec("neural network (64, 32)", mlp, True, "neural network"),
        spec(
            "neural network (32,)",
            MLPRegressor(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=3000, random_state=random_state),
            True,
            "neural network",
        ),
        spec("kernel ridge (RBF)", KernelRidge(alpha=0.1, kernel="rbf", gamma=0.05), True, "kernel method"),
        spec("support vector regression (RBF)", SVR(C=3, epsilon=0.05, gamma="scale"), True, "kernel method"),
        spec(
            "k-nearest neighbours (k=10)",
            KNeighborsRegressor(n_neighbors=10, weights="distance"),
            True,
            "instance-based",
        ),
    ]
    return {e.name: e for e in entries}


def tuning_summary(result: BenchmarkResult) -> pd.DataFrame:
    """The hyperparameters each tuned model chose in every outer fold (nested CV) and on the full training set.

    One row per tuned model and fold (``fold`` = ``final`` for the model fitted on all training rows, which is the
    one evaluated on the test set), with the inner-CV RMSE and one column per hyperparameter. Frequent agreement
    across folds means the tuning is stable.
    """
    rows = []
    for name, folds in result.collected.items():
        for i, info in enumerate(folds):
            if info:
                rows.append({"model": name, "fold": i, "inner_cv_rmse": info["inner_cv_rmse"], **info["best_params"]})
        final = result.fitted.get(name)
        if isinstance(final, TunedRegressor):
            rows.append(
                {
                    "model": name,
                    "fold": "final",
                    "inner_cv_rmse": final.best_score_,
                    **_clean_params(final.best_params_),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Analysis of the comparison
# --------------------------------------------------------------------------- #
def error_by_group(y_true: Any, predictions: pd.DataFrame, groups: Any, n_bands: Optional[int] = None) -> pd.DataFrame:
    """Errors of every model within groups: a categorical variable, or ``n_bands`` quantile bands of the true price.

    Pass ``groups`` as labels (for example ``location_type``) or, with ``n_bands``, anything (it is ignored) to split
    the true prices into bands Q1 (cheapest) to Qn. Returns a tidy table with one row per (group, model):
    ``n, mae, bias, bias_pct, rmse``.
    """
    truth = np.asarray(y_true, dtype=float)
    if n_bands is not None:
        banded = pd.qcut(truth, n_bands, labels=[f"Q{i + 1}" for i in range(n_bands)], duplicates="drop")
        labels = pd.Series(np.asarray(banded, dtype=object))
    else:
        labels = pd.Series(np.asarray(groups, dtype=object)).fillna("Missing")
    if len(labels) != len(truth):
        raise ValueError("groups must have the same length as y_true")
    rows = []
    for group in pd.unique(labels):
        mask = (labels == group).to_numpy()
        for model in predictions.columns:
            err = predictions[model].to_numpy(dtype=float)[mask] - truth[mask]
            rows.append(
                {
                    "group": group,
                    "model": model,
                    "n": int(mask.sum()),
                    "mae": float(np.abs(err).mean()),
                    "bias": float(err.mean()),
                    "bias_pct": float(100 * np.mean(err / truth[mask])),
                    "rmse": float(np.sqrt(np.mean(err**2))),
                }
            )
    table = pd.DataFrame(rows)
    if n_bands is not None:
        table["group"] = pd.Categorical(table["group"], [f"Q{i + 1}" for i in range(n_bands)], ordered=True)
        table = table.sort_values(["group", "model"]).reset_index(drop=True)
        table["group"] = table["group"].astype(str)
    return table


def bootstrap_rmse_difference(
    y_true: Any, pred_a: Any, pred_b: Any, n_boot: int = 2000, random_state: int = 0
) -> Dict[str, float]:
    """Bootstrap the test set to ask whether model B's RMSE really differs from model A's.

    The difference is ``RMSE(A) - RMSE(B)`` (positive: B is better). Returns the observed difference, a 95%
    percentile interval, and the share of bootstrap samples in which B beats A. An interval that excludes 0 means the
    difference is unlikely to be due to which homes happened to land in the test set. (It does not account for the
    models having been tuned, only for the finite test sample.)
    """
    y, a, b = (np.asarray(v, dtype=float) for v in (y_true, pred_a, pred_b))
    if not (len(y) == len(a) == len(b)):
        raise ValueError("y_true and both prediction arrays must have the same length")
    rng = np.random.default_rng(random_state)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = np.sqrt(np.mean((a[idx] - y[idx]) ** 2)) - np.sqrt(np.mean((b[idx] - y[idx]) ** 2))
    observed = float(np.sqrt(np.mean((a - y) ** 2)) - np.sqrt(np.mean((b - y) ** 2)))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "rmse_difference": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "share_b_better": float(np.mean(diffs > 0)),
    }


def _final_estimator(model: Any) -> Any:
    """Peel off tuning, target-transform and pipeline wrappers to reach the model itself."""
    current = model
    for _ in range(6):
        if isinstance(current, TunedRegressor):
            current = current.best_estimator_
        elif isinstance(current, TransformedTargetRegressor):
            current = current.regressor_
        elif isinstance(current, Pipeline):
            current = current.steps[-1][1]
        else:
            break
    return current


def model_complexity(model: Any) -> Dict[str, Any]:
    """Kind of model, number of fitted parameters (coefficients, tree nodes or network weights) and file size."""
    final = _final_estimator(model)
    kind, n_params = type(final).__name__, float("nan")
    if isinstance(final, SeedAveragedRegressor):
        nets = final.models_
        kind = f"neural network x {len(nets)} seeds"
        n_params = float(sum(w.size for net in nets for w in [*net.coefs_, *net.intercepts_]))
    elif isinstance(final, GradientBoostingRegressor):
        kind = f"boosted trees ({final.n_estimators_} trees)"
        n_params = float(sum(t.tree_.node_count for t in final.estimators_.ravel()))
    elif hasattr(final, "coef_"):
        kind = "linear model"
        n_params = float(np.size(final.coef_) + 1)
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return {"kind": kind, "parameters": n_params, "size_kb": buffer.tell() / 1024}


def permutation_importance_table(
    model: Any, X: pd.DataFrame, y: Any, n_repeats: int = 10, random_state: int = 0
) -> pd.DataFrame:
    """How much does shuffling each raw feature hurt the model? (increase in price-scale RMSE)

    Model-agnostic, so it compares what the baseline, the trees and the network each rely on. Computed on the data
    given (for interpretation; do not use it to choose models). Returns ``importance`` (mean RMSE increase),
    ``std`` and ``share_%`` (importance as a share of the total), sorted by importance.
    """
    result = permutation_importance(
        model, X, y, scoring="neg_root_mean_squared_error", n_repeats=n_repeats, random_state=random_state, n_jobs=1
    )
    table = pd.DataFrame({"importance": result.importances_mean, "std": result.importances_std}, index=X.columns)
    positive = table["importance"].clip(lower=0)
    table["share_%"] = 100 * positive / positive.sum() if positive.sum() > 0 else 0.0
    return table.sort_values("importance", ascending=False)


@dataclass
class ModelAnalysis:
    """Everything :func:`analyse_comparison` computes beyond the benchmark tables."""

    by_price_band: pd.DataFrame
    by_group: Optional[pd.DataFrame]
    versus_reference: pd.DataFrame
    residual_correlation: pd.DataFrame
    blends: pd.DataFrame
    complexity: pd.DataFrame
    importance: Dict[str, pd.DataFrame] = field(default_factory=dict)
    tuning: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "by_price_band": self.by_price_band,
            "by_group": self.by_group,
            "versus_reference": self.versus_reference,
            "residual_correlation": self.residual_correlation,
            "blends": self.blends,
            "complexity": self.complexity,
            "importance": self.importance,
            "tuning": self.tuning,
        }


def analyse_comparison(
    result: BenchmarkResult,
    X_test: pd.DataFrame,
    y_test: Any,
    group_column: Optional[str] = "location_type",
    n_boot: int = 2000,
    importance_repeats: int = 10,
    random_state: int = 0,
) -> ModelAnalysis:
    """Where, how often and why do the models differ on the held-out test set?

    * ``by_price_band`` / ``by_group``: MAE and bias in each price fifth and location;
    * ``versus_reference``: for every other model, the share of homes where it beats the reference (smaller absolute
      error), and the bootstrap interval of the RMSE difference;
    * ``residual_correlation``: correlation of the models' errors (low means they make different mistakes);
    * ``blends``: test RMSE of the simple average of each pair of models (for insight only; not tuned);
    * ``complexity``: parameters, file size, fit and predict time; ``importance``: permutation importance per model.
    """
    floors = [n for n, spec in result.specs.items() if spec.tier == "floor"]
    preds = result.predictions.drop(columns=["true", *floors])  # the mean predictor is not a model worth comparing
    truth = result.predictions["true"].to_numpy()
    reference = result.reference

    bands = error_by_group(truth, preds, None, n_bands=5)
    groups = error_by_group(truth, preds, X_test[group_column]) if group_column in X_test.columns else None

    rows = {}
    ref_abs = np.abs(preds[reference].to_numpy() - truth)
    for name in preds.columns:
        if name == reference:
            continue
        boot = bootstrap_rmse_difference(truth, preds[reference], preds[name], n_boot, random_state)
        wins = float(np.mean(np.abs(preds[name].to_numpy() - truth) < ref_abs) * 100)
        rows[name] = {"wins_vs_reference_%": wins, **boot}
    versus = pd.DataFrame(rows).T

    residuals = preds.sub(truth, axis=0)
    names = list(preds.columns)
    blends = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            blend = (preds[a] + preds[b]) / 2
            blends.append({"model_a": a, "model_b": b, "blend_rmse": regression_metrics(truth, blend)["rmse"]})

    complexity = {}
    for name, model in result.fitted.items():
        if name not in preds.columns:
            continue
        fit_seconds: Any = result.holdout.loc[name, "fit_seconds"]
        predict_seconds: Any = result.holdout.loc[name, "predict_seconds"]
        complexity[name] = {
            **model_complexity(model),
            "fit_seconds": float(fit_seconds),
            "predict_ms_per_home": 1000 * float(predict_seconds) / len(truth),
        }

    importance = {}
    for name, model in result.fitted.items():
        if name in preds.columns:
            importance[name] = permutation_importance_table(model, X_test, truth, importance_repeats, random_state)

    return ModelAnalysis(
        by_price_band=bands,
        by_group=groups,
        versus_reference=versus,
        residual_correlation=residuals.corr(),
        blends=pd.DataFrame(blends),
        complexity=pd.DataFrame(complexity).T,
        importance=importance,
        tuning=tuning_summary(result),
    )


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def plot_comparison(
    result: BenchmarkResult,
    analysis: ModelAnalysis,
    path: Union[str, Path],
    short_names: Optional[Dict[str, str]] = None,
) -> Path:
    """Four-panel figure: cross-validated RMSE, test predictions, bias by price band, permutation importance."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short = short_names or {}
    label = lambda n: short.get(n, n)  # noqa: E731
    models = [n for n in result.comparison.index if result.specs[n].tier != "floor"]
    colours = plt.get_cmap("tab10")(np.arange(len(models)))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    cv = result.comparison.loc[models]
    axes[0, 0].barh([label(n) for n in models], cv["rmse"], xerr=cv["rmse_std"], color=colours, capsize=3)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set(
        xlabel="RMSE (price units), mean and std over CV folds", title="Cross-validated error (nested for tuned models)"
    )

    truth = result.predictions["true"]
    lim = [truth.min(), truth.max()]
    for colour, name in zip(colours, models):
        axes[0, 1].scatter(truth, result.predictions[name], s=10, alpha=0.5, color=colour, label=label(name))
    axes[0, 1].plot(lim, lim, color="black", lw=1)
    axes[0, 1].set(xlabel="actual price", ylabel="predicted price", title="Test-set predictions")
    axes[0, 1].legend(fontsize=7)

    bands = analysis.by_price_band
    for colour, name in zip(colours, models):
        part = bands[bands["model"] == name]
        axes[1, 0].plot(part["group"], part["bias_pct"], marker="o", color=colour, label=label(name))
    axes[1, 0].axhline(0, color="black", lw=1)
    axes[1, 0].set(
        xlabel="true-price band (Q1 = cheapest)",
        ylabel="mean error (% of price)",
        title="Bias by price band (test set)",
    )

    imp = pd.DataFrame({label(n): t["share_%"] for n, t in analysis.importance.items() if n in models})
    imp = imp.loc[imp.mean(axis=1).sort_values(ascending=False).index]
    imp.plot.barh(ax=axes[1, 1], width=0.8)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set(xlabel="share of permutation importance (%)", title="What each model relies on")
    axes[1, 1].legend(fontsize=7)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
