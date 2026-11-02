"""
Evaluation metrics and diagnostics for the housing-price regression problem.

Every metric is computed on the **original price scale** (not on log prices), so numbers are
directly comparable across models whatever target transform they use internally.

Why these metrics (the EDA found a right-skewed price with a few very expensive homes):

* **RMSE** (primary): squares errors, so expensive homes dominate it. It is the standard headline
  number and is in price units, but it is sensitive to a handful of large misses.
* **MAE**: the typical absolute miss, robust to the expensive tail; reported next to RMSE.
* **MAPE** and **within 10%**: relative errors. A 20,000 miss matters far more on a 150,000 home
  than on a 900,000 one; "share of predictions within 10% of the true price" is also easy to explain.
* **R-squared** and adjusted R-squared: the share of price variance explained, comparable across
  problems; the mean predictor scores about 0.
* **Bias** (mean signed error): systematic over- or under-prediction. RMSE and MAE hide its sign.
* **RMSLE**: error measured on a log scale (symmetric in relative terms).
* **Median absolute error** and **max error**: the typical case and the worst case.

``residual_report`` shows where the errors are (by price band and group), ``cross_validate_metrics``
computes all metrics per cross-validation fold, and ``corrected_resampled_ttest`` compares two models
fairly across overlapping folds.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.model_selection import RepeatedKFold

__all__ = [
    "regression_metrics",
    "metric_definitions",
    "residual_report",
    "cross_validate_metrics",
    "corrected_resampled_ttest",
    "plot_diagnostics",
    "PRIMARY_METRIC",
    "HIGHER_IS_BETTER",
]

logger = logging.getLogger(__name__)

PRIMARY_METRIC = "rmse"
#: Metrics where a larger value is better (all others: smaller is better; ``bias`` is best near 0).
HIGHER_IS_BETTER = ("r2", "adj_r2", "explained_variance", "within_10pct")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _as_1d(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr


def regression_metrics(
    y_true: Any, y_pred: Any, n_features: Optional[int] = None, tolerance: float = 0.10
) -> Dict[str, float]:
    """All regression metrics for one set of predictions (price scale).

    Parameters
    ----------
    n_features:
        Number of predictors; needed for adjusted R-squared (``nan`` when omitted or when the
        sample is too small).
    tolerance:
        Relative tolerance of the "within X%" hit rate (default 10%); the key is ``within_10pct``
        for the default and ``within_<n>pct`` otherwise.

    Rows where the true price is 0 are excluded from the relative metrics (MAPE, within-X%);
    RMSLE clips negative predictions at 0 so that it stays defined for models that can predict
    below zero.
    """
    yt, yp = _as_1d(y_true, "y_true"), _as_1d(y_pred, "y_pred")
    if len(yt) != len(yp):
        raise ValueError(f"y_true and y_pred have different lengths ({len(yt)} vs {len(yp)})")
    if len(yt) == 0:
        raise ValueError("Cannot compute metrics on empty arrays")
    if not (np.isfinite(yt).all() and np.isfinite(yp).all()):
        raise ValueError("y_true and y_pred must not contain NaN or infinite values")
    if not 0 < tolerance < 1:
        raise ValueError("tolerance must be between 0 and 1")

    err = yp - yt
    abs_err = np.abs(err)
    n = len(yt)
    mse = float(np.mean(err**2))
    nonzero = yt != 0
    rel = abs_err[nonzero] / np.abs(yt[nonzero])
    r2 = float(r2_score(yt, yp)) if n > 1 else float("nan")

    adj_r2 = float("nan")
    if n_features is not None and n > n_features + 1 and np.isfinite(r2):
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1)

    rmsle = float("nan")
    if (yt >= 0).all():
        rmsle = float(np.sqrt(np.mean((np.log1p(np.clip(yp, 0, None)) - np.log1p(yt)) ** 2)))

    key = f"within_{round(tolerance * 100)}pct"
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(abs_err.mean()),
        "mape": float(100 * rel.mean()) if rel.size else float("nan"),
        "r2": r2,
        "adj_r2": adj_r2,
        "explained_variance": float(explained_variance_score(yt, yp)) if n > 1 else float("nan"),
        "median_ae": float(np.median(abs_err)),
        "max_error": float(abs_err.max()),
        "bias": float(err.mean()),
        "rmsle": rmsle,
        "rmse_pct_of_mean": float(100 * np.sqrt(mse) / abs(yt.mean())) if yt.mean() != 0 else float("nan"),
        key: float(100 * np.mean(rel <= tolerance)) if rel.size else float("nan"),
    }


def metric_definitions() -> pd.DataFrame:
    """Table explaining every metric: definition, direction and why it matters for this problem."""
    rows = [
        (
            "rmse",
            "sqrt(mean((pred - true)^2))",
            "lower",
            "Primary metric. Punishes large misses; dominated by expensive homes.",
        ),
        ("mae", "mean(|pred - true|)", "lower", "Typical miss in price units; robust to the expensive tail."),
        (
            "mape",
            "100 * mean(|pred - true| / true)",
            "lower",
            "Relative error in percent: a miss matters more on a cheap home.",
        ),
        ("r2", "1 - SSE / SST", "higher", "Share of price variance explained; the mean predictor scores about 0."),
        (
            "adj_r2",
            "R2 penalised for the number of predictors",
            "higher",
            "R2 that does not rise just because features were added.",
        ),
        ("explained_variance", "1 - Var(err) / Var(true)", "higher", "Like R2 but ignores a constant bias."),
        ("median_ae", "median(|pred - true|)", "lower", "Typical error, ignoring outliers."),
        ("max_error", "max(|pred - true|)", "lower", "Worst single miss."),
        ("bias", "mean(pred - true)", "near 0", "Systematic over- (positive) or under-prediction (negative)."),
        ("rmsle", "sqrt(mean((log(1+pred) - log(1+true))^2))", "lower", "Error on a log scale (relative, symmetric)."),
        ("rmse_pct_of_mean", "100 * RMSE / mean(true)", "lower", "RMSE as a share of the average price."),
        ("within_10pct", "share of predictions with |error| <= 10% of true", "higher", "Easy-to-explain hit rate."),
    ]
    return pd.DataFrame(rows, columns=["metric", "definition", "better", "why it matters"]).set_index("metric")


# --------------------------------------------------------------------------- #
# Residual diagnostics
# --------------------------------------------------------------------------- #
def residual_report(
    y_true: Any,
    y_pred: Any,
    groups: Optional[Any] = None,
    n_bands: int = 5,
) -> Dict[str, pd.DataFrame]:
    """Where are the errors? Summaries of the residuals overall, by price band and by group.

    ``by_price_band`` splits the *true* prices into ``n_bands`` quantile bands and reports bias,
    MAE and relative error in each, which exposes the common pattern that a model over-predicts
    cheap homes and under-predicts expensive ones. ``by_group`` does the same for a categorical
    variable such as ``location_type``. Residual = prediction - true.
    """
    yt, yp = _as_1d(y_true, "y_true"), _as_1d(y_pred, "y_pred")
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred have different lengths")
    if n_bands < 2:
        raise ValueError("n_bands must be at least 2")
    frame = pd.DataFrame({"true": yt, "pred": yp})
    frame["residual"] = frame["pred"] - frame["true"]
    frame["abs_error"] = frame["residual"].abs()
    frame["rel_error_pct"] = 100 * frame["residual"] / frame["true"].replace(0, np.nan)

    def summarise(part: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "n": len(part),
                "mean_true": part["true"].mean(),
                "bias": part["residual"].mean(),
                "bias_pct": part["rel_error_pct"].mean(),
                "mae": part["abs_error"].mean(),
                "rmse": float(np.sqrt((part["residual"] ** 2).mean())),
                "mape": part["rel_error_pct"].abs().mean(),
            }
        )

    report: Dict[str, pd.DataFrame] = {"overall": summarise(frame).to_frame("all").T}
    bands = pd.qcut(frame["true"], n_bands, labels=[f"Q{i + 1}" for i in range(n_bands)], duplicates="drop")
    report["by_price_band"] = frame.groupby(bands, observed=True).apply(summarise, include_groups=False)
    report["by_price_band"].index.name = "price_band"
    if groups is not None:
        labels = pd.Series(np.asarray(groups, dtype=object), name="group")
        if len(labels) != len(frame):
            raise ValueError("groups must have the same length as y_true")
        report["by_group"] = frame.groupby(labels.fillna("Missing").to_numpy()).apply(summarise, include_groups=False)
        report["by_group"].index.name = "group"
    spread = frame.groupby(bands, observed=True)["abs_error"].mean()
    report["error_trend"] = pd.DataFrame(
        {
            "corr_abs_error_vs_true_price": [
                float(np.corrcoef(frame["abs_error"], frame["true"])[0, 1])
                if frame["abs_error"].std() > 0 and frame["true"].std() > 0
                else float("nan")
            ],
            "mae_top_band_over_bottom_band": [
                float(spread.iloc[-1] / spread.iloc[0]) if spread.iloc[0] else float("nan")
            ],
        }
    )
    return report


# --------------------------------------------------------------------------- #
# Cross-validation and model comparison
# --------------------------------------------------------------------------- #
def cross_validate_metrics(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
    n_features: Optional[int] = None,
) -> pd.DataFrame:
    """Repeated K-fold cross-validation returning *all* metrics for every fold.

    The estimator is cloned and refitted inside each fold (so any preprocessing it contains never
    sees the validation fold) and metrics are computed on the price scale. The folds depend only on
    ``random_state``, so different models evaluated with the same settings share identical folds.
    The result has one row per fold with the columns of :func:`regression_metrics` plus
    ``fit_seconds``; the row positions of each validation fold are kept in
    ``result.attrs["validation_indices"]`` so that identical folds can be verified.
    """
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    rows = []
    validation_indices = []
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X)):
        validation_indices.append(valid_idx)
        model = clone(estimator)
        started = time.perf_counter()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fit_seconds = time.perf_counter() - started
        pred = model.predict(X.iloc[valid_idx])
        rows.append(
            {"fold": fold, **regression_metrics(y.iloc[valid_idx], pred, n_features), "fit_seconds": fit_seconds}
        )
    result = pd.DataFrame(rows).set_index("fold")
    result.attrs["validation_indices"] = validation_indices  # row positions of every validation fold
    return result


def corrected_resampled_ttest(
    baseline_scores: Union[Sequence[float], np.ndarray], scores: Union[Sequence[float], np.ndarray], n_splits: int
) -> Tuple[float, float, float]:
    """Compare two models over the same cross-validation folds (Nadeau and Bengio correction).

    An ordinary paired t-test over repeated-CV folds is over-confident, because the training sets
    overlap. The correction inflates the variance by ``1/k + n_test/n_train`` (here
    ``n_test/n_train = 1/(n_splits - 1)``). Returns ``(mean_improvement, t_statistic, p_value)``
    where ``mean_improvement = mean(baseline_scores - scores)`` (positive means the model has the
    smaller, better, error). The p-value is NaN when the differences have no variance.
    """
    diff = np.asarray(baseline_scores, dtype=float) - np.asarray(scores, dtype=float)
    k = len(diff)
    if k < 3:
        raise ValueError("At least three paired scores are needed")
    variance = float(np.var(diff, ddof=1))
    if variance == 0:
        return float(diff.mean()), float("nan"), float("nan")
    t_stat = float(diff.mean() / np.sqrt((1.0 / k + 1.0 / (n_splits - 1)) * variance))
    return float(diff.mean()), t_stat, float(2 * stats.t.sf(abs(t_stat), k - 1))


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot_diagnostics(
    y_true: Any, y_pred: Any, path: Union[str, Path], title: str = "Model diagnostics", n_bands: int = 5
) -> Path:
    """Save a four-panel diagnostics figure: predicted vs actual, residuals vs prediction,
    residual histogram and bias by price band. Needs matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plot_diagnostics needs matplotlib: pip install matplotlib") from exc

    yt, yp = _as_1d(y_true, "y_true"), _as_1d(y_pred, "y_pred")
    resid = yp - yt
    bands = residual_report(yt, yp, n_bands=n_bands)["by_price_band"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    axes[0, 0].scatter(yt, yp, s=14, alpha=0.5)
    axes[0, 0].plot(lim, lim, color="red", lw=1.5, label="perfect prediction")
    axes[0, 0].set(xlabel="actual price", ylabel="predicted price", title="Predicted vs actual")
    axes[0, 0].legend()
    axes[0, 1].scatter(yp, resid, s=14, alpha=0.5)
    axes[0, 1].axhline(0, color="red", lw=1.5)
    axes[0, 1].set(xlabel="predicted price", ylabel="residual (pred - true)", title="Residuals vs prediction")
    axes[1, 0].hist(resid, bins=30, color="steelblue", edgecolor="white")
    axes[1, 0].axvline(0, color="red", lw=1.5)
    axes[1, 0].set(xlabel="residual", ylabel="count", title=f"Residual distribution (mean {resid.mean():,.0f})")
    axes[1, 1].bar(bands.index.astype(str), bands["bias"], color=np.where(bands["bias"] >= 0, "indianred", "steelblue"))
    axes[1, 1].axhline(0, color="black", lw=1)
    axes[1, 1].set(xlabel="true-price band (Q1 = cheapest)", ylabel="mean residual", title="Bias by price band")
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
