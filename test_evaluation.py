"""Tests for src/evaluation.py: metrics, residual diagnostics, cross-validation and model comparison."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.evaluation import (
    HIGHER_IS_BETTER,
    PRIMARY_METRIC,
    corrected_resampled_ttest,
    cross_validate_metrics,
    metric_definitions,
    plot_diagnostics,
    regression_metrics,
    residual_report,
)

Y = np.array([100.0, 200.0, 300.0, 400.0])
P = np.array([110.0, 190.0, 330.0, 380.0])  # errors: +10, -10, +30, -20


# --------------------------------------------------------------------------- #
# regression_metrics
# --------------------------------------------------------------------------- #
def test_metrics_match_hand_calculation():
    m = regression_metrics(Y, P, n_features=1)
    assert m["rmse"] == pytest.approx(np.sqrt((100 + 100 + 900 + 400) / 4))
    assert m["mae"] == pytest.approx(17.5)
    assert m["mape"] == pytest.approx(7.5)  # 10%, 5%, 10%, 5%
    assert m["median_ae"] == pytest.approx(15.0) and m["max_error"] == pytest.approx(30.0)
    assert m["bias"] == pytest.approx(2.5)
    assert m["r2"] == pytest.approx(1 - 1500 / 50000)
    assert m["adj_r2"] == pytest.approx(1 - (1 - 0.97) * 3 / 2)
    assert m["rmse_pct_of_mean"] == pytest.approx(100 * m["rmse"] / 250)
    assert m["within_10pct"] == pytest.approx(100.0)
    expected_rmsle = np.sqrt(np.mean((np.log1p(P) - np.log1p(Y)) ** 2))
    assert m["rmsle"] == pytest.approx(expected_rmsle)


def test_metrics_agree_with_scikit_learn(raw_df):
    rng = np.random.default_rng(0)
    y = raw_df["house_sale_price"].to_numpy()
    pred = y * rng.normal(1, 0.1, len(y))
    m = regression_metrics(y, pred)
    assert m["rmse"] == pytest.approx(mean_squared_error(y, pred) ** 0.5)
    assert m["mae"] == pytest.approx(mean_absolute_error(y, pred))
    assert m["r2"] == pytest.approx(r2_score(y, pred))


def test_perfect_prediction_and_mean_predictor():
    perfect = regression_metrics(Y, Y)
    assert perfect["rmse"] == 0 and perfect["r2"] == 1 and perfect["bias"] == 0 and perfect["within_10pct"] == 100
    mean_pred = regression_metrics(Y, np.full(4, Y.mean()))
    assert mean_pred["r2"] == pytest.approx(0) and mean_pred["bias"] == pytest.approx(0)


def test_bias_sign_convention():
    assert regression_metrics(Y, Y + 5)["bias"] == pytest.approx(5)  # over-prediction is positive
    assert regression_metrics(Y, Y - 5)["bias"] == pytest.approx(-5)


def test_within_tolerance_key_and_value():
    m = regression_metrics(Y, P, tolerance=0.05)
    assert "within_5pct" in m and "within_10pct" not in m
    assert m["within_5pct"] == pytest.approx(50.0)  # only the two 5% errors qualify


def test_adjusted_r2_needs_enough_rows():
    assert np.isnan(regression_metrics(Y, P)["adj_r2"])  # n_features not given
    assert np.isnan(regression_metrics(Y, P, n_features=3)["adj_r2"])  # n <= p + 1


def test_zero_true_prices_are_excluded_from_relative_metrics():
    m = regression_metrics([0.0, 100.0, 200.0], [10.0, 110.0, 180.0])
    assert m["mape"] == pytest.approx(100 * np.mean([0.1, 0.1]))
    assert np.isfinite(m["rmse"])


def test_rmsle_clips_negative_predictions_and_is_nan_for_negative_truth():
    assert np.isfinite(regression_metrics([100.0, 200.0], [-5.0, 210.0])["rmsle"])
    assert np.isnan(regression_metrics([-1.0, 2.0], [0.0, 2.0])["rmsle"])


@pytest.mark.parametrize(
    "y_true,y_pred,match",
    [
        ([1.0, 2.0], [1.0], "different lengths"),
        ([], [], "empty"),
        ([1.0, np.nan], [1.0, 2.0], "NaN"),
        ([1.0, 2.0], [1.0, np.inf], "NaN"),
    ],
)
def test_metrics_reject_bad_input(y_true, y_pred, match):
    with pytest.raises(ValueError, match=match):
        regression_metrics(y_true, y_pred)


def test_metrics_reject_bad_tolerance_and_shape():
    with pytest.raises(ValueError):
        regression_metrics(Y, P, tolerance=1.5)
    with pytest.raises(ValueError, match="one-dimensional"):
        regression_metrics(np.ones((2, 2)), np.ones((2, 2)))


def test_metrics_accept_series_and_column_vectors():
    a = regression_metrics(pd.Series(Y), pd.Series(P))
    b = regression_metrics(Y.reshape(-1, 1), P.reshape(-1, 1))
    assert pd.Series(a).equals(pd.Series(b))  # (NaN == NaN here)


def test_metric_definitions_cover_every_metric_and_the_constants_are_consistent():
    defs = metric_definitions()
    metrics = set(regression_metrics(Y, P, n_features=1))
    assert metrics == set(defs.index)
    assert PRIMARY_METRIC in defs.index and set(HIGHER_IS_BETTER) <= set(defs.index)
    assert (defs["why it matters"].str.len() > 10).all()


# --------------------------------------------------------------------------- #
# residual_report
# --------------------------------------------------------------------------- #
@pytest.fixture()
def heteroscedastic():
    rng = np.random.default_rng(1)
    y = rng.uniform(100_000, 900_000, 600)
    pred = y + rng.normal(0, 0.05, 600) * y  # error proportional to price
    groups = rng.choice(["A", "B", "C"], 600)
    return y, pred, groups


def test_residual_report_structure_and_counts(heteroscedastic):
    y, pred, groups = heteroscedastic
    report = residual_report(y, pred, groups=groups, n_bands=5)
    assert {"overall", "by_price_band", "by_group", "error_trend"} <= set(report)
    assert list(report["by_price_band"].index) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert report["by_price_band"]["n"].sum() == 600 and report["by_group"]["n"].sum() == 600
    assert report["overall"].loc["all", "n"] == 600
    assert report["overall"].loc["all", "bias"] == pytest.approx(np.mean(pred - y))


def test_residual_report_detects_errors_growing_with_price(heteroscedastic):
    y, pred, _ = heteroscedastic
    trend = residual_report(y, pred)["error_trend"].iloc[0]
    assert trend["corr_abs_error_vs_true_price"] > 0.4
    assert trend["mae_top_band_over_bottom_band"] > 3


def test_residual_report_detects_regression_to_the_mean():
    y = np.linspace(100, 500, 200)
    pred = 300 + 0.7 * (y - 300)  # shrinks towards the mean
    bands = residual_report(y, pred)["by_price_band"]
    assert bands["bias"].iloc[0] > 0 > bands["bias"].iloc[-1]  # over-predicts cheap, under-predicts expensive


def test_residual_report_validates_input():
    with pytest.raises(ValueError):
        residual_report([1, 2, 3], [1, 2])
    with pytest.raises(ValueError, match="n_bands"):
        residual_report([1, 2, 3, 4], [1, 2, 3, 4], n_bands=1)
    with pytest.raises(ValueError, match="same length"):
        residual_report([1, 2, 3, 4], [1, 2, 3, 4], groups=["a", "b"])


def test_residual_report_missing_group_label_is_kept():
    report = residual_report([1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5], groups=["a", None, "a", "b"], n_bands=2)
    assert "Missing" in report["by_group"].index


# --------------------------------------------------------------------------- #
# cross_validate_metrics
# --------------------------------------------------------------------------- #
def test_cross_validation_returns_all_metrics_per_fold(split):
    raw_train, _, y_train, _ = split
    num = raw_train.drop(columns="location_type")
    folds = cross_validate_metrics(
        LinearRegression(), num, y_train, n_splits=4, n_repeats=2, random_state=0, n_features=7
    )
    assert len(folds) == 8 and set(regression_metrics(y_train, y_train, 7)) <= set(folds.columns)
    assert "fit_seconds" in folds.columns and (folds["fit_seconds"] >= 0).all()
    assert folds["r2"].notna().all()


def test_cross_validation_is_deterministic_and_shares_folds(split):
    raw_train, _, y_train, _ = split
    num = raw_train.drop(columns="location_type")
    a = cross_validate_metrics(DummyRegressor(), num, y_train, 3, 2, 5)["rmse"]
    b = cross_validate_metrics(DummyRegressor(), num, y_train, 3, 2, 5)["rmse"]
    c = cross_validate_metrics(DummyRegressor(), num, y_train, 3, 2, 6)["rmse"]
    pd.testing.assert_series_equal(a, b)
    assert not np.allclose(a, c)
    folds_a = cross_validate_metrics(DummyRegressor(), num, y_train, 3, 2, 5).attrs["validation_indices"]
    folds_b = cross_validate_metrics(LinearRegression(), num, y_train, 3, 2, 5).attrs["validation_indices"]
    assert all(np.array_equal(x, y) for x, y in zip(folds_a, folds_b))  # different models, same folds


def test_cross_validation_refits_a_clone_on_the_training_part_only(split):
    raw_train, _, y_train, _ = split
    seen = []

    class Recorder(BaseEstimator, RegressorMixin):
        def fit(self, X, y):
            seen.append((len(X), set(X.index)))
            self.mean_ = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(len(X), self.mean_)

    est = Recorder()
    cross_validate_metrics(est, raw_train.drop(columns="location_type"), y_train, 4, 1, 0)
    assert len(seen) == 4 and all(n == 300 for n, _ in seen)  # 400 rows, 4 folds -> 300 train rows
    assert not hasattr(est, "mean_")  # the original estimator was never fitted
    assert len({frozenset(s) for _, s in seen}) == 4


# --------------------------------------------------------------------------- #
# corrected_resampled_ttest
# --------------------------------------------------------------------------- #
def test_corrected_ttest_matches_manual_formula():
    base = np.array([10.0, 11.0, 12.0, 9.0, 10.5, 11.5, 10.2, 9.8, 12.2, 10.9])
    model = base - np.array([1.0, 1.2, 0.8, 1.1, 0.9, 1.3, 1.0, 0.7, 1.2, 1.1])
    diff = base - model
    mean, t_stat, p_value = corrected_resampled_ttest(base, model, n_splits=5)
    k = len(diff)
    t_manual = diff.mean() / np.sqrt((1 / k + 1 / 4) * diff.var(ddof=1))
    assert mean == pytest.approx(diff.mean()) and t_stat == pytest.approx(t_manual)
    assert p_value == pytest.approx(2 * stats.t.sf(abs(t_manual), k - 1))
    assert mean > 0  # lower error = improvement


def test_corrected_ttest_is_more_cautious_than_the_plain_paired_test():
    rng = np.random.default_rng(0)
    base = rng.normal(10, 1, 15)
    model = base - rng.normal(0.4, 0.5, 15)
    plain = stats.ttest_rel(base, model).pvalue
    assert corrected_resampled_ttest(base, model, 5)[2] > plain


def test_corrected_ttest_edge_cases():
    same = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isnan(corrected_resampled_ttest(same, same, 5)[2])  # no variance in the differences
    with pytest.raises(ValueError, match="three"):
        corrected_resampled_ttest([1.0, 2.0], [1.0, 2.0], 5)


# --------------------------------------------------------------------------- #
# plot
# --------------------------------------------------------------------------- #
def test_diagnostics_plot_is_written(tmp_path, heteroscedastic):
    y, pred, _ = heteroscedastic
    path = plot_diagnostics(y, pred, tmp_path / "plots" / "diag.png", title="test")
    assert path.exists() and path.stat().st_size > 10_000
