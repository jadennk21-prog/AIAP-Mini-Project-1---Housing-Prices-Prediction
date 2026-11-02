"""Tests for src/models.py: baseline estimators, the baseline ladder, the benchmark harness."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from src.evaluation import regression_metrics
from src.models import (
    REFERENCE_BASELINE,
    BaselineModel,
    BestSingleFeatureRegressor,
    GroupMeanRegressor,
    baseline_specs,
    candidate_specs,
    run_benchmark,
)
from src.preprocessing import PreprocessingConfig


@pytest.fixture()
def small_frame():
    return pd.DataFrame(
        {
            "location_type": ["Rural", "Rural", "Urban", "Urban", "Urban", "Suburban"],
            "size": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "noise": [5.0, 1.0, 4.0, 2.0, 9.0, 3.0],
        }
    ), pd.Series([100.0, 120.0, 200.0, 220.0, 240.0, 300.0])


# --------------------------------------------------------------------------- #
# GroupMeanRegressor
# --------------------------------------------------------------------------- #
def test_group_mean_predicts_group_averages(small_frame):
    X, y = small_frame
    model = GroupMeanRegressor("location_type").fit(X, y)
    assert model.group_values_ == {"Rural": 110.0, "Suburban": 300.0, "Urban": 220.0}
    assert model.predict(X).tolist() == [110.0, 110.0, 220.0, 220.0, 220.0, 300.0]


def test_group_mean_unseen_missing_and_messy_labels_get_the_global_mean(small_frame):
    X, y = small_frame
    model = GroupMeanRegressor("location_type").fit(X, y)
    new = pd.DataFrame({"location_type": ["  urban ", "Coastal", None, "RURAL"]})
    assert model.predict(new).tolist() == [220.0, y.mean(), y.mean(), 110.0]


def test_group_mean_median_statistic_and_errors(small_frame):
    X, y = small_frame
    assert GroupMeanRegressor("location_type", statistic="median").fit(X, y).group_values_["Urban"] == 220.0
    with pytest.raises(ValueError, match="statistic"):
        GroupMeanRegressor(statistic="mode").fit(X, y)
    with pytest.raises(ValueError, match="not found"):
        GroupMeanRegressor("nope").fit(X, y)
    with pytest.raises(NotFittedError):
        GroupMeanRegressor().predict(X)


# --------------------------------------------------------------------------- #
# BestSingleFeatureRegressor
# --------------------------------------------------------------------------- #
def test_best_single_feature_picks_the_most_correlated_column_and_fits_the_line(small_frame):
    X, y = small_frame
    model = BestSingleFeatureRegressor().fit(X.drop(columns="location_type"), y)
    assert model.feature_ == "size"
    slope, intercept = np.polyfit(X["size"], y, 1)
    assert model.coef_ == pytest.approx(slope) and model.intercept_ == pytest.approx(intercept)
    assert np.allclose(model.predict(X), intercept + slope * X["size"])
    assert list(model.correlations_.index)[0] == "size"


def test_best_single_feature_restricts_to_given_features_and_handles_missing(small_frame):
    X, y = small_frame
    only_noise = BestSingleFeatureRegressor(features=["noise"]).fit(X, y)
    assert only_noise.feature_ == "noise"
    holed = X.copy()
    holed.loc[0, "size"] = np.nan
    model = BestSingleFeatureRegressor(features=["size"]).fit(holed, y)
    assert np.isfinite(model.predict(holed)).all()
    assert np.isfinite(model.predict(pd.DataFrame({"size": [np.nan]}))).all()  # filled with the training median


def test_best_single_feature_errors():
    with pytest.raises(ValueError, match="No usable"):
        BestSingleFeatureRegressor().fit(pd.DataFrame({"a": ["x", "y", "z"]}), pd.Series([1.0, 2.0, 3.0]))
    with pytest.raises(NotFittedError):
        BestSingleFeatureRegressor().predict(pd.DataFrame({"a": [1.0]}))


def test_simple_estimators_are_clonable(small_frame):
    X, y = small_frame
    for est in (GroupMeanRegressor("location_type"), BestSingleFeatureRegressor(features=["size"])):
        assert np.allclose(clone(est).fit(X, y).predict(X), est.fit(X, y).predict(X))


# --------------------------------------------------------------------------- #
# The baseline ladder
# --------------------------------------------------------------------------- #
def test_ladder_order_tiers_and_reference():
    specs = baseline_specs()
    names = list(specs)
    assert names[0].startswith("mean price") and names[-1] == REFERENCE_BASELINE
    assert [s.tier for s in specs.values()] == ["floor", "floor", "naive", "naive", "naive", "reference"]
    assert all(len(s.description) > 20 for s in specs.values())


def test_every_baseline_fits_and_predicts_finite_prices(split):
    raw_train, raw_test, y_train, _ = split
    for name, spec in baseline_specs().items():
        pred = np.asarray(spec.build().fit(raw_train, y_train).predict(raw_test))
        assert pred.shape == (len(raw_test),) and np.isfinite(pred).all(), name


def test_builders_return_fresh_unfitted_estimators():
    spec = baseline_specs()[REFERENCE_BASELINE]
    assert spec.build() is not spec.build()


def test_ladder_improves_step_by_step_on_held_out_data(split):
    raw_train, raw_test, y_train, y_test = split
    rmse = {}
    for name, spec in baseline_specs().items():
        pred = spec.build().fit(raw_train, y_train).predict(raw_test)
        rmse[name] = regression_metrics(y_test, pred)["rmse"]
    names = list(rmse)
    assert (
        rmse[names[0]] > rmse[names[3]] > rmse[names[4]] > rmse[names[5]]
    )  # mean > single feature > raw linear > reference
    mean_r2 = regression_metrics(y_test, baseline_specs()[names[0]].build().fit(raw_train, y_train).predict(raw_test))[
        "r2"
    ]
    assert abs(mean_r2) < 0.15


def test_baselines_survive_missing_values(split):
    raw_train, raw_test, y_train, _ = split
    rng = np.random.default_rng(0)
    holed_train, holed_test = (
        raw_train.mask(rng.random(raw_train.shape) < 0.05),
        raw_test.mask(rng.random(raw_test.shape) < 0.05),
    )
    for name, spec in baseline_specs().items():
        pred = np.asarray(spec.build().fit(holed_train, y_train).predict(holed_test))
        assert np.isfinite(pred).all(), name


def test_candidate_specs_are_challengers_and_run(split):
    raw_train, raw_test, y_train, _ = split
    specs = candidate_specs()
    assert {s.tier for s in specs.values()} == {"challenger"} and len(specs) == 6
    assert "linear regression + engineered features (VIF)" in specs
    for name in ("ridge (alpha=10) + preprocessing", "linear regression + engineered features (VIF)"):
        pred = specs[name].build().fit(raw_train, y_train).predict(raw_test)
        assert np.isfinite(pred).all()


# --------------------------------------------------------------------------- #
# BaselineModel
# --------------------------------------------------------------------------- #
def test_baseline_model_fit_predict_evaluate_save_load(split, tmp_path):
    raw_train, raw_test, y_train, y_test = split
    model = BaselineModel().fit(raw_train, y_train)
    pred = model.predict(raw_test)
    metrics = model.evaluate(raw_test, y_test)
    assert metrics["r2"] > 0.8 and metrics["rmse"] == pytest.approx(regression_metrics(y_test, pred)["rmse"])
    loaded = BaselineModel.load(model.save(tmp_path / "m" / "baseline.joblib"))
    assert np.allclose(loaded.predict(raw_test), pred)


def test_baseline_model_matches_the_reference_ladder_entry(split):
    raw_train, raw_test, y_train, _ = split
    ladder = baseline_specs()[REFERENCE_BASELINE].build().fit(raw_train, y_train).predict(raw_test)
    assert np.allclose(BaselineModel().fit(raw_train, y_train).predict(raw_test), ladder)


def test_baseline_model_custom_config_and_unfitted_errors(split):
    raw_train, raw_test, y_train, _ = split
    model = BaselineModel(PreprocessingConfig(target_transform="none")).fit(raw_train, y_train)
    assert np.isfinite(model.predict(raw_test)).all()
    with pytest.raises(NotFittedError):
        BaselineModel().predict(raw_test)
    with pytest.raises(NotFittedError):
        BaselineModel().save("x.joblib")


# --------------------------------------------------------------------------- #
# run_benchmark
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def benchmark(raw_df):
    from src.preprocessing import clean_data, split_data

    cfg = PreprocessingConfig()
    cleaned, _ = clean_data(raw_df, cfg)
    X_train, X_test, y_train, y_test = split_data(cleaned, cfg)
    return run_benchmark(X_train, y_train, X_test, y_test, n_splits=3, n_repeats=2, random_state=0), (
        X_train,
        y_train,
        X_test,
        y_test,
    )


def test_benchmark_tables_have_one_row_per_model(benchmark):
    result, _ = benchmark
    names = list(baseline_specs())
    for table in (result.cv_table, result.comparison, result.holdout):
        assert list(table.index) == names
    assert result.reference == REFERENCE_BASELINE
    assert all(len(f) == 6 for f in result.fold_metrics.values())  # 3 folds x 2 repeats


def test_benchmark_comparison_semantics(benchmark):
    result, _ = benchmark
    comp = result.comparison
    assert comp.loc[REFERENCE_BASELINE, "rmse_gain_vs_reference_%"] == pytest.approx(0)
    assert np.isnan(comp.loc[REFERENCE_BASELINE, "p_value_vs_reference"])
    assert comp.iloc[0]["skill_vs_mean_%"] == pytest.approx(0)
    assert comp.loc[REFERENCE_BASELINE, "skill_vs_mean_%"] > 60
    assert (comp.drop(index=REFERENCE_BASELINE)["rmse_gain_vs_reference_%"] < 0).all()  # everything else is worse
    assert comp.drop(index=REFERENCE_BASELINE)["p_value_vs_reference"].between(0, 1).all()
    assert {"rmse", "mae", "mape", "r2", "within_10pct", "bias", "fit_seconds"} <= set(comp.columns)


def test_benchmark_predictions_and_holdout_agree(benchmark):
    result, (_, _, X_test, y_test) = benchmark
    assert list(result.predictions.columns) == ["true", *baseline_specs()]
    assert np.allclose(result.predictions["true"], y_test)
    for name in result.holdout.index:
        assert result.holdout.loc[name, "rmse"] == pytest.approx(
            regression_metrics(y_test, result.predictions[name])["rmse"]
        )


def test_benchmark_uses_identical_folds_for_every_model(benchmark):
    result, _ = benchmark
    reference_folds = result.fold_metrics[REFERENCE_BASELINE].attrs["validation_indices"]
    assert len(reference_folds) == 6
    for name, folds in result.fold_metrics.items():
        for mine, theirs in zip(folds.attrs["validation_indices"], reference_folds):
            assert np.array_equal(mine, theirs), name  # the very same validation rows in every fold


def test_benchmark_residual_report_describes_the_reference(benchmark):
    result, (_, _, X_test, _) = benchmark
    assert {"overall", "by_price_band", "by_group", "error_trend"} <= set(result.residuals)
    assert result.residuals["by_group"]["n"].sum() == len(X_test)
    assert result.residuals["overall"].loc["all", "n"] == len(X_test)


def test_benchmark_is_deterministic(benchmark):
    result, (X_train, y_train, X_test, y_test) = benchmark
    again = run_benchmark(X_train, y_train, X_test, y_test, n_splits=3, n_repeats=2, random_state=0)
    pd.testing.assert_frame_equal(
        result.comparison.drop(columns="fit_seconds"), again.comparison.drop(columns="fit_seconds")
    )
    pd.testing.assert_frame_equal(result.holdout.drop(columns="fit_seconds"), again.holdout.drop(columns="fit_seconds"))


def test_benchmark_save_writes_tables_predictions_and_plot(benchmark, tmp_path):
    result, _ = benchmark
    written = result.save(tmp_path / "out")
    for key in ("comparison", "holdout", "predictions", "by_price_band", "by_group", "plot"):
        assert written[key].exists(), key
    assert pd.read_csv(written["comparison"], index_col=0).shape[0] == 6
    assert not (result.save(tmp_path / "noplot", diagnostics_plot=False).get("plot"))


def test_benchmark_to_dict_is_json_ready(benchmark):
    import json

    from src.experiment import to_jsonable

    result, _ = benchmark
    payload = to_jsonable(result.to_dict())
    json.dumps(payload)
    assert payload["reference"] == REFERENCE_BASELINE and payload["protocol"]["n_splits"] == 3
    assert len(payload["fold_metrics"][REFERENCE_BASELINE]["rmse"]) == 6


def test_benchmark_validates_arguments(benchmark):
    _, (X_train, y_train, X_test, y_test) = benchmark
    with pytest.raises(ValueError, match="Reference model"):
        run_benchmark(X_train, y_train, X_test, y_test, reference="nope", n_splits=3, n_repeats=1)
    with pytest.raises(ValueError, match="n_splits"):
        run_benchmark(X_train, y_train, X_test, y_test, n_splits=1)


def test_benchmark_can_include_challengers(benchmark):
    _, (X_train, y_train, X_test, y_test) = benchmark
    specs = {
        **baseline_specs(),
        "ridge (alpha=10) + preprocessing": candidate_specs()["ridge (alpha=10) + preprocessing"],
    }
    result = run_benchmark(X_train, y_train, X_test, y_test, specs, n_splits=3, n_repeats=1)
    assert result.comparison.shape[0] == 7
    assert (
        abs(result.comparison.loc["ridge (alpha=10) + preprocessing", "rmse_gain_vs_reference_%"]) < 10
    )  # close to the reference


# --------------------------------------------------------------------------- #
# Validation set in the benchmark
# --------------------------------------------------------------------------- #
def test_benchmark_scores_every_model_on_the_validation_set(raw_df):
    from src.preprocessing import run_preprocessing

    res = run_preprocessing(raw_df, PreprocessingConfig(validation_size=0.1))
    result = run_benchmark(
        res.raw_X_train,
        res.y_train,
        res.raw_X_test,
        res.y_test,
        n_splits=3,
        n_repeats=1,
        X_val=res.raw_X_val,
        y_val=res.y_val,
    )
    assert list(result.validation.index) == list(baseline_specs()) and len(res.raw_X_val) == 50
    ref = result.validation.loc[REFERENCE_BASELINE]
    expected = regression_metrics(
        res.y_val, result.fitted[REFERENCE_BASELINE].predict(res.raw_X_val), res.raw_X_train.shape[1]
    )
    assert ref["rmse"] == pytest.approx(expected["rmse"]) and ref["r2"] > 0.8
    assert result.protocol["n_validation"] == 50
    assert result.to_dict()["validation"] is not None


def test_benchmark_without_validation_has_no_validation_table(benchmark):
    result, _ = benchmark
    assert result.validation is None and result.protocol["n_validation"] == 0


def test_benchmark_saves_the_validation_table(raw_df, tmp_path):
    from src.preprocessing import run_preprocessing

    res = run_preprocessing(raw_df, PreprocessingConfig(validation_size=0.1))
    result = run_benchmark(
        res.raw_X_train,
        res.y_train,
        res.raw_X_test,
        res.y_test,
        n_splits=3,
        n_repeats=1,
        X_val=res.raw_X_val,
        y_val=res.y_val,
    )
    written = result.save(tmp_path, diagnostics_plot=False)
    assert pd.read_csv(written["validation"], index_col=0).shape[0] == 6


def test_validation_inputs_must_come_together(benchmark):
    _, (X_train, y_train, X_test, y_test) = benchmark
    with pytest.raises(ValueError, match="together"):
        run_benchmark(X_train, y_train, X_test, y_test, n_splits=3, n_repeats=1, X_val=X_test)
