"""Tests for src/advanced_models.py: the two advanced architectures, nested tuning and the comparison analysis."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.advanced_models import (
    ENGINEERED_LINEAR_NAME,
    GRADIENT_BOOSTING_DEFAULTS,
    GRADIENT_BOOSTING_NAME,
    GRADIENT_BOOSTING_SPACE,
    NEURAL_NETWORK_NAME,
    NEURAL_NETWORK_SPACE,
    SeedAveragedRegressor,
    TunedRegressor,
    _collect_tuning,
    advanced_specs,
    analyse_comparison,
    bootstrap_rmse_difference,
    comparison_specs,
    error_by_group,
    make_gradient_boosting,
    make_neural_network,
    model_complexity,
    permutation_importance_table,
    plot_comparison,
    screening_specs,
    tuning_summary,
)
from src.evaluation import cross_validate_metrics
from src.models import REFERENCE_BASELINE, baseline_specs, run_benchmark
from src.preprocessing import PreprocessingConfig, clean_data, make_model_pipeline, split_data


def xy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = np.sin(X["a"]) + 0.5 * X["b"] + rng.normal(0, 0.05, n)
    return X, y


# --------------------------------------------------------------------------- #
# SeedAveragedRegressor
# --------------------------------------------------------------------------- #
def test_seed_average_is_the_mean_of_its_members():
    X, y = xy()
    model = SeedAveragedRegressor(MLPRegressor(hidden_layer_sizes=(8,), max_iter=300), n_models=3, random_state=5)
    model.fit(X, y)
    assert len(model.models_) == 3
    assert np.allclose(model.predict(X), np.mean([m.predict(X) for m in model.models_], axis=0))


def test_members_differ_only_by_their_seed():
    X, y = xy()
    model = SeedAveragedRegressor(MLPRegressor(hidden_layer_sizes=(8,), max_iter=300), n_models=3, random_state=5).fit(
        X, y
    )
    assert [m.random_state for m in model.models_] == [5, 6, 7]
    first, second = model.models_[0].predict(X), model.models_[1].predict(X)
    assert not np.allclose(first, second)  # different initialisations
    assert {m.hidden_layer_sizes for m in model.models_} == {(8,)}


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_a_single_member_equals_the_plain_network():
    X, y = xy()
    base = MLPRegressor(hidden_layer_sizes=(8,), max_iter=300)
    averaged = SeedAveragedRegressor(base, n_models=1, random_state=3).fit(X, y)
    plain = clone(base).set_params(random_state=3).fit(X, y)
    assert np.allclose(averaged.predict(X), plain.predict(X))


def test_seed_average_is_reproducible_and_clonable():
    X, y = xy()
    make = lambda: SeedAveragedRegressor(MLPRegressor(hidden_layer_sizes=(8,), max_iter=200), 2, random_state=1)  # noqa: E731
    assert np.array_equal(make().fit(X, y).predict(X), clone(make()).fit(X, y).predict(X))


def test_seed_average_validates_its_arguments():
    X, y = xy()
    with pytest.raises(ValueError, match="base estimator"):
        SeedAveragedRegressor().fit(X, y)
    with pytest.raises(ValueError, match="at least 1"):
        SeedAveragedRegressor(MLPRegressor(), n_models=0).fit(X, y)


def test_convergence_is_reported_instead_of_warned():
    X, y = xy()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any ConvergenceWarning would fail the test
        stopped_early = SeedAveragedRegressor(MLPRegressor(hidden_layer_sizes=(8,), max_iter=1), 2).fit(X, y)
    assert stopped_early.converged_fraction_ == 0.0
    converged = SeedAveragedRegressor(MLPRegressor(hidden_layer_sizes=(8,), max_iter=3000), 2).fit(X, y)
    assert 0.0 <= converged.converged_fraction_ <= 1.0


# --------------------------------------------------------------------------- #
# TunedRegressor
# --------------------------------------------------------------------------- #
@pytest.fixture()
def ridge_pipeline():
    return make_model_pipeline(Ridge(), PreprocessingConfig())


def test_tuned_regressor_picks_the_better_setting_and_predicts_prices(split, ridge_pipeline):
    raw_train, raw_test, y_train, y_test = split
    space = {"regressor__model__alpha": [1e-6, 1e8]}  # a huge penalty flattens the model
    tuned = TunedRegressor(ridge_pipeline, space, n_iter=2, inner_folds=3, random_state=0).fit(raw_train, y_train)
    assert tuned.best_params_["regressor__model__alpha"] == 1e-6
    assert tuned.best_score_ > 0 and len(tuned.cv_results_) == 2
    pred = tuned.predict(raw_test)
    assert np.isfinite(pred).all() and 0.3 * y_test.median() < np.median(pred) < 3 * y_test.median()


def test_final_params_apply_only_to_the_final_refit(split, ridge_pipeline):
    raw_train, _, y_train, _ = split
    space = {"regressor__model__alpha": [1e-6, 1e-3]}
    tuned = TunedRegressor(
        ridge_pipeline, space, n_iter=2, inner_folds=2, final_params={"regressor__model__alpha": 123.0}
    )
    tuned.fit(raw_train, y_train)
    assert tuned.best_estimator_.regressor_.named_steps["model"].alpha == 123.0
    assert tuned.best_params_["regressor__model__alpha"] in (1e-6, 1e-3)  # the search itself used the sampled values


def test_tuning_is_deterministic_for_a_seed(split, ridge_pipeline):
    raw_train, _, y_train, _ = split
    space = {"regressor__model__alpha": [1e-4, 1e-2, 1.0, 100.0]}
    make = lambda: TunedRegressor(ridge_pipeline, space, n_iter=3, inner_folds=3, random_state=4)  # noqa: E731
    a, b = make().fit(raw_train, y_train), make().fit(raw_train, y_train)
    assert a.best_params_ == b.best_params_ and np.isclose(a.best_score_, b.best_score_)


def test_tuned_regressor_requires_an_estimator_and_a_space(split):
    raw_train, _, y_train, _ = split
    with pytest.raises(ValueError, match="needs"):
        TunedRegressor().fit(raw_train, y_train)


def test_nested_cross_validation_collects_the_choices_of_every_outer_fold(split, ridge_pipeline):
    raw_train, _, y_train, _ = split
    tuned = TunedRegressor(ridge_pipeline, {"regressor__model__alpha": [1e-6, 1e8]}, n_iter=2, inner_folds=2)
    folds = cross_validate_metrics(tuned, raw_train, y_train, 3, 1, 0, collect=_collect_tuning)
    collected = folds.attrs["collected"]
    assert len(collected) == 3 and all(set(c) == {"best_params", "inner_cv_rmse"} for c in collected)
    assert all(c["best_params"] == {"alpha": 1e-6} for c in collected)
    assert (folds["r2"] > 0.5).all()


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def test_untuned_gradient_boosting_uses_the_documented_settings():
    inner = make_gradient_boosting(tune=False, random_state=7).regressor.named_steps["model"]
    assert isinstance(inner, GradientBoostingRegressor) and inner.random_state == 7
    for name, value in GRADIENT_BOOSTING_DEFAULTS.items():
        assert getattr(inner, name) == value


def test_tuned_builders_return_a_tuned_regressor_with_the_right_budget():
    gb = make_gradient_boosting(tune=True, n_iter=5, inner_folds=4, random_state=3)
    nn = make_neural_network(tune=True, n_iter=6, inner_folds=2, n_seeds=3, random_state=3)
    assert isinstance(gb, TunedRegressor) and (gb.n_iter, gb.inner_folds, gb.random_state) == (5, 4, 3)
    assert isinstance(nn, TunedRegressor) and (nn.n_iter, nn.inner_folds) == (6, 2)


@pytest.mark.parametrize(
    "builder,space", [(make_gradient_boosting, GRADIENT_BOOSTING_SPACE), (make_neural_network, NEURAL_NETWORK_SPACE)]
)
def test_every_search_parameter_exists_on_the_estimator(builder, space):
    tuned = builder(tune=True)
    assert len(tuned.param_distributions) == len(space)
    for name, values in tuned.param_distributions.items():
        sample = values[0] if isinstance(values, list) else values.rvs(random_state=0)
        clone(tuned.estimator).set_params(**{name: sample})  # raises ValueError for an unknown parameter


def test_the_network_trains_on_a_standardised_target_and_the_trees_do_not():
    nn = make_neural_network(tune=False)
    assert isinstance(nn.transformer, Pipeline) and isinstance(nn.transformer.steps[-1][1], StandardScaler)
    assert isinstance(make_gradient_boosting(tune=False).transformer, FunctionTransformer)


def test_the_search_uses_one_network_and_the_final_model_a_seed_average():
    tuned = make_neural_network(tune=True, n_seeds=4)
    assert tuned.estimator.regressor.named_steps["model"].n_models == 1
    assert tuned.final_params == {"regressor__model__n_models": 4}
    assert make_neural_network(tune=False, n_seeds=4).regressor.named_steps["model"].n_models == 4


@pytest.mark.parametrize("builder", [make_gradient_boosting, make_neural_network])
def test_both_models_fit_and_predict_prices_on_the_raw_data(split, builder):
    raw_train, raw_test, y_train, y_test = split
    kwargs = {"n_seeds": 2} if builder is make_neural_network else {}
    pred = builder(tune=False, **kwargs).fit(raw_train, y_train).predict(raw_test)
    assert pred.shape == (len(raw_test),) and np.isfinite(pred).all()
    assert 0.3 * y_test.median() < np.median(pred) < 3 * y_test.median()  # price units, not log units
    from sklearn.metrics import r2_score

    assert r2_score(y_test, pred) > 0.5


@pytest.mark.parametrize("builder", [make_gradient_boosting, make_neural_network])
def test_both_models_cope_with_missing_values(split, builder):
    raw_train, raw_test, y_train, _ = split
    rng = np.random.default_rng(0)
    kwargs = {"n_seeds": 2} if builder is make_neural_network else {}
    model = builder(tune=False, **kwargs).fit(raw_train.mask(rng.random(raw_train.shape) < 0.05), y_train)
    assert np.isfinite(model.predict(raw_test.mask(rng.random(raw_test.shape) < 0.05))).all()


def test_the_seed_controls_every_random_choice(split):
    raw_train, raw_test, y_train, _ = split
    same = [
        make_neural_network(tune=False, n_seeds=2, random_state=1).fit(raw_train, y_train).predict(raw_test)
        for _ in range(2)
    ]
    other = make_neural_network(tune=False, n_seeds=2, random_state=2).fit(raw_train, y_train).predict(raw_test)
    assert np.array_equal(same[0], same[1]) and not np.allclose(same[0], other)
    gb = [
        make_gradient_boosting(tune=False, random_state=s).fit(raw_train, y_train).predict(raw_test) for s in (1, 1, 2)
    ]
    assert np.array_equal(gb[0], gb[1]) and not np.allclose(gb[0], gb[2])  # subsample < 1 makes the seed matter


# --------------------------------------------------------------------------- #
# Catalogues
# --------------------------------------------------------------------------- #
def test_advanced_specs_describe_two_distinct_architectures():
    specs = advanced_specs()
    assert list(specs) == [GRADIENT_BOOSTING_NAME, NEURAL_NETWORK_NAME]
    assert {s.tier for s in specs.values()} == {"advanced"} and all(callable(s.collect) for s in specs.values())
    assert "randomized search" in specs[GRADIENT_BOOSTING_NAME].description
    assert "fixed settings" in advanced_specs(tune=False)[NEURAL_NETWORK_NAME].description
    assert specs[GRADIENT_BOOSTING_NAME].build() is not specs[GRADIENT_BOOSTING_NAME].build()


def test_comparison_specs_order_and_options():
    names = list(comparison_specs())
    assert names == [
        "mean price (ignores all features)",
        REFERENCE_BASELINE,
        ENGINEERED_LINEAR_NAME,
        GRADIENT_BOOSTING_NAME,
        NEURAL_NETWORK_NAME,
    ]
    assert ENGINEERED_LINEAR_NAME not in comparison_specs(include_engineered_linear=False)
    assert (
        list(comparison_specs())[1] == list(baseline_specs())[-1]
    )  # the same reference baseline as the baseline sprint


def test_screening_covers_distinct_architecture_families():
    specs = screening_specs()
    assert len(specs) == len(set(specs)) == 13
    families = {s.description.split(":")[0] for s in specs.values() if s.tier == "screen"}
    assert {
        "linear",
        "bagged trees",
        "randomised trees",
        "boosted trees",
        "neural network",
        "kernel method",
        "instance-based",
    } <= families
    assert sum(s.tier == "floor" for s in specs.values()) == 1


@pytest.mark.parametrize(
    "name",
    [
        "linear regression",
        "ridge (alpha=10)",
        "kernel ridge (RBF)",
        "k-nearest neighbours (k=10)",
        "support vector regression (RBF)",
        "neural network (32,)",
    ],
)
def test_screening_models_run(split, name):
    raw_train, raw_test, y_train, _ = split
    pred = screening_specs()[name].build().fit(raw_train, y_train).predict(raw_test)
    assert np.isfinite(pred).all() and pred.shape == (len(raw_test),)


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #
def test_error_by_price_band_table():
    y = np.linspace(100, 500, 100)
    preds = pd.DataFrame({"over": y * 1.1, "exact": y})
    table = error_by_group(y, preds, None, n_bands=5)
    assert len(table) == 10 and list(table["group"].unique()) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    over, exact = table[table["model"] == "over"], table[table["model"] == "exact"]
    assert np.allclose(over["bias_pct"], 10.0) and np.allclose(exact["mae"], 0.0) and (over["n"] == 20).all()
    assert over["mae"].is_monotonic_increasing  # the same 10% matters more in money for dear homes


def test_error_by_group_labels_and_validation():
    y = np.array([100.0, 200.0, 300.0, 400.0])
    preds = pd.DataFrame({"m": y + 10})
    table = error_by_group(y, preds, ["a", "a", "b", None])
    assert set(table["group"]) == {"a", "b", "Missing"} and (table["mae"] == 10).all()
    with pytest.raises(ValueError, match="same length"):
        error_by_group(y, preds, ["a", "b"])


def test_bootstrap_difference_cases():
    rng = np.random.default_rng(0)
    y = rng.uniform(100, 500, 200)
    noisy, exact = y + rng.normal(0, 30, 200), y.copy()
    same = bootstrap_rmse_difference(y, noisy, noisy, n_boot=200)
    assert same["rmse_difference"] == 0 and same["ci_low"] == 0 == same["ci_high"] and same["share_b_better"] == 0
    better = bootstrap_rmse_difference(y, noisy, exact, n_boot=500)
    assert better["rmse_difference"] > 0 and better["ci_low"] > 0 and better["share_b_better"] == 1.0
    assert better["ci_low"] <= better["rmse_difference"] <= better["ci_high"]
    worse = bootstrap_rmse_difference(y, exact, noisy, n_boot=200)
    assert worse["rmse_difference"] < 0 and worse["share_b_better"] == 0.0
    assert bootstrap_rmse_difference(y, noisy, exact, n_boot=100, random_state=1) == bootstrap_rmse_difference(
        y, noisy, exact, n_boot=100, random_state=1
    )
    with pytest.raises(ValueError, match="same length"):
        bootstrap_rmse_difference(y, noisy[:10], exact)


def test_model_complexity_counts_parameters(split):
    raw_train, _, y_train, _ = split
    linear = make_model_pipeline(LinearRegression(), PreprocessingConfig()).fit(raw_train, y_train)
    info = model_complexity(linear)
    assert (
        info["kind"] == "linear model" and info["parameters"] == 10 and info["size_kb"] > 0
    )  # 9 coefficients + intercept
    nn = make_neural_network(tune=False, n_seeds=2).fit(raw_train, y_train)
    assert (
        model_complexity(nn)["parameters"] == 2 * (9 * 32 + 32 + 32 * 1 + 1)
        and "2 seeds" in model_complexity(nn)["kind"]
    )
    gb = make_gradient_boosting(tune=False).fit(raw_train, y_train)
    gb_info = model_complexity(gb)
    assert "800 trees" in gb_info["kind"] and gb_info["parameters"] > 800 * 3


def test_permutation_importance_finds_the_informative_feature():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"signal": rng.normal(size=300), "noise": rng.normal(size=300)})
    y = 10 * X["signal"] + 100
    model = LinearRegression().fit(X, y)
    table = permutation_importance_table(model, X, y, n_repeats=5)
    assert table.index[0] == "signal" and table.loc["signal", "share_%"] > 95
    assert table["share_%"].sum() == pytest.approx(100) and table["importance"].is_monotonic_decreasing


# --------------------------------------------------------------------------- #
# The full comparison on a small problem
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def small_comparison(raw_df):
    cfg = PreprocessingConfig()
    cleaned, _ = clean_data(raw_df, cfg)
    X_train, X_test, y_train, y_test = split_data(cleaned, cfg)
    specs = comparison_specs(cfg, tune=False, n_seeds=2)
    result = run_benchmark(X_train, y_train, X_test, y_test, specs, n_splits=3, n_repeats=1, random_state=0)
    analysis = analyse_comparison(result, X_test, y_test, n_boot=100, importance_repeats=2)
    return result, analysis, (X_train, y_train, X_test, y_test)


def test_comparison_benchmark_contains_every_model(small_comparison):
    result, _, _ = small_comparison
    assert list(result.comparison.index) == list(comparison_specs())
    assert "predict_seconds" in result.holdout.columns and (result.holdout["predict_seconds"] >= 0).all()
    assert result.comparison.loc[REFERENCE_BASELINE, "rmse_gain_vs_reference_%"] == pytest.approx(0)


def test_analysis_excludes_the_mean_predictor_and_has_the_right_shapes(small_comparison):
    result, analysis, (_, _, X_test, _) = small_comparison
    models = [n for n in result.comparison.index if result.specs[n].tier != "floor"]
    assert len(models) == 4
    assert set(analysis.versus_reference.index) == set(models) - {REFERENCE_BASELINE}
    assert len(analysis.blends) == 6 and not analysis.blends["model_a"].str.contains("mean price").any()
    assert analysis.residual_correlation.shape == (4, 4) and np.allclose(np.diag(analysis.residual_correlation), 1)
    assert len(analysis.by_price_band) == 5 * 4 and len(analysis.by_group) == 3 * 4
    assert set(analysis.importance) == set(models) and set(analysis.complexity.index) == set(models)
    assert {"wins_vs_reference_%", "rmse_difference", "ci_low", "ci_high", "share_b_better"} <= set(
        analysis.versus_reference.columns
    )
    assert analysis.tuning.empty


def test_analysis_numbers_agree_with_the_predictions(small_comparison):
    result, analysis, (_, _, _, y_test) = small_comparison
    name = NEURAL_NETWORK_NAME
    band_mae = analysis.by_price_band.query("model == @name").set_index("group")["mae"]
    truth = result.predictions["true"].to_numpy()
    q1 = truth <= np.quantile(truth, 0.2)
    assert band_mae["Q1"] == pytest.approx(np.abs(result.predictions[name].to_numpy()[q1] - truth[q1]).mean(), rel=0.2)
    wins = analysis.versus_reference.loc[name, "wins_vs_reference_%"]
    expected = 100 * np.mean(
        np.abs(result.predictions[name] - truth) < np.abs(result.predictions[REFERENCE_BASELINE] - truth)
    )
    assert wins == pytest.approx(expected)


def test_analysis_to_dict_and_figure(small_comparison, tmp_path):
    result, analysis, _ = small_comparison
    assert set(analysis.to_dict()) == {
        "by_price_band",
        "by_group",
        "versus_reference",
        "residual_correlation",
        "blends",
        "complexity",
        "importance",
        "tuning",
    }
    figure = plot_comparison(result, analysis, tmp_path / "plots" / "cmp.png", {NEURAL_NETWORK_NAME: "neural network"})
    assert figure.exists() and figure.stat().st_size > 20_000


def test_tuning_summary_lists_every_outer_fold_and_the_final_model(raw_df):
    cfg = PreprocessingConfig()
    cleaned, _ = clean_data(raw_df, cfg)
    X_train, X_test, y_train, y_test = split_data(cleaned, cfg)
    specs = {
        REFERENCE_BASELINE: baseline_specs(cfg)[REFERENCE_BASELINE],
        **advanced_specs(cfg, tune=True, n_iter=2, inner_folds=2, n_seeds=1),
    }
    result = run_benchmark(X_train, y_train, X_test, y_test, specs, n_splits=3, n_repeats=1, random_state=0)
    summary = tuning_summary(result)
    assert set(summary["model"]) == {GRADIENT_BOOSTING_NAME, NEURAL_NETWORK_NAME} and len(summary) == 8
    for name in (GRADIENT_BOOSTING_NAME, NEURAL_NETWORK_NAME):
        part = summary[summary["model"] == name]
        assert list(part["fold"].astype(str)) == ["0", "1", "2", "final"] and (part["inner_cv_rmse"] > 0).all()
    assert {"n_estimators", "learning_rate", "max_depth"} <= set(summary.columns)
    assert {"hidden_layer_sizes", "alpha", "activation"} <= set(summary.columns)
    assert len(result.collected[GRADIENT_BOOSTING_NAME]) == 3
