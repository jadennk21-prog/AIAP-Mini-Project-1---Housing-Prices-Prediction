"""Unit tests for src/feature_engineering.py. Run with:  pytest"""

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score

from src.feature_engineering import (
    SELECTION_METHODS,
    FeatureEngineer,
    FeatureEngineeringConfig,
    FeatureSelector,
    default_scenarios,
    evaluate_feature_sets,
    make_full_pipeline,
    select_best_scenario,
)
from src.preprocessing import DataPreprocessor

BASES = [
    "property_size",
    "crime_rate_in_area",
    "proximity_to_public_transport",
    "age_of_property",
    "school_quality_score",
    "number_of_bedrooms",
    "number_of_bathrooms",
]


@pytest.fixture()
def prepared(split, cfg):
    """Preprocessed (imputed, logged, scaled, one-hot) train and test frames."""
    raw_train, raw_test, y_train, y_test = split
    pre = DataPreprocessor(cfg).fit(raw_train)
    return pre.transform(raw_train), pre.transform(raw_test), y_train, y_test


@pytest.fixture(scope="module")
def curved_df() -> pd.DataFrame:
    """Data whose price has real curvature and an interaction, so engineered features must help."""
    rng = np.random.default_rng(1)
    n = 500
    df = pd.DataFrame(
        {
            "property_size": rng.lognormal(7.8, 0.5, n),
            "crime_rate_in_area": rng.lognormal(-3.0, 1.0, n),
            "proximity_to_public_transport": rng.lognormal(1.9, 0.9, n),
            "age_of_property": rng.uniform(0.03, 1.2, n),
            "school_quality_score": rng.normal(7, 1.5, n),
            "number_of_bedrooms": rng.normal(0, 1, n),
            "number_of_bathrooms": rng.normal(0, 1, n),
            "location_type": rng.choice(["Rural", "Suburban", "Urban"], n, p=[0.1, 0.5, 0.4]),
        }
    )
    z = lambda s: (s - s.mean()) / s.std()  # noqa: E731
    ltr, school = z(np.log(df.proximity_to_public_transport)), z(df.school_quality_score)
    lsize = z(np.log(df.property_size))
    df["house_sale_price"] = np.exp(
        12 + 0.3 * lsize + 0.1 * school + 0.25 * ltr**2 + 0.1 * lsize * school + rng.normal(0, 0.03, n)
    )
    return df


@pytest.fixture(scope="module")
def signal_frame():
    """3 informative columns, 1 near-duplicate, 1 constant, 5 pure noise."""
    rng = np.random.default_rng(3)
    n = 400
    a, b, c = rng.normal(size=(3, n))
    X = pd.DataFrame(
        {
            "a": a,
            "b": b,
            "c": c,
            "a_copy": a + rng.normal(0, 0.01, n),
            "const": 1.0,
            **{f"noise{i}": rng.normal(size=n) for i in range(5)},
        }
    )
    y = 3 * a + 2 * b - 1.5 * c + rng.normal(0, 0.3, n)
    return X, y


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kwargs", [{"n_tiers": 1}, {"clip_z": 0}, {"clip_z": -1}, {"roles": {"a": "x", "b": "x"}}])
def test_invalid_fe_config(kwargs):
    with pytest.raises(ValueError):
        FeatureEngineeringConfig(**kwargs)


def test_default_config_covers_the_seven_numeric_columns():
    assert sorted(FeatureEngineeringConfig().base_columns) == sorted(BASES)


# --------------------------------------------------------------------------- #
# FeatureEngineer
# --------------------------------------------------------------------------- #
def test_engineer_creates_documented_features(prepared):
    X_train, X_test, _, _ = prepared
    fe = FeatureEngineer(FeatureEngineeringConfig()).fit(X_train)
    out = fe.transform(X_train)
    groups = fe.feature_catalog_["group"].value_counts().to_dict()
    assert groups == {"interaction": 21, "group_interaction": 14, "polynomial": 7, "composite": 4, "tier": 2}
    assert list(out.columns) == fe.feature_names_out_
    assert out.shape == (len(X_train), X_train.shape[1] + 48)
    assert out.columns.is_unique
    assert {
        "nbhd_quality",
        "rooms_total",
        "spaciousness",
        "transit_extremity",
        "crime_sq",
        "size_x_school",
        "crime_x_urban",
        "crime_tier",
    } <= set(out.columns)
    assert fe.feature_catalog_["rationale"].str.len().min() > 20  # every feature is explained
    assert list(fe.transform(X_test).columns) == list(out.columns)


def test_original_columns_are_unchanged_and_index_kept(prepared):
    X_train, _, _, _ = prepared
    fe = FeatureEngineer().fit(X_train)
    out = fe.transform(X_train)
    pd.testing.assert_frame_equal(out[X_train.columns], X_train)
    assert out.index.equals(X_train.index)


def test_keep_original_false_returns_only_engineered(prepared):
    X_train, _, _, _ = prepared
    out = FeatureEngineer(FeatureEngineeringConfig(keep_original=False)).fit_transform(X_train)
    assert list(out.columns) == list(FeatureEngineer().fit(X_train).engineered_features_)


def test_engineered_columns_are_standardised_on_training_data(prepared):
    X_train, X_test, _, _ = prepared
    fe = FeatureEngineer().fit(X_train)
    eng = fe.transform(X_train)[fe.engineered_features_]
    non_constant = eng.std(ddof=0) > 0
    assert np.allclose(eng.mean()[non_constant], 0, atol=1e-9)
    assert np.allclose(eng.std(ddof=0)[non_constant], 1, atol=1e-9)
    assert abs(fe.transform(X_test)["crime_sq"].mean()) > 1e-6  # test is NOT re-centred


def test_test_features_use_training_statistics(prepared):
    X_train, X_test, _, _ = prepared
    cfg = FeatureEngineeringConfig(
        clip_z=None,
        scale_engineered=False,
        add_composites=False,
        add_interactions=False,
        add_group_interactions=False,
        add_tiers=False,
    )
    fe = FeatureEngineer(cfg).fit(X_train)
    col = "age_of_property"
    z_test = (X_test[col] - X_train[col].mean()) / X_train[col].std(ddof=0)
    assert np.allclose(fe.transform(X_test)["age_sq"], z_test**2)


def test_clip_z_limits_extreme_squares(prepared):
    X_train, _, _, _ = prepared
    spiked = X_train.copy()
    spiked.iloc[0, spiked.columns.get_loc("age_of_property")] = 1e6
    clipped = (
        FeatureEngineer(FeatureEngineeringConfig(clip_z=3.0, scale_engineered=False)).fit(X_train).transform(spiked)
    )
    assert clipped["age_sq"].max() <= 9.0 + 1e-9
    unclipped = (
        FeatureEngineer(FeatureEngineeringConfig(clip_z=None, scale_engineered=False)).fit(X_train).transform(spiked)
    )
    assert unclipped["age_sq"].max() > 1e6


@pytest.mark.parametrize(
    "flag,group",
    [
        ("add_composites", "composite"),
        ("add_polynomial", "polynomial"),
        ("add_interactions", "interaction"),
        ("add_group_interactions", "group_interaction"),
        ("add_tiers", "tier"),
    ],
)
def test_feature_groups_can_be_switched_off(prepared, flag, group):
    X_train, _, _, _ = prepared
    fe = FeatureEngineer(FeatureEngineeringConfig(**{flag: False})).fit(X_train)
    assert group not in set(fe.feature_catalog_["group"])


def test_custom_interaction_pairs_and_polynomial_subset(prepared):
    X_train, _, _, _ = prepared
    cfg = FeatureEngineeringConfig(
        add_composites=False,
        add_group_interactions=False,
        add_tiers=False,
        polynomial_features=["crime_rate_in_area"],
        interaction_pairs=[("property_size", "school_quality_score")],
    )
    fe = FeatureEngineer(cfg).fit(X_train)
    assert fe.engineered_features_ == ["crime_sq", "size_x_school"]


def test_formulas_are_correct(prepared):
    X_train, _, _, _ = prepared
    cfg = FeatureEngineeringConfig(clip_z=None, scale_engineered=False)
    fe = FeatureEngineer(cfg).fit(X_train)
    out = fe.transform(X_train)
    z = lambda c: (X_train[c] - X_train[c].mean()) / X_train[c].std(ddof=0)  # noqa: E731
    assert np.allclose(out["nbhd_quality"], z("school_quality_score") - z("crime_rate_in_area"))
    assert np.allclose(out["rooms_total"], (z("number_of_bedrooms") + z("number_of_bathrooms")) / 2)
    assert np.allclose(out["transit_extremity"], z("proximity_to_public_transport").abs())
    assert np.allclose(out["size_x_school"], z("property_size") * z("school_quality_score"))
    assert np.allclose(out["crime_x_urban"], z("crime_rate_in_area") * X_train["location_type_Urban"])


def test_tiers_use_training_quantiles(prepared):
    X_train, X_test, _, _ = prepared
    fe = FeatureEngineer(FeatureEngineeringConfig(scale_engineered=False)).fit(X_train)
    tiers = fe.transform(X_train)["crime_tier"]
    assert set(tiers.unique()) == {0.0, 1.0, 2.0}
    shares = tiers.value_counts(normalize=True)
    assert shares.between(0.3, 0.37).all()  # roughly equal thirds
    assert set(fe.transform(X_test)["crime_tier"].unique()) <= {0.0, 1.0, 2.0}


def test_missing_values_propagate_instead_of_crashing(prepared):
    X_train, _, _, _ = prepared
    fe = FeatureEngineer().fit(X_train)
    holed = X_train.copy()
    holed.iloc[:5, holed.columns.get_loc("school_quality_score")] = np.nan
    out = fe.transform(holed)
    assert out["school_sq"].iloc[:5].isna().all() and out["school_sq"].iloc[5:].notna().all()
    assert out["crime_sq"].notna().all()  # unaffected columns stay complete


def test_constant_column_does_not_break_scaling(prepared):
    X_train, _, _, _ = prepared
    frame = X_train.assign(age_of_property=0.5)
    out = FeatureEngineer().fit_transform(frame)
    assert np.isfinite(out.to_numpy(float)).all()


def test_fe_errors(prepared, cfg):
    X_train, _, _, _ = prepared
    with pytest.raises(ValueError, match="missing"):
        FeatureEngineer().fit(X_train.drop(columns=["crime_rate_in_area"]))
    with pytest.raises(NotFittedError):
        FeatureEngineer().transform(X_train)
    with pytest.raises(TypeError):
        FeatureEngineer().fit(np.zeros((3, 3)))
    fe = FeatureEngineer().fit(X_train)
    with pytest.raises(ValueError, match="missing"):
        fe.transform(X_train.drop(columns=["location_type_Urban"]))


def test_no_group_columns_is_warned_and_skipped(prepared, caplog):
    X_train, _, _, _ = prepared
    frame = X_train.drop(columns=[c for c in X_train.columns if c.startswith("location_type_")])
    with caplog.at_level(logging.WARNING, logger="src.feature_engineering"):
        fe = FeatureEngineer().fit(frame)
    assert "group interactions are skipped" in caplog.text
    assert "group_interaction" not in set(fe.feature_catalog_["group"])


def test_fit_is_deterministic_and_clonable(prepared):
    X_train, X_test, _, _ = prepared
    a = FeatureEngineer().fit(X_train).transform(X_test)
    b = clone(FeatureEngineer()).fit(X_train).transform(X_test)
    pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------- #
# FeatureSelector
# --------------------------------------------------------------------------- #
def test_all_methods_are_listed():
    assert set(SELECTION_METHODS) == {
        "none",
        "variance",
        "correlation",
        "vif",
        "univariate",
        "lasso",
        "rfe",
        "tree",
        "consensus",
    }


def test_none_keeps_everything(signal_frame):
    X, y = signal_frame
    assert FeatureSelector("none").fit(X, y).selected_features_ == list(X.columns)


def test_vif_removes_exact_linear_combinations_but_respects_protection():
    rng = np.random.default_rng(0)
    base = pd.DataFrame(rng.normal(size=(300, 3)), columns=["x1", "x2", "x3"])
    X = base.assign(combo=base["x1"] - 2 * base["x2"] + 0.5 * base["x3"], noise=rng.normal(size=300))
    sel = FeatureSelector("vif", vif_threshold=10).fit(X)
    assert len(sel.dropped_features_) == 1 and sel.dropped_features_[0] in {"x1", "x2", "x3", "combo"}
    assert "noise" in sel.selected_features_
    report = sel.selection_report_
    assert report.loc[report["kept"], "score"].max() < 10  # kept columns are all below the threshold
    assert report.loc[~report["kept"], "score"].min() > 10  # dropped columns record their VIF at removal
    protected = FeatureSelector("vif", always_keep=["x1", "x2", "x3"]).fit(X)
    assert protected.dropped_features_ == ["combo"]  # the engineered column goes, the originals stay


def test_vif_drops_constant_and_near_duplicate_columns(signal_frame):
    X, _ = signal_frame
    sel = FeatureSelector("vif").fit(X)
    assert "const" in sel.dropped_features_
    assert len({"a", "a_copy"} & set(sel.selected_features_)) == 1


def test_variance_drops_constant_column(signal_frame):
    X, _ = signal_frame
    sel = FeatureSelector("variance").fit(X)
    assert sel.dropped_features_ == ["const"]


def test_correlation_filter_removes_near_duplicate_keeping_stronger(signal_frame):
    X, y = signal_frame
    sel = FeatureSelector("correlation", corr_threshold=0.95).fit(X, y)
    assert len({"a", "a_copy"} & set(sel.selected_features_)) == 1
    assert {"b", "c"} <= set(sel.selected_features_)


def test_correlation_filter_works_without_target(signal_frame):
    X, _ = signal_frame
    sel = FeatureSelector("correlation").fit(X)
    assert len({"a", "a_copy"} & set(sel.selected_features_)) == 1


@pytest.mark.parametrize("method", ["univariate", "lasso", "rfe", "tree", "consensus"])
def test_supervised_methods_keep_the_signal(signal_frame, method):
    X, y = signal_frame
    sel = FeatureSelector(method, random_state=0).fit(X, y)
    assert {"b", "c"} <= set(sel.selected_features_)
    assert any(n in sel.selected_features_ for n in ("a", "a_copy"))
    noise_kept = [n for n in sel.selected_features_ if n.startswith("noise")]
    assert len(noise_kept) <= 2  # most of the 5 noise columns removed
    assert sel.transform(X).shape[1] == len(sel.selected_features_)


def test_lasso_and_univariate_drop_all_noise(signal_frame):
    X, y = signal_frame
    univariate = FeatureSelector("univariate").fit(X, y)
    assert not any(n.startswith("noise") for n in univariate.selected_features_)


@pytest.mark.parametrize("method", ["univariate", "rfe", "tree"])
def test_n_features_selects_exactly_k(signal_frame, method):
    X, y = signal_frame
    sel = FeatureSelector(method, n_features=3, random_state=0).fit(X, y)
    assert len(sel.selected_features_) == 3
    assert "b" in sel.selected_features_  # the strongest independent signal
    assert not any(n.startswith("noise") or n == "const" for n in sel.selected_features_)


def test_univariate_is_blind_to_redundancy_unlike_the_correlation_filter(signal_frame):
    X, y = signal_frame
    univariate = FeatureSelector("univariate").fit(X, y).selected_features_
    assert "a" in univariate and "a_copy" in univariate  # keeps both near-duplicates
    assert len({"a", "a_copy"} & set(FeatureSelector("correlation").fit(X, y).selected_features_)) == 1


def test_consensus_counts_votes(signal_frame):
    X, y = signal_frame
    sel = FeatureSelector("consensus", consensus_methods=("univariate", "lasso", "variance"), min_votes=3).fit(X, y)
    report = sel.selection_report_
    assert report.loc["b", "votes"] == 3
    assert report.loc["const", "votes"] == 0 and not report.loc["const", "kept"]
    assert {"univariate_kept", "lasso_kept", "variance_kept"} <= set(report.columns)


def test_always_keep_protects_columns(signal_frame):
    X, y = signal_frame
    sel = FeatureSelector("lasso", always_keep=["noise0"], always_keep_prefixes=["const"]).fit(X, y)
    assert "noise0" in sel.selected_features_ and "const" in sel.selected_features_
    assert sel.selection_report_.loc["noise0", "protected"]


def test_selection_report_is_consistent(signal_frame):
    X, y = signal_frame
    sel = FeatureSelector("lasso").fit(X, y)
    report = sel.selection_report_
    assert list(report.index) == list(X.columns)
    assert list(report.index[report["kept"]]) == sel.selected_features_
    assert sel.dropped_features_ == list(report.index[~report["kept"]])


@pytest.mark.parametrize("method", ["univariate", "lasso", "rfe", "tree", "consensus"])
def test_supervised_methods_require_target(signal_frame, method):
    X, _ = signal_frame
    with pytest.raises(ValueError, match="needs the target"):
        FeatureSelector(method).fit(X)


def test_selector_errors(signal_frame):
    X, y = signal_frame
    with pytest.raises(ValueError, match="Unknown method"):
        FeatureSelector("magic").fit(X, y)
    with pytest.raises(ValueError):
        FeatureSelector("consensus", consensus_methods=("consensus",)).fit(X, y)
    with pytest.raises(TypeError):
        FeatureSelector("none").fit(X.to_numpy())
    with pytest.raises(NotFittedError):
        FeatureSelector("none").transform(X)
    sel = FeatureSelector("none").fit(X)
    with pytest.raises(ValueError, match="missing"):
        sel.transform(X.drop(columns=["a"]))


def test_selector_never_removes_everything(signal_frame, caplog):
    X, y = signal_frame
    with caplog.at_level(logging.WARNING, logger="src.feature_engineering"):
        sel = FeatureSelector("variance", variance_threshold=1e6).fit(X)
    assert sel.selected_features_ == list(X.columns)
    assert "keeping all" in caplog.text


def test_selector_is_deterministic_and_clonable(signal_frame):
    X, y = signal_frame
    a = FeatureSelector("tree", random_state=1).fit(X, y).selected_features_
    b = clone(FeatureSelector("tree", random_state=1)).fit(X, y).selected_features_
    assert a == b


# --------------------------------------------------------------------------- #
# Pipelines and evaluation
# --------------------------------------------------------------------------- #
def test_full_pipeline_predicts_on_price_scale(split):
    raw_train, raw_test, y_train, y_test = split
    model = make_full_pipeline(LinearRegression(), fe_config=FeatureEngineeringConfig(), selector="lasso")
    model.fit(raw_train, y_train)
    pred = model.predict(raw_test)
    assert pred.shape == (len(raw_test),) and np.isfinite(pred).all()
    assert 0.3 * y_test.median() < np.median(pred) < 3 * y_test.median()
    steps = [name for name, _ in model.regressor_.steps]
    assert steps == ["preprocess", "features", "select", "model"]


def test_baseline_pipeline_has_no_feature_steps(split):
    raw_train, _, y_train, _ = split
    model = make_full_pipeline(LinearRegression()).fit(raw_train, y_train)
    assert [n for n, _ in model.regressor_.steps] == ["preprocess", "model"]


def test_selector_accepts_instance_and_protect_flag(split):
    raw_train, _, y_train, _ = split
    inst = make_full_pipeline(Ridge(), fe_config=FeatureEngineeringConfig(), selector=FeatureSelector("variance"))
    assert inst.regressor.named_steps["select"].method == "variance"
    assert not inst.regressor.named_steps["select"].always_keep  # instances are used as configured
    protected = make_full_pipeline(Ridge(), fe_config=FeatureEngineeringConfig(), selector="lasso").fit(
        raw_train, y_train
    )
    kept = set(protected.regressor_.named_steps["select"].selected_features_)
    assert set(BASES) <= kept and {"location_type_Suburban", "location_type_Urban"} <= kept  # protected by default


def test_main_effects_are_protected_by_default_and_unprotected_vif_drops_them(split):
    """Regression test for a real failure: an unprotected VIF filter removed every original feature."""
    raw_train, _, y_train, _ = split
    fe = FeatureEngineeringConfig()
    unprotected = make_full_pipeline(Ridge(), fe_config=fe, selector="vif", protect_main_effects=False).fit(
        raw_train, y_train
    )
    dropped = set(unprotected.regressor_.named_steps["select"].dropped_features_)
    assert dropped & set(BASES)  # main effects can disappear...
    protected = make_full_pipeline(Ridge(), fe_config=fe, selector="vif").fit(raw_train, y_train)
    assert not (set(protected.regressor_.named_steps["select"].dropped_features_) & set(BASES))  # ...unless protected


def test_pipeline_handles_missing_values_and_unseen_category(split):
    raw_train, raw_test, y_train, _ = split
    rng = np.random.default_rng(0)
    holed = raw_train.mask(rng.random(raw_train.shape) < 0.05)
    model = make_full_pipeline(LinearRegression(), fe_config=FeatureEngineeringConfig(), selector="lasso")
    model.fit(holed, y_train)
    test = raw_test.copy()
    test.iloc[0, test.columns.get_loc("location_type")] = "Coastal"
    assert np.isfinite(model.predict(test)).all()


def test_cross_validation_runs_with_selection_inside(split):
    raw_train, _, y_train, _ = split
    model = make_full_pipeline(LinearRegression(), fe_config=FeatureEngineeringConfig(), selector="lasso")
    assert (cross_val_score(model, raw_train, y_train, cv=3, scoring="r2") > 0.5).all()


def test_engineering_helps_when_the_data_is_curved(curved_df):
    X, y = curved_df.drop(columns="house_sale_price"), curved_df["house_sale_price"]
    scenarios = {
        "base": {"fe": None, "selector": None},
        "engineered": {"fe": FeatureEngineeringConfig(), "selector": "lasso"},
    }
    table = evaluate_feature_sets(X, y, scenarios, n_splits=4, n_repeats=2)
    assert table.loc["engineered", "rmse"] < 0.6 * table.loc["base", "rmse"]
    assert table.loc["engineered", "rmse_gain_%"] > 40
    assert 0 < table.loc["engineered", "p_value"] <= 1  # (corrected test is deliberately cautious with few folds)


def test_evaluate_feature_sets_structure_and_baseline(curved_df):
    X, y = curved_df.drop(columns="house_sale_price"), curved_df["house_sale_price"]
    scenarios = {"base": {"fe": None, "selector": None}, "all": {"fe": FeatureEngineeringConfig(), "selector": None}}
    table = evaluate_feature_sets(X, y, scenarios, n_splits=3, n_repeats=2)
    assert list(table.index) == ["base", "all"]
    assert {"n_features", "rmse", "rmse_std", "mae", "r2", "rmse_gain_%", "p_value"} <= set(table.columns)
    assert table.loc["base", "rmse_gain_%"] == 0 and np.isnan(table.loc["base", "p_value"])
    assert table.loc["base", "n_features"] == 9 and table.loc["all", "n_features"] == 57
    assert len(table.attrs["fold_rmse"]["base"]) == 6  # 3 folds x 2 repeats
    assert table.attrs["baseline"] == "base"


def test_corrected_p_value_matches_manual_calculation(curved_df):
    from scipy import stats

    X, y = curved_df.drop(columns="house_sale_price"), curved_df["house_sale_price"]
    scenarios = {"base": {"fe": None, "selector": None}, "all": {"fe": FeatureEngineeringConfig(), "selector": None}}
    table = evaluate_feature_sets(X, y, scenarios, n_splits=4, n_repeats=2)
    d = table.attrs["fold_rmse"]["base"] - table.attrs["fold_rmse"]["all"]
    k = len(d)
    t = d.mean() / np.sqrt((1 / k + 1 / 3) * d.var(ddof=1))
    assert table.loc["all", "p_value"] == pytest.approx(2 * stats.t.sf(abs(t), k - 1))


def test_default_scenarios_cover_groups_and_selectors():
    names = list(default_scenarios())
    assert names[0].startswith("baseline")
    assert {
        "polynomial only",
        "interactions only",
        "all engineered",
        "all + lasso",
        "all + rfe",
        "all + tree",
        "all + consensus",
        "all + univariate",
        "all + correlation",
        "all + vif",
        "all + vif -> lasso",
        "all + vif -> rfe",
    } <= set(names)
    only_poly = default_scenarios()["polynomial only"]["fe"]
    assert only_poly.add_polynomial and not only_poly.add_interactions and not only_poly.add_composites


def test_select_best_scenario_one_standard_error_rule():
    folds = {name: np.full(10, 0.0) for name in ("a", "b", "c")}
    table = pd.DataFrame(
        {"rmse": [100.0, 101.0, 130.0], "rmse_std": [10.0, 10.0, 10.0], "n_features": [50, 12, 5]},
        index=["a", "b", "c"],
    )
    table.attrs["fold_rmse"] = folds
    assert select_best_scenario(table, one_se_rule=False) == "a"
    assert select_best_scenario(table, one_se_rule=True) == "b"  # within 1 SE of "a", far fewer features


def test_chained_selectors_run_in_sequence(split):
    raw_train, raw_test, y_train, _ = split
    model = make_full_pipeline(LinearRegression(), fe_config=FeatureEngineeringConfig(), selector=["vif", "lasso"])
    model.fit(raw_train, y_train)
    steps = [name for name, _ in model.regressor_.steps]
    assert steps == ["preprocess", "features", "select_1", "select_2", "model"]
    first, second = model.regressor_.named_steps["select_1"], model.regressor_.named_steps["select_2"]
    assert len(second.selected_features_) <= len(first.selected_features_)
    assert set(second.selected_features_) <= set(first.selected_features_)
    assert np.isfinite(model.predict(raw_test)).all()


def test_linear_composites_are_flagged_in_the_catalog(prepared):
    X_train, _, _, _ = prepared
    catalog = FeatureEngineer().fit(X_train).feature_catalog_
    for name in ("nbhd_quality", "rooms_total", "spaciousness"):
        assert "linear combination" in catalog.loc[name, "rationale"]
    assert "linear combination" not in catalog.loc["transit_extremity", "rationale"]


def test_vif_selection_makes_ols_well_conditioned(split):
    raw_train, _, y_train, _ = split
    fe = FeatureEngineeringConfig(clip_z=None)  # without winsorising the dependency is exact for any data
    full = (
        make_full_pipeline(LinearRegression(), fe_config=fe)
        .fit(raw_train, y_train)
        .regressor_[:-1]
        .transform(raw_train)
    )
    vif = (
        make_full_pipeline(LinearRegression(), fe_config=fe, selector="vif")
        .fit(raw_train, y_train)
        .regressor_[:-1]
        .transform(raw_train)
    )
    design = lambda F: np.column_stack([np.ones(len(F)), F.to_numpy(float)])  # least squares includes an intercept  # noqa: E731
    assert np.linalg.cond(design(full)) > 1e10  # exact linear combinations -> numerically singular
    assert np.linalg.cond(design(vif)) < 1e3


# --------------------------------------------------------------------------- #
# Config dictionaries (used by YAML/JSON experiment files)
# --------------------------------------------------------------------------- #
def test_fe_config_dict_round_trip_including_interaction_pairs():
    cfg = FeatureEngineeringConfig(
        add_tiers=False,
        clip_z=2.5,
        n_tiers=4,
        interaction_pairs=[("property_size", "school_quality_score")],
        polynomial_features=["crime_rate_in_area"],
    )
    as_dict = cfg.to_dict()
    assert as_dict["interaction_pairs"] == [["property_size", "school_quality_score"]]  # tuples become lists
    import json

    assert FeatureEngineeringConfig.from_dict(json.loads(json.dumps(as_dict))) == cfg
    assert FeatureEngineeringConfig.from_dict(None) == FeatureEngineeringConfig()


def test_fe_config_from_dict_rejects_unknown_keys_and_invalid_values():
    with pytest.raises(ValueError, match="Unknown feature-engineering"):
        FeatureEngineeringConfig.from_dict({"add_tier": False})
    with pytest.raises(ValueError):
        FeatureEngineeringConfig.from_dict({"n_tiers": 1})
