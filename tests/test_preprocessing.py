"""Unit tests for src/preprocessing.py. Run with:  pytest"""

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score

from src.preprocessing import (
    DataPreprocessor,
    IQROutlierClipper,
    PreprocessingConfig,
    SafeLogTransformer,
    build_target_transformer,
    clean_data,
    make_model_pipeline,
    missing_value_report,
    run_preprocessing,
    split_data,
    split_distribution_report,
)


def with_missing(frame: pd.DataFrame, frac: float = 0.08, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = frame.copy().reset_index(drop=True)
    return out.mask(rng.random(out.shape) < frac)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        {"scaler": "banana"},
        {"outlier_strategy": "drop"},
        {"target_transform": "sqrt"},
        {"numeric_imputer_strategy": "mode"},
        {"categorical_imputer_strategy": "median"},
        {"feature_selection": "pca"},
        {"output_dtype": "float16"},
        {"test_size": 1.5},
        {"outlier_iqr_factor": 0},
        {"variance_threshold": -1},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PreprocessingConfig(**kwargs)


def test_column_cannot_have_two_roles():
    with pytest.raises(ValueError):
        PreprocessingConfig(log_features=["property_size"], standardised_features=["property_size"])


def test_target_cannot_be_a_feature():
    with pytest.raises(ValueError):
        PreprocessingConfig(numeric_features=["house_sale_price"])


# --------------------------------------------------------------------------- #
# Cleaning and splitting
# --------------------------------------------------------------------------- #
def test_clean_data_reports_and_fixes_problems(cfg):
    demo = pd.DataFrame(
        {
            "house_sale_price": [100.0, 100.0, np.nan, -5.0, 200.0],
            "location_type": [" urban", " urban", "Rural", "Rural", "Suburban"],
            "property_size": ["10", "10", 5, 6, "x"],
        }
    )
    cleaned, report = clean_data(demo, cfg)
    assert report["duplicates_removed"] == 1
    assert report["missing_target_removed"] == 1
    assert report["non_positive_target_removed"] == 1
    assert report["rows_out"] == 2
    assert cleaned["location_type"].iloc[0] == "Urban"
    assert cleaned["property_size"].isna().sum() == 1  # "x" became NaN


def test_clean_data_does_not_modify_input_by_default(cfg):
    demo = pd.DataFrame({"house_sale_price": [1.0, 2.0], "location_type": [" a ", "b"], "x": ["1", "2"]})
    snapshot = demo.copy()
    clean_data(demo, cfg)
    pd.testing.assert_frame_equal(demo, snapshot)


def test_clean_data_needs_target(cfg):
    with pytest.raises(KeyError):
        clean_data(pd.DataFrame({"a": [1]}), cfg)


def test_non_positive_target_kept_when_no_log_transform():
    cfg = PreprocessingConfig(target_transform="none")
    _, report = clean_data(pd.DataFrame({"house_sale_price": [-1.0, 2.0], "location_type": ["A", "B"]}), cfg)
    assert report["non_positive_target_removed"] == 0


def test_split_is_stratified_and_reproducible(raw_df, cfg):
    cleaned, _ = clean_data(raw_df, cfg)
    a = split_data(cleaned, cfg)
    b = split_data(cleaned, cfg)
    pd.testing.assert_frame_equal(a[0], b[0])
    train_share = a[0]["location_type"].value_counts(normalize=True)
    test_share = a[1]["location_type"].value_counts(normalize=True)
    assert (train_share - test_share).abs().max() < 0.02
    assert len(a[0]) + len(a[1]) == len(cleaned)
    assert len(a[1]) == pytest.approx(0.2 * len(cleaned), abs=1)


def test_split_without_stratify_column_still_works(raw_df):
    cfg = PreprocessingConfig(stratify_column=None)
    cleaned, _ = clean_data(raw_df, cfg)
    X_train, X_test, _, _ = split_data(cleaned, cfg)
    assert len(X_train) + len(X_test) == len(cleaned)


def test_missing_value_report_counts(raw_df):
    frame = raw_df.copy()
    frame.loc[:9, "age_of_property"] = np.nan
    report = missing_value_report(frame)
    assert report.loc["age_of_property", "missing"] == 10
    assert report.iloc[0].name == "age_of_property"


def test_split_distribution_report(split, cfg):
    raw_train, raw_test, y_train, y_test = split
    report = split_distribution_report(raw_train, raw_test, y_train, y_test, cfg)
    table = report["stratify_column"]
    assert table["train_n"].sum() == len(raw_train)
    assert table["gap_points"].max() < 2.0
    assert {"train", "test"} == set(report["target"].columns)
    assert "skew" in report["target"].index


# --------------------------------------------------------------------------- #
# Custom transformers
# --------------------------------------------------------------------------- #
def test_clipper_caps_extremes_and_keeps_nan():
    x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [1000.0], [np.nan]])
    clipper = IQROutlierClipper(factor=1.5).fit(x)
    out = clipper.transform(x)
    assert out[5, 0] == pytest.approx(clipper.upper_[0])
    assert np.isnan(out[6, 0])
    assert out[0, 0] == 1.0  # inside values untouched


def test_clipper_leaves_constant_column_unchanged():
    x = np.full((10, 1), 7.0)
    assert np.array_equal(IQROutlierClipper().fit_transform(x), x)


def test_clipper_uses_training_fences_on_new_data():
    clipper = IQROutlierClipper(factor=1.5).fit(np.arange(100.0).reshape(-1, 1))
    out = clipper.transform(np.array([[1e6], [-1e6]]))
    assert out[0, 0] == pytest.approx(clipper.upper_[0])
    assert out[1, 0] == pytest.approx(clipper.lower_[0])


def test_safe_log_matches_log_for_positive_and_is_finite_otherwise():
    train = np.array([[0.5], [1.0], [2.0]])
    log = SafeLogTransformer().fit(train)
    assert np.allclose(log.transform(train), np.log(train))
    out = log.transform(np.array([[0.0], [-3.0], [np.nan]]))
    assert np.isfinite(out[:2]).all() and np.isnan(out[2, 0])
    assert out[0, 0] == pytest.approx(np.log(0.5))  # floor = smallest positive training value


# --------------------------------------------------------------------------- #
# DataPreprocessor: structure and outputs
# --------------------------------------------------------------------------- #
def test_output_shape_names_and_no_nan(fitted, split):
    raw_train, raw_test, _, _ = split
    X_train, X_test = fitted.transform(raw_train), fitted.transform(raw_test)
    assert list(X_train.columns) == list(X_test.columns) == fitted.feature_names_out_
    assert X_train.shape == (len(raw_train), 9)  # 7 numeric + 2 location dummies
    assert not X_train.isna().any().any() and not X_test.isna().any().any()
    assert X_train.index.equals(raw_train.index)
    assert {"location_type_Suburban", "location_type_Urban"} <= set(X_train.columns)


def test_standardised_features_are_only_imputed_not_rescaled(fitted, split):
    raw_train, _, _, _ = split
    out = fitted.transform(raw_train)
    for col in ["number_of_bedrooms", "number_of_bathrooms"]:
        assert np.allclose(out[col], raw_train[col])


@pytest.mark.parametrize("scaler", ["standard", "robust", "minmax", "none"])
def test_scalers(split, scaler):
    raw_train, _, _, _ = split
    out = DataPreprocessor(PreprocessingConfig(scaler=scaler, outlier_strategy="none")).fit_transform(raw_train)
    col = out["age_of_property"]
    if scaler == "standard":
        assert col.mean() == pytest.approx(0, abs=1e-9) and col.std(ddof=0) == pytest.approx(1, abs=1e-9)
    elif scaler == "robust":
        assert col.median() == pytest.approx(0, abs=1e-9)
    elif scaler == "minmax":
        assert col.min() == pytest.approx(0) and col.max() == pytest.approx(1)
    else:
        assert np.allclose(col, raw_train["age_of_property"])  # "none": values pass through unchanged


def test_test_set_uses_training_statistics(fitted, split):
    raw_train, raw_test, _, _ = split
    col = "age_of_property"
    lo, hi = fitted.outlier_bounds_.loc[col, ["lower", "upper"]]
    train_clipped = raw_train[col].clip(lo, hi)
    expected = (raw_test[col].clip(lo, hi) - train_clipped.mean()) / train_clipped.std(ddof=0)
    assert np.allclose(fitted.transform(raw_test)[col], expected)
    assert abs(fitted.transform(raw_test)[col].mean()) > 1e-6  # not re-centred on the test set


def test_log_transform_reduces_skew(fitted, split):
    raw_train, _, _, _ = split
    out = fitted.transform(raw_train)
    for col in ["crime_rate_in_area", "property_size", "proximity_to_public_transport"]:
        assert abs(out[col].skew()) < abs(raw_train[col].skew())


def test_outlier_clipping_can_be_disabled(split):
    raw_train, _, _, _ = split
    pre = DataPreprocessor(PreprocessingConfig(outlier_strategy="none")).fit(raw_train)
    assert pre.outlier_bounds_.empty


def test_extreme_value_is_mapped_to_fence_value(fitted, split):
    _, raw_test, _, _ = split
    upper = fitted.outlier_bounds_.loc["property_size", "upper"]
    a, b = raw_test.iloc[[0]].copy(), raw_test.iloc[[0]].copy()
    a["property_size"], b["property_size"] = upper * 1e6, upper
    assert np.allclose(fitted.transform(a), fitted.transform(b))


def test_unseen_and_messy_categories(fitted, split):
    _, raw_test, _, _ = split
    row = raw_test.iloc[[0, 1]].copy()
    row["location_type"] = ["Coastal", "  rural "]
    out = fitted.transform(row)
    assert (out.iloc[0][["location_type_Suburban", "location_type_Urban"]] == 0).all()  # unseen -> zeros
    assert (out.iloc[1][["location_type_Suburban", "location_type_Urban"]] == 0).all()  # Rural is the dropped level


def test_extra_columns_are_ignored_and_logged(split, cfg, caplog):
    raw_train, _, _, _ = split
    extra = raw_train.assign(notes="x")
    with caplog.at_level(logging.WARNING, logger="src.preprocessing"):
        pre = DataPreprocessor(cfg).fit(extra)
    assert pre.ignored_columns_ == ["notes"]
    assert "notes" in caplog.text
    assert pre.transform(extra).shape[1] == 9


def test_target_in_features_is_rejected(split, cfg):
    raw_train, _, y_train, _ = split
    with pytest.raises(ValueError, match="Target column"):
        DataPreprocessor(cfg).fit(raw_train.assign(house_sale_price=y_train))


def test_missing_required_column_is_rejected(split, cfg):
    raw_train, _, _, _ = split
    with pytest.raises(ValueError, match="missing"):
        DataPreprocessor(cfg).fit(raw_train.drop(columns=["crime_rate_in_area"]))


def test_transform_before_fit_fails(split, cfg):
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        DataPreprocessor(cfg).transform(split[0])


def test_requires_dataframe(cfg):
    with pytest.raises(TypeError):
        DataPreprocessor(cfg).fit(np.zeros((3, 3)))


def test_fit_is_deterministic(split, cfg):
    raw_train, raw_test, _, _ = split
    a = DataPreprocessor(cfg).fit(raw_train).transform(raw_test)
    b = DataPreprocessor(cfg).fit(raw_train).transform(raw_test)
    pd.testing.assert_frame_equal(a, b)


def test_output_dtype_float32_halves_memory(split):
    raw_train, _, _, _ = split
    f64 = DataPreprocessor(PreprocessingConfig()).fit_transform(raw_train)
    f32 = DataPreprocessor(PreprocessingConfig(output_dtype="float32")).fit_transform(raw_train)
    assert (f32.dtypes == "float32").all()
    assert f32.memory_usage(deep=True).sum() < 0.6 * f64.memory_usage(deep=True).sum()


# --------------------------------------------------------------------------- #
# Imputation
# --------------------------------------------------------------------------- #
def test_imputation_report_counts_match_injected_missing(split, cfg):
    raw_train, _, _, _ = split
    messy = with_missing(raw_train, frac=0.08)
    pre = DataPreprocessor(cfg).fit(messy)
    report = pre.imputation_report_
    for col in report.index:
        assert report.loc[col, "n_missing"] == messy[col].isna().sum()
    assert report.loc["location_type", "strategy"] == "most_frequent"
    assert report.loc["age_of_property", "strategy"] == "median"
    assert report.loc["age_of_property", "fill_value"] == pytest.approx(messy["age_of_property"].median())


def test_imputation_report_for_new_data_uses_training_fill_values(fitted, split):
    _, raw_test, _, _ = split
    messy = with_missing(raw_test, frac=0.1, seed=3)
    report = fitted.imputation_report(messy)
    assert report.loc["school_quality_score", "n_missing"] == messy["school_quality_score"].isna().sum()
    assert (
        report.loc["school_quality_score", "fill_value"]
        == fitted.imputation_report_.loc["school_quality_score", "fill_value"]
    )
    assert fitted.imputation_report_["n_missing"].sum() == 0  # the training report is unchanged


def test_unparseable_text_counts_as_missing(fitted, split):
    raw_train, _, _, _ = split
    frame = raw_train.copy()
    frame["age_of_property"] = frame["age_of_property"].astype(object)
    frame.iloc[0, frame.columns.get_loc("age_of_property")] = "n/a"
    assert fitted.imputation_report(frame).loc["age_of_property", "n_missing"] == 1
    assert fitted.transform(frame).notna().all().all()


@pytest.mark.parametrize("strategy", ["median", "mean", "constant"])
def test_numeric_imputation_strategies(split, strategy):
    raw_train, _, _, _ = split
    messy = with_missing(raw_train)
    cfg = PreprocessingConfig(
        numeric_imputer_strategy=strategy, numeric_fill_value=-1.0, scaler="none", outlier_strategy="none"
    )
    pre = DataPreprocessor(cfg).fit(messy)
    out = pre.transform(messy)
    assert out.notna().all().all()
    fill = pre.imputation_report_.loc["number_of_bedrooms", "fill_value"]
    expected = {
        "median": messy["number_of_bedrooms"].median(),
        "mean": messy["number_of_bedrooms"].mean(),
        "constant": -1.0,
    }
    assert fill == pytest.approx(expected[strategy])
    was_missing = messy["number_of_bedrooms"].isna()
    assert np.allclose(out.loc[was_missing.values, "number_of_bedrooms"], fill)


def test_constant_categorical_imputation_creates_missing_level(split):
    raw_train, _, _, _ = split
    messy = with_missing(raw_train)
    cfg = PreprocessingConfig(categorical_imputer_strategy="constant", categorical_fill_value="Unknown")
    pre = DataPreprocessor(cfg).fit(messy)
    assert "location_type_Unknown" in pre.feature_names_out_


def test_missing_indicator_columns(split):
    raw_train, _, _, _ = split
    messy = with_missing(raw_train, frac=0.05)
    pre = DataPreprocessor(PreprocessingConfig(add_missing_indicator=True)).fit(messy)
    out = pre.transform(messy)
    flags = [c for c in out.columns if c.startswith("missingindicator_")]
    assert len(flags) == len(pre.missing_flag_columns_) > 0
    col = pre.missing_flag_columns_[0]
    assert (out[f"missingindicator_{col}"].astype(bool).values == messy[col].isna().values).all()


def test_no_missing_indicator_columns_when_data_is_complete(split):
    raw_train, _, _, _ = split
    out = DataPreprocessor(PreprocessingConfig(add_missing_indicator=True)).fit_transform(raw_train)
    assert not any(c.startswith("missingindicator_") for c in out.columns)


def test_fit_logs_a_summary(split, cfg, caplog):
    raw_train, _, _, _ = split
    messy = with_missing(raw_train)
    with caplog.at_level(logging.INFO, logger="src.preprocessing"):
        pre = DataPreprocessor(cfg).fit(messy)
    text = caplog.text
    assert "Imputation" in text and "Outlier clipping" in text and "Scaler: standard" in text
    assert "One-hot encoding created 2" in text and "Output feature space" in text
    assert pre.fit_summary_["n_output_features"] == 9
    assert pre.fit_summary_["imputed_values_per_column"]


# --------------------------------------------------------------------------- #
# Feature selection
# --------------------------------------------------------------------------- #
def test_variance_selection_drops_constant_column(split):
    raw_train, _, _, _ = split
    frame = raw_train.copy()
    frame["age_of_property"] = 5.0  # a constant column carries no information
    pre = DataPreprocessor(PreprocessingConfig(feature_selection="variance")).fit(frame)
    assert pre.n_features_before_selection_ == 9
    assert pre.dropped_features_ == ["age_of_property"]
    assert pre.transform(frame).shape[1] == 8
    assert not pre.feature_selection_report_.loc["age_of_property", "kept"]


def test_lasso_selection_requires_target(split):
    raw_train, _, _, _ = split
    with pytest.raises(ValueError, match="needs the target"):
        DataPreprocessor(PreprocessingConfig(feature_selection="lasso")).fit(raw_train)


def test_lasso_selection_keeps_informative_and_drops_noise(split):
    raw_train, _, y_train, _ = split
    noisy = raw_train.assign(age_of_property=np.random.default_rng(5).normal(size=len(raw_train)))
    pre = DataPreprocessor(PreprocessingConfig(feature_selection="lasso")).fit(noisy, np.log(y_train))
    assert "property_size" in pre.selected_features_
    report = pre.feature_selection_report_
    assert report.loc["property_size", "kept"] and "abs_lasso_coef" in report.columns
    assert pre.transform(noisy).shape[1] == len(pre.selected_features_)


# --------------------------------------------------------------------------- #
# Target transform and sklearn integration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strategy", ["log", "boxcox"])
def test_target_transform_round_trip_and_skew(split, strategy):
    _, _, y_train, _ = split
    values = y_train.to_numpy().reshape(-1, 1)
    t = build_target_transformer(strategy).fit(values)
    z = np.asarray(t.transform(values))
    back = np.asarray(t.inverse_transform(z)).ravel()
    assert np.allclose(back, y_train.to_numpy(), rtol=1e-6)
    assert abs(pd.Series(z.ravel()).skew()) < abs(y_train.skew())


def test_target_transform_none_returns_none():
    assert build_target_transformer("none") is None


def test_unknown_target_transform_rejected():
    with pytest.raises(ValueError):
        build_target_transformer("square")


def test_clone_and_get_params(cfg):
    pre = DataPreprocessor(cfg)
    copy = clone(pre)
    assert copy.get_params()["config"] == cfg and copy is not pre


def test_save_and_load_round_trip(fitted, split, tmp_path):
    _, raw_test, _, _ = split
    path = fitted.save(tmp_path / "nested" / "pre.joblib")
    assert path.exists()
    pd.testing.assert_frame_equal(DataPreprocessor.load(path).transform(raw_test), fitted.transform(raw_test))


def test_save_before_fit_fails(cfg, tmp_path):
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        DataPreprocessor(cfg).save(tmp_path / "x.joblib")


@pytest.mark.parametrize("target_transform", ["log", "none"])
def test_model_pipeline_predicts_on_price_scale(split, target_transform):
    raw_train, raw_test, y_train, y_test = split
    cfg = PreprocessingConfig(target_transform=target_transform)
    model = make_model_pipeline(Ridge(alpha=1.0), cfg).fit(raw_train, y_train)
    pred = model.predict(raw_test)
    assert pred.shape == (len(raw_test),) and np.isfinite(pred).all()
    assert 0.3 * y_test.median() < np.median(pred) < 3 * y_test.median()  # right order of magnitude (not log scale)


def test_cross_validation_runs_without_leakage_machinery_errors(split, cfg):
    raw_train, _, y_train, _ = split
    scores = cross_val_score(make_model_pipeline(LinearRegression(), cfg), raw_train, y_train, cv=3, scoring="r2")
    assert (scores > 0.5).all()


def test_model_pipeline_handles_missing_values_end_to_end(split, cfg):
    raw_train, raw_test, y_train, _ = split
    model = make_model_pipeline(LinearRegression(), cfg).fit(with_missing(raw_train), y_train.reset_index(drop=True))
    assert np.isfinite(model.predict(with_missing(raw_test, seed=9))).all()


def test_lasso_selection_works_inside_model_pipeline(split):
    raw_train, raw_test, y_train, _ = split
    cfg = PreprocessingConfig(feature_selection="lasso")
    model = make_model_pipeline(LinearRegression(), cfg).fit(raw_train, y_train)
    assert np.isfinite(model.predict(raw_test)).all()


# --------------------------------------------------------------------------- #
# End-to-end helper
# --------------------------------------------------------------------------- #
def test_run_preprocessing_end_to_end(raw_df, cfg):
    res = run_preprocessing(raw_df, cfg)
    assert res.X_train.shape[1] == res.X_test.shape[1] == 9
    assert len(res.X_train) + len(res.X_test) == len(raw_df)
    assert np.allclose(np.exp(res.y_train_transformed), res.y_train)
    assert not res.X_train.isna().any().any()
    assert "stratify_column" in res.split_report and "target" in res.split_report


def test_run_preprocessing_with_dirty_data(raw_df, cfg):
    dirty = with_missing(raw_df.drop(columns="house_sale_price"), 0.03).assign(
        house_sale_price=raw_df["house_sale_price"].values
    )
    dirty = pd.concat([dirty, dirty.iloc[:5]], ignore_index=True)  # 5 exact duplicates
    res = run_preprocessing(dirty, cfg)
    assert res.cleaning_report["duplicates_removed"] >= 5
    assert not res.X_train.isna().any().any() and not res.X_test.isna().any().any()


# --------------------------------------------------------------------------- #
# Command-line script (uses the real CSV when it is present)
# --------------------------------------------------------------------------- #
def test_main_validation_suite_passes():
    from pathlib import Path

    import main

    if not (Path(main.ROOT) / "Housing_Price_Dataset.csv").exists():
        pytest.skip("real dataset not available")
    assert main.main(["--no-save", "--validate"]) == 0


# --------------------------------------------------------------------------- #
# Per-column / KNN imputation
# --------------------------------------------------------------------------- #
from src.preprocessing import ColumnImputer, FrequencyEncoder, HashingEncoder  # noqa: E402


def test_column_imputer_applies_a_strategy_per_column():
    X = pd.DataFrame({"a": [1.0, 2.0, np.nan, 100.0], "b": [1.0, 2.0, np.nan, 100.0], "c": [np.nan, 5.0, 5.0, 7.0]})
    imp = ColumnImputer(strategies={"b": "mean", "c": "constant"}, default_strategy="median", fill_value=-1).fit(X)
    out = imp.transform(X)
    assert out[2, 0] == 2.0  # median of 1, 2, 100
    assert out[2, 1] == pytest.approx(103 / 3)  # mean
    assert out[0, 2] == -1  # constant
    assert imp.strategies_ == ["median", "mean", "constant"]
    assert not np.isnan(out).any()


def test_column_imputer_uses_training_values_on_new_data():
    imp = ColumnImputer(default_strategy="median").fit(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    assert imp.transform(pd.DataFrame({"a": [np.nan, 50.0]}))[0, 0] == 2.0


def test_column_imputer_all_missing_column_falls_back_to_zero():
    imp = ColumnImputer().fit(pd.DataFrame({"a": [np.nan, np.nan]}))
    assert imp.transform(pd.DataFrame({"a": [np.nan]}))[0, 0] == 0.0


def test_column_imputer_knn_uses_similar_rows():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    X = pd.DataFrame({"x": x, "y": 2 * x + rng.normal(0, 0.05, 300)})
    holed = X.copy()
    holed.loc[:19, "y"] = np.nan
    imp = ColumnImputer(default_strategy="knn", n_neighbors=5).fit(holed)
    out = imp.transform(holed)
    knn_error = np.abs(out[:20, 1] - X["y"][:20]).mean()
    median_error = np.abs(holed["y"].median() - X["y"][:20]).mean()
    assert knn_error < 0.2 * median_error  # neighbours on the correlated column beat the median
    assert np.isnan(imp.statistics_).all()


def test_column_imputer_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown imputation"):
        ColumnImputer(default_strategy="mode").fit(pd.DataFrame({"a": [1.0]}))


def test_preprocessor_imputer_overrides_and_group_strategies(split):
    raw_train, _, _, _ = split
    messy = with_missing(raw_train, frac=0.1)
    cfg = PreprocessingConfig(
        numeric_imputer_strategy="median",
        group_imputer_strategies={"standardised": "constant"},
        imputer_overrides={"age_of_property": "mean"},
        numeric_fill_value=0.0,
    )
    pre = DataPreprocessor(cfg).fit(messy)
    rep = pre.imputation_report_
    assert rep.loc["age_of_property", "strategy"] == "mean"
    assert rep.loc["age_of_property", "fill_value"] == pytest.approx(messy["age_of_property"].mean())
    assert rep.loc["number_of_bedrooms", "strategy"] == "constant"
    assert rep.loc["school_quality_score", "strategy"] == "median"
    assert pre.transform(messy).notna().all().all()


def test_preprocessor_knn_strategy_reports_knn_and_has_no_nan(split):
    raw_train, _, _, _ = split
    messy = with_missing(raw_train, frac=0.08)
    pre = DataPreprocessor(PreprocessingConfig(numeric_imputer_strategy="knn")).fit(messy)
    assert (pre.imputation_report_.query("group != 'categorical_onehot'")["strategy"] == "knn").all()
    assert pre.transform(messy).notna().all().all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"imputer_overrides": {"a": "mode"}},
        {"group_imputer_strategies": {"nonsense": "mean"}},
        {"group_imputer_strategies": {"numeric": "mode"}},
        {"knn_neighbors": 0},
        {"categorical_encoding": "ordinal"},
        {"high_cardinality_encoding": "onehot"},
        {"encoding_overrides": {"a": "auto"}},
        {"hashing_n_features": 0},
        {"max_onehot_cardinality": 0},
    ],
)
def test_new_config_options_are_validated(kwargs):
    with pytest.raises(ValueError):
        PreprocessingConfig(**kwargs)


def test_preprocessing_config_dict_round_trip_and_unknown_keys():
    cfg = PreprocessingConfig(scaler="robust", imputer_overrides={"age_of_property": "knn"})
    assert PreprocessingConfig.from_dict(cfg.to_dict()) == cfg
    assert PreprocessingConfig.from_dict(None) == PreprocessingConfig()
    with pytest.raises(ValueError, match="Unknown preprocessing"):
        PreprocessingConfig.from_dict({"scalar": "standard"})


# --------------------------------------------------------------------------- #
# Categorical encoders
# --------------------------------------------------------------------------- #
def test_frequency_encoder_uses_training_shares_and_zero_for_unseen():
    train = pd.DataFrame({"c": ["a", "a", "a", "b"]})
    enc = FrequencyEncoder().fit(train)
    out = enc.transform(pd.DataFrame({"c": ["a", "b", "zzz", None]}))
    assert out.ravel().tolist() == [0.75, 0.25, 0.0, 0.0]
    assert list(enc.get_feature_names_out()) == ["c_freq"]


def test_hashing_encoder_is_deterministic_fixed_width_and_handles_unseen():
    train = pd.DataFrame({"c": [f"k{i}" for i in range(50)]})
    enc = HashingEncoder(n_features=6).fit(train)
    a, b = enc.transform(train), enc.transform(train)
    assert a.shape == (50, 6) and np.array_equal(a, b)
    assert (a.sum(axis=1) == 1).all()  # one token per row, one bucket hit
    unseen = enc.transform(pd.DataFrame({"c": ["never-seen"]}))
    assert unseen.shape == (1, 6) and unseen.sum() == 1
    assert list(enc.get_feature_names_out()) == [f"hash_{i}" for i in range(6)]


@pytest.fixture(scope="module")
def high_card_df(raw_df):
    rng = np.random.default_rng(7)
    cats = [f"area{i:03d}" for i in range(80)]
    effect = {c.title(): v for c, v in zip(cats, rng.normal(0, 0.4, 80))}
    out = raw_df.copy()
    out["neighbourhood"] = rng.choice(cats, len(out))
    out["house_sale_price"] = np.exp(
        11
        + 0.3 * np.log(out["property_size"])
        + out["neighbourhood"].str.title().map(effect)
        + rng.normal(0, 0.25, len(out))
    )
    return out, effect


def _hc_config(**kw):
    return PreprocessingConfig(categorical_features=["location_type", "neighbourhood"], **kw)


@pytest.mark.parametrize(
    "encoding,width", [("onehot", 79 + 2), ("frequency", 1 + 2), ("target", 1 + 2), ("hashing", 8 + 2)]
)
def test_each_encoding_produces_the_expected_columns(high_card_df, encoding, width):
    df, _ = high_card_df
    cfg = _hc_config(categorical_encoding=encoding, encoding_overrides={"location_type": "onehot"})
    X = df.drop(columns="house_sale_price")
    pre = DataPreprocessor(cfg).fit(X, np.log(df["house_sale_price"]))
    out = pre.transform(X)
    assert out.shape[1] == 7 + width and out.notna().all().all()
    assert pre.encoding_report_.loc["neighbourhood", "encoding"] == encoding
    assert pre.encoding_report_.loc["neighbourhood", "n_output_columns"] == width - 2


def test_auto_encoding_chooses_by_cardinality(high_card_df):
    df, _ = high_card_df
    X = df.drop(columns="house_sale_price")
    pre = DataPreprocessor(
        _hc_config(categorical_encoding="auto", max_onehot_cardinality=10, high_cardinality_encoding="frequency")
    ).fit(X)
    plan = pre.encoding_plan_
    assert plan["location_type"] == ("onehot", 3) and plan["neighbourhood"] == ("frequency", 80)
    low = DataPreprocessor(_hc_config(categorical_encoding="auto", max_onehot_cardinality=100)).fit(X)
    assert low.encoding_plan_["neighbourhood"][0] == "onehot"


def test_encoding_overrides_beat_the_global_setting(high_card_df):
    df, _ = high_card_df
    X = df.drop(columns="house_sale_price")
    pre = DataPreprocessor(
        _hc_config(categorical_encoding="onehot", encoding_overrides={"neighbourhood": "hashing"})
    ).fit(X)
    assert pre.encoding_plan_["neighbourhood"][0] == "hashing" and pre.encoding_plan_["location_type"][0] == "onehot"


def test_target_encoding_requires_the_target(high_card_df):
    df, _ = high_card_df
    with pytest.raises(ValueError, match="Target encoding needs the target"):
        DataPreprocessor(_hc_config(categorical_encoding="target")).fit(df.drop(columns="house_sale_price"))


def test_target_encoding_training_output_is_cross_fitted(high_card_df):
    """fit_transform must not encode a row using its own target (that would leak)."""
    df, effect = high_card_df
    X, y = df.drop(columns="house_sale_price"), np.log(df["house_sale_price"])
    pre = DataPreprocessor(_hc_config(categorical_encoding="target"))
    crossfit = pre.fit_transform(X, y)["neighbourhood"]
    leaky = pre.transform(X)["neighbourhood"]
    resid = y - 0.3 * np.log(X["property_size"])
    assert np.corrcoef(crossfit, resid)[0, 1] < np.corrcoef(leaky, resid)[0, 1] - 0.02
    true_effect = X["neighbourhood"].str.title().map(effect)
    assert np.corrcoef(crossfit, true_effect)[0, 1] > 0.8  # still captures the real category effect


def test_target_encoding_unseen_category_gets_a_finite_value(high_card_df):
    df, _ = high_card_df
    X, y = df.drop(columns="house_sale_price"), np.log(df["house_sale_price"])
    pre = DataPreprocessor(_hc_config(categorical_encoding="target")).fit(X, y)
    test = X.iloc[:3].copy()
    test["neighbourhood"] = "brand-new-area"
    assert np.isfinite(pre.transform(test).to_numpy(float)).all()


def test_target_encoding_inside_a_model_pipeline_and_cv(high_card_df):
    df, _ = high_card_df
    X, y = df.drop(columns="house_sale_price"), df["house_sale_price"]
    cfg = _hc_config(categorical_encoding="target")
    scores = cross_val_score(make_model_pipeline(LinearRegression(), cfg), X, y, cv=3, scoring="r2")
    assert (scores > 0.4).all()
    onehot = cross_val_score(make_model_pipeline(LinearRegression(), _hc_config()), X, y, cv=3, scoring="r2")
    assert scores.mean() > 0.3 and onehot.mean() > 0.3


def test_default_encoding_is_unchanged_one_hot(fitted):
    assert fitted.encoding_plan_ == {"location_type": ("onehot", 3)}
    assert {"location_type_Suburban", "location_type_Urban"} <= set(fitted.feature_names_out_)
