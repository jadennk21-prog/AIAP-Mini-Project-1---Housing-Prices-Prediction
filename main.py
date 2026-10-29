"""
Command-line entry point for the housing-price preprocessing pipeline.

Examples
--------
Run the pipeline and write the processed train/test files::

    python main.py

Run it and execute the validation suite and a baseline model::

    python main.py --validate

Engineer and select features on top of the preprocessing (saves the engineered data and reports)::

    python main.py --features --selection-method lasso

Benchmark the baseline models (a ladder from "predict the mean" to the preprocessed linear regression)::

    python main.py --baseline                    # about 5 seconds
    python main.py --baseline --candidates       # also Ridge, Lasso, trees and the engineered-feature model

Compare feature sets with cross-validation (training data only) and confirm on the held-out test set::

    python main.py --compare-features            # about 2 minutes
    python main.py --compare-features --compare-models   # also checks Ridge / tree models

Run a whole experiment from a version-controlled config file (command-line flags override it)::

    python main.py --config configs/default.yaml --features
    python main.py --save-config my_experiment.yaml --scaler robust   # write the resolved config and exit

Every run that saves files writes into versioned folders (``artifacts/<run_id>/``, ``reports/<run_id>/``,
``processed/<run_id>/``) plus a ``results.json`` record, the resolved ``config.yaml`` and a line in
``reports/runs.csv``. Use ``--no-versioning`` for the old flat layout and ``--profile`` to track memory.

Try other choices (robust scaling, no outlier clipping, Box-Cox target)::

    python main.py --scaler robust --outlier-strategy none --target-transform boxcox
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

from src.experiment import (
    RunRecorder,
    StageProfiler,
    ensure_unique_run_id,
    load_config,
    make_run_id,
    save_config,
    versioned_dir,
)
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
from src.models import (
    REFERENCE_BASELINE,
    BaselineModel,
    baseline_specs,
    candidate_specs,
    regression_metrics,
    run_benchmark,
)
from src.preprocessing import (
    DataPreprocessor,
    PreprocessedData,
    PreprocessingConfig,
    clean_data,
    load_data,
    make_model_pipeline,
    missing_value_report,
    run_preprocessing,
)

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("main")


# --------------------------------------------------------------------------- #
# Validation suite
# --------------------------------------------------------------------------- #
Check = Tuple[str, bool, str]


def _num(value: Any) -> float:
    """Plain float from a pandas/NumPy scalar (keeps type checkers and formatting happy)."""
    return float(value)


def _check_output_quality(res: PreprocessedData) -> Check:
    frames = {"train": res.X_train, "test": res.X_test}
    problems = []
    for name, frame in frames.items():
        if frame.isna().any().any():
            problems.append(f"{name} contains NaN")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            problems.append(f"{name} contains inf")
    if list(res.X_train.columns) != list(res.X_test.columns):
        problems.append("train and test columns differ")
    detail = f"{res.X_train.shape[1]} features; train {res.X_train.shape}, test {res.X_test.shape}"
    return "Output has no NaN/inf and train/test columns match", not problems, "; ".join(problems) or detail


def _check_scaling(res: PreprocessedData) -> Check:
    cfg = res.config
    if cfg.scaler != "standard":
        return "Scaled train features have mean 0 / std 1", True, f"skipped (scaler = {cfg.scaler})"
    cols = [c for c in res.X_train.columns if c in cfg.log_features or c in res.preprocessor.numeric_features_]
    means = res.X_train[cols].mean().abs().max()
    stds = (res.X_train[cols].std(ddof=0) - 1).abs().max()
    ok = means < 1e-8 and stds < 1e-8
    return "Scaled train features have mean 0 / std 1", ok, f"max |mean| = {means:.1e}, max |std-1| = {stds:.1e}"


def _check_no_leakage(res: PreprocessedData) -> Check:
    """Test features must be scaled with *training* statistics, not their own."""
    cfg, pre = res.config, res.preprocessor
    col = next((c for c in pre.numeric_features_), None)
    if col is None or cfg.scaler != "standard" or cfg.outlier_strategy != "clip":
        return "Test set uses training statistics (no leakage)", True, "skipped (needs standard scaler and clipping)"
    lo, hi = pre.outlier_bounds_.loc[col, ["lower", "upper"]]
    train_clipped = res.raw_X_train[col].clip(lo, hi)
    expected = (res.raw_X_test[col].clip(lo, hi) - train_clipped.mean()) / train_clipped.std(ddof=0)
    max_diff = float((expected - res.X_test[col]).abs().max())
    own_mean = float(abs(res.X_test[col].mean()))
    ok = max_diff < 1e-9 and own_mean > 1e-6
    return (
        "Test set uses training statistics (no leakage)",
        ok,
        f"'{col}': max diff vs hand-computed = {max_diff:.1e}; test mean = {own_mean:.3f} (not forced to 0)",
    )


def _check_log_effect(res: PreprocessedData) -> Check:
    cfg = res.config
    rows, ok = [], True
    for col in cfg.log_features:
        before, after = _num(res.raw_X_train[col].skew()), _num(res.X_train[col].skew())
        rows.append(f"{col}: {before:.2f} -> {after:.2f}")
        ok &= abs(after) < abs(before)
    return "Log transform reduces skewness of skewed features", ok, "; ".join(rows)


def _check_outlier_clipping(res: PreprocessedData) -> Check:
    cfg, pre = res.config, res.preprocessor
    if cfg.outlier_strategy != "clip":
        return "Extreme values are capped", True, "skipped (outlier_strategy = none)"
    col = cfg.log_features[0] if cfg.log_features else pre.numeric_features_[0]
    upper = pre.outlier_bounds_.loc[col, "upper"]
    base = res.raw_X_test.iloc[[0]].copy()
    extreme, at_fence = base.copy(), base.copy()
    extreme[col], at_fence[col] = upper * 1e6, upper
    same = np.allclose(pre.transform(extreme).to_numpy(float), pre.transform(at_fence).to_numpy(float))
    n_clipped = int(pre.count_clipped(res.raw_X_train).sum())
    return (
        "Extreme values are capped at the learned fence",
        bool(same),
        (
            f"x1e6 value in '{col}' maps to the same output as the fence value; "
            f"{n_clipped} training values clipped in total"
        ),
    )


def _check_robustness(res: PreprocessedData) -> Check:
    """Fit and transform on deliberately corrupted data."""
    cfg = res.config
    rng = np.random.default_rng(0)

    def corrupt(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy().reset_index(drop=True)
        mask = rng.random(out.shape) < 0.05
        out = out.mask(mask)  # 5% of all cells missing
        num_col = cfg.log_features[0]
        out[num_col] = out[num_col].astype(object)
        out.loc[1, num_col] = "not a number"  # unparseable text in a numeric column
        cat = cfg.categorical_features[0]
        out[cat] = out[cat].astype(object)
        out.loc[2, cat] = "  rural "  # whitespace / case problems
        return out

    bad_train, bad_test = corrupt(res.raw_X_train), corrupt(res.raw_X_test)
    bad_test.loc[0, cfg.categorical_features[0]] = "Coastal"  # category never seen in training
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # the unseen-category warning is expected here
            pre = DataPreprocessor(cfg).fit(bad_train)
            out = pre.transform(bad_test)
    except Exception as exc:  # noqa: BLE001
        return "Handles missing values, bad text and unseen categories", False, f"raised {exc!r}"
    dummies = [c for c in out.columns if c.startswith(cfg.categorical_features[0] + "_")]
    unseen_ok = (out.loc[0, dummies] == 0).all() if dummies else True
    finite = bool(out.notna().all().all() and np.isfinite(out.to_numpy(float)).all())
    return (
        "Handles missing values, bad text and unseen categories",
        finite and bool(unseen_ok),
        "output finite; unseen category encoded as all zeros" if finite and unseen_ok else "see output",
    )


def _check_imputation_report(res: PreprocessedData) -> Check:
    """The imputation report must match the real number of missing values."""
    cfg = res.config
    rng = np.random.default_rng(1)
    frame = res.raw_X_train.copy().reset_index(drop=True)
    frame = frame.mask(rng.random(frame.shape) < 0.08)
    num_col = cfg.log_features[0]
    frame[num_col] = frame[num_col].astype(object)
    frame.loc[3, num_col] = "bad text"  # becomes NaN when coerced to a number
    expected = {}
    for col in frame.columns:
        if col in cfg.categorical_features:
            expected[col] = int(frame[col].isna().sum())
        else:
            expected[col] = int(pd.to_numeric(frame[col], errors="coerce").isna().sum())
    pre = DataPreprocessor(cfg).fit(frame)
    reported = pre.imputation_report_["n_missing"].to_dict()
    ok = all(reported[c] == expected[c] for c in expected)
    total = sum(expected.values())
    return (
        "Imputation report matches the true missing counts",
        ok,
        (
            f"{total} missing values injected; report agrees for all {len(expected)} columns"
            if ok
            else f"expected {expected}, reported {reported}"
        ),
    )


def _check_target(res: PreprocessedData) -> Check:
    if res.target_transformer is None:
        return "Target transform is reversible and reduces skew", True, "skipped (target_transform = none)"
    back = np.asarray(res.target_transformer.inverse_transform(res.y_train_transformed.to_frame())).ravel()
    max_rel = float(np.max(np.abs(back - res.y_train.to_numpy()) / res.y_train.to_numpy()))
    before, after = _num(res.y_train.skew()), _num(res.y_train_transformed.skew())
    ok = max_rel < 1e-6 and abs(after) < abs(before)
    return (
        "Target transform is reversible and reduces skew",
        ok,
        (f"max relative round-trip error = {max_rel:.1e}; skew {before:.2f} -> {after:.2f}"),
    )


def _check_persistence(res: PreprocessedData) -> Check:
    with tempfile.TemporaryDirectory() as tmp:
        path = res.preprocessor.save(Path(tmp) / "pre.joblib")
        loaded = DataPreprocessor.load(path)
        try:
            pd.testing.assert_frame_equal(loaded.transform(res.raw_X_test), res.X_test)
        except AssertionError as exc:
            return "Saved preprocessor reproduces identical output", False, str(exc)[:120]
    return "Saved preprocessor reproduces identical output", True, "save/load round trip identical"


def _check_split(res: PreprocessedData) -> Check:
    col = res.config.stratify_column
    if not col or col not in res.raw_X_train.columns:
        return "Split preserves category proportions", True, "skipped (no stratify column)"
    train = res.raw_X_train[col].value_counts(normalize=True)
    test = res.raw_X_test[col].value_counts(normalize=True)
    gap = float((train - test).abs().max())
    return "Split preserves category proportions", gap < 0.02, f"largest train/test share gap = {gap * 100:.1f} points"


def _check_cleaning(res: PreprocessedData) -> Check:
    cfg = res.config
    demo = pd.DataFrame(
        {
            cfg.target: [100.0, 100.0, np.nan, -5.0, 200.0],
            "location_type": [" urban", " urban", "Rural", "Rural", "Suburban"],
            "property_size": ["10", "10", 5, 6, "x"],
        }
    )
    logging.getLogger("src.preprocessing").setLevel(logging.ERROR)  # demo data is deliberately dirty
    try:
        cleaned, rep = clean_data(demo, cfg)
    finally:
        logging.getLogger("src.preprocessing").setLevel(logging.NOTSET)
    ok = rep["duplicates_removed"] == 1 and rep["missing_target_removed"] == 1
    if cfg.target_transform in ("log", "boxcox"):
        ok &= rep["non_positive_target_removed"] == 1
    ok &= cleaned["location_type"].iloc[0] == "Urban" and cleaned["property_size"].isna().sum() == 1
    return "clean_data removes duplicates / bad targets and fixes text", bool(ok), str(rep)


def _check_config() -> Check:
    try:
        PreprocessingConfig(scaler="banana")
    except ValueError:
        return "Invalid configuration is rejected", True, "ValueError raised for scaler='banana'"
    return "Invalid configuration is rejected", False, "no error raised"


def _check_model(res: PreprocessedData) -> Check:
    """End-to-end: preprocessing + linear regression, evaluated on the original price scale."""
    cfg = res.config
    lines, ok = [], True
    for label, transform in (("raw target", "none"), (f"{cfg.target_transform} target", cfg.target_transform)):
        local = PreprocessingConfig(**{**cfg.to_dict(), "target_transform": transform})
        model = make_model_pipeline(LinearRegression(), local)
        model.fit(res.raw_X_train, res.y_train)
        pred = model.predict(res.raw_X_test)
        ok &= bool(np.isfinite(pred).all())
        rmse = mean_squared_error(res.y_test, pred) ** 0.5
        cv = cross_val_score(
            make_model_pipeline(LinearRegression(), local),
            res.raw_X_train,
            res.y_train,
            cv=KFold(5, shuffle=True, random_state=cfg.random_state),
            scoring="r2",
        )
        lines.append(
            f"[{label}] test R2 = {r2_score(res.y_test, pred):.3f}, "
            f"MAE = {mean_absolute_error(res.y_test, pred):,.0f}, "
            f"RMSE = {rmse:,.0f}, 5-fold CV R2 = {cv.mean():.3f}"
        )
    return "End-to-end linear baseline runs inside sklearn pipelines (no leakage in CV)", ok, " | ".join(lines)


def _check_feature_engineering(res: PreprocessedData) -> Check:
    fe = FeatureEngineer(FeatureEngineeringConfig()).fit(res.X_train)
    train, test = fe.transform(res.X_train), fe.transform(res.X_test)
    eng = fe.engineered_features_
    problems = []
    if not np.isfinite(train.to_numpy(float)).all() or not np.isfinite(test.to_numpy(float)).all():
        problems.append("non-finite values")
    if not train.columns.is_unique:
        problems.append("duplicate column names")
    if list(train.columns) != list(test.columns):
        problems.append("train/test columns differ")
    varying = train[eng].std(ddof=0) > 0
    if not np.allclose(train[eng].mean()[varying], 0, atol=1e-8):
        problems.append("engineered training columns are not centred")
    if abs(test["crime_sq"].mean()) < 1e-6:
        problems.append("test set appears re-centred on its own statistics")
    again = FeatureEngineer(FeatureEngineeringConfig()).fit(res.X_train).transform(res.X_test)
    if not again.equals(test):
        problems.append("not deterministic")
    groups = fe.feature_catalog_["group"].value_counts().to_dict()
    return (
        "Feature engineering is finite, leak-free and deterministic",
        not problems,
        ("; ".join(problems) or f"{len(eng)} engineered features {groups}"),
    )


def _check_feature_pipeline(res: PreprocessedData) -> Check:
    """The engineered + selected pipeline must beat the baseline on the held-out test set."""
    cfg = res.config
    scores = {}
    for label, fe_cfg, selector in (("baseline", None, None), ("engineered + vif", FeatureEngineeringConfig(), "vif")):
        model = make_full_pipeline(LinearRegression(), cfg, fe_cfg, selector).fit(res.raw_X_train, res.y_train)
        pred = model.predict(res.raw_X_test)
        scores[label] = (
            mean_squared_error(res.y_test, pred) ** 0.5,
            r2_score(res.y_test, pred),
            np.isfinite(pred).all(),
        )
    ok = all(v[2] for v in scores.values()) and scores["engineered + vif"][0] < scores["baseline"][0]
    detail = " | ".join(f"[{k}] test RMSE = {v[0]:,.0f}, R2 = {v[1]:.3f}" for k, v in scores.items())
    return "Engineered + selected pipeline beats the baseline on the test set", bool(ok), detail


def _check_metrics() -> Check:
    """Metrics must match values worked out by hand (errors 10, -10, 30, -20 on prices 100..400)."""
    m = regression_metrics([100.0, 200.0, 300.0, 400.0], [110.0, 190.0, 330.0, 380.0], n_features=1)
    expected = {
        "rmse": np.sqrt(375.0),
        "mae": 17.5,
        "mape": 7.5,
        "r2": 0.97,
        "adj_r2": 0.955,
        "bias": 2.5,
        "median_ae": 15.0,
        "max_error": 30.0,
        "within_10pct": 100.0,
    }
    wrong = {k: (m[k], v) for k, v in expected.items() if not np.isclose(m[k], v)}
    return (
        "Evaluation metrics match hand-computed values",
        not wrong,
        (
            f"mismatch {wrong}"
            if wrong
            else f"{len(expected)} metrics checked (RMSE {m['rmse']:.3f}, MAE {m['mae']:.1f}, R2 {m['r2']:.2f})"
        ),
    )


def _check_baseline_ladder(res: PreprocessedData) -> Check:
    """The baseline ladder must be ordered: each rung is better than the one below it on the test set."""
    specs = baseline_specs(res.config)
    rmse = {}
    r2 = {}
    for name, spec in specs.items():
        pred = spec.build().fit(res.raw_X_train, res.y_train).predict(res.raw_X_test)
        scores = regression_metrics(res.y_test, pred)
        rmse[name], r2[name] = scores["rmse"], scores["r2"]
    names = list(specs)
    floor_r2 = r2[names[0]]
    ordered = all(rmse[names[i]] >= rmse[names[i + 1]] for i in range(1, len(names) - 1))
    ok = abs(floor_r2) < 0.1 and ordered and rmse[REFERENCE_BASELINE] == min(rmse.values())
    detail = " > ".join(f"{rmse[n]:,.0f}" for n in names[1:])
    return (
        "Baseline ladder is ordered and the mean predictor has R2 near 0",
        bool(ok),
        (f"mean predictor R2 = {floor_r2:.3f}; test RMSE down the ladder: {detail}"),
    )


def run_validation(res: PreprocessedData) -> Tuple[bool, List[Dict[str, Any]]]:
    checks: List[Callable[[], Check]] = [
        lambda: _check_output_quality(res),
        lambda: _check_scaling(res),
        lambda: _check_no_leakage(res),
        lambda: _check_log_effect(res),
        lambda: _check_outlier_clipping(res),
        lambda: _check_robustness(res),
        lambda: _check_imputation_report(res),
        lambda: _check_target(res),
        lambda: _check_persistence(res),
        lambda: _check_split(res),
        lambda: _check_cleaning(res),
        _check_config,
        lambda: _check_model(res),
        lambda: _check_feature_engineering(res),
        lambda: _check_feature_pipeline(res),
        _check_metrics,
        lambda: _check_baseline_ladder(res),
    ]
    results: List[Check] = []
    for fn in checks:
        try:
            results.append(fn())
        except Exception as exc:  # noqa: BLE001
            results.append(("Check crashed", False, repr(exc)))
    print("\nValidation results")
    print("-" * 78)
    for name, passed, detail in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}\n       {detail}")
    n_pass = sum(p for _, p, _ in results)
    print("-" * 78)
    print(f"{n_pass}/{len(results)} checks passed")
    return n_pass == len(results), [{"check": n, "passed": bool(p), "detail": d} for n, p, d in results]


# --------------------------------------------------------------------------- #
# Settings: config file + command-line overrides
# --------------------------------------------------------------------------- #
#: command-line option -> (config section, key). Options left unset on the command line (None)
#: do not override the config file or the defaults.
OVERRIDES: Dict[str, Tuple[str, str]] = {
    "test_size": ("preprocessing", "test_size"),
    "seed": ("preprocessing", "random_state"),
    "scaler": ("preprocessing", "scaler"),
    "outlier_strategy": ("preprocessing", "outlier_strategy"),
    "outlier_iqr_factor": ("preprocessing", "outlier_iqr_factor"),
    "target_transform": ("preprocessing", "target_transform"),
    "numeric_imputer": ("preprocessing", "numeric_imputer_strategy"),
    "categorical_imputer": ("preprocessing", "categorical_imputer_strategy"),
    "missing_indicator": ("preprocessing", "add_missing_indicator"),
    "feature_selection": ("preprocessing", "feature_selection"),
    "output_dtype": ("preprocessing", "output_dtype"),
    "categorical_encoding": ("preprocessing", "categorical_encoding"),
    "clip_z": ("feature_engineering", "clip_z"),
    "selection_method": ("selection", "method"),
    "cv_folds": ("cv", "folds"),
    "cv_repeats": ("cv", "repeats"),
}


@dataclass
class Settings:
    """Fully resolved experiment settings (defaults < config file < command line)."""

    preprocessing: PreprocessingConfig
    feature_engineering: FeatureEngineeringConfig
    selection: str
    cv_folds: int
    cv_repeats: int
    data: Path

    def as_config(self) -> Dict[str, Any]:
        """The resolved configuration in the same layout a config file uses."""
        return {
            "run": {"data": str(self.data)},
            "preprocessing": self.preprocessing.to_dict(),
            "feature_engineering": self.feature_engineering.to_dict(),
            "selection": {"method": self.selection},
            "cv": {"folds": self.cv_folds, "repeats": self.cv_repeats},
        }


def resolve_settings(args: argparse.Namespace) -> Settings:
    """Merge defaults, the optional config file and explicit command-line options (in that order)."""
    file_cfg = load_config(args.config) if args.config else {}
    sections: Dict[str, Dict[str, Any]] = {
        name: dict(file_cfg.get(name) or {}) for name in ("preprocessing", "feature_engineering", "selection", "cv")
    }
    for dest, (section, key) in OVERRIDES.items():
        value = getattr(args, dest, None)
        if value is None:
            continue
        if dest == "clip_z" and value <= 0:
            value = None  # 0 switches winsorising off
        sections[section][key] = value

    data = args.data or (file_cfg.get("run") or {}).get("data") or ROOT / "Housing_Price_Dataset.csv"
    data = Path(data)
    if not data.is_absolute():
        data = (Path.cwd() / data) if (Path.cwd() / data).exists() else ROOT / data
    return Settings(
        preprocessing=PreprocessingConfig.from_dict(sections["preprocessing"]),
        feature_engineering=FeatureEngineeringConfig.from_dict(sections["feature_engineering"]),
        selection=str(sections["selection"].get("method", "vif")),
        cv_folds=int(sections["cv"].get("folds", 5)),
        cv_repeats=int(sections["cv"].get("repeats", 3)),
        data=data,
    )


@dataclass
class RunPaths:
    """Where this run writes its files (versioned sub-folders unless versioning is off)."""

    processed: Path
    artifacts: Path
    reports: Path
    reports_base: Path

    @classmethod
    def create(cls, args: argparse.Namespace, run_id: str) -> RunPaths:
        on = not args.no_versioning
        return cls(
            processed=versioned_dir(args.output_dir, run_id, on),
            artifacts=versioned_dir(args.artifact_dir, run_id, on),
            reports=versioned_dir(args.reports_dir, run_id, on),
            reports_base=Path(args.reports_dir),
        )


# --------------------------------------------------------------------------- #
# Baseline benchmark
# --------------------------------------------------------------------------- #
def run_baseline_stage(
    res: PreprocessedData,
    settings: Settings,
    paths: Optional[RunPaths],
    recorder: RunRecorder,
    include_candidates: bool,
) -> Dict[str, Any]:
    """Benchmark the baseline ladder (and optionally challengers) and describe the reference model's errors."""
    cfg = res.config
    specs = baseline_specs(cfg)
    if include_candidates:
        specs.update(candidate_specs(cfg, settings.feature_engineering, cfg.random_state))
    result = run_benchmark(
        res.raw_X_train,
        res.y_train,
        res.raw_X_test,
        res.y_test,
        specs,
        n_splits=settings.cv_folds,
        n_repeats=settings.cv_repeats,
        random_state=cfg.random_state,
    )

    print("\n" + "=" * 78)
    print(f"BASELINE BENCHMARK ({settings.cv_folds}-fold CV x {settings.cv_repeats} repeats on the training set)")
    print("=" * 78)
    cols = [
        "rmse",
        "mae",
        "mape",
        "r2",
        "within_10pct",
        "skill_vs_mean_%",
        "rmse_gain_vs_reference_%",
        "p_value_vs_reference",
    ]
    shown = result.comparison[cols].copy()
    shown[["rmse", "mae"]] = shown[["rmse", "mae"]].round(0)
    shown[["mape", "r2", "within_10pct", "skill_vs_mean_%", "rmse_gain_vs_reference_%"]] = shown[
        ["mape", "r2", "within_10pct", "skill_vs_mean_%", "rmse_gain_vs_reference_%"]
    ].round(3)
    shown["p_value_vs_reference"] = shown["p_value_vs_reference"].map(lambda p: "" if pd.isna(p) else f"{p:.4f}")
    print(shown.to_string())
    print("\nmape and within_10pct are in percent. skill_vs_mean_% = RMSE reduction versus predicting the mean price;")
    print("rmse_gain_vs_reference_% = RMSE reduction versus the reference (negative = worse); the p-value uses the")
    print("corrected resampled t-test (cautious: it accounts for overlapping training sets).")

    print("\nHold-out test set (each model evaluated once, after cross-validation):")
    hold = result.holdout[["rmse", "mae", "mape", "r2", "within_10pct", "bias", "rmsle", "fit_seconds"]].copy()
    hold[["rmse", "mae", "bias"]] = hold[["rmse", "mae", "bias"]].round(0)
    print(hold.round(3).to_string())

    ref = result.reference
    print(f"\nWhere does the reference baseline ('{ref}') make its errors? (hold-out, residual = prediction - truth)")
    print(result.residuals["by_price_band"].round(2).to_string())
    if "by_group" in result.residuals:
        print()
        print(result.residuals["by_group"].round(2).to_string())
    trend = result.residuals["error_trend"].iloc[0]
    print(
        f"\nError grows with price: the correlation of absolute error with true price is {trend.iloc[0]:.2f}; "
        f"the most expensive fifth of homes has {trend.iloc[1]:.1f}x the MAE of the cheapest fifth."
    )

    if paths is not None:
        written = result.save(paths.reports)
        BaselineModel(cfg).fit(res.raw_X_train, res.y_train).save(paths.artifacts / "baseline_model.joblib")
        print(f"\nSaved to {paths.reports}/: {', '.join(p.name for p in written.values())}")
        print(f"Saved the fitted baseline model to {paths.artifacts}/baseline_model.joblib")
    recorder.add("baseline", result.to_dict())
    floor = next(n for n, s in result.specs.items() if s.tier == "floor")
    return {
        "baseline_cv_rmse": _num(result.cv_table.loc[ref, "rmse"]),
        "baseline_holdout_rmse": _num(result.holdout.loc[ref, "rmse"]),
        "baseline_holdout_r2": _num(result.holdout.loc[ref, "r2"]),
        "mean_predictor_holdout_rmse": _num(result.holdout.loc[floor, "rmse"]),
    }


# --------------------------------------------------------------------------- #
# Feature engineering stages
# --------------------------------------------------------------------------- #
def _holdout_metrics(model, res: PreprocessedData) -> Tuple[float, float, float]:
    pred = model.predict(res.raw_X_test)
    return (
        mean_squared_error(res.y_test, pred) ** 0.5,
        mean_absolute_error(res.y_test, pred),
        r2_score(res.y_test, pred),
    )


def run_feature_stage(
    res: PreprocessedData, settings: Settings, paths: Optional[RunPaths], recorder: RunRecorder
) -> Dict[str, Any]:
    """Engineer and select features on the preprocessed data (fitted on the training set only)."""
    cfg = res.config
    fe = FeatureEngineer(settings.feature_engineering).fit(res.X_train)
    train_f, test_f = fe.transform(res.X_train), fe.transform(res.X_test)

    methods = settings.selection.split("+")
    bad = [x for x in methods if x not in SELECTION_METHODS]
    if bad:
        raise SystemExit(f"Unknown selection method(s) {bad}; choose from {SELECTION_METHODS}")
    protect = tuple(fe.base_features_)
    prefix = settings.feature_engineering.group_prefix
    selectors, train_s, test_s = [], train_f, test_f
    for method in methods:
        sel = FeatureSelector(
            method=method,
            random_state=cfg.random_state,
            always_keep=protect,
            always_keep_prefixes=(prefix,) if prefix else (),
        )
        sel.fit(train_s, res.y_train_transformed)
        train_s, test_s = sel.transform(train_s), sel.transform(test_s)
        selectors.append(sel)
    selected = list(train_s.columns)
    dropped = [c for c in train_f.columns if c not in selected]
    report = selectors[-1].selection_report_

    print("\n" + "=" * 78)
    print("FEATURE ENGINEERING AND SELECTION")
    print("=" * 78)
    counts = fe.feature_catalog_["group"].value_counts().to_dict()
    print(
        f"Created {len(fe.engineered_features_)} engineered features from "
        f"{res.X_train.shape[1]} preprocessed columns: {counts}"
    )
    kept_eng = [f for f in selected if f in fe.engineered_features_]
    print(
        f"Selection '{settings.selection}' (original features protected) kept {len(selected)} of "
        f"{train_f.shape[1]} columns ({len(kept_eng)} engineered)."
    )
    print("Dropped:", ", ".join(dropped) if dropped else "nothing")
    top = report.query("kept").sort_values("score", ascending=False).head(10)
    print(f"\nTop kept features by the last selector's score ({selectors[-1].method}):")
    print(top[["score"]].round(4).to_string())

    summary = {
        "engineered": len(fe.engineered_features_),
        "groups": counts,
        "selection": settings.selection,
        "kept": selected,
        "dropped": dropped,
    }
    recorder.add("features", {**summary, "catalog": fe.feature_catalog_, "selection_report": report})
    if paths is not None:
        train_s.to_csv(paths.processed / "features_X_train.csv", index_label="row_id")
        test_s.to_csv(paths.processed / "features_X_test.csv", index_label="row_id")
        fe.feature_catalog_.to_csv(paths.reports / "feature_catalog.csv")
        report.to_csv(paths.reports / "feature_selection_report.csv")
        pipeline = make_full_pipeline(LinearRegression(), cfg, settings.feature_engineering, methods)
        pipeline.fit(res.raw_X_train, res.y_train)
        joblib.dump(pipeline, paths.artifacts / "feature_pipeline.joblib")
        print(
            f"\nSaved engineered data to {paths.processed}/, reports to {paths.reports}/, "
            f"and the fitted end-to-end pipeline to {paths.artifacts}/feature_pipeline.joblib"
        )
    return summary


def run_feature_comparison(
    res: PreprocessedData, settings: Settings, paths: Optional[RunPaths], recorder: RunRecorder, compare_models: bool
) -> Dict[str, Any]:
    """Compare feature sets by repeated cross-validation on the training data, then test once."""
    cfg = res.config
    scenarios = default_scenarios(settings.feature_engineering)
    print("\n" + "=" * 78)
    print(
        f"FEATURE-SET COMPARISON ({settings.cv_folds}-fold CV x {settings.cv_repeats} repeats "
        "on the training set; LinearRegression, log target)"
    )
    print("=" * 78)
    table = evaluate_feature_sets(
        res.raw_X_train,
        res.y_train,
        scenarios,
        preprocessing_config=cfg,
        n_splits=settings.cv_folds,
        n_repeats=settings.cv_repeats,
        random_state=cfg.random_state,
    )
    shown = table[["n_features", "rmse", "mae", "r2", "rmse_gain_%", "p_value"]].copy()
    shown["rmse"], shown["mae"] = shown["rmse"].round(0), shown["mae"].round(0)
    shown[["n_features", "r2", "rmse_gain_%"]] = shown[["n_features", "r2", "rmse_gain_%"]].round(3)
    shown["p_value"] = shown["p_value"].map(lambda p: "" if pd.isna(p) else f"{p:.4f}")
    print(shown.to_string())
    print(
        "\nrmse_gain_% = RMSE reduction versus the baseline (positive is better); p_value = corrected "
        "resampled t-test (cautious: it accounts for overlapping training sets)."
    )

    baseline = table.attrs["baseline"]
    best = select_best_scenario(table, one_se_rule=True)
    lowest = select_best_scenario(table, one_se_rule=False)
    print(f"\nLowest mean CV RMSE: '{lowest}'. Simplest scenario within one standard error of it: '{best}'.")

    print("\nHold-out test set (used once, after the choices above; both selection rules are reported):")
    rows = {}
    candidates = [("baseline", baseline), (f"lowest CV RMSE: {lowest}", lowest)]
    if best != lowest:
        candidates.append((f"simplest within 1 SE: {best}", best))
    for label, name in candidates:
        spec = scenarios[name]
        model = make_full_pipeline(LinearRegression(), cfg, spec["fe"], spec["selector"]).fit(
            res.raw_X_train, res.y_train
        )
        rmse, mae, r2 = _holdout_metrics(model, res)
        rows[label] = {"rmse": round(rmse), "mae": round(mae), "r2": round(r2, 4)}
    holdout = pd.DataFrame(rows).T
    holdout["rmse_gain_%"] = (
        100 * (holdout.loc["baseline", "rmse"] - holdout["rmse"]) / holdout.loc["baseline", "rmse"]
    ).round(1)
    print(holdout.to_string())

    extra: Dict[str, Any] = {}
    if compare_models:
        print("\n" + "-" * 78)
        print("DOES FEATURE ENGINEERING HELP OTHER MODELS? (CV on the training set, 1 repeat for the tree models)")
        print("-" * 78)
        compact = {k: scenarios[k] for k in (baseline, "all engineered", "all + vif")}
        for label, model, repeats in (
            ("Ridge(alpha=10)", Ridge(alpha=10), settings.cv_repeats),
            ("GradientBoosting", GradientBoostingRegressor(random_state=cfg.random_state), 1),
            ("RandomForest", RandomForestRegressor(n_estimators=200, random_state=cfg.random_state, n_jobs=-1), 1),
        ):
            t = evaluate_feature_sets(
                res.raw_X_train,
                res.y_train,
                compact,
                model=model,
                preprocessing_config=cfg,
                n_splits=settings.cv_folds,
                n_repeats=repeats,
                random_state=cfg.random_state,
            )
            print(f"\n{label}:")
            print(t[["n_features", "rmse", "r2", "rmse_gain_%", "p_value"]].round(3).to_string())
            extra[label] = t.to_dict(orient="index")

    recorder.add(
        "comparison",
        {
            "table": table,
            "fold_rmse": table.attrs["fold_rmse"],
            "baseline": baseline,
            "lowest_cv_rmse": lowest,
            "simplest_within_one_se": best,
            "holdout": holdout,
            "other_models": extra,
        },
    )
    if paths is not None:
        table.to_csv(paths.reports / "feature_set_comparison.csv")
        holdout.to_csv(paths.reports / "holdout_comparison.csv")
        print(f"\nSaved comparison tables to {paths.reports}/")
    return {
        "best_scenario": lowest,
        "cv_rmse_best": _num(table.loc[lowest, "rmse"]),
        "cv_rmse_baseline": _num(table.loc[baseline, "rmse"]),
        "holdout_rmse_best": _num(holdout.iloc[1]["rmse"]),
        "holdout_rmse_baseline": float(holdout.loc["baseline", "rmse"]),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    """Options that map to settings default to ``None`` so that they only override a config file when given."""
    p = argparse.ArgumentParser(description="Housing-price preprocessing, feature engineering and evaluation pipeline")
    p.add_argument("--config", type=Path, help="YAML or JSON experiment config (command-line options override it)")
    p.add_argument("--save-config", type=Path, metavar="PATH", help="write the resolved configuration to PATH and exit")
    p.add_argument("--data", type=Path, default=None, help="path to the raw CSV (default: Housing_Price_Dataset.csv)")
    p.add_argument("--output-dir", type=Path, default=ROOT / "processed", help="where to write processed data")
    p.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts", help="where to save fitted objects")
    p.add_argument("--reports-dir", type=Path, default=ROOT / "reports", help="where to write reports and run records")
    p.add_argument("--run-label", default=None, help="optional text appended to the run ID")
    p.add_argument(
        "--no-versioning", action="store_true", help="write into the base folders instead of run_id sub-folders"
    )
    p.add_argument("--profile", action="store_true", help="also track peak Python memory per stage (slower)")
    p.add_argument("--test-size", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--scaler", choices=["standard", "robust", "minmax", "none"], default=None)
    p.add_argument("--outlier-strategy", choices=["clip", "none"], default=None)
    p.add_argument("--outlier-iqr-factor", type=float, default=None)
    p.add_argument("--target-transform", choices=["log", "boxcox", "none"], default=None)
    p.add_argument("--numeric-imputer", choices=["median", "mean", "constant", "knn"], default=None)
    p.add_argument("--categorical-imputer", choices=["most_frequent", "constant"], default=None)
    p.add_argument(
        "--categorical-encoding",
        choices=["onehot", "frequency", "hashing", "target", "auto"],
        default=None,
        help="encoding for categorical columns ('auto' picks by cardinality)",
    )
    p.add_argument(
        "--missing-indicator",
        action="store_true",
        default=None,
        help="add 0/1 flag columns for features with missing values",
    )
    p.add_argument(
        "--feature-selection",
        choices=["none", "variance", "lasso"],
        default=None,
        help="selection built into the preprocessor (see --selection-method for the feature-engineering stage)",
    )
    p.add_argument("--output-dtype", choices=["float64", "float32"], default=None)
    p.add_argument(
        "--baseline", action="store_true", help="benchmark the baseline models (mean predictor ... preprocessed linear)"
    )
    p.add_argument(
        "--candidates",
        action="store_true",
        help="with --baseline: also benchmark Ridge, Lasso, trees and engineered features",
    )
    p.add_argument("--features", action="store_true", help="engineer and select features after preprocessing")
    p.add_argument(
        "--selection-method",
        default=None,
        help=f"feature selection technique, one of {', '.join(SELECTION_METHODS)}; "
        "chain several with '+' (e.g. vif+rfe)",
    )
    p.add_argument(
        "--clip-z",
        type=float,
        default=None,
        help="winsorise standardised features at +/- this many SD before engineering (0 = off)",
    )
    p.add_argument(
        "--compare-features", action="store_true", help="cross-validated comparison of feature sets + test check"
    )
    p.add_argument(
        "--compare-models", action="store_true", help="with --compare-features: also try Ridge and tree models"
    )
    p.add_argument("--cv-folds", type=int, default=None)
    p.add_argument("--cv-repeats", type=int, default=None)
    p.add_argument("--validate", action="store_true", help="run the validation suite and a baseline model")
    p.add_argument("--no-save", action="store_true", help="do not write any files")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")
    settings = resolve_settings(args)
    resolved = settings.as_config()
    if args.save_config:
        print(f"Wrote resolved configuration to {save_config(resolved, args.save_config)}")
        return 0

    cfg = settings.preprocessing
    run_id = make_run_id(resolved, label=args.run_label)
    if not args.no_save and not args.no_versioning:
        run_id = ensure_unique_run_id(run_id, [args.output_dir, args.artifact_dir, args.reports_dir])
    recorder = RunRecorder(run_id, resolved, argv)
    profiler = StageProfiler(track_memory=args.profile)
    paths = None if args.no_save else RunPaths.create(args, run_id)
    print(f"Run ID: {run_id}")

    with profiler.stage("load data"):
        raw = load_data(settings.data)
    recorder.set_data(settings.data, raw)
    print(f"Loaded {raw.shape[0]} rows x {raw.shape[1]} columns")
    print("\nMissing values in raw data:")
    missing = missing_value_report(raw).query("missing > 0")
    print(missing.to_string() if not missing.empty else "  none")

    with profiler.stage("preprocessing"):
        res = run_preprocessing(raw, cfg)
    print("\nCleaning report:", res.cleaning_report)
    print(f"Train: {res.X_train.shape}, Test: {res.X_test.shape}")
    print("\nOutput features:", ", ".join(res.X_train.columns))
    pre = res.preprocessor
    print("\nImputation report (training data):")
    print(pre.imputation_report_.to_string())
    if not pre.encoding_report_.empty:
        print("\nCategorical encoding:")
        print(pre.encoding_report_.to_string())
    if cfg.feature_selection != "none":
        print(f"\nFeature selection ({cfg.feature_selection}) dropped: {pre.dropped_features_ or 'nothing'}")
    print("\nDistribution of the stratification column across the split:")
    if "stratify_column" in res.split_report:
        print(res.split_report["stratify_column"].to_string())
    print("\nTarget distribution in train and test:")
    print(res.split_report["target"].to_string())
    if not pre.outlier_bounds_.empty:
        table = pre.outlier_bounds_.copy()
        table["train_values_clipped"] = pre.count_clipped(res.raw_X_train)
        print("\nOutlier fences learned on the training set:")
        print(table.round(3).to_string())
    recorder.add(
        "preprocessing",
        {
            "cleaning_report": res.cleaning_report,
            "train_shape": res.X_train.shape,
            "test_shape": res.X_test.shape,
            "output_features": list(res.X_train.columns),
            "fit_summary": pre.fit_summary_,
            "imputation_report": pre.imputation_report_,
            "encoding_report": pre.encoding_report_,
            "outlier_bounds": pre.outlier_bounds_,
            "split_report": res.split_report,
        },
    )

    summary: Dict[str, Any] = {"rows": raw.shape[0], "output_features": res.X_train.shape[1]}
    if paths is not None:
        with profiler.stage("save preprocessed data"):
            res.X_train.to_csv(paths.processed / "X_train.csv", index_label="row_id")
            res.X_test.to_csv(paths.processed / "X_test.csv", index_label="row_id")
            pd.DataFrame({cfg.target: res.y_train, f"{cfg.target}_transformed": res.y_train_transformed}).to_csv(
                paths.processed / "y_train.csv", index_label="row_id"
            )
            pd.DataFrame({cfg.target: res.y_test, f"{cfg.target}_transformed": res.y_test_transformed}).to_csv(
                paths.processed / "y_test.csv", index_label="row_id"
            )
            saved = pre.save(paths.artifacts / "preprocessor.joblib")
        print(f"\nSaved processed data to {paths.processed}/ and the fitted preprocessor to {saved}")

    if args.baseline or args.candidates:
        with profiler.stage("baseline benchmark"):
            summary.update(run_baseline_stage(res, settings, paths, recorder, args.candidates))
    if args.features:
        with profiler.stage("feature engineering + selection"):
            stage = run_feature_stage(res, settings, paths, recorder)
        summary.update(
            {
                "selection": settings.selection,
                "engineered_features": stage["engineered"],
                "kept_features": len(stage["kept"]),
            }
        )
    if args.compare_features:
        with profiler.stage("feature-set comparison"):
            summary.update(run_feature_comparison(res, settings, paths, recorder, args.compare_models))

    exit_code = 0
    if args.validate:
        with profiler.stage("validation"):
            ok, checks = run_validation(res)
        recorder.add("validation", {"passed": ok, "checks": checks})
        summary["validation_passed"] = ok
        exit_code = 0 if ok else 1

    print("\nStage timings" + (" and peak Python memory" if args.profile else "") + ":")
    print(profiler.table().to_string())
    print(f"Total: {profiler.total_seconds():.1f}s")

    if paths is not None:
        save_config(resolved, paths.reports / "config.yaml")
        record_path = recorder.write(paths.reports, profiler)
        registry = recorder.register(paths.reports_base, {**summary, "total_seconds": profiler.total_seconds()})
        print(
            f"Run record: {record_path}\nResolved config (rerun with --config): {paths.reports / 'config.yaml'}"
            f"\nRun registry: {registry}"
        )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
