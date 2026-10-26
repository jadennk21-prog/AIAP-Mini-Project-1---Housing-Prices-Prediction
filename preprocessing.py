"""
Data preprocessing pipeline for the housing-price regression project.

The design follows the findings of the exploratory data analysis (``eda.ipynb``):

* No missing values or duplicates were found in the raw data, but the pipeline still
  handles both so that it stays reusable on new or messier extracts.
* ``crime_rate_in_area``, ``property_size`` and ``proximity_to_public_transport`` are
  strongly right-skewed, and the log of crime is much more linear against price,
  so these three features are log-transformed.
* The IQR outliers in the features and in the target look genuine, not erroneous.
  The target is therefore never filtered; features are only *clipped* at extreme
  (3 x IQR) fences to limit leverage, and this can be switched off.
* ``number_of_bedrooms`` and ``number_of_bathrooms`` are already z-scores, so they are
  imputed but not rescaled again.
* ``house_sale_price`` is right-skewed; a log (or Box-Cox) target is provided as a
  reversible target transformer.
* ``location_type`` is a three-level nominal variable with a small Rural group, so
  it is one-hot encoded and the train/test split is stratified on it.

Everything that learns from data (imputation values, outlier fences, log floors,
scaler statistics, encoder categories) is fitted on the training set only and then
reused unchanged, which prevents data leakage.

Typical use::

    from src.preprocessing import PreprocessingConfig, load_data, run_preprocessing

    cfg = PreprocessingConfig()
    result = run_preprocessing(load_data("Housing_Price_Dataset.csv"), cfg)
    result.X_train, result.y_train_transformed

or, inside a model pipeline::

    from sklearn.linear_model import Ridge
    model = make_model_pipeline(Ridge(alpha=1.0), cfg)
    model.fit(X_train, y_train)          # y on the original price scale
    model.predict(X_test)                # predictions back on the price scale
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils.validation import check_is_fitted

__all__ = [
    "PreprocessingConfig",
    "IQROutlierClipper",
    "SafeLogTransformer",
    "DataPreprocessor",
    "PreprocessedData",
    "load_data",
    "clean_data",
    "missing_value_report",
    "split_data",
    "build_target_transformer",
    "make_model_pipeline",
    "run_preprocessing",
]

logger = logging.getLogger(__name__)

_SCALERS = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}
_VALID_OUTLIER = ("clip", "none")
_VALID_TARGET = ("log", "boxcox", "none")
_VALID_NUM_IMPUTE = ("median", "mean")
_VALID_CAT_IMPUTE = ("most_frequent",)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessingConfig:
    """All tunable choices of the pipeline in one place.

    Defaults encode the conclusions of the EDA notebook.
    """

    # Column roles
    target: str = "house_sale_price"
    categorical_features: List[str] = field(default_factory=lambda: ["location_type"])
    log_features: List[str] = field(
        default_factory=lambda: [
            "crime_rate_in_area",
            "property_size",
            "proximity_to_public_transport",
        ]
    )
    #: Already z-scored in the raw data: imputed only, never rescaled again.
    standardised_features: List[str] = field(
        default_factory=lambda: ["number_of_bedrooms", "number_of_bathrooms"]
    )
    #: Remaining numeric predictors. ``None`` means "infer every other numeric column".
    numeric_features: Optional[List[str]] = None

    # Cleaning and splitting
    drop_duplicates: bool = True
    test_size: float = 0.2
    random_state: int = 42
    stratify_column: Optional[str] = "location_type"

    # Missing values
    numeric_imputer_strategy: str = "median"
    categorical_imputer_strategy: str = "most_frequent"

    # Outliers: "clip" winsorises features at Q1 - k*IQR / Q3 + k*IQR, "none" disables it.
    outlier_strategy: str = "clip"
    outlier_iqr_factor: float = 3.0

    # Scaling of numeric features: "standard", "robust", "minmax" or "none"
    scaler: str = "standard"

    # Encoding
    drop_first_category: bool = True

    # Target: "log", "boxcox" or "none"
    target_transform: str = "log"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` for an inconsistent configuration."""
        if self.outlier_strategy not in _VALID_OUTLIER:
            raise ValueError(f"outlier_strategy must be one of {_VALID_OUTLIER}")
        if self.scaler not in (*_SCALERS, "none"):
            raise ValueError(f"scaler must be one of {(*_SCALERS, 'none')}")
        if self.target_transform not in _VALID_TARGET:
            raise ValueError(f"target_transform must be one of {_VALID_TARGET}")
        if self.numeric_imputer_strategy not in _VALID_NUM_IMPUTE:
            raise ValueError(f"numeric_imputer_strategy must be one of {_VALID_NUM_IMPUTE}")
        if self.categorical_imputer_strategy not in _VALID_CAT_IMPUTE:
            raise ValueError(f"categorical_imputer_strategy must be one of {_VALID_CAT_IMPUTE}")
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1")
        if self.outlier_iqr_factor <= 0:
            raise ValueError("outlier_iqr_factor must be positive")
        roles = (
            list(self.categorical_features)
            + list(self.log_features)
            + list(self.standardised_features)
            + list(self.numeric_features or [])
        )
        duplicated = sorted({c for c in roles if roles.count(c) > 1})
        if duplicated:
            raise ValueError(f"Columns assigned to more than one role: {duplicated}")
        if self.target in roles:
            raise ValueError("The target column cannot also be a feature")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Custom transformers
# --------------------------------------------------------------------------- #
def _feature_names(transformer: BaseEstimator, n_features: int, input_features) -> np.ndarray:
    """Resolve output feature names for a one-to-one transformer."""
    if input_features is not None:
        return np.asarray(input_features, dtype=object)
    if hasattr(transformer, "feature_names_in_"):
        return np.asarray(transformer.feature_names_in_, dtype=object)
    return np.asarray([f"x{i}" for i in range(n_features)], dtype=object)


class IQROutlierClipper(BaseEstimator, TransformerMixin):
    """Clip each column to ``[Q1 - factor*IQR, Q3 + factor*IQR]`` learned on the training data.

    Outliers are capped, not removed, so no rows are lost and NaNs pass through untouched
    (they are imputed in a separate step). Columns with a zero IQR are left unchanged.
    """

    def __init__(self, factor: float = 3.0):
        self.factor = factor

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        q1 = np.nanpercentile(arr, 25, axis=0)
        q3 = np.nanpercentile(arr, 75, axis=0)
        iqr = q3 - q1
        lower = q1 - self.factor * iqr
        upper = q3 + self.factor * iqr
        degenerate = iqr == 0
        self.lower_ = np.where(degenerate, -np.inf, lower)
        self.upper_ = np.where(degenerate, np.inf, upper)
        self.n_features_in_ = arr.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, ["lower_", "upper_"])
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return np.clip(arr, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["lower_"])
        return _feature_names(self, self.n_features_in_, input_features)


class SafeLogTransformer(BaseEstimator, TransformerMixin):
    """Natural log that cannot fail on zero or negative values.

    ``fit`` records the smallest strictly positive value of every column; at transform time
    values below that floor are raised to it before taking the log. This keeps the exact
    ``log(x)`` shape that worked best in the EDA (``log1p`` would be almost the identity for
    a feature like crime rate, whose values are all below 1) without producing ``-inf``.
    """

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        floors = []
        for j in range(arr.shape[1]):
            col = arr[:, j]
            positive = col[np.isfinite(col) & (col > 0)]
            floors.append(positive.min() if positive.size else 1.0)
        self.floor_ = np.asarray(floors, dtype=float)
        self.n_features_in_ = arr.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, ["floor_"])
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return np.log(np.maximum(arr, self.floor_))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["floor_"])
        return _feature_names(self, self.n_features_in_, input_features)


# --------------------------------------------------------------------------- #
# Cleaning, reporting and splitting helpers
# --------------------------------------------------------------------------- #
def load_data(path: Union[str, Path]) -> pd.DataFrame:
    """Read the raw CSV file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded %s rows and %s columns from %s", *df.shape, path)
    return df


def _normalise_text(series: pd.Series) -> pd.Series:
    """Strip whitespace and normalise case, keeping missing values as NaN."""
    text = series.astype("string").str.strip().str.title()
    return text.astype(object).where(text.notna(), np.nan)


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of missing values per column (sorted, worst first)."""
    report = pd.DataFrame({"missing": df.isna().sum(), "missing_pct": df.isna().mean() * 100})
    return report.sort_values("missing", ascending=False).round(2)


def clean_data(
    df: pd.DataFrame, config: Optional[PreprocessingConfig] = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Row-level cleaning that does not learn anything from the data.

    * strips column names and normalises category text (``" rural "`` -> ``"Rural"``),
    * coerces numeric columns (unparseable text becomes NaN and is imputed later),
    * removes exact duplicate rows (optional),
    * removes rows whose target is missing, or non-positive when a log / Box-Cox
      target transform is requested (those values cannot be transformed).

    Because nothing is learned here, it is safe to run before the train/test split.
    Returns the cleaned frame and a dictionary of what was changed.
    """
    cfg = config or PreprocessingConfig()
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    if cfg.target not in out.columns:
        raise KeyError(f"Target column '{cfg.target}' not found. Columns: {list(out.columns)}")

    report: Dict[str, int] = {"rows_in": len(out)}

    for col in out.columns:
        if col in cfg.categorical_features:
            out[col] = _normalise_text(out[col])
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if cfg.drop_duplicates:
        before = len(out)
        out = out.drop_duplicates()
        report["duplicates_removed"] = before - len(out)
    else:
        report["duplicates_removed"] = 0

    before = len(out)
    out = out[out[cfg.target].notna()]
    report["missing_target_removed"] = before - len(out)

    if cfg.target_transform in ("log", "boxcox"):
        before = len(out)
        out = out[out[cfg.target] > 0]
        report["non_positive_target_removed"] = before - len(out)
    else:
        report["non_positive_target_removed"] = 0

    out = out.reset_index(drop=True)
    report["rows_out"] = len(out)
    for key, value in report.items():
        if key not in ("rows_in", "rows_out") and value:
            logger.warning("clean_data: %s = %s", key, value)
    return out, report


def split_data(
    df: pd.DataFrame, config: Optional[PreprocessingConfig] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train/test split, stratified on ``config.stratify_column`` when that is possible.

    Stratifying keeps the small Rural group represented in both parts.
    """
    cfg = config or PreprocessingConfig()
    X = df.drop(columns=[cfg.target])
    y = df[cfg.target]

    stratify = None
    if cfg.stratify_column and cfg.stratify_column in X.columns:
        labels = X[cfg.stratify_column].fillna("Missing")
        if labels.value_counts().min() >= 2:
            stratify = labels
        else:
            logger.warning("A stratification group has fewer than 2 rows; splitting without stratification")
    elif cfg.stratify_column:
        logger.warning("Stratify column '%s' not found; splitting without stratification", cfg.stratify_column)

    return train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=stratify
    )


# --------------------------------------------------------------------------- #
# Feature preprocessor
# --------------------------------------------------------------------------- #
class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible feature preprocessor (predictors only, no target).

    Per column group, in this order:

    ===================  =====================================================================
    group                steps
    ===================  =====================================================================
    log features         impute -> clip outliers -> safe log -> scale
    other numeric        impute -> clip outliers -> scale
    standardised         impute only (already z-scores)
    categorical          impute (mode) -> one-hot encode (unknown categories become all zeros)
    ===================  =====================================================================

    ``fit`` learns everything from the data it is given (pass the training set only);
    ``transform`` applies those learned values and returns a ``pandas.DataFrame`` with
    readable column names and the original index. Because it follows the estimator API it
    can be dropped into a :class:`sklearn.pipeline.Pipeline`, cross-validated and cloned.

    After fitting, inspect ``feature_names_out_``, ``outlier_bounds_`` and ``log_floors_``.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config

    # -- helpers ----------------------------------------------------------------
    @property
    def _cfg(self) -> PreprocessingConfig:
        return self.config if self.config is not None else PreprocessingConfig()

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Coerce types so that fit and transform see identically formatted data."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DataPreprocessor expects a pandas DataFrame")
        cfg = self._cfg
        if cfg.target in X.columns:
            raise ValueError(
                f"Target column '{cfg.target}' found in the features; remove it before preprocessing"
            )
        out = X.copy()
        out.columns = [str(c).strip() for c in out.columns]
        for col in self.categorical_features_:
            if col in out.columns:
                out[col] = _normalise_text(out[col])
        for col in self.numeric_columns_:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    def _numeric_pipeline(self, cfg: PreprocessingConfig, log: bool) -> Pipeline:
        steps = [
            (
                "impute",
                SimpleImputer(strategy=cfg.numeric_imputer_strategy, keep_empty_features=True),
            )
        ]
        if cfg.outlier_strategy == "clip":
            steps.append(("clip", IQROutlierClipper(factor=cfg.outlier_iqr_factor)))
        if log:
            steps.append(("log", SafeLogTransformer()))
        if cfg.scaler != "none":
            steps.append(("scale", _SCALERS[cfg.scaler]()))
        return Pipeline(steps)

    # -- estimator API ----------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None) -> "DataPreprocessor":
        cfg = self._cfg
        cfg.validate()
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DataPreprocessor expects a pandas DataFrame")
        columns = [str(c).strip() for c in X.columns]

        self.categorical_features_ = list(cfg.categorical_features)
        self.log_features_ = list(cfg.log_features)
        self.standardised_features_ = list(cfg.standardised_features)

        required = self.categorical_features_ + self.log_features_ + self.standardised_features_
        required += list(cfg.numeric_features or [])
        missing = [c for c in required if c not in columns]
        if missing:
            raise ValueError(f"Required feature columns are missing from the data: {missing}")

        assigned = set(required)
        if cfg.numeric_features is not None:
            self.numeric_features_ = list(cfg.numeric_features)
        else:
            named = X.copy()
            named.columns = columns
            self.numeric_features_ = [
                c
                for c in columns
                if c not in assigned and c != cfg.target and pd.api.types.is_numeric_dtype(named[c])
            ]
        self.numeric_columns_ = self.log_features_ + self.standardised_features_ + self.numeric_features_

        ignored = [c for c in columns if c not in set(self.numeric_columns_ + self.categorical_features_)]
        if ignored:
            logger.warning("Columns not covered by any group and therefore dropped: %s", ignored)
        self.ignored_columns_ = ignored

        prepared = self._prepare(X)

        transformers = []
        if self.log_features_:
            transformers.append(("log_numeric", self._numeric_pipeline(cfg, log=True), self.log_features_))
        if self.numeric_features_:
            transformers.append(("numeric", self._numeric_pipeline(cfg, log=False), self.numeric_features_))
        if self.standardised_features_:
            transformers.append(
                (
                    "standardised",
                    Pipeline(
                        [
                            (
                                "impute",
                                SimpleImputer(strategy=cfg.numeric_imputer_strategy, keep_empty_features=True),
                            )
                        ]
                    ),
                    self.standardised_features_,
                )
            )
        if self.categorical_features_:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            (
                                "impute",
                                SimpleImputer(
                                    strategy=cfg.categorical_imputer_strategy,
                                    keep_empty_features=True,
                                ),
                            ),
                            (
                                "encode",
                                OneHotEncoder(
                                    handle_unknown="ignore",
                                    drop="first" if cfg.drop_first_category else None,
                                    sparse_output=False,
                                ),
                            ),
                        ]
                    ),
                    self.categorical_features_,
                )
            )

        column_transformer = ColumnTransformer(
            transformers, remainder="drop", verbose_feature_names_out=False
        )
        column_transformer.set_output(transform="pandas")
        column_transformer.fit(prepared)

        self.column_transformer_ = column_transformer
        self.feature_names_in_ = np.asarray(columns, dtype=object)
        self.n_features_in_ = len(columns)
        self.feature_names_out_ = list(column_transformer.get_feature_names_out())
        self._collect_fitted_summary()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["column_transformer_"])
        prepared = self._prepare(X)
        missing = [c for c in self.numeric_columns_ + self.categorical_features_ if c not in prepared.columns]
        if missing:
            raise ValueError(f"Columns required by the fitted preprocessor are missing: {missing}")
        return self.column_transformer_.transform(prepared)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, ["feature_names_out_"])
        return np.asarray(self.feature_names_out_, dtype=object)

    def set_output(self, *, transform=None):
        """Output is always a DataFrame; accepted for scikit-learn pipeline compatibility."""
        return self

    # -- inspection and persistence ---------------------------------------------
    def _collect_fitted_summary(self) -> None:
        bounds, floors = {}, {}
        for name, pipe, cols in self.column_transformer_.transformers_:
            if not isinstance(pipe, Pipeline):
                continue
            clip = pipe.named_steps.get("clip")
            if clip is not None:
                for col, lo, hi in zip(cols, clip.lower_, clip.upper_):
                    bounds[col] = {"lower": lo, "upper": hi}
            log = pipe.named_steps.get("log")
            if log is not None:
                floors.update(dict(zip(cols, log.floor_)))
        self.outlier_bounds_ = pd.DataFrame(bounds).T if bounds else pd.DataFrame(columns=["lower", "upper"])
        self.log_floors_ = pd.Series(floors, dtype=float, name="log_floor")

    def count_clipped(self, X: pd.DataFrame) -> pd.Series:
        """Number of values in ``X`` outside the learned outlier fences, per column."""
        check_is_fitted(self, ["outlier_bounds_"])
        prepared = self._prepare(X)
        counts = {}
        for col, row in self.outlier_bounds_.iterrows():
            values = prepared[col]
            counts[col] = int(((values < row["lower"]) | (values > row["upper"])).sum())
        return pd.Series(counts, name="values_clipped", dtype=int)

    def save(self, path: Union[str, Path]) -> Path:
        """Persist the fitted preprocessor with joblib."""
        check_is_fitted(self, ["column_transformer_"])
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Union[str, Path]) -> "DataPreprocessor":
        """Load a preprocessor saved with :meth:`save`."""
        return joblib.load(path)


# --------------------------------------------------------------------------- #
# Target handling and model integration
# --------------------------------------------------------------------------- #
def build_target_transformer(strategy: str = "log"):
    """Return a reversible target transformer, or ``None`` for ``"none"``.

    The result is meant for :class:`sklearn.compose.TransformedTargetRegressor`, which fits it
    on the training target only and converts predictions back to the original price scale.
    """
    if strategy == "none":
        return None
    if strategy == "log":
        return FunctionTransformer(np.log, inverse_func=np.exp, check_inverse=True)
    if strategy == "boxcox":
        return PowerTransformer(method="box-cox", standardize=False)
    raise ValueError(f"Unknown target transform '{strategy}'. Choose from {_VALID_TARGET}")


def make_model_pipeline(model: BaseEstimator, config: Optional[PreprocessingConfig] = None):
    """Wrap ``model`` with the preprocessor and the configured target transform.

    The returned estimator is trained on features as they appear in the raw data and on the
    raw price, and predicts on the raw price scale, so it can be used directly with
    ``cross_val_score`` or ``GridSearchCV`` without leakage.
    """
    cfg = config or PreprocessingConfig()
    pipeline = Pipeline([("preprocess", DataPreprocessor(cfg)), ("model", model)])
    transformer = build_target_transformer(cfg.target_transform)
    if transformer is None:
        return pipeline
    return TransformedTargetRegressor(regressor=pipeline, transformer=transformer)


# --------------------------------------------------------------------------- #
# End-to-end convenience
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessedData:
    """Everything produced by :func:`run_preprocessing`."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_train_transformed: pd.Series
    y_test_transformed: pd.Series
    preprocessor: DataPreprocessor
    target_transformer: Any
    cleaning_report: Dict[str, int]
    config: PreprocessingConfig
    raw_X_train: pd.DataFrame
    raw_X_test: pd.DataFrame


def run_preprocessing(
    df: pd.DataFrame, config: Optional[PreprocessingConfig] = None
) -> PreprocessedData:
    """Clean, split, fit the preprocessor on the training part and transform both parts."""
    cfg = config or PreprocessingConfig()
    cleaned, report = clean_data(df, cfg)
    raw_train, raw_test, y_train, y_test = split_data(cleaned, cfg)

    preprocessor = DataPreprocessor(cfg).fit(raw_train)
    X_train = preprocessor.transform(raw_train)
    X_test = preprocessor.transform(raw_test)

    target_transformer = build_target_transformer(cfg.target_transform)
    if target_transformer is None:
        y_train_t, y_test_t = y_train.copy(), y_test.copy()
    else:
        target_transformer.fit(y_train.to_frame())
        y_train_t = pd.Series(
            np.asarray(target_transformer.transform(y_train.to_frame())).ravel(), index=y_train.index, name=cfg.target
        )
        y_test_t = pd.Series(
            np.asarray(target_transformer.transform(y_test.to_frame())).ravel(), index=y_test.index, name=cfg.target
        )

    logger.info("Preprocessed: %d train rows, %d test rows, %d features",
                len(X_train), len(X_test), X_train.shape[1])
    return PreprocessedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_train_transformed=y_train_t,
        y_test_transformed=y_test_t,
        preprocessor=preprocessor,
        target_transformer=target_transformer,
        cleaning_report=report,
        config=cfg,
        raw_X_train=raw_train,
        raw_X_test=raw_test,
    )
