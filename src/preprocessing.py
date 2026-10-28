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
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.impute import KNNImputer, MissingIndicator, SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
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
    "split_distribution_report",
    "build_target_transformer",
    "make_model_pipeline",
    "run_preprocessing",
]

logger = logging.getLogger(__name__)

_SCALERS = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}
_VALID_OUTLIER = ("clip", "none")
_VALID_TARGET = ("log", "boxcox", "none")
_VALID_NUM_IMPUTE = ("median", "mean", "constant", "knn")
_VALID_CAT_IMPUTE = ("most_frequent", "constant")
_VALID_SELECTION = ("none", "variance", "lasso")
_VALID_DTYPE = ("float64", "float32")
_VALID_ENCODING = ("onehot", "frequency", "hashing", "target")
_VALID_CAT_ENCODING = (*_VALID_ENCODING, "auto")
_VALID_HIGH_CARD = ("frequency", "hashing", "target")
_IMPUTE_GROUPS = ("log_numeric", "numeric", "standardised")


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
    standardised_features: List[str] = field(default_factory=lambda: ["number_of_bedrooms", "number_of_bathrooms"])
    #: Remaining numeric predictors. ``None`` means "infer every other numeric column".
    numeric_features: Optional[List[str]] = None

    # Cleaning and splitting
    drop_duplicates: bool = True
    test_size: float = 0.2
    random_state: int = 42
    stratify_column: Optional[str] = "location_type"

    # Missing values
    numeric_imputer_strategy: str = "median"  # "median", "mean", "constant" or "knn"
    categorical_imputer_strategy: str = "most_frequent"  # "most_frequent" or "constant"
    numeric_fill_value: float = 0.0  # used when the numeric strategy is "constant"
    categorical_fill_value: str = "Missing"  # used when the categorical strategy is "constant"
    #: Add 0/1 flag columns for every feature that had missing values in the training data.
    add_missing_indicator: bool = False
    #: Neighbours for the "knn" strategy (columns using "knn" in the same group are imputed together).
    knn_neighbors: int = 5
    #: Numeric imputation strategy per column, overriding everything else: {"age_of_property": "mean"}.
    imputer_overrides: Dict[str, str] = field(default_factory=dict)
    #: Numeric imputation strategy per column group ("log_numeric", "numeric" or "standardised").
    group_imputer_strategies: Dict[str, str] = field(default_factory=dict)

    # Outliers: "clip" winsorises features at Q1 - k*IQR / Q3 + k*IQR, "none" disables it.
    outlier_strategy: str = "clip"
    outlier_iqr_factor: float = 3.0

    # Scaling of numeric features: "standard", "robust", "minmax" or "none"
    scaler: str = "standard"

    # Categorical encoding. "onehot" suits low-cardinality columns such as location_type. For
    # high-cardinality columns choose "frequency" (category share), "hashing" (fixed number of
    # hashed columns) or "target" (smoothed, cross-fitted target mean; needs y in fit). "auto" picks
    # one-hot when a column has at most max_onehot_cardinality categories in the training data and
    # high_cardinality_encoding otherwise. encoding_overrides sets the encoding per column.
    categorical_encoding: str = "onehot"
    encoding_overrides: Dict[str, str] = field(default_factory=dict)
    max_onehot_cardinality: int = 10
    high_cardinality_encoding: str = "target"
    hashing_n_features: int = 8
    target_encoder_smoothing: Union[float, str] = "auto"
    drop_first_category: bool = True

    # Target: "log", "boxcox" or "none"
    target_transform: str = "log"

    # Optional feature selection after encoding and scaling:
    #   "variance" drops columns whose variance is <= variance_threshold (0.0 = constant columns),
    #   "lasso" keeps columns with a non-zero LassoCV coefficient (needs y in fit).
    feature_selection: str = "none"
    variance_threshold: float = 0.0

    # Memory: "float32" halves the size of the processed output
    output_dtype: str = "float64"

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
        if self.categorical_encoding not in _VALID_CAT_ENCODING:
            raise ValueError(f"categorical_encoding must be one of {_VALID_CAT_ENCODING}")
        if self.high_cardinality_encoding not in _VALID_HIGH_CARD:
            raise ValueError(f"high_cardinality_encoding must be one of {_VALID_HIGH_CARD}")
        bad_enc = {c: e for c, e in self.encoding_overrides.items() if e not in _VALID_ENCODING}
        if bad_enc:
            raise ValueError(f"encoding_overrides contains unknown encodings {bad_enc}; choose from {_VALID_ENCODING}")
        bad_imp = {c: v for c, v in self.imputer_overrides.items() if v not in _VALID_NUM_IMPUTE}
        if bad_imp:
            raise ValueError(
                f"imputer_overrides contains unknown strategies {bad_imp}; choose from {_VALID_NUM_IMPUTE}"
            )
        bad_grp = [g for g in self.group_imputer_strategies if g not in _IMPUTE_GROUPS]
        if bad_grp:
            raise ValueError(f"group_imputer_strategies has unknown groups {bad_grp}; choose from {_IMPUTE_GROUPS}")
        bad_grp_val = {g: v for g, v in self.group_imputer_strategies.items() if v not in _VALID_NUM_IMPUTE}
        if bad_grp_val:
            raise ValueError(f"group_imputer_strategies has unknown strategies {bad_grp_val}")
        if self.knn_neighbors < 1:
            raise ValueError("knn_neighbors must be at least 1")
        if self.max_onehot_cardinality < 1 or self.hashing_n_features < 1:
            raise ValueError("max_onehot_cardinality and hashing_n_features must be at least 1")
        if self.feature_selection not in _VALID_SELECTION:
            raise ValueError(f"feature_selection must be one of {_VALID_SELECTION}")
        if self.output_dtype not in _VALID_DTYPE:
            raise ValueError(f"output_dtype must be one of {_VALID_DTYPE}")
        if self.variance_threshold < 0:
            raise ValueError("variance_threshold must be non-negative")
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
        """JSON/YAML-friendly dictionary of every setting."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> PreprocessingConfig:
        """Build a config from a dictionary (for example loaded from YAML), rejecting unknown keys."""
        data = dict(data or {})
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(
                f"Unknown preprocessing setting(s) {unknown}; valid settings: {sorted(cls.__dataclass_fields__)}"
            )
        return cls(**data)


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


class ColumnImputer(BaseEstimator, TransformerMixin):
    """Numeric imputation with a strategy per column.

    Strategies: ``median``, ``mean``, ``constant`` (``fill_value``) or ``knn``. ``strategies`` maps
    column names to a strategy; other columns use ``default_strategy``. Columns using ``knn`` are
    imputed together with :class:`sklearn.impute.KNNImputer`, so each of them is filled from rows that
    are similar on the *other* knn columns (a single knn column falls back to its mean). Everything is
    learned in ``fit``; after fitting, ``strategies_`` and ``statistics_`` describe each column
    (``statistics_`` is NaN for knn columns).
    """

    def __init__(
        self,
        strategies: Optional[Dict[str, str]] = None,
        default_strategy: str = "median",
        fill_value: float = 0.0,
        n_neighbors: int = 5,
    ):
        self.strategies = strategies
        self.default_strategy = default_strategy
        self.fill_value = fill_value
        self.n_neighbors = n_neighbors

    @staticmethod
    def _array(X) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        return arr.reshape(-1, 1) if arr.ndim == 1 else arr

    def fit(self, X, y=None):
        arr = self._array(X)
        names = list(X.columns) if hasattr(X, "columns") else [f"x{i}" for i in range(arr.shape[1])]
        overrides = self.strategies or {}
        strategies = [overrides.get(n, self.default_strategy) for n in names]
        bad = sorted({s for s in strategies if s not in _VALID_NUM_IMPUTE})
        if bad:
            raise ValueError(f"Unknown imputation strategies {bad}; choose from {_VALID_NUM_IMPUTE}")
        stats_ = np.full(arr.shape[1], np.nan)
        for j, strategy in enumerate(strategies):
            observed = arr[~np.isnan(arr[:, j]), j]
            if strategy == "median":
                stats_[j] = np.median(observed) if observed.size else 0.0
            elif strategy == "mean":
                stats_[j] = observed.mean() if observed.size else 0.0
            elif strategy == "constant":
                stats_[j] = self.fill_value
        knn_columns = [j for j, s in enumerate(strategies) if s == "knn"]
        self._knn_ = (
            KNNImputer(n_neighbors=self.n_neighbors, keep_empty_features=True).fit(arr[:, knn_columns])
            if knn_columns
            else None
        )
        self.strategies_, self.statistics_, self.knn_columns_ = strategies, stats_, knn_columns
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, ["statistics_"])
        arr = self._array(X).copy()
        for j, strategy in enumerate(self.strategies_):
            if strategy != "knn":
                arr[np.isnan(arr[:, j]), j] = self.statistics_[j]
        if self._knn_ is not None:
            arr[:, self.knn_columns_] = self._knn_.transform(arr[:, self.knn_columns_])
        return arr

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["statistics_"])
        return _feature_names(self, self.n_features_in_, input_features)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category by its share of the training rows (one output column per input column).

    Works for any cardinality and adds exactly one column per feature. Categories not seen in
    training are encoded as 0.0. Different categories with the same frequency become
    indistinguishable, which is the method's main limitation.
    """

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.columns_ = [str(c) for c in frame.columns]
        frame.columns = self.columns_
        self.frequencies_ = {c: frame[c].value_counts(normalize=True, dropna=True).to_dict() for c in self.columns_}
        self.feature_names_in_ = np.asarray(self.columns_, dtype=object)
        self.n_features_in_ = len(self.columns_)
        return self

    def transform(self, X):
        check_is_fitted(self, ["frequencies_"])
        frame = pd.DataFrame(X)
        frame.columns = self.columns_
        out = {c: frame[c].map(self.frequencies_[c]).fillna(0.0).astype(float) for c in self.columns_}
        return pd.DataFrame(out).to_numpy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["frequencies_"])
        return np.asarray([f"{c}_freq" for c in self.columns_], dtype=object)


class HashingEncoder(BaseEstimator, TransformerMixin):
    """Hash ``column=category`` tokens into a fixed number of count columns.

    The output width (``n_features``) does not depend on the number of categories, and unseen
    categories need no special handling. Collisions are possible by design; more buckets mean
    fewer collisions. The hash is deterministic, so results are reproducible across runs.
    """

    def __init__(self, n_features: int = 8):
        self.n_features = n_features

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.columns_ = [str(c) for c in frame.columns]
        self.feature_names_in_ = np.asarray(self.columns_, dtype=object)
        self.n_features_in_ = len(self.columns_)
        return self

    def transform(self, X):
        check_is_fitted(self, ["columns_"])
        frame = pd.DataFrame(X)
        frame.columns = self.columns_
        tokens = [[f"{c}={v}" for c, v in zip(self.columns_, row)] for row in frame.to_numpy()]
        hasher = FeatureHasher(n_features=self.n_features, input_type="string", alternate_sign=False)
        return hasher.transform(tokens).toarray()

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"hash_{i}" for i in range(self.n_features)], dtype=object)


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
    df: pd.DataFrame, config: Optional[PreprocessingConfig] = None, copy: bool = True
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Row-level cleaning that does not learn anything from the data.

    * strips column names and normalises category text (``" rural "`` -> ``"Rural"``),
    * coerces numeric columns (unparseable text becomes NaN and is imputed later),
    * removes exact duplicate rows (optional),
    * removes rows whose target is missing, or non-positive when a log / Box-Cox
      target transform is requested (those values cannot be transformed).

    Because nothing is learned here, it is safe to run before the train/test split.
    Returns the cleaned frame and a dictionary of what was changed.

    ``copy=True`` (default) leaves the input untouched, which is safest. ``copy=False``
    skips the defensive copy, but then the input DataFrame is modified in place (column names
    and dtypes), so only use it if the original is not needed. The saving is modest (about 7%
    of peak memory in a 300,000-row measurement) because type conversion allocates new arrays
    anyway; ``output_dtype="float32"`` is the bigger memory lever.
    """
    cfg = config or PreprocessingConfig()
    out = df.copy() if copy else df
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

    return train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=stratify)


def split_distribution_report(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: Optional[PreprocessingConfig] = None,
) -> Dict[str, pd.DataFrame]:
    """Show how well the split preserved the distribution of the stratification column and target.

    Returns a dictionary with ``"stratify_column"`` (counts and shares per category in train and
    test, plus the gap in percentage points) and ``"target"`` (summary statistics and skewness
    of the target in train and test). Pass the *raw* (untransformed) feature frames.
    """
    cfg = config or PreprocessingConfig()
    report: Dict[str, pd.DataFrame] = {}

    col = cfg.stratify_column
    if col and col in X_train.columns and col in X_test.columns:
        train = X_train[col].fillna("Missing").value_counts()
        test = X_test[col].fillna("Missing").value_counts()
        table = pd.DataFrame({"train_n": train, "test_n": test}).fillna(0).astype(int)
        table["train_%"] = 100 * table["train_n"] / table["train_n"].sum()
        table["test_%"] = 100 * table["test_n"] / table["test_n"].sum()
        table["gap_points"] = (table["train_%"] - table["test_%"]).abs()
        report["stratify_column"] = table.round(2)

    stats = pd.DataFrame({"train": y_train.describe(), "test": y_test.describe()})
    stats.loc["skew"] = [y_train.skew(), y_test.skew()]
    report["target"] = stats.round(3)
    return report


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
    categorical          impute -> one-hot encode (unknown categories become all zeros)
    missing flags        optional 0/1 columns for features that had missing values in training
    ===================  =====================================================================

    An optional feature-selection step (``feature_selection`` in the config) can then drop
    uninformative columns.

    ``fit`` learns everything from the data it is given (pass the training set only);
    ``transform`` applies those learned values and returns a ``pandas.DataFrame`` with
    readable column names and the original index. Because it follows the estimator API it
    can be dropped into a :class:`sklearn.pipeline.Pipeline`, cross-validated and cloned.

    After fitting, the following attributes describe what was learned:

    ``feature_names_out_``
        final output columns
    ``imputation_report_``
        per column: group, strategy, fill value and how many training values were missing
    ``outlier_bounds_``, ``log_floors_``
        clipping fences and log floors
    ``feature_selection_report_``, ``dropped_features_``
        which output columns were kept or dropped by the selector
    ``fit_summary_``
        a dictionary with the headline facts (also written to the log at INFO level)
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config

    # -- helpers ----------------------------------------------------------------
    @property
    def _cfg(self) -> PreprocessingConfig:
        return self.config if self.config is not None else PreprocessingConfig()

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select the columns the pipeline uses and coerce types (identically at fit and transform).

        Only the needed columns are copied, so unused columns never add to memory use.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DataPreprocessor expects a pandas DataFrame")
        cfg = self._cfg
        names = [str(c).strip() for c in X.columns]
        if cfg.target in names:
            raise ValueError(f"Target column '{cfg.target}' found in the features; remove it before preprocessing")
        needed = [c for c in self.numeric_columns_ + self.categorical_features_ if c in names]
        out = X.iloc[:, [names.index(c) for c in needed]].copy()
        out.columns = needed
        for col in self.categorical_features_:
            if col in out.columns:
                out[col] = _normalise_text(out[col])
        for col in self.numeric_columns_:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    def _categorical_imputer(self) -> SimpleImputer:
        cfg = self._cfg
        kwargs: Dict[str, Any] = {"strategy": cfg.categorical_imputer_strategy, "keep_empty_features": True}
        if cfg.categorical_imputer_strategy == "constant":
            kwargs["fill_value"] = cfg.categorical_fill_value
        return SimpleImputer(**kwargs)

    def _numeric_imputer(self, group: str, columns: List[str]) -> ColumnImputer:
        cfg = self._cfg
        return ColumnImputer(
            strategies={c: cfg.imputer_overrides[c] for c in columns if c in cfg.imputer_overrides},
            default_strategy=cfg.group_imputer_strategies.get(group, cfg.numeric_imputer_strategy),
            fill_value=cfg.numeric_fill_value,
            n_neighbors=cfg.knn_neighbors,
        )

    def _encoding_plan(self, prepared: pd.DataFrame) -> Dict[str, Tuple[str, int]]:
        """Choose an encoding for every categorical column from the training cardinality."""
        cfg = self._cfg
        plan = {}
        for col in self.categorical_features_:
            cardinality = int(prepared[col].nunique(dropna=True))
            if col in cfg.encoding_overrides:
                encoding = cfg.encoding_overrides[col]
            elif cfg.categorical_encoding == "auto":
                encoding = "onehot" if cardinality <= cfg.max_onehot_cardinality else cfg.high_cardinality_encoding
            else:
                encoding = cfg.categorical_encoding
            plan[col] = (encoding, cardinality)
        return plan

    def _encoder(self, encoding: str):
        cfg = self._cfg
        if encoding == "onehot":
            return OneHotEncoder(
                handle_unknown="ignore", drop="first" if cfg.drop_first_category else None, sparse_output=False
            )
        if encoding == "frequency":
            return FrequencyEncoder()
        if encoding == "hashing":
            return HashingEncoder(n_features=cfg.hashing_n_features)
        return TargetEncoder(
            target_type="continuous", smooth=cfg.target_encoder_smoothing, cv=5, random_state=cfg.random_state
        )

    def _numeric_pipeline(self, cfg: PreprocessingConfig, log: bool, group: str, columns: List[str]) -> Pipeline:
        steps = [("impute", self._numeric_imputer(group, columns))]
        if cfg.outlier_strategy == "clip":
            steps.append(("clip", IQROutlierClipper(factor=cfg.outlier_iqr_factor)))
        if log:
            steps.append(("log", SafeLogTransformer()))
        if cfg.scaler != "none":
            steps.append(("scale", _SCALERS[cfg.scaler]()))
        return Pipeline(steps)

    # -- estimator API ----------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None) -> DataPreprocessor:
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
            named = X.copy(deep=False)  # shallow: only the labels change, no data is copied
            named.columns = columns
            self.numeric_features_ = [
                c for c in columns if c not in assigned and c != cfg.target and pd.api.types.is_numeric_dtype(named[c])
            ]
        self.numeric_columns_ = self.log_features_ + self.standardised_features_ + self.numeric_features_

        ignored = [c for c in columns if c not in set(self.numeric_columns_ + self.categorical_features_)]
        if ignored:
            logger.warning("Columns not covered by any group and therefore dropped: %s", ignored)
        self.ignored_columns_ = ignored

        prepared = self._prepare(X)

        transformers = []
        if self.log_features_:
            transformers.append(
                (
                    "log_numeric",
                    self._numeric_pipeline(cfg, True, "log_numeric", self.log_features_),
                    self.log_features_,
                )
            )
        if self.numeric_features_:
            transformers.append(
                (
                    "numeric",
                    self._numeric_pipeline(cfg, False, "numeric", self.numeric_features_),
                    self.numeric_features_,
                )
            )
        if self.standardised_features_:
            transformers.append(
                (
                    "standardised",
                    Pipeline([("impute", self._numeric_imputer("standardised", self.standardised_features_))]),
                    self.standardised_features_,
                )
            )
        self.encoding_plan_ = self._encoding_plan(prepared)
        for encoding in _VALID_ENCODING:
            cols = [c for c in self.categorical_features_ if self.encoding_plan_[c][0] == encoding]
            if cols:
                transformers.append(
                    (
                        f"categorical_{encoding}",
                        Pipeline([("impute", self._categorical_imputer()), ("encode", self._encoder(encoding))]),
                        cols,
                    )
                )
        self.uses_target_encoding_ = any(e == "target" for e, _ in self.encoding_plan_.values())
        if self.uses_target_encoding_ and y is None:
            raise ValueError("Target encoding needs the target: call fit(X, y)")
        if cfg.add_missing_indicator:
            flagged = [c for c in self.numeric_columns_ + self.categorical_features_ if prepared[c].isna().any()]
            if flagged:
                transformers.append(("missing_flags", MissingIndicator(features="all"), flagged))
            self.missing_flag_columns_ = flagged
        else:
            self.missing_flag_columns_ = []

        column_transformer = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
        column_transformer.set_output(transform="pandas")
        transformed = column_transformer.fit_transform(
            prepared, None if y is None else np.asarray(y, dtype=float).ravel() if self.uses_target_encoding_ else None
        ).astype(float)

        self.column_transformer_ = column_transformer
        self.feature_names_in_ = np.asarray(columns, dtype=object)
        self.n_features_in_ = len(columns)
        self.n_features_before_selection_ = transformed.shape[1]
        self._fit_feature_selection(transformed, y)
        self.feature_names_out_ = list(self.selected_features_)
        # With target encoding the training features must be the cross-fitted ones that fit_transform
        # produced (re-transforming the training data would leak each row's own target).
        self._fit_output_ = (
            transformed[self.selected_features_].astype(cfg.output_dtype) if self.uses_target_encoding_ else None
        )
        self.encoding_report_ = self._build_encoding_report()
        self.imputation_report_ = self._build_imputation_report(prepared)
        self._collect_fitted_summary()
        self._build_fit_summary(prepared)
        return self

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        """Fit and transform the training data.

        Identical to ``fit(X, y).transform(X)`` except when target encoding is used: then the training
        output comes from cross-fitting, so no row's encoding is computed from its own target.
        """
        self.fit(X, y)
        if self._fit_output_ is not None:
            return self._fit_output_.copy()
        return self.transform(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["column_transformer_"])
        prepared = self._prepare(X)
        missing = [c for c in self.numeric_columns_ + self.categorical_features_ if c not in prepared.columns]
        if missing:
            raise ValueError(f"Columns required by the fitted preprocessor are missing: {missing}")
        out = self.column_transformer_.transform(prepared)
        return out[self.selected_features_].astype(self._cfg.output_dtype)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, ["feature_names_out_"])
        return np.asarray(self.feature_names_out_, dtype=object)

    def set_output(self, *, transform=None):
        """Output is always a DataFrame; accepted for scikit-learn pipeline compatibility."""
        return self

    # -- feature selection ------------------------------------------------------
    def _fit_feature_selection(self, transformed: pd.DataFrame, y) -> None:
        cfg = self._cfg
        names = list(transformed.columns)
        self.selected_features_ = names
        self.dropped_features_: List[str] = []
        self.feature_selection_report_ = pd.DataFrame({"kept": True}, index=names)
        if cfg.feature_selection == "none":
            return

        if cfg.feature_selection == "variance":
            label = "variance"
            try:
                selector = VarianceThreshold(threshold=cfg.variance_threshold).fit(transformed)
            except ValueError:
                logger.warning("No feature exceeds the variance threshold; keeping all features")
                return
            keep, score = selector.get_support(), selector.variances_
        else:  # "lasso"
            label = "abs_lasso_coef"
            if y is None:
                raise ValueError("feature_selection='lasso' needs the target: call fit(X, y)")
            y_arr = np.asarray(y, dtype=float).ravel()
            if len(y_arr) != len(transformed):
                raise ValueError("X and y must have the same number of rows")
            selector = SelectFromModel(LassoCV(cv=5, max_iter=10000, random_state=cfg.random_state)).fit(
                transformed, y_arr
            )
            keep, score = selector.get_support(), np.abs(selector.estimator_.coef_)

        if not keep.any():
            logger.warning("Feature selection would remove every feature; keeping all features")
            keep = np.ones_like(keep, dtype=bool)
        self.selected_features_ = [n for n, k in zip(names, keep) if k]
        self.dropped_features_ = [n for n, k in zip(names, keep) if not k]
        self.feature_selection_report_ = pd.DataFrame({"kept": keep, label: score}, index=names)

    # -- reporting --------------------------------------------------------------
    def _build_imputation_report(self, prepared: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for name, pipe, cols in self.column_transformer_.transformers_:
            if not isinstance(pipe, Pipeline) or "impute" not in pipe.named_steps:
                continue
            imputer = pipe.named_steps["impute"]
            strategies = getattr(imputer, "strategies_", None) or [imputer.strategy] * len(cols)
            for col, strategy, fill in zip(cols, strategies, imputer.statistics_):
                n_missing = int(prepared[col].isna().sum())
                rows.append(
                    {
                        "column": col,
                        "group": name,
                        "strategy": strategy,
                        "fill_value": "knn" if strategy == "knn" else fill,
                        "n_missing": n_missing,
                        "pct_missing": round(100 * n_missing / max(len(prepared), 1), 2),
                    }
                )
        return pd.DataFrame(rows).set_index("column")

    def _build_encoding_report(self) -> pd.DataFrame:
        """Per categorical column: the encoding used, training cardinality and number of output columns."""
        rows = []
        for name, pipe, cols in self.column_transformer_.transformers_:
            if not isinstance(pipe, Pipeline) or "encode" not in pipe.named_steps:
                continue
            encoder = pipe.named_steps["encode"]
            if isinstance(encoder, OneHotEncoder):
                dropped = 0 if encoder.drop_idx_ is None else 1
                widths = [max(len(categories) - dropped, 0) for categories in encoder.categories_]
            elif isinstance(encoder, HashingEncoder):
                widths = [encoder.n_features] * len(cols)  # all columns share the hashed output
            else:
                widths = [1] * len(cols)
            for col, width in zip(cols, widths):
                rows.append(
                    {
                        "column": col,
                        "encoding": name.replace("categorical_", ""),
                        "cardinality": self.encoding_plan_[col][1],
                        "n_output_columns": width,
                    }
                )
        if not rows:
            return pd.DataFrame(columns=["encoding", "cardinality", "n_output_columns"])
        return pd.DataFrame(rows).set_index("column")

    def imputation_report(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """How many values were (or would be) imputed per column.

        Without an argument this is the report for the training data seen in ``fit``.
        With a DataFrame it counts the missing values in that data (for example the test
        set or a new batch) and shows the training-derived fill value that would be used.
        Unparseable text in a numeric column counts as missing.
        """
        check_is_fitted(self, ["imputation_report_"])
        if X is None:
            return self.imputation_report_.copy()
        prepared = self._prepare(X)
        base = self.imputation_report_[["group", "strategy", "fill_value"]].copy()
        counts = prepared[base.index].isna().sum()
        base["n_missing"] = counts.astype(int)
        base["pct_missing"] = (100 * counts / max(len(prepared), 1)).round(2)
        return base

    def _collect_fitted_summary(self) -> None:
        bounds, floors = {}, {}
        for _, pipe, cols in self.column_transformer_.transformers_:
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

    def _count_clipped_prepared(self, prepared: pd.DataFrame) -> pd.Series:
        counts = {}
        for col, row in self.outlier_bounds_.iterrows():
            values = prepared[col]
            counts[col] = int(((values < row["lower"]) | (values > row["upper"])).sum())
        return pd.Series(counts, name="values_clipped", dtype=int)

    def count_clipped(self, X: pd.DataFrame) -> pd.Series:
        """Number of values in ``X`` outside the learned outlier fences, per column."""
        check_is_fitted(self, ["outlier_bounds_"])
        return self._count_clipped_prepared(self._prepare(X))

    def _build_fit_summary(self, prepared: pd.DataFrame) -> None:
        """Collect the headline facts about the fit and write them to the log (INFO level)."""
        cfg = self._cfg
        report = self.imputation_report_
        imputed = {c: int(n) for c, n in report["n_missing"].items() if n > 0}
        clipped = {c: int(n) for c, n in self._count_clipped_prepared(prepared).items() if n > 0}
        n_encoded = 0
        for _, pipe, cols in self.column_transformer_.transformers_:
            if isinstance(pipe, Pipeline) and isinstance(pipe.named_steps.get("encode"), OneHotEncoder):
                n_encoded += len(pipe.named_steps["encode"].get_feature_names_out(cols))
        encodings = {c: e for c, (e, _) in self.encoding_plan_.items()}

        self.fit_summary_ = {
            "n_rows": len(prepared),
            "n_input_columns": self.n_features_in_,
            "ignored_columns": list(self.ignored_columns_),
            "imputed_values_per_column": imputed,
            "missing_flag_columns": list(self.missing_flag_columns_),
            "outlier_strategy": cfg.outlier_strategy,
            "values_clipped_per_column": clipped,
            "scaler": cfg.scaler,
            "n_one_hot_columns": n_encoded,
            "encodings": encodings,
            "feature_selection": cfg.feature_selection,
            "dropped_features": list(self.dropped_features_),
            "n_output_features": len(self.selected_features_),
        }
        logger.info("Fitted DataPreprocessor on %d rows and %d input columns", len(prepared), self.n_features_in_)
        if imputed:
            logger.info(
                "Imputation (numeric=%s, categorical=%s): %s",
                cfg.numeric_imputer_strategy,
                cfg.categorical_imputer_strategy,
                imputed,
            )
        else:
            logger.info("Imputation: no missing values found in the training data")
        if cfg.outlier_strategy == "clip":
            logger.info(
                "Outlier clipping at %.1f x IQR: %s",
                cfg.outlier_iqr_factor,
                clipped if clipped else "no values outside the fences",
            )
        else:
            logger.info("Outlier clipping: disabled")
        logger.info("Scaler: %s", cfg.scaler)
        logger.info("One-hot encoding created %d column(s) from %s", n_encoded, self.categorical_features_)
        if self.missing_flag_columns_:
            logger.info("Missing-value flag columns added for: %s", self.missing_flag_columns_)
        if cfg.feature_selection != "none":
            logger.info(
                "Feature selection (%s) dropped %d of %d feature(s): %s",
                cfg.feature_selection,
                len(self.dropped_features_),
                self.n_features_before_selection_,
                self.dropped_features_,
            )
        logger.info(
            "Output feature space: %d feature(s) from %d input column(s)",
            len(self.selected_features_),
            self.n_features_in_,
        )

    # -- persistence ------------------------------------------------------------
    def save(self, path: Union[str, Path]) -> Path:
        """Persist the fitted preprocessor with joblib."""
        check_is_fitted(self, ["column_transformer_"])
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Union[str, Path]) -> DataPreprocessor:
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
    split_report: Dict[str, pd.DataFrame]


def run_preprocessing(df: pd.DataFrame, config: Optional[PreprocessingConfig] = None) -> PreprocessedData:
    """Clean, split, fit the preprocessor on the training part and transform both parts."""
    cfg = config or PreprocessingConfig()
    cleaned, report = clean_data(df, cfg)
    raw_train, raw_test, y_train, y_test = split_data(cleaned, cfg)

    # The target transformer is fitted on the training target only. Its output is also the
    # target used by the (optional) model-based feature selection.
    target_transformer = build_target_transformer(cfg.target_transform)
    if target_transformer is None:
        y_train_t, y_test_t = y_train.copy(), y_test.copy()
    else:
        target_transformer.fit(y_train.to_frame())
        y_train_t = pd.Series(
            np.asarray(target_transformer.transform(y_train.to_frame())).ravel(),
            index=y_train.index,
            name=cfg.target,
        )
        y_test_t = pd.Series(
            np.asarray(target_transformer.transform(y_test.to_frame())).ravel(),
            index=y_test.index,
            name=cfg.target,
        )

    preprocessor = DataPreprocessor(cfg)
    X_train = preprocessor.fit_transform(raw_train, y_train_t)  # cross-fitted when target encoding is used
    X_test = preprocessor.transform(raw_test)

    logger.info("Preprocessed: %d train rows, %d test rows, %d features", len(X_train), len(X_test), X_train.shape[1])
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
        split_report=split_distribution_report(raw_train, raw_test, y_train, y_test, cfg),
    )
