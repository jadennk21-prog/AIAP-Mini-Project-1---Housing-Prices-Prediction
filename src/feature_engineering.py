"""
Feature engineering and feature selection for the housing-price regression project.

Position in the workflow
------------------------
::

    raw data -> DataPreprocessor -> FeatureEngineer -> FeatureSelector -> model
                (src/preprocessing)   (this module)     (this module)

``FeatureEngineer`` is designed to run **after** :class:`src.preprocessing.DataPreprocessor`.
At that point the data is complete (no missing values), skewed features are already
log-transformed, numeric features are standardised, and ``location_type`` is one-hot encoded.
Working in this "model space" keeps every engineered term on a comparable, centred scale
(a product of two z-scores is an interaction, a squared z-score is a curvature term).

Where the features come from
----------------------------
Every feature has a documented reason (see ``FeatureEngineer.feature_catalog_`` and the README):

* **EDA findings** (``eda.ipynb``): non-linear crime effect (section 3.3), location-specific
  size slopes (5.1), school x crime combination (5.3, 5.4), the transport variable's odd
  behaviour (5.5), strongly correlated size features (section 4).
* **Domain knowledge**: buyers pay for a *neighbourhood package* (good schools and low crime),
  total living space, how spacious a house is for its number of rooms, and both very central
  and very remote locations can carry a premium.
* **Residual diagnostics** of the baseline linear model on the training data, which showed
  U-shaped or hump-shaped residual patterns for transport, age and crime.

A caution about the data: ``number_of_bedrooms`` and ``number_of_bathrooms`` are standardised
values (the EDA found negative "counts"), so true ratios such as "size per bedroom" cannot be
computed. The composite features below therefore use differences and averages in z-units.

Lessons from validating these features (see the README for the numbers)
----------------------------------------------------------------------
* Squares of the base features give most of the gain; interactions add some; location-specific
  interactions and tiers add little on their own.
* The composites ``nbhd_quality``, ``rooms_total`` and ``spaciousness`` are exact linear
  combinations of existing columns. They add no information for a linear model and make plain
  least squares numerically singular, so a collinearity filter (``vif``) is part of the
  recommended selection. ``transit_extremity`` is non-linear and does carry signal.
* Engineered features help linear models; they did not help gradient boosting or random forests.

Feature selection
-----------------
Creating many candidate features risks over-fitting, so :class:`FeatureSelector` offers several
techniques (variance, correlation filter, univariate F-test, Lasso, RFE, random-forest
importance and a consensus vote). It is a scikit-learn transformer: inside a pipeline it is
fitted on each training fold only, so selection never sees validation or test data.

Typical use::

    from sklearn.linear_model import LinearRegression
    from src.feature_engineering import FeatureEngineeringConfig, make_full_pipeline

    model = make_full_pipeline(LinearRegression(), fe_config=FeatureEngineeringConfig(),
                               selector="lasso")
    model.fit(X_train, y_train)          # raw features, raw price
    model.predict(X_test)                # predictions on the price scale
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, RFECV, VarianceThreshold, f_regression
from sklearn.linear_model import LassoCV, Ridge
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .preprocessing import DataPreprocessor, PreprocessingConfig, build_target_transformer

__all__ = [
    "FeatureEngineeringConfig",
    "FeatureEngineer",
    "FeatureSelector",
    "SELECTION_METHODS",
    "make_full_pipeline",
    "default_scenarios",
    "evaluate_feature_sets",
    "select_best_scenario",
]

logger = logging.getLogger(__name__)

SELECTION_METHODS = ("none", "variance", "correlation", "vif", "univariate", "lasso", "rfe", "tree", "consensus")

#: Column roles used by the domain features. Values are the column names *after* preprocessing.
_DEFAULT_ROLES: Dict[str, str] = {
    "size": "property_size",
    "crime": "crime_rate_in_area",
    "transit": "proximity_to_public_transport",
    "age": "age_of_property",
    "school": "school_quality_score",
    "bedrooms": "number_of_bedrooms",
    "bathrooms": "number_of_bathrooms",
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class FeatureEngineeringConfig:
    """Which engineered features to create. Defaults follow the EDA of this dataset."""

    #: role name -> column name. Role names are used to name engineered features
    #: (for example ``crime_sq``, ``size_x_school``) and to define the domain composites.
    roles: Dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_ROLES))
    #: Prefix of the one-hot group columns (here the location dummies) used for group interactions.
    group_prefix: Optional[str] = "location_type_"

    #: Re-centre and re-scale the base columns with training statistics (harmless on
    #: already-standardised data, and makes products of columns centred on any input).
    standardise_bases: bool = True
    #: Winsorise standardised bases at +/- this many standard deviations before building
    #: features (limits the leverage of extreme values in squares and products). ``None`` = off.
    clip_z: Optional[float] = 3.0

    add_composites: bool = True
    add_polynomial: bool = True
    polynomial_features: Optional[List[str]] = None            # None = every base numeric column
    add_interactions: bool = True
    interaction_pairs: Optional[List[Tuple[str, str]]] = None  # None = every pair of base columns
    add_group_interactions: bool = True
    group_interaction_features: Optional[List[str]] = None     # None = every base numeric column
    add_tiers: bool = True
    tier_features: List[str] = field(default_factory=lambda: ["crime_rate_in_area", "school_quality_score"])
    n_tiers: int = 3

    #: Standardise engineered columns with training statistics (helps penalised models).
    scale_engineered: bool = True
    keep_original: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.n_tiers < 2:
            raise ValueError("n_tiers must be at least 2")
        if self.clip_z is not None and self.clip_z <= 0:
            raise ValueError("clip_z must be positive (or None)")
        if len(set(self.roles.values())) != len(self.roles):
            raise ValueError("Two roles point at the same column")

    @property
    def base_columns(self) -> List[str]:
        return list(self.roles.values())

    def to_dict(self) -> Dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create domain- and EDA-driven features from a (preprocessed) numeric DataFrame.

    Feature groups
    --------------
    ``composite``
        ``nbhd_quality`` (school score minus crime: the "safe area, good schools" package),
        ``rooms_total`` (average of the standardised bedroom and bathroom counts),
        ``spaciousness`` (size relative to the number of rooms) and ``transit_extremity``
        (distance of the transport value from typical, capturing the U-shape in which both
        very central and very remote homes cost more).
    ``polynomial``
        squares of the base features (``crime_sq`` ...), allowing diminishing returns,
        accelerating effects and U-shapes in a linear model.
    ``interaction``
        products of pairs of base features (``size_x_school`` ...) so that one feature can change
        the effect of another.
    ``group_interaction``
        base features multiplied by the location dummies (``crime_x_rural`` ...), giving each
        location its own slope.
    ``tier``
        quantile tiers (0, 1, 2) of crime and school score, learned on the training data.

    Everything learned from data (means, standard deviations, tier edges, engineered-column
    scaling) is computed in ``fit`` on the training data only. Missing values stay missing, so
    run this after imputation (``DataPreprocessor`` does that). Original columns are kept by
    default and engineered columns are appended.

    Attributes after fitting: ``engineered_features_`` and ``feature_catalog_`` (name, group,
    formula and rationale for each engineered feature).
    """

    def __init__(self, config: Optional[FeatureEngineeringConfig] = None):
        self.config = config

    @property
    def _cfg(self) -> FeatureEngineeringConfig:
        return self.config if self.config is not None else FeatureEngineeringConfig()

    # -- helpers ----------------------------------------------------------------
    def _alias(self, column: str) -> str:
        for role, col in self._cfg.roles.items():
            if col == column:
                return role
        return column

    def _bases(self, X: pd.DataFrame) -> pd.DataFrame:
        cfg = self._cfg
        out = {}
        for col in self.base_features_:
            values = pd.to_numeric(X[col], errors="coerce").astype(float)
            if cfg.standardise_bases:
                values = (values - self.mean_[col]) / self.std_[col]
            if cfg.clip_z is not None:
                values = values.clip(-cfg.clip_z, cfg.clip_z)
            out[col] = values
        return pd.DataFrame(out, index=X.index)

    def _build(self, X: pd.DataFrame, record: bool = False) -> pd.DataFrame:
        """Create the unscaled engineered columns; ``record=True`` also fills the catalog."""
        cfg = self._cfg
        Z = self._bases(X)
        feats: Dict[str, pd.Series] = {}
        info: Dict[str, Tuple[str, str, str]] = {}

        def add(name: str, values: pd.Series, group: str, formula: str, why: str) -> None:
            feats[name] = values
            info[name] = (group, formula, why)

        role = {r: c for r, c in cfg.roles.items() if c in Z.columns}

        if cfg.add_composites:
            if {"school", "crime"} <= role.keys():
                add("nbhd_quality", Z[role["school"]] - Z[role["crime"]], "composite",
                    "school_z - crime_z",
                    "Buyers pay for a safe area with good schools; the EDA found school quality and "
                    "crime (r = -0.44) combine into one price gradient (sections 5.3-5.4)."
                    " Note: an exact linear combination of existing columns, so it adds no information for a "
                    "linear model (the VIF selector removes it); it is kept for interpretability.")
            if {"bedrooms", "bathrooms"} <= role.keys():
                add("rooms_total", (Z[role["bedrooms"]] + Z[role["bathrooms"]]) / 2, "composite",
                    "(bedrooms_z + bathrooms_z) / 2",
                    "Bedrooms and bathrooms are strongly correlated (0.68); one living-space index "
                    "summarises both (section 4)."
                    " Note: an exact linear combination of existing columns, so it adds no information for a "
                    "linear model (the VIF selector removes it); it is kept for interpretability.")
            if {"size", "bedrooms", "bathrooms"} <= role.keys():
                rooms = (Z[role["bedrooms"]] + Z[role["bathrooms"]]) / 2
                add("spaciousness", Z[role["size"]] - rooms, "composite", "size_z - rooms_total",
                    "Property size relative to its number of rooms: large plots with few rooms feel "
                    "spacious. (An exact size-per-room ratio is impossible because room counts are "
                    "standardised.) Note: an exact linear combination of existing columns, so it adds no "
                    "information for a linear model (the VIF selector removes it).")
            if "transit" in role:
                add("transit_extremity", Z[role["transit"]].abs(), "composite", "|transit_z|",
                    "Residual diagnostics show a U-shape: both very central and very remote homes "
                    "price higher than average (EDA section 5.5 found the transport effect odd).")

        if cfg.add_polynomial:
            for col in (cfg.polynomial_features if cfg.polynomial_features is not None else list(Z.columns)):
                if col in Z.columns:
                    add(f"{self._alias(col)}_sq", Z[col] ** 2, "polynomial", f"{self._alias(col)}_z ** 2",
                        "Allows a curved effect (the EDA's LOWESS curves for crime, section 3.3, and "
                        "residual plots for transport and age show non-linearity).")

        if cfg.add_interactions:
            pairs = cfg.interaction_pairs if cfg.interaction_pairs is not None else list(combinations(Z.columns, 2))
            for a, b in pairs:
                if a in Z.columns and b in Z.columns:
                    add(f"{self._alias(a)}_x_{self._alias(b)}", Z[a] * Z[b], "interaction",
                        f"{self._alias(a)}_z * {self._alias(b)}_z",
                        "Lets one feature change the effect of another (EDA interaction analysis, "
                        "sections 5.1-5.4).")

        if cfg.add_group_interactions and self.group_columns_:
            targets = cfg.group_interaction_features if cfg.group_interaction_features is not None else list(Z.columns)
            for g in self.group_columns_:
                gname = g[len(cfg.group_prefix):].lower()
                gvals = pd.to_numeric(X[g], errors="coerce").astype(float)
                for col in targets:
                    if col in Z.columns:
                        add(f"{self._alias(col)}_x_{gname}", Z[col] * gvals, "group_interaction",
                            f"{self._alias(col)}_z * is_{gname}",
                            "Location-specific slope: the size-price slope differed by location "
                            "(87 / 80 / 66 per unit, section 5.1).")

        if cfg.add_tiers:
            for col in cfg.tier_features:
                if col in Z.columns and col in self.tier_edges_:
                    edges = self.tier_edges_[col]
                    codes = np.digitize(Z[col].to_numpy(), edges).astype(float)
                    codes[Z[col].isna().to_numpy()] = np.nan
                    add(f"{self._alias(col)}_tier", pd.Series(codes, index=Z.index), "tier",
                        f"quantile tier (0..{len(edges)}) of {self._alias(col)}",
                        "Low / mid / high tiers capture threshold effects (the 3 x 3 tier grid of "
                        "section 5.4 showed a steady gradient).")

        frame = pd.DataFrame(feats, index=X.index)
        if record:
            self._info = info
        return frame

    # -- estimator API ----------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        cfg = self._cfg
        cfg.validate()
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer expects a pandas DataFrame")
        missing = [c for c in cfg.base_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Columns required by the feature-engineering config are missing: {missing}")

        self.base_features_ = list(cfg.base_columns)
        self.group_columns_ = (
            [c for c in X.columns if cfg.group_prefix and str(c).startswith(cfg.group_prefix)]
            if cfg.add_group_interactions else []
        )
        if cfg.add_group_interactions and cfg.group_prefix and not self.group_columns_:
            logger.warning("No columns start with '%s'; group interactions are skipped", cfg.group_prefix)

        self.mean_, self.std_ = {}, {}
        for col in self.base_features_:
            values = pd.to_numeric(X[col], errors="coerce").astype(float)
            mean, std = float(np.nanmean(values)), float(np.nanstd(values))
            self.mean_[col] = mean
            self.std_[col] = std if std > 0 else 1.0

        Z = self._bases(X)
        self.tier_edges_ = {}
        if cfg.add_tiers:
            quantiles = np.linspace(0, 1, cfg.n_tiers + 1)[1:-1]
            for col in cfg.tier_features:
                if col in Z.columns:
                    self.tier_edges_[col] = np.unique(np.nanquantile(Z[col].to_numpy(), quantiles))

        raw = self._build(X, record=True)
        if cfg.scale_engineered and not raw.empty:
            self.eng_mean_ = raw.mean(skipna=True)
            std = raw.std(ddof=0, skipna=True)
            self.eng_std_ = std.where(std > 0, 1.0)
        else:
            self.eng_mean_ = pd.Series(0.0, index=raw.columns)
            self.eng_std_ = pd.Series(1.0, index=raw.columns)

        self.engineered_features_ = list(raw.columns)
        self.feature_catalog_ = pd.DataFrame(
            [(n, *self._info[n]) for n in self.engineered_features_],
            columns=["feature", "group", "formula", "rationale"],
        ).set_index("feature")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.feature_names_out_ = (list(X.columns) if cfg.keep_original else []) + self.engineered_features_
        logger.info(
            "FeatureEngineer created %d feature(s): %s",
            len(self.engineered_features_),
            self.feature_catalog_["group"].value_counts().to_dict(),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["engineered_features_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer expects a pandas DataFrame")
        missing = [c for c in self.base_features_ + self.group_columns_ if c not in X.columns]
        if missing:
            raise ValueError(f"Columns required by the fitted feature engineer are missing: {missing}")
        engineered = self._build(X)
        if self._cfg.scale_engineered and not engineered.empty:
            engineered = (engineered - self.eng_mean_) / self.eng_std_
        if self._cfg.keep_original:
            return pd.concat([X, engineered], axis=1)
        return engineered

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, ["feature_names_out_"])
        return np.asarray(self.feature_names_out_, dtype=object)

    def set_output(self, *, transform=None):
        """Output is always a DataFrame; accepted for scikit-learn pipeline compatibility."""
        return self


# --------------------------------------------------------------------------- #
# Feature selection
# --------------------------------------------------------------------------- #
class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select features with one of several techniques (all fitted on the training data only).

    ===============  ====================================================================
    method           rule
    ===============  ====================================================================
    ``none``         keep everything
    ``variance``     drop columns whose variance is <= ``variance_threshold``
    ``correlation``  among columns correlated above ``corr_threshold`` keep the one most
                     related to the target (removes near-duplicates)
    ``vif``          repeatedly drop the column with the highest variance inflation factor
                     until all are below ``vif_threshold`` (removes exact and near-exact
                     linear combinations, which a pairwise correlation filter cannot see)
    ``univariate``   F-test against the target; keep the top ``n_features`` or, if not
                     given, those with Bonferroni-adjusted p-value below ``alpha``
    ``lasso``        keep columns with a non-zero LassoCV coefficient
    ``rfe``          recursive feature elimination with Ridge (``n_features`` or chosen by
                     cross-validation)
    ``tree``         random-forest importance; top ``n_features`` or the smallest set that
                     covers 95% of total importance
    ``consensus``    keep columns chosen by at least ``min_votes`` of ``consensus_methods``
    ===============  ====================================================================

    ``always_keep`` / ``always_keep_prefixes`` protect columns from removal (for example the
    original features, so that a retained interaction keeps its main effects). Methods other
    than ``none``, ``variance`` and (partly) ``correlation`` need ``y`` in ``fit``; inside a
    pipeline that is passed automatically.

    After fitting: ``selected_features_``, ``dropped_features_`` and ``selection_report_`` (kept
    flag and score per feature; votes for ``consensus``).
    """

    def __init__(
        self,
        method: str = "lasso",
        n_features: Optional[int] = None,
        alpha: float = 0.05,
        corr_threshold: float = 0.95,
        vif_threshold: float = 10.0,
        variance_threshold: float = 0.0,
        always_keep: Sequence[str] = (),
        always_keep_prefixes: Sequence[str] = (),
        consensus_methods: Sequence[str] = ("univariate", "lasso", "tree"),
        min_votes: int = 2,
        cv: int = 5,
        random_state: int = 42,
    ):
        self.method = method
        self.n_features = n_features
        self.alpha = alpha
        self.corr_threshold = corr_threshold
        self.vif_threshold = vif_threshold
        self.variance_threshold = variance_threshold
        self.always_keep = always_keep
        self.always_keep_prefixes = always_keep_prefixes
        self.consensus_methods = consensus_methods
        self.min_votes = min_votes
        self.cv = cv
        self.random_state = random_state

    # -- individual methods -----------------------------------------------------
    def _protected(self, names: Sequence[str]) -> np.ndarray:
        exact = set(self.always_keep)
        return np.array([n in exact or any(str(n).startswith(p) for p in self.always_keep_prefixes) for n in names])

    def _vif_filter(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Iteratively remove the highest-VIF column (never a protected one) until all VIFs <= threshold."""
        names = list(X.columns)
        values = X.to_numpy(float)
        protected = self._protected(names)
        keep = np.ones(len(names), bool)
        score = np.full(len(names), np.nan)
        keep[values.std(axis=0) == 0] = False                      # constant columns carry no information
        keep |= protected & (values.std(axis=0) > 0)
        while keep.sum() > 1:
            idx = np.where(keep)[0]
            corr = np.corrcoef(values[:, idx], rowvar=False)
            vif = np.diag(np.linalg.inv(corr + 1e-9 * np.eye(len(idx))))
            score[idx] = vif                                       # last value seen for each column
            candidates = [(v, i) for v, i in zip(vif, idx) if v > self.vif_threshold and not protected[i]]
            if not candidates:
                break
            keep[max(candidates)[1]] = False
        return keep, score

    def _need_y(self, y, method: str) -> np.ndarray:
        if y is None:
            raise ValueError(f"method='{method}' needs the target: call fit(X, y)")
        return np.asarray(y, dtype=float).ravel()

    def _run(self, method: str, X: pd.DataFrame, y) -> Tuple[np.ndarray, np.ndarray]:
        """Return (boolean keep-mask, score per feature) for one method."""
        p = X.shape[1]
        names = list(X.columns)
        if method == "none":
            return np.ones(p, bool), np.full(p, np.nan)

        if method == "variance":
            try:
                selector = VarianceThreshold(self.variance_threshold).fit(X)
            except ValueError:
                logger.warning("No feature exceeds the variance threshold; keeping all features")
                return np.ones(p, bool), X.var(ddof=0).to_numpy()
            return selector.get_support(), selector.variances_

        if method == "correlation":
            corr = X.corr().abs().fillna(0.0)
            if y is not None:
                target_corr = X.apply(lambda col: abs(np.corrcoef(col, np.asarray(y, float).ravel())[0, 1])
                                      if col.std() > 0 else 0.0)
                order = list(target_corr.sort_values(ascending=False).index)
                score = target_corr.reindex(names).to_numpy()
            else:
                order, score = names, np.full(p, np.nan)
            kept: List[str] = []
            for name in order:
                if all(corr.loc[name, other] < self.corr_threshold for other in kept):
                    kept.append(name)
            return np.array([n in kept for n in names]), score

        if method == "vif":
            return self._vif_filter(X)

        y_arr = self._need_y(y, method)
        scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=names, index=X.index)

        if method == "univariate":
            f_stat, p_val = f_regression(X, y_arr)
            f_stat = np.nan_to_num(f_stat)
            if self.n_features is not None:
                keep = np.zeros(p, bool)
                keep[np.argsort(-f_stat)[: self.n_features]] = True
            else:
                keep = np.minimum(np.nan_to_num(p_val, nan=1.0) * p, 1.0) < self.alpha
            return keep, f_stat

        if method == "lasso":
            model = LassoCV(cv=self.cv, max_iter=20000, random_state=self.random_state).fit(scaled, y_arr)
            coef = np.abs(model.coef_)
            return coef > 1e-10, coef

        if method == "rfe":
            if self.n_features is not None:
                rfe = RFE(Ridge(alpha=1.0), n_features_to_select=min(self.n_features, p)).fit(scaled, y_arr)
            else:
                rfe = RFECV(Ridge(alpha=1.0), step=1, cv=self.cv, min_features_to_select=1,
                            scoring="neg_root_mean_squared_error").fit(scaled, y_arr)
            return rfe.support_, 1.0 / rfe.ranking_

        if method == "tree":
            forest = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                           random_state=self.random_state, n_jobs=1).fit(X, y_arr)
            importance = forest.feature_importances_
            order = np.argsort(-importance)
            if self.n_features is not None:
                count = self.n_features
            else:
                count = int(np.searchsorted(np.cumsum(importance[order]), 0.95) + 1)
            keep = np.zeros(p, bool)
            keep[order[: max(1, count)]] = True
            return keep, importance

        raise ValueError(f"Unknown method '{method}'. Choose from {SELECTION_METHODS}")

    # -- estimator API ----------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None) -> "FeatureSelector":
        if self.method not in SELECTION_METHODS:
            raise ValueError(f"Unknown method '{self.method}'. Choose from {SELECTION_METHODS}")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureSelector expects a pandas DataFrame")
        names = list(X.columns)
        report = pd.DataFrame(index=names)

        if self.method == "consensus":
            votes = np.zeros(len(names), int)
            for sub in self.consensus_methods:
                if sub == "consensus":
                    raise ValueError("consensus_methods cannot contain 'consensus'")
                keep_sub, score_sub = self._run(sub, X, y)
                votes += keep_sub.astype(int)
                report[f"{sub}_kept"] = keep_sub
                report[f"{sub}_score"] = score_sub
            keep, score = votes >= self.min_votes, votes.astype(float)
            report["votes"] = votes
        else:
            keep, score = self._run(self.method, X, y)

        protected = self._protected(names)
        keep = np.asarray(keep, bool) | protected
        if not keep.any():
            logger.warning("Selection would remove every feature; keeping all features")
            keep = np.ones(len(names), bool)

        report.insert(0, "score", score)
        report.insert(0, "kept", keep)
        report["protected"] = protected
        self.selection_report_ = report
        self.selected_features_ = [n for n, k in zip(names, keep) if k]
        self.dropped_features_ = [n for n, k in zip(names, keep) if not k]
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.n_features_in_ = len(names)
        self.feature_names_out_ = list(self.selected_features_)
        logger.info("FeatureSelector(%s) kept %d of %d feature(s)", self.method, len(self.selected_features_), len(names))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["selected_features_"])
        missing = [c for c in self.selected_features_ if c not in X.columns]
        if missing:
            raise ValueError(f"Columns required by the fitted selector are missing: {missing}")
        return X[self.selected_features_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, ["selected_features_"])
        return np.asarray(self.selected_features_, dtype=object)

    def set_output(self, *, transform=None):
        """Output is always a DataFrame; accepted for scikit-learn pipeline compatibility."""
        return self


# --------------------------------------------------------------------------- #
# Integration
# --------------------------------------------------------------------------- #
SelectorLike = Union[None, str, FeatureSelector, Sequence[Union[str, FeatureSelector]]]


def _as_selectors(selector: SelectorLike, fe_config: Optional[FeatureEngineeringConfig],
                  protect_main_effects: bool) -> List[FeatureSelector]:
    """Normalise ``selector`` (None, name, instance, or a sequence of those) into a list of selectors."""
    if selector is None:
        return []
    if isinstance(selector, (str, FeatureSelector)):
        one = _as_selector(selector, fe_config, protect_main_effects)
        return [one] if one is not None else []
    return [s for item in selector for s in _as_selectors(item, fe_config, protect_main_effects)]


def _as_selector(selector: Union[str, FeatureSelector], fe_config: Optional[FeatureEngineeringConfig],
                 protect_main_effects: bool) -> Optional[FeatureSelector]:
    if isinstance(selector, str) and selector == "none":
        return None
    if isinstance(selector, FeatureSelector):
        return clone(selector)
    protect: Dict[str, Any] = {}
    if protect_main_effects:
        base = list((fe_config or FeatureEngineeringConfig()).base_columns)
        prefix = (fe_config.group_prefix if fe_config else "location_type_")
        protect = {"always_keep": tuple(base), "always_keep_prefixes": (prefix,) if prefix else ()}
    return FeatureSelector(method=selector, **protect)


def make_full_pipeline(
    model: BaseEstimator,
    preprocessing_config: Optional[PreprocessingConfig] = None,
    fe_config: Optional[FeatureEngineeringConfig] = None,
    selector: SelectorLike = None,
    protect_main_effects: bool = True,
):
    """Preprocessing -> feature engineering -> feature selection -> model, with target transform.

    ``fe_config=None`` skips feature engineering (the baseline). ``selector`` can be a method
    name, a :class:`FeatureSelector`, a sequence of those applied one after another (for example
    ``["vif", "lasso"]``), or ``None``. With ``protect_main_effects=True`` (the default)
    the original features and location dummies are never dropped when ``selector`` is given as a
    method name (the hierarchy principle: keep the main effects of any interaction or square that is
    retained; without it, a VIF filter can remove every original feature and leave a model that is
    far worse). A ready-made :class:`FeatureSelector` instance is used exactly as configured. The returned
    estimator takes raw features and a raw price, and predicts on the price scale, so it is safe to
    cross-validate: every learned quantity (imputation, scaling, engineered-feature statistics,
    selected columns) is fitted on the training fold only.
    """
    prep_cfg = preprocessing_config or PreprocessingConfig()
    steps: List[Tuple[str, Any]] = [("preprocess", DataPreprocessor(prep_cfg))]
    if fe_config is not None:
        steps.append(("features", FeatureEngineer(fe_config)))
    selectors = _as_selectors(selector, fe_config, protect_main_effects)
    for i, sel in enumerate(selectors):
        steps.append(("select" if len(selectors) == 1 else f"select_{i + 1}", sel))
    steps.append(("model", model))
    pipeline = Pipeline(steps)
    transformer = build_target_transformer(prep_cfg.target_transform)
    if transformer is None:
        return pipeline
    return TransformedTargetRegressor(regressor=pipeline, transformer=transformer)


# --------------------------------------------------------------------------- #
# Evaluation of feature sets
# --------------------------------------------------------------------------- #
Scenario = Dict[str, Any]   # {"fe": FeatureEngineeringConfig | None, "selector": SelectorLike, "protect": bool}


def default_scenarios(fe_config: Optional[FeatureEngineeringConfig] = None) -> Dict[str, Scenario]:
    """Baseline, one-group-at-a-time ablations, all features, and all features + each selector."""
    full = fe_config or FeatureEngineeringConfig()
    groups = ("add_composites", "add_polynomial", "add_interactions", "add_group_interactions", "add_tiers")

    def only(group: str) -> FeatureEngineeringConfig:
        return replace(full, **{g: (g == group) for g in groups})

    scenarios: Dict[str, Scenario] = {
        "baseline (no engineering)": {"fe": None, "selector": None},
        "composites only": {"fe": only("add_composites"), "selector": None},
        "polynomial only": {"fe": only("add_polynomial"), "selector": None},
        "interactions only": {"fe": only("add_interactions"), "selector": None},
        "group interactions only": {"fe": only("add_group_interactions"), "selector": None},
        "tiers only": {"fe": only("add_tiers"), "selector": None},
        "all engineered": {"fe": full, "selector": None},
    }
    for method in ("correlation", "vif", "univariate", "lasso", "rfe", "tree", "consensus"):
        scenarios[f"all + {method}"] = {"fe": full, "selector": method}
    scenarios["all + vif -> lasso"] = {"fe": full, "selector": ["vif", "lasso"]}
    scenarios["all + vif -> rfe"] = {"fe": full, "selector": ["vif", "rfe"]}
    return scenarios


def evaluate_feature_sets(
    X: pd.DataFrame,
    y: pd.Series,
    scenarios: Dict[str, Scenario],
    model: Optional[BaseEstimator] = None,
    preprocessing_config: Optional[PreprocessingConfig] = None,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
    baseline: Optional[str] = None,
) -> pd.DataFrame:
    """Compare feature sets with repeated K-fold cross-validation on the original price scale.

    Every scenario is a complete pipeline, refitted inside each fold, so feature creation and
    selection never see the validation fold. All scenarios share identical folds, which allows a
    paired comparison against the baseline. The p-value uses the corrected resampled t-test
    (Nadeau and Bengio), which accounts for the overlap between training sets in repeated CV and
    is therefore more cautious than an ordinary paired t-test.

    Returns one row per scenario with mean and standard deviation of RMSE, MAE, R-squared, the
    average number of features used, the RMSE change versus the baseline (positive = better) and
    that change's p-value. Per-fold RMSE values are stored in ``result.attrs["fold_rmse"]``.
    """
    from sklearn.linear_model import LinearRegression

    estimator = model if model is not None else LinearRegression()
    names = list(scenarios)
    baseline = baseline or names[0]
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}

    rows, fold_rmse = {}, {}
    for name, spec in scenarios.items():
        pipe = make_full_pipeline(clone(estimator), preprocessing_config, spec.get("fe"),
                                  spec.get("selector"), spec.get("protect", True))
        res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, return_estimator=True, n_jobs=1)
        rmse = -res["test_rmse"]
        counts = []
        for est in res["estimator"]:
            inner = est.regressor_ if hasattr(est, "regressor_") else est
            last = inner.steps[-2][1]                     # the step just before the model
            counts.append(len(last.feature_names_out_) if hasattr(last, "feature_names_out_") else np.nan)
        fold_rmse[name] = rmse
        rows[name] = {
            "n_features": float(np.nanmean(counts)),
            "rmse": rmse.mean(), "rmse_std": rmse.std(ddof=1),
            "mae": (-res["test_mae"]).mean(), "r2": res["test_r2"].mean(),
        }

    table = pd.DataFrame(rows).T
    k = len(fold_rmse[baseline])
    ratio = 1.0 / (n_splits - 1)                          # n_test / n_train in K-fold
    gains, pvals = [], []
    for name in names:
        diff = fold_rmse[baseline] - fold_rmse[name]
        gains.append(100 * diff.mean() / fold_rmse[baseline].mean())
        if name == baseline or diff.std(ddof=1) == 0:
            pvals.append(np.nan)
        else:
            t_stat = diff.mean() / np.sqrt((1.0 / k + ratio) * diff.var(ddof=1))
            pvals.append(float(2 * stats.t.sf(abs(t_stat), k - 1)))
    table["rmse_gain_%"] = gains
    table["p_value"] = pvals
    table.attrs["fold_rmse"] = fold_rmse
    table.attrs["baseline"] = baseline
    return table


def select_best_scenario(table: pd.DataFrame, one_se_rule: bool = True) -> str:
    """Pick a scenario from :func:`evaluate_feature_sets` output.

    With ``one_se_rule`` the choice is the *simplest* scenario (fewest features) whose mean RMSE is
    within one standard error of the best; otherwise it is simply the lowest mean RMSE.
    """
    best = table["rmse"].idxmin()
    if not one_se_rule:
        return best
    k = len(table.attrs["fold_rmse"][best])
    threshold = table.loc[best, "rmse"] + table.loc[best, "rmse_std"] / np.sqrt(k)
    eligible = table[table["rmse"] <= threshold]
    return eligible["n_features"].idxmin()