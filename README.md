# Housing Price Regression: EDA, Preprocessing and Feature Engineering

This repository contains an exploratory analysis of a housing-price dataset (`eda.ipynb`), a
reusable, leakage-safe preprocessing pipeline built from its findings (`src/preprocessing.py`), and a
feature engineering and selection module (`src/feature_engineering.py`) that is validated against the
preprocessing-only baseline. Experiments are configured with version-controlled YAML/JSON files, and every
run is recorded under its own run ID (`src/experiment.py`). The regression target is `house_sale_price`.

```
raw data -> DataPreprocessor -> FeatureEngineer -> FeatureSelector -> model
            (preprocessing.py)   (feature_engineering.py)
```

## Quick start

```bash
pip install -r requirements.txt          # developed and tested with Python 3.12, scikit-learn 1.8, pandas 3.0

python main.py                           # preprocess the data and print what was learned
python main.py --features                # add feature engineering + selection and save the results
python main.py --validate                # run the 15-check validation suite and a baseline model
python main.py --compare-features        # cross-validated comparison of feature sets (about 3 minutes with --compare-models)
python main.py --config configs/default.yaml --features   # run from a config file
python -m pytest                         # run the 208 unit tests
python main.py --help                    # every option
```

Each run that saves files writes into a folder named after its **run ID** (for example
`reports/20261105-142530-3fa9c1d2/`), so runs never overwrite each other.

### Command-line reference

| Option | Meaning |
|---|---|
| `--config FILE` | load a YAML/JSON experiment config; command-line options override it |
| `--save-config FILE` | write the resolved configuration (defaults + config file + options) and exit |
| `--features` | engineer and select features after preprocessing |
| `--selection-method M` | `none, variance, correlation, vif, univariate, lasso, rfe, tree, consensus`; chain with `+` (`vif+rfe`) |
| `--compare-features` | cross-validated comparison of feature sets, then one check on the test set |
| `--compare-models` | with `--compare-features`: also try Ridge, gradient boosting and random forest |
| `--validate` | run the validation suite and a baseline model (non-zero exit code on failure) |
| `--scaler`, `--outlier-strategy`, `--outlier-iqr-factor`, `--target-transform` | preprocessing choices |
| `--numeric-imputer {median,mean,constant,knn}`, `--categorical-imputer`, `--missing-indicator` | imputation |
| `--categorical-encoding {onehot,frequency,hashing,target,auto}` | categorical encoding |
| `--clip-z X`, `--cv-folds N`, `--cv-repeats N`, `--seed N`, `--test-size X` | feature engineering, evaluation and split settings (`--clip-z 0` switches winsorising off) |
| `--data`, `--output-dir`, `--artifact-dir`, `--reports-dir` | input and output locations |
| `--run-label TEXT`, `--no-versioning`, `--no-save` | name a run, use the flat folder layout, or write nothing |
| `--profile`, `-v` | track peak memory per stage, print log messages (including what each fit learned) |

### Where the outputs go

| Folder | Content (per run, inside `<folder>/<run_id>/`) |
|---|---|
| `processed/` | `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`; with `--features` also `features_X_train.csv`, `features_X_test.csv` |
| `artifacts/` | `preprocessor.joblib`; with `--features` also `feature_pipeline.joblib` (fitted end-to-end pipeline) |
| `reports/` | `results.json` (full run record), `config.yaml` (resolved config), `feature_catalog.csv`, `feature_selection_report.csv`; with `--compare-features` also `feature_set_comparison.csv`, `holdout_comparison.csv` |
| `reports/runs.csv` | one line per run (run ID, config hash, command, key results) to list and compare runs |
| `<folder>/latest.txt` | the ID of the newest run in that folder |

All three folders are git-ignored. `--no-versioning` writes directly into `processed/`, `artifacts/` and `reports/` instead.

## Repository structure

```
your-repo-name/
├── README.md
├── eda.ipynb                     # exploratory analysis with executed outputs
├── requirements.txt              # runtime and test dependencies
├── requirements-dev.txt          # + ruff, mypy, pre-commit
├── main.py                       # CLI: preprocess, engineer features, compare, validate, record runs
├── pyproject.toml                # ruff and mypy configuration
├── pytest.ini                    # test configuration
├── .pre-commit-config.yaml       # git hooks (lint, format, types, tests before push)
├── .github/workflows/ci.yml      # CI: lint, type check, tests, validation on Python 3.9 / 3.11 / 3.12
├── Housing_Price_Dataset.csv     # raw data (1,000 rows, 9 columns)
├── configs/
│   ├── default.yaml              # every default setting, ready to copy
│   └── example_advanced.yaml     # KNN / per-column imputation, auto encoding, chained selection
├── examples/
│   └── compare_encoders.py       # encoder comparison on synthetic high-cardinality data
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # cleaning, imputation, outliers, scaling, encoding
│   ├── feature_engineering.py    # feature creation, feature selection, feature-set evaluation
│   └── experiment.py             # config files, run IDs, profiling, run records
└── tests/
    ├── conftest.py               # shared fixtures (small synthetic dataset)
    ├── test_preprocessing.py     # 100 tests
    ├── test_feature_engineering.py   # 68 tests
    ├── test_experiment.py        # 21 tests
    └── test_main.py              # 19 tests (command line, config precedence, versioned outputs)
```

## Configuration files

An experiment is described by a YAML or JSON file with up to five sections: `run`, `preprocessing`
(every field of `PreprocessingConfig`), `feature_engineering` (every field of `FeatureEngineeringConfig`),
`selection` and `cv`. Missing settings use the defaults, and **unknown keys are rejected** so typos are caught
immediately. Precedence is *defaults < config file < command-line options*.

```yaml
# my_experiment.yaml
preprocessing:
  scaler: robust
  group_imputer_strategies: {numeric: knn}
  imputer_overrides: {age_of_property: mean}
feature_engineering:
  add_tiers: false
selection:
  method: vif+rfe
cv: {folds: 5, repeats: 3}
```

```bash
python main.py --save-config configs/mine.yaml --scaler robust       # create a config from options
python main.py --config configs/mine.yaml --features                  # run it
python main.py --config reports/<run_id>/config.yaml --features       # exactly repeat an earlier run
```

Every saving run stores the **resolved** configuration as `config.yaml` next to its results, so any run can be repeated. Rerunning from a
saved config gives the same config hash and the same selected features (this is tested).

## Experiment tracking

Each run that saves files produces `reports/<run_id>/results.json`, a single machine-readable record with:

- the run ID (`UTC timestamp - configuration hash`, plus a label if given; a suffix `-2`, `-3` is added if the same ID already exists), start/finish time and command line
- the environment (Python, platform, package versions) and a fingerprint of the data (SHA-256, rows, columns)
- the resolved configuration and its hash
- stage timings (and peak memory with `--profile`)
- results: preprocessing summary, imputation and encoding reports, feature catalog and selection report, the full cross-validation
  table **with fold-level scores**, the hold-out results and the validation checks

`reports/runs.csv` accumulates one summary line per run. To load a run in Python: `json.load(open("reports/<run_id>/results.json"))`.

## How the EDA findings map to pipeline decisions

| EDA finding | Pipeline decision |
|---|---|
| No missing values or duplicates in the raw data | Still handled so the module is reusable: median / most-frequent imputation, duplicate removal, rows with a missing target dropped |
| `crime_rate_in_area` (skew 2.55), `property_size` (2.02), `proximity_to_public_transport` (1.52) are right-skewed; `log(crime)` lifts the correlation with price from -0.40 to -0.50 | These three features are log-transformed (`SafeLogTransformer`) |
| IQR outliers in features and target look genuine, not errors | The target is never filtered. Features are only **clipped** at extreme fences (Q1 - 3 x IQR, Q3 + 3 x IQR), which touches about 0.5% of numeric training values (22 of 4,000); switch off with `--outlier-strategy none` |
| `number_of_bedrooms` / `number_of_bathrooms` are already z-scores (mean about 0, std about 1) | Imputed only, not scaled again |
| Target `house_sale_price` is right-skewed (1.57); log gives 0.27 | Reversible log target (Box-Cox optional), applied through `TransformedTargetRegressor` so predictions return to the price scale |
| `location_type` has three nominal levels and only 100 Rural rows | One-hot encoding (first level dropped, unseen categories become all zeros); train/test split stratified on `location_type` |
| VIF is at most about 2.8 | No feature removal or decorrelation step needed |

## What the pipeline does

| Column group | Steps (all fitted on the training set only) |
|---|---|
| Log features | impute (median) -> clip outliers -> safe log -> standard scale |
| Other numeric (`age_of_property`, `school_quality_score`) | impute -> clip outliers -> standard scale |
| Standardised (`number_of_bedrooms`, `number_of_bathrooms`) | impute only |
| Categorical (`location_type`) | impute (most frequent) -> one-hot encode |
| Missing flags (optional) | 0/1 column for every feature that had missing values in training |
| Feature selection (optional) | `variance` drops constant columns; `lasso` keeps columns with a non-zero LassoCV coefficient |

Row-level cleaning (`clean_data`) happens before the split because it learns nothing from the data:
whitespace and case in categories are normalised, unparseable numbers become NaN (and are then
imputed), exact duplicates are removed, and rows whose target is missing or non-positive are removed.

## What the pipeline reports about itself

After `fit`, the preprocessor explains what it learned:

| Attribute / function | Content |
|---|---|
| `imputation_report_` | per column: group, strategy, fill value, number and share of missing training values |
| `imputation_report(X)` | the same counts for any new data (test set, new batch), using the training fill values |
| `outlier_bounds_`, `count_clipped(X)` | learned fences and how many values fall outside them |
| `log_floors_` | the floor used by the safe log for each log feature |
| `feature_selection_report_`, `dropped_features_` | which output columns were kept or dropped, with their variance or Lasso score |
| `fit_summary_` | headline facts: rows, imputed values, clipped values, scaler, one-hot columns, final feature count |
| `split_distribution_report(...)` | train/test counts and shares of the stratification column, and train/test target statistics |

The same facts are written to the log at INFO level when the preprocessor is fitted (run `python main.py -v`).
For example, on data with missing cells:

```
Imputation (numeric=median, categorical=most_frequent): {'age_of_property': 32, 'location_type': 41, ...}
Outlier clipping at 3.0 x IQR: {'crime_rate_in_area': 10, 'property_size': 9, ...}
Scaler: standard
One-hot encoding created 2 column(s) from ['location_type']
Output feature space: 9 feature(s) from 8 input column(s)
```

## Feature engineering and selection

### Where the features come from

`FeatureEngineer` runs **after** preprocessing, so it works on data that is already complete, log-transformed,
standardised and one-hot encoded. Each engineered column is a product or transformation of standardised
columns, so everything is centred and on a comparable scale. 48 features are created from the 9 preprocessed
columns. Each one has a documented reason in `FeatureEngineer.feature_catalog_` (also saved to `reports/feature_catalog.csv`).

| Group | Count | Examples | Why (EDA, domain knowledge, diagnostics) |
|---|---|---|---|
| Polynomial (squares) | 7 | `crime_sq`, `transit_sq`, `age_sq` | The EDA found a curved crime effect (section 3.3). Residual plots of the baseline model showed a U-shape for transport (both very central and very remote homes cost more), a hump for crime and a curve for age. Squares let a linear model fit these. |
| Interactions | 21 | `size_x_school`, `age_x_transit` | One feature changing the effect of another (EDA sections 5.1-5.4). Pairs with the strongest signal in diagnostics included age x transit and size x school. |
| Group interactions | 14 | `crime_x_rural`, `size_x_urban` | Location-specific slopes: the size-price slope differed by location (87 / 80 / 66, section 5.1). |
| Composites | 4 | `nbhd_quality`, `rooms_total`, `spaciousness`, `transit_extremity` | Domain ideas: buyers pay for a "safe area with good schools" package, bedrooms and bathrooms (r = 0.68) describe one living-space dimension, size relative to rooms, and distance of the transport value from typical. |
| Tiers | 2 | `crime_tier`, `school_tier` | Low / mid / high thresholds, as in the EDA's 3 x 3 tier grid (section 5.4). |

A data caveat shaped the design: `number_of_bedrooms` and `number_of_bathrooms` are standardised values (the EDA
found negative "counts"), so true ratios like "size per bedroom" are impossible. The composites use differences and
averages in z-units instead.

### Feature selection techniques

`FeatureSelector` is a scikit-learn transformer. Inside a pipeline it is fitted on each training fold only, so
selection never sees validation or test data.

| Method | Rule |
|---|---|
| `variance` | drop (near-)constant columns |
| `correlation` | among columns correlated above a threshold keep the one most related to the target |
| `vif` | repeatedly drop the highest variance-inflation-factor column until all are below a threshold (default 10); removes exact and near-exact linear combinations |
| `univariate` | F-test against the target, Bonferroni-adjusted, or top-k |
| `lasso` | keep columns with a non-zero LassoCV coefficient |
| `rfe` | recursive feature elimination with Ridge, size chosen by cross-validation or fixed |
| `tree` | random-forest importance (top-k or 95% cumulative importance) |
| `consensus` | keep columns chosen by at least *m* of several methods |

Methods can be chained (`selector=["vif", "rfe"]`). `always_keep` protects columns; `make_full_pipeline` protects the
original features and location dummies by default (see the lessons below for why).

### Results

All numbers below come from `python main.py --compare-features --compare-models`, using a linear regression on the
log target. Scenarios are compared by 5-fold cross-validation repeated 3 times **on the 800 training rows only**
(feature creation and selection are refitted inside every fold). The test set is used once at the end. RMSE and MAE are
in price units; the p-value is a corrected resampled t-test (cautious because repeated CV folds overlap).

| Feature set | Features | CV RMSE | CV MAE | CV R2 | RMSE gain | p-value |
|---|---|---|---|---|---|---|
| baseline (no engineering) | 9 | 25,149 | 17,250 | 0.960 | | |
| composites only | 13 | 27,495 | 16,550 | 0.934 | -9.3% | 0.84 |
| **polynomial (squares) only** | 16 | 19,299 | 14,018 | 0.976 | 23.3% | 0.0001 |
| interactions only | 30 | 22,127 | 15,925 | 0.969 | 12.0% | 0.002 |
| group interactions only | 23 | 24,820 | 16,992 | 0.961 | 1.3% | 0.48 |
| tiers only | 11 | 24,227 | 16,833 | 0.963 | 3.7% | 0.054 |
| all engineered, no selection | 57 | 21,881 | 14,193 | 0.964 | 13.0% | 0.64 |
| all + correlation filter | 57 | 21,881 | 14,193 | 0.964 | 13.0% | 0.64 |
| all + univariate F-test | 31 | 22,988 | 15,131 | 0.963 | 8.6% | 0.70 |
| all + Lasso | 42 | 21,795 | 14,114 | 0.964 | 13.3% | 0.63 |
| all + RFE | 31 | 22,079 | 14,207 | 0.963 | 12.2% | 0.67 |
| all + random-forest importance | 36 | 26,039 | 16,377 | 0.948 | -3.5% | 0.92 |
| all + consensus | 40 | 23,033 | 14,768 | 0.960 | 8.4% | 0.77 |
| **all + VIF** | 48 | 19,151 | 13,766 | 0.977 | 23.9% | 0.0002 |
| all + VIF then Lasso | 37 | 19,303 | 13,835 | 0.977 | 23.2% | 0.0004 |
| **all + VIF then RFE** | 28 | 19,158 | 13,752 | 0.977 | 23.8% | 0.0001 |

**Hold-out test set** (200 rows, used once after the choices above):

| Model | RMSE | MAE | R2 |
|---|---|---|---|
| baseline | 29,191 | 19,084 | 0.958 |
| all + VIF (lowest CV RMSE) | 19,397 | 13,985 | 0.982 |
| polynomial only (simplest within 1 standard error) | 22,800 | 14,860 | 0.974 |

### What the evidence says (and the lessons)

1. **Squares carry most of the gain.** Seven squared terms alone cut CV RMSE by 23%. Interactions add more on top;
   group interactions and tiers add little by themselves.
2. **Three of the four composites add nothing for a linear model.** `nbhd_quality`, `rooms_total` and `spaciousness`
   are exact linear combinations of existing columns. Removing all composites, with no selection at all, gives the same
   error as the best model (19,102 vs 19,151). They also make the design matrix numerically singular (condition number about
   2 x 10^15), so plain least squares can produce wild predictions for a few extreme houses; this is why "all engineered, no selection" has a
   *better MAE* than the baseline but a *worse RMSE* (its five worst predictions account for 71% of its squared error). They are kept
   for interpretability and because a VIF filter removes them. `transit_extremity` is different: it is non-linear and does
   carry signal (a model with only the composites and VIF improved RMSE by about 10%).
3. **Selection must suit the model.** Random-forest importance selection hurt a linear model (-3.5%), because it
   keeps what trees use rather than what a linear model needs. The univariate test ignores redundancy. The VIF filter
   fits the actual problem (collinearity) and was best.
4. **Protect the main effects.** An unprotected VIF filter removed all seven original features (they correlate most
   with the engineered terms) and left a model of interactions without main effects: RMSE 50,062 versus 19,151 when protected.
   This is the hierarchy principle, and there is a regression test for it.
5. **Feature engineering helps linear models, not tree ensembles.** Ridge improved by about 21 to 22%. Gradient boosting
   (+2%, p = 0.73) and random forest (+3%, p = 0.24) did not improve significantly, and after VIF selection they got
   worse. Trees find curvature and interactions on their own. Also note that a plain linear model on this data beats gradient boosting and
   random forests with default settings (CV RMSE 25k versus 32k and 45k), which fits the EDA's finding of smooth, mostly additive relationships.

### Honest limitations of this evaluation

- **The feature ideas were not designed on the training set alone.** Residual diagnostics and exploratory tests used all 1,000 rows,
  including what later became the test set. The cross-validated comparison and the hold-out test are clean for *choosing among
  the scenarios*, but the choice of feature families, the VIF default and `clip_z = 3` were partly informed by looking at all
  the data. The hold-out improvement (about 34%) is therefore likely a little optimistic. The cross-validated gain
  (about 24%) is the more cautious number.
- `clip_z = 3` (winsorising standardised features before building squares and products) was chosen after seeing it help slightly in
  cross-validation. Its effect is small.
- 200 test rows from one random split is a small sample. The test set also contains the most expensive homes (target skew 1.90 against 1.45 in training), which inflates its RMSE for every model.
- One dataset, one target. The findings (especially "composites add nothing" and "trees do not benefit") describe this data.
- Several feature-set comparisons are not statistically significant (see p-values); the claims above are limited to the clear
  differences (squares, VIF, the baseline gap).

### Using the module

```python
from sklearn.linear_model import LinearRegression
from src.feature_engineering import FeatureEngineeringConfig, FeatureEngineer, FeatureSelector, make_full_pipeline

# End-to-end: preprocessing -> feature engineering -> VIF selection -> model (raw data in, price-scale predictions out)
model = make_full_pipeline(LinearRegression(), fe_config=FeatureEngineeringConfig(), selector="vif")
model.fit(raw_X_train, y_train)
model.predict(raw_X_test)

# Inspect what was created and kept
fe = model.regressor_.named_steps["features"]
fe.feature_catalog_  # name, group, formula and rationale of every engineered feature
model.regressor_.named_steps["select"].dropped_features_
model.regressor_.named_steps["select"].selection_report_

# Chain selectors, or switch feature groups off
make_full_pipeline(LinearRegression(), fe_config=FeatureEngineeringConfig(add_tiers=False), selector=["vif", "rfe"])
```

`FeatureEngineeringConfig` switches each group on or off (`add_composites`, `add_polynomial`, `add_interactions`,
`add_group_interactions`, `add_tiers`), restricts the features or pairs used, sets `clip_z` and the number of tiers, and maps
roles to column names so the module can be reused on other data. `evaluate_feature_sets(...)` compares any set of scenarios by
repeated cross-validation; `select_best_scenario(...)` picks the lowest-RMSE scenario or, with the one-standard-error rule, the
simplest scenario within one standard error of it.

## Using the preprocessing module

```python
from sklearn.linear_model import Ridge
from src.preprocessing import (
    PreprocessingConfig,
    load_data,
    run_preprocessing,
    make_model_pipeline,
)

cfg = PreprocessingConfig()  # defaults follow the EDA
df = load_data("Housing_Price_Dataset.csv")

# 1) Step by step: processed arrays for any model
res = run_preprocessing(df, cfg)
res.X_train, res.X_test  # DataFrames with readable column names
res.y_train_transformed  # log(price)
res.preprocessor.outlier_bounds_  # fences learned on the training data

# 2) As a single estimator: cross-validation / grid search without leakage
model = make_model_pipeline(Ridge(alpha=1.0), cfg)
model.fit(res.raw_X_train, res.y_train)  # raw features, raw price
model.predict(res.raw_X_test)  # predictions back on the price scale
```

`DataPreprocessor` follows the scikit-learn estimator API (`fit`, `transform`, `get_feature_names_out`,
`clone`), so it can be used in any `Pipeline`. `save()` and `load()` persist a fitted preprocessor.

### Configuration

All choices live in `PreprocessingConfig`; the most useful are:

| Option | Default | Alternatives |
|---|---|---|
| `scaler` | `"standard"` | `"robust"`, `"minmax"`, `"none"` |
| `outlier_strategy` / `outlier_iqr_factor` | `"clip"` / `3.0` | `"none"`; 1.5 for stricter capping |
| `target_transform` | `"log"` | `"boxcox"`, `"none"` |
| `numeric_imputer_strategy` / `numeric_fill_value` | `"median"` / `0.0` | `"mean"`, `"constant"`, `"knn"` (with `knn_neighbors`) |
| `imputer_overrides`, `group_imputer_strategies` | `{}` | per-column / per-group strategy, e.g. `{age_of_property: mean}`, `{numeric: knn}` |
| `categorical_encoding` | `"onehot"` | `"frequency"`, `"hashing"`, `"target"`, `"auto"` (see below) |
| `encoding_overrides`, `max_onehot_cardinality`, `high_cardinality_encoding`, `hashing_n_features` | `{}`, `10`, `"target"`, `8` | per-column encoding; the rule used by `"auto"` |
| `categorical_imputer_strategy` / `categorical_fill_value` | `"most_frequent"` / `"Missing"` | `"constant"` |
| `add_missing_indicator` | `False` | `True` adds 0/1 flag columns |
| `feature_selection` / `variance_threshold` | `"none"` / `0.0` | `"variance"`, `"lasso"` |
| `output_dtype` | `"float64"` | `"float32"` (halves output memory) |
| `log_features`, `standardised_features`, `categorical_features` | from the EDA | any column lists |
| `test_size`, `random_state`, `stratify_column` | `0.2`, `42`, `"location_type"` | |

### Imputation and categorical encoding options

**Imputation.** `median`, `mean`, `constant` and `knn` are available for numeric columns, set globally, per column group
(`group_imputer_strategies`: `log_numeric`, `numeric`, `standardised`) or per column (`imputer_overrides`). Columns that use
`knn` in the same group are imputed together from similar rows; a single knn column falls back to its mean. Categorical columns use
`most_frequent` or `constant`. `add_missing_indicator` adds 0/1 flag columns.

**Encoding.** The dataset's only categorical column, `location_type`, has 3 levels, so one-hot encoding is the default and is used
throughout the EDA-driven design (the location interactions in the feature engineering rely on its dummy columns). For higher-cardinality
columns the pipeline offers:

| Encoding | What it does | Notes |
|---|---|---|
| `onehot` | one 0/1 column per category (first dropped) | best when there are few categories; unseen categories become all zeros |
| `frequency` | the category's share of the training rows (1 column) | works for any cardinality but cannot tell apart categories with the same share |
| `hashing` | hashes categories into `hashing_n_features` count columns | fixed width, handles unseen categories, but collisions blur categories |
| `target` | smoothed mean of the target per category (1 column) | needs `y`; training features are **cross-fitted** (5-fold) so no row is encoded with its own target, which `DataPreprocessor.fit_transform` handles; unseen categories get the overall target mean |
| `auto` | one-hot when a column has at most `max_onehot_cardinality` categories in the training data, else `high_cardinality_encoding` | the plan is stored in `encoding_plan_` / `encoding_report_` |

`python examples/compare_encoders.py` compares them on synthetic data with a 60-category column whose categories each have their own price
effect (5-fold CV R-squared with linear regression): one-hot 0.931 (68 features), **target 0.925 (9 features)**, hashing 0.264, frequency 0.104.
Frequency and hashing do poorly here because category identity matters; they are mainly useful when the *share* of a category or a fixed memory
budget matters. These numbers come from synthetic data and show behaviour, not a result about the housing data.

## Testing and validation

**Unit tests** (`python -m pytest`, 208 tests, about 25 seconds) use a small synthetic dataset from `tests/conftest.py`
so they do not depend on the real CSV. They cover:

- configuration validation (invalid values and conflicting column roles are rejected)
- `clean_data` (duplicates, missing and non-positive targets, text normalisation, input not modified)
- `split_data` (stratification, reproducibility) and `split_distribution_report`
- `IQROutlierClipper` and `SafeLogTransformer` in isolation
- `DataPreprocessor`: output shape and names, no NaN, leakage (test features scaled with training statistics,
  checked against a hand computation), unseen categories, extra or missing columns, determinism, scalers,
  float32 output, save/load, `clone`
- imputation: all strategies, exact missing counts in the report, text treated as missing, missing-indicator columns, fit-time logging
- feature selection in the preprocessor (variance and Lasso, including the "needs y" error)
- `FeatureEngineer`: every group and its formulas checked by hand, standardisation with training statistics (no leakage),
  winsorising, NaN propagation, constant columns, tiers from training quantiles, switches and custom pairs, error handling
- `FeatureSelector`: every method on a synthetic frame with known signal, noise, a near-duplicate and a constant column;
  `n_features`, protection, consensus votes, the "needs y" error, VIF behaviour (including the main-effects regression test) and chained selectors
- imputation options (per-column, per-group, KNN) and every categorical encoder, including a test that target-encoded training features are cross-fitted
- configuration: dictionary and YAML/JSON round trips, unknown keys rejected, defaults < file < command line precedence
- `src/experiment.py`: config hashing, run IDs (including collisions), versioned folders, JSON conversion, profiling, run record and registry
- the command line end to end: versioned outputs, `results.json`, rerunning from a saved config, `--no-save`, `--no-versioning`
- pipelines and evaluation: price-scale predictions, missing values and unseen categories, cross-validation with selection inside,
  the corrected p-value checked against a manual calculation, the one-standard-error rule, and a curved-data test where engineering must help
- target transforms (round trip and skew), model pipelines on the original price scale, cross-validation
  with missing values, and an end-to-end run on deliberately dirty data

To check that the tests are meaningful, bugs were injected on purpose (a clipper that does not clip, a scaler that uses the
test set's own statistics, a wrong composite formula, an engineered-feature scaler that leaks test statistics, selector protection removed, leaky target-encoding output, an ignored
per-column imputer override, run-ID collisions, a config file overriding the command line)
and the suite failed as it should each time.

**Validation script** (`python main.py --validate`) runs 15 checks on the real data and exits with a non-zero status if any fail:
no NaN/inf in outputs, scaling statistics, leakage, skew reduction, outlier capping, robustness to corrupted data,
correct imputation report, reversible target transform, save/load, split proportions, cleaning, config validation, an
end-to-end linear baseline, leak-free and deterministic feature engineering, and the engineered + selected pipeline beating the baseline on the test set. The baseline prints test R-squared of about 0.86 with a raw target and about 0.96 with the log target
(plain linear regression, seed 42). It is only a sanity check of the pipeline, not a modelling result.

## Performance notes and profiling

The pipeline is written for clarity and safety first. Measured on 300,000 rows (about 51 MB as a DataFrame, one unused column):

| | float64 (default) | float32 |
|---|---|---|
| fit | about 0.9 s | about 0.7 s |
| transform | about 0.3 s | about 0.4 s |
| processed output size | 21.6 MB | 10.8 MB |

- `DataPreprocessor` copies **only the columns it uses**, so unused columns add no memory.
- `output_dtype="float32"` halves the size of the processed data (use it for large data; keep float64 when you need full numeric precision).
- `clean_data` copies the whole frame by default so the caller's data is never modified. `copy=False` skips that copy but then modifies the
  input in place; it saved only about 7% of peak memory in the same measurement, so it is a minor option.
- **Stage timings** are printed at the end of every run and stored in `results.json`; `--profile` also records peak Python memory per stage
  (via `tracemalloc`, which slows the run, so it is off by default). On the 1,000-row dataset: loading 0.006 s, preprocessing 0.07 to 0.23 s,
  feature engineering and selection about 0.17 s (peak 1.7 MB), the 15 validation checks about 1 s, and the full
  `--compare-features --compare-models` comparison (about 16 scenarios of 15 cross-validation fits each, plus Ridge and tree models) about 163 s.
  Nearly all of that time is model fitting inside cross-validation, not feature creation.
- The pipeline works on in-memory pandas DataFrames. For data that does not fit in memory it would need chunked or out-of-core processing,
  which is not implemented. `LassoCV` selection is the slowest optional step because it cross-validates.

## Scope decisions

- **No temporal features.** Lags, rolling statistics, seasonality and "time since" features were deliberately left out: the dataset contains no
  timestamp or date column, so there is nothing to derive them from. If dated sales were added, a time-aware split (train on earlier, test on later
  data) would also be needed instead of the random stratified split.
- **No higher-cardinality columns in this dataset.** The extra encoders exist so the pipeline is reusable, and they are tested and demonstrated on
  synthetic data only. They were not needed for the housing data, where one-hot encoding of `location_type` is used.
- **No class-balancing methods.** This is a regression problem, so oversampling and class weights do not apply. The split report instead
  checks that the stratification column and the target distribution are preserved.

## Developer tooling

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .    # lint and formatting (configured in pyproject.toml)
mypy                                     # static type check of src/ and main.py
pre-commit install                       # run the checks automatically on every commit (and the tests before every push)
```

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs the unit tests and the validation suite on Python 3.9, 3.11 and 3.12, and lint, format and type checks on 3.12.
The workflow has not been executed yet (it needs the repository on GitHub); everything in this repository was run locally on Python 3.12 only, so 3.9 and 3.11 support is intended but unverified.
The type checker is configured leniently because scikit-learn's estimator API (attributes created in `fit`) does not fit strict typing; the code in
`src/` and `main.py` is annotated and currently passes `mypy` and `ruff` with no findings.

## Assumptions and limitations

- Preprocessing expects the column names of this dataset; other datasets need an updated config.
- Stratification is on `location_type` only. The split report shows that the target itself can still differ between
  train and test (here the test set happens to contain the most expensive homes, so its skew is higher); with a small dataset a repeated or
  cross-validated evaluation is more reliable than one split.
- The extra encoders, KNN imputation and per-column strategies add flexibility but add no accuracy on this dataset (no missing values, one 3-level categorical column).
- Feature selection built into the preprocessor is optional and off by default: with only 8 predictors and low multicollinearity (VIF at most about 2.8) the EDA found nothing to remove.
- `proximity_to_public_transport` has a counterintuitive positive relation with price. It is
  kept as is; confirm its definition with the data owner before interpreting it.
- Clipping and log floors are learned from 800 training rows, so the Rural group (about 80 rows in
  training) contributes little to those statistics.
