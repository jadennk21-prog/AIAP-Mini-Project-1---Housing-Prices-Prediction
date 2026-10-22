## 1. Project objective and problem definition

This project predicts **`house_sale_price`**, a continuous numeric value, from property characteristics. Therefore, the supplied dataset represents a **supervised regression problem**, not a classification problem.

> **Important clarification:** The task description uses both “classification” and “None regression.” These conflict. Since `house_sale_price` is a continuous target, this README treats the project as **regression**. If your project brief specifically requires classification, confirm the intended target with your instructor before modelling. One possible derived classification target would be whether a sale price is above or below the dataset median, but that would answer a different question and discard price magnitude.

The literature review informs the modelling workflow: data inspection, preprocessing, feature engineering, candidate model selection, validation, evaluation, interpretation, and limitations.

## 2. Dataset inspection

The supplied file, `Housing_Price_Dataset.csv`, was inspected for this review.

| Item | Finding |
|---|---|
| Rows / columns | 1,000 rows × 9 columns |
| Target | `house_sale_price` |
| Predictors | 8 features |
| Missing values | None detected in the supplied file |
| Categorical feature | `location_type` |
| Categories | Suburban: 500; Urban: 400; Rural: 100 |
| Target mean | 350,137.92 |
| Target median | 323,881.58 |
| Target standard deviation | 130,904.60 |
| Target range | 120,520.43–1,202,878.27 |

### Features

| Feature | Type | Initial handling / questions |
|---|---|---|
| `property_size` | Numeric | Inspect distribution and extreme values; units are not specified in the file |
| `number_of_bedrooms` | Numeric in file | Values include decimals and negatives, which are unusual for a bedroom count; verify whether these values are synthetic/standardized or erroneous before treating them as literal counts |
| `number_of_bathrooms` | Numeric in file | Inspect decimals and negative values; verify meaning |
| `location_type` | Categorical nominal | One-hot encode; do not impose an arbitrary order on Rural, Suburban, and Urban |
| `age_of_property` | Numeric | Inspect range and relationship with price |
| `proximity_to_public_transport` | Numeric | Confirm whether a larger value means closer or farther away; the column name alone does not define the direction |
| `crime_rate_in_area` | Numeric | Inspect scale, skew and possible outliers |
| `school_quality_score` | Numeric | Confirm score range and direction |

The Pearson correlations with the target in this dataset are approximately:

- `property_size`: **+0.708**
- `number_of_bedrooms`: **+0.648**
- `school_quality_score`: **+0.606**
- `number_of_bathrooms`: **+0.570**
- `proximity_to_public_transport`: **+0.297**
- `age_of_property`: **−0.284**
- `crime_rate_in_area`: **−0.401**

These are exploratory associations, not evidence of causation and not proof that a feature will improve out-of-sample performance. Correlation captures linear association and can miss nonlinear relationships. Compute correlations using the training data during the modelling workflow, and examine scatterplots and grouped summaries as well.

## 3. Literature review method

A useful review should connect reliable sources to decisions in this specific project rather than merely list algorithms. Use the following search and synthesis method:

1. Start with peer-reviewed foundational papers for model families.
2. Use official scikit-learn documentation for practical preprocessing, validation, pipelines, and metric definitions.
3. Record each source's question, method, strengths, limitations, and relevance to this dataset.
4. Convert findings into explicit decisions and test them empirically on a held-out test set.
5. Keep a small evidence table so every major methodological choice can be traced to a source.

This is a focused applied literature review, not a systematic review of every published house-price study. For a formal systematic review, add databases searched, search dates, exact search strings, inclusion/exclusion criteria, and a screening log.

## 4. Findings and implications

### 4.1 Recommended modelling workflow

Use a reproducible supervised-learning workflow:

1. Validate data types, ranges, duplicates, target distribution, and feature meanings.
2. Split into training and test sets before fitting transformations or selecting features.
3. Use cross-validation on the training data for model selection and hyperparameter tuning.
4. Put preprocessing and the estimator in a single pipeline.
5. Compare simple baselines and a small set of complementary regression models.
6. Choose a primary metric before tuning; report additional metrics for context.
7. Evaluate the selected model once on the held-out test set.
8. Inspect residuals, subgroup performance, and feature importance; document limitations.

For this dataset, a random split is a reasonable starting point **only if rows are independent and represent the same data-generating period/population**. If records are grouped by neighbourhood, seller, property development, or time, use a group-aware or time-aware split to avoid leakage and better represent deployment.

### 4.2 Candidate model architectures

| Model | Why include it | Main limitations / tuning considerations |
|---|---|---|
| Mean/median dummy regressor | Establishes a minimum baseline; models must beat it | Not intended as a useful production predictor |
| Linear Regression | Interpretable baseline; tests whether additive linear relationships are adequate | Sensitive to multicollinearity and outliers; may underfit nonlinear relationships |
| Ridge Regression | Linear baseline with L2 regularization | Tune `alpha`; standardize numeric predictors because regularization depends on scale |
| Random Forest Regressor | Captures nonlinearities and feature interactions with limited manual feature construction | Can be less effective at extrapolating beyond observed target/feature ranges; tune tree count, depth, leaf size, and feature sampling |
| Gradient Boosting Regressor (e.g. HistGradientBoostingRegressor) | Often effective on structured/tabular data; learns sequential corrections to previous trees | Tune learning rate, iteration count, leaf complexity and regularization; monitor overfitting |
| Support Vector Regression (SVR) | RBF kernel can represent nonlinear relationships in smaller/medium-sized datasets | Requires numeric scaling; tune `C`, `epsilon`, and `gamma`; may be less convenient to interpret |
| K-Nearest Neighbours Regressor | Simple nonlinear local baseline | Sensitive to scaling and distance choices; can struggle in higher dimensions and uneven-density regions |

**Recommended initial comparison:** DummyRegressor, Ridge/Linear Regression, RandomForestRegressor, and HistGradientBoostingRegressor. Add SVR or KNN if time allows. With 1,000 rows and 8 original features, begin with classical tabular models rather than a neural network: deep learning is not justified by dataset size alone.

**Evidence:** Breiman (2001) introduced Random Forests and discussed generalization and variable importance. Official scikit-learn documentation provides practical estimator and evaluation guidance. See references below.

### 4.3 Data preprocessing

- **Numeric features:** Check units, valid ranges, skew, and outliers. Imputation is not currently required by the file's missing-value check, but a pipeline can include median imputation for robustness if future data may be incomplete.
- **Categorical features:** One-hot encode `location_type`. The categories are nominal, so arbitrary ordinal encoding would introduce a false ordering. Configure unknown-category handling for inference data.
- **Scaling:** Standardize numeric features for Ridge, SVR, and KNN. Tree-based models generally do not require feature scaling.
- **Target:** Keep `house_sale_price` out of the predictor matrix. Avoid transforming the target unless diagnostics and business interpretation justify it; if a log transform is tested, inverse-transform predictions and evaluate them in the original currency units.
- **Pipeline:** Fit encoders, scalers, feature selection, and any other learned transformation only on training folds. A scikit-learn `Pipeline`/`ColumnTransformer` helps prevent train-test leakage and keeps inference consistent.

Do not automatically remove outliers just because they are extreme. First determine whether they are errors or valid high-value properties. Report results with and without a justified treatment only if there is a defensible reason to do so.

### 4.4 Feature engineering

Candidate features to investigate, subject to confirming the data's meaning:

- **Nonlinear size effect:** compare raw `property_size` with a log transform or spline/polynomial terms for linear models.
- **Size interactions:** test `property_size × location_type` or size-by-quality interactions if domain reasoning and validation support them.
- **Age nonlinearity:** property age may have a nonlinear relationship with price; compare a spline or tree-based model.
- **Ratios:** price-per-size is a useful *analysis* variable but must not be used as a predictor when it is calculated from the target. A bedroom-per-size ratio could be explored only if bedroom values are confirmed to be meaningful.
- **Outlier flags:** consider only when the extreme values are valid and the flag can be computed consistently at inference time.
- **Feature selection:** use domain reasoning, cross-validated performance, and stability rather than correlation alone.

Feature engineering should be hypothesis-driven. Compare each change against a baseline with the same validation folds. Do not create many features simply because they are possible.

### 4.5 Evaluation strategy and metrics

Use a **train / validation-by-cross-validation / test** design. For example, hold out 20% as a final test set and use 5-fold cross-validation on the remaining 80% for model and hyperparameter selection. Fix a random seed for reproducibility. If data is grouped or temporal, replace random splitting with an appropriate group/time split.

| Metric | Interpretation | Role in this project |
|---|---|---|
| MAE (Mean Absolute Error) | Average absolute prediction error in the target's units; relatively easy to explain | **Recommended primary metric** because it communicates a typical absolute error |
| RMSE (Root Mean Squared Error) | Square-root of mean squared error; penalizes large errors more strongly than MAE | Secondary metric to highlight costly large misses |
| R² (coefficient of determination) | Relative to predicting the mean, measures variance explained under the evaluation sample; can be negative | Contextual metric, not an error in currency units |
| Median Absolute Error | Median absolute error; less influenced by a small number of very large errors | Robust supplementary metric |
| MAPE | Average absolute percentage error | Use cautiously: percentage errors can be unstable when true values are near zero; target here is positive, but MAE/RMSE are more directly interpretable |

Report at minimum **MAE, RMSE, and R²** on cross-validation and the final test set. Include the units/currency assumption clearly; the CSV does not specify a currency. Compare to the dummy baseline. Do not select a model using test-set performance repeatedly, because that leaks information from the test set into model selection.

### 4.6 Hyperparameter tuning

Use `RandomizedSearchCV` or a modest `GridSearchCV` over the training data with a fixed cross-validation strategy. Example parameters:

- Ridge: `alpha`
- Random Forest: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
- HistGradientBoosting: `learning_rate`, `max_iter`, `max_leaf_nodes`, `l2_regularization`
- SVR: `C`, `epsilon`, `gamma`

Tune against negative MAE if MAE is the primary metric, then inspect RMSE and R² as secondary metrics. Search spaces should be small and purposeful given the dataset size. Use the same folds when comparing models to reduce noise in the comparison.

### 4.7 Residual analysis, interpretability, and reliability

After choosing a model:

- Plot actual versus predicted prices and residuals versus predicted values.
- Check whether errors vary with price, property size, location type, or other relevant groups.
- Look for systematic overprediction/underprediction and heteroscedasticity.
- Use permutation importance on held-out data or cross-validation to assess feature reliance; correlated features can divide or distort importance.
- Consider partial dependence or individual explanations only with awareness of correlated predictors.
- Report uncertainty honestly. A point prediction is not a guarantee of sale price.
- Reassess performance when the market, geography, property mix, or data collection process changes.

Fairness and subgroup checks should be framed around legitimate use and available data. `location_type` may act as a proxy for socioeconomic or geographic differences. Investigate errors by group, but do not infer sensitive attributes or causal effects from these features alone.

## 5. Project decisions derived from the review

| Decision | Proposed approach | Reason |
|---|---|---|
| Problem type | Regression | `house_sale_price` is continuous |
| Baseline | Mean/median DummyRegressor | Establishes a reference point |
| Main candidates | Ridge, Random Forest, HistGradientBoosting | Balances interpretability and nonlinear capacity |
| Categorical preprocessing | OneHotEncoder for `location_type` | Nominal categories have no natural rank |
| Scaling | StandardScaler for linear regularized / distance-based models | Helps scale-sensitive algorithms |
| Validation | 5-fold CV on training split; untouched test split | Estimates generalization and limits leakage |
| Primary metric | MAE | Interpretable typical error |
| Secondary metrics | RMSE and R² | Large-error sensitivity and variance-explanation context |
| Interpretability | Residual analysis and permutation importance | Reveals systematic error and model reliance |
| Key data-quality action | Investigate decimal/negative bedroom and bathroom values | Their semantics are unclear and may be invalid if literal counts |

These are starting decisions, not results. The final model should be selected by empirical validation and documented limitations.

## 6. Risks and limitations

1. **Dataset representativeness:** only 1,000 rows are supplied. The file does not describe the collection process, geography, date range, or whether it represents actual transactions.
2. **Unclear feature semantics:** bedroom/bathroom values include decimals and negatives; the units/direction of some fields are undocumented.
3. **Missing context:** sale price currency, market date, property type, floor, condition, and neighbourhood-level identifiers are not specified.
4. **Correlation is not causation:** observed associations should not be interpreted as causal price drivers.
5. **Synthetic-data possibility:** unusual values may indicate generated or transformed data. Confirm this with the dataset provider before interpreting features literally.
6. **Distribution shift:** a model trained on this file may not generalize to a different region or future market.
7. **Evaluation uncertainty:** with a modest dataset, scores can vary by split. Use cross-validation and report the split strategy.

## 7. Eight-day execution plan

| Day | Activities | Output |
|---|---|---|
| 1 | Confirm task is regression; define objective, target, scope, and search strategy | Problem statement and review plan |
| 2 | Gather foundational papers and official documentation | Annotated source list |
| 3 | Review regression metrics, validation, leakage, and preprocessing | Evidence notes |
| 4 | Review linear and regularized models plus feature engineering | Model comparison notes |
| 5 | Review tree ensembles, SVR/KNN, interpretability, and limitations | Candidate-model rationale |
| 6 | Inspect dataset and connect findings to literature | Dataset profile and proposed pipeline |
| 7 | Synthesize decisions, metrics, risks, and references in README | Draft README |
| 8 | Verify citations, consistency, reproducibility, and task wording | Final README.md |

## 8. Suggested implementation sequence after the review

1. Create `README.md` at the repository root.
2. Keep `Housing_Price_Dataset.csv` in the project data location, and document the path without committing sensitive data.
3. Build a notebook or scripts for data validation and exploratory analysis.
4. Implement preprocessing with `ColumnTransformer` and model-specific pipelines.
5. Compare the baseline and candidate models using identical cross-validation folds.
6. Tune the strongest candidate, evaluate it once on the test set, and save the metrics and plots.
7. Update this README with actual experimental results only after running the experiments. Do not invent scores in the literature review.

## 9. References

1. Breiman, L. (2001). **Random Forests.** *Machine Learning*, 45, 5–32. https://doi.org/10.1023/A:1010933404324
2. scikit-learn developers. **Model evaluation: quantifying the quality of predictions.** Official documentation. https://scikit-learn.org/stable/modules/model_evaluation.html
3. scikit-learn developers. **OneHotEncoder.** Official API documentation. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
4. scikit-learn developers. **Preprocessing data.** Official documentation. https://scikit-learn.org/stable/modules/preprocessing.html
5. scikit-learn developers. **Pipelines and composite estimators.** Official documentation. https://scikit-learn.org/stable/modules/compose.html
6. scikit-learn developers. **Cross-validation: evaluating estimator performance.** Official documentation. https://scikit-learn.org/stable/modules/cross_validation.html
7. scikit-learn developers. **Permutation feature importance.** Official documentation. https://scikit-learn.org/stable/modules/permutation_importance.html
8. scikit-learn developers. **RandomForestRegressor.** Official API documentation. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
9. scikit-learn developers. **HistGradientBoostingRegressor.** Official API documentation. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html

## 10. Final conclusion

The literature supports starting with a reproducible tabular-regression workflow, a simple baseline, interpretable linear models, and tree-based ensembles. For this dataset, preprocessing should treat `location_type` as nominal, scale features for scale-sensitive algorithms, and keep all learned transformations inside cross-validation pipelines. MAE is a practical primary metric, complemented by RMSE and R². Before modelling, clarify the unusual bedroom/bathroom values and confirm feature definitions. The final model choice must be based on held-out performance, residual analysis, and the intended deployment context—not on literature or correlation alone.

## Exploratory Data Analysis(EDA)

## Initial Data Exploration and Profiling 

### Objective

The initial exploratory data analysis (EDA) examines the structure, data types, completeness, and basic statistical characteristics of the housing-price dataset before preprocessing and model development.

### Dataset Overview

The dataset contains 1,000 observations and 9 columns, consisting of 8 predictors and the target variable `house_sale_price`.

The target is continuous, so the project is treated as a supervised regression problem.

### Data Quality

Initial inspection found no missing values or exact duplicate rows in the supplied dataset.

However, `number_of_bedrooms` and `number_of_bathrooms` contain negative and fractional values. These values require clarification because they are unusual if the features represent literal room counts.

### Feature Types

`location_type` is a categorical feature. The remaining columns are stored as numeric variables.

The categorical feature will be investigated further to determine the appropriate encoding strategy.

### Target Variable

The target variable is `house_sale_price`. Its mean is approximately 350,137.92, while its median is approximately 323,881.58.

The difference between the mean and median suggests right skewness, which will be investigated using distribution plots and further statistical analysis.

### Next Steps

The next EDA stage will examine feature distributions, potential outliers, correlations, and relationships between predictors and the target. These findings will inform preprocessing and feature engineering decisions.
