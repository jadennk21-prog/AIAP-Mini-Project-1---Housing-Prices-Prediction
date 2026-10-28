"""Compare categorical encodings on synthetic data with a high-cardinality column.

Run from the repository root:   python examples/compare_encoders.py

The housing dataset has no high-cardinality column, so this example adds a synthetic
``neighbourhood`` column (60 categories, each with its own price effect) to a copy of the
data and shows how each encoding behaves. Results depend on the data: here category identity
matters, which is exactly the case that separates the encoders.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing import DataPreprocessor, PreprocessingConfig, make_model_pipeline  # noqa: E402

rng = np.random.default_rng(0)
n, n_categories = 1500, 60
categories = [f"hood{i:02d}" for i in range(n_categories)]
effect = {c.title(): v for c, v in zip(categories, rng.normal(0, 0.4, n_categories))}
df = pd.DataFrame(
    {
        "property_size": rng.lognormal(7.8, 0.5, n),
        "crime_rate_in_area": rng.lognormal(-3, 1, n),
        "proximity_to_public_transport": rng.lognormal(1.9, 0.9, n),
        "age_of_property": rng.uniform(0.03, 1.2, n),
        "school_quality_score": rng.normal(7, 1.5, n),
        "number_of_bedrooms": rng.normal(0, 1, n),
        "number_of_bathrooms": rng.normal(0, 1, n),
        "location_type": rng.choice(["Rural", "Suburban", "Urban"], n),
        "neighbourhood": rng.choice(categories, n),
    }
)
df["house_sale_price"] = np.exp(
    12 + 0.3 * np.log(df["property_size"]) + df["neighbourhood"].str.title().map(effect) + rng.normal(0, 0.1, n)
)
X, y = df.drop(columns="house_sale_price"), df["house_sale_price"]

rows = []
for encoding in ["onehot", "frequency", "hashing", "target", "auto"]:
    cfg = PreprocessingConfig(categorical_features=["location_type", "neighbourhood"], categorical_encoding=encoding)
    n_features = DataPreprocessor(cfg).fit_transform(X, np.log(y)).shape[1]
    r2 = cross_val_score(
        make_model_pipeline(LinearRegression(), cfg), X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="r2"
    ).mean()
    rows.append({"encoding": encoding, "features": n_features, "cv_r2": round(r2, 3)})
print(pd.DataFrame(rows).set_index("encoding").to_string())
print("\n'auto' uses one-hot for location_type (3 categories) and target encoding for neighbourhood (60 > 10).")
