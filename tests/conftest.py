"""Shared fixtures: a small synthetic housing dataset so tests do not depend on the real CSV."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import PreprocessingConfig, clean_data, split_data


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 500
    size = rng.lognormal(7.8, 0.5, n)
    crime = rng.lognormal(-3.0, 1.0, n)
    transport = rng.lognormal(1.9, 0.9, n)
    age = rng.normal(0, 1, n)
    school = rng.normal(7, 1.5, n)
    beds = rng.normal(0, 1, n)
    baths = rng.normal(0, 1, n)
    location = rng.choice(["Rural", "Suburban", "Urban"], n, p=[0.1, 0.5, 0.4])
    price = np.exp(11.0 + 0.5 * np.log(size) - 0.1 * np.log(crime) + 0.05 * school + 0.1 * beds + rng.normal(0, 0.1, n))
    return pd.DataFrame(
        {
            "property_size": size,
            "crime_rate_in_area": crime,
            "proximity_to_public_transport": transport,
            "age_of_property": age,
            "school_quality_score": school,
            "number_of_bedrooms": beds,
            "number_of_bathrooms": baths,
            "location_type": location,
            "house_sale_price": price,
        }
    )


@pytest.fixture()
def cfg() -> PreprocessingConfig:
    return PreprocessingConfig()


@pytest.fixture()
def split(raw_df, cfg):
    cleaned, _ = clean_data(raw_df, cfg)
    return split_data(cleaned, cfg)  # raw_train, raw_test, y_train, y_test


@pytest.fixture()
def fitted(split, cfg):
    from src.preprocessing import DataPreprocessor

    raw_train, _, _, _ = split
    return DataPreprocessor(cfg).fit(raw_train)
