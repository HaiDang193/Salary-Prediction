
"""Preprocessing for Car Price Prediction."""

import pandas as pd

from src.config import EXPECTED_COLUMNS


REFERENCE_YEAR = 2026

OWNER_MAP = {
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4,
    "Test Drive Car": 0,
}

CATEGORICAL = [
    "brand",
    "fuel",
    "seller_type",
    "transmission",
]


def extract_brand(name: str) -> str:
    words = str(name).strip().split()

    if (
        len(words) >= 2
        and words[0].lower() == "land"
        and words[1].lower() == "rover"
    ):
        return "Land Rover"

    return words[0] if words else "Unknown"


def preprocess_raw(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:

    # Validate input columns
    missing = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing input columns: {sorted(missing)}"
        )

    data = df[EXPECTED_COLUMNS].copy()

    # Validate numeric values
    for column in ["year", "km_driven"]:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    if data[["year", "km_driven"]].isna().any().any():
        raise ValueError("Missing numeric values")

    if (data["km_driven"] < 0).any():
        raise ValueError("Mileage must be non-negative")

    if not data["owner"].isin(OWNER_MAP).all():
        raise ValueError("Unknown owner category")

    # Feature engineering
    brand = data["name"].apply(extract_brand)

    output = pd.DataFrame(index=data.index)

    output["km_driven"] = data["km_driven"]
    output["owner"] = data["owner"].map(OWNER_MAP).astype(int)
    output["vehicle_age"] = REFERENCE_YEAR - data["year"]

    sources = {
        "brand": brand,
        "fuel": data["fuel"],
        "seller_type": data["seller_type"],
        "transmission": data["transmission"],
    }

    # Generate exactly the features used during training
    for feature in feature_columns:

        if feature in output.columns:
            continue

        prefix = next(
            (
                col for col in CATEGORICAL
                if feature.startswith(col + "_")
            ),
            None,
        )

        if prefix is None:
            raise ValueError(
                f"Unknown model feature: {feature}"
            )

        category = feature[len(prefix) + 1:]

        output[feature] = (
            sources[prefix] == category
        ).astype(int)

    return output.reindex(columns=feature_columns)
