
"""Raw dataset validation."""

import pandas as pd

from src.config import EXPECTED_COLUMNS


def validate_cars(df: pd.DataFrame) -> dict:

    missing = sorted(
        set(EXPECTED_COLUMNS) - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    prices = pd.to_numeric(
        df["selling_price"],
        errors="coerce",
    )

    return {
        "rows": len(df),
        "duplicates": int(df.duplicated().sum()),
        "missing_values": (
            df[EXPECTED_COLUMNS].isna().sum().to_dict()
        ),
        "invalid_prices": int(
            (prices.isna() | (prices <= 0)).sum()
        ),
    }
