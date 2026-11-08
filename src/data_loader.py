
"""Load and validate raw CSV."""

from pathlib import Path

import pandas as pd

from src.config import EXPECTED_COLUMNS


def load_csv(file_path: str | Path) -> pd.DataFrame:

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    missing = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df = df[EXPECTED_COLUMNS].copy()

    if df.empty:
        raise ValueError("Dataset is empty")

    if df.isna().any().any():
        raise ValueError("Dataset contains missing values")

    for column in [
        "year",
        "selling_price",
        "km_driven",
    ]:
        values = pd.to_numeric(
            df[column],
            errors="raise",
        )

        if (values % 1 != 0).any():
            raise ValueError(
                f"{column} must contain integers"
            )

        df[column] = values.astype("int64")

    print(f"[DATA] Loaded {len(df):,} rows")

    return df
