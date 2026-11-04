# File này chịu trách nhiệm đọc CSV thành Pandas DataFrame và kiểm tra cấu trúc dữ liệu.
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
            f"Missing required columns: {sorted(missing)}"
        )

    df = df[EXPECTED_COLUMNS].copy()

    if df.empty:
        raise ValueError("Dataset is empty")

    if df.isna().any().any():
        raise ValueError(
            "Dataset contains missing values"
        )

    numeric_columns = [
        "year",
        "selling_price",
        "km_driven",
    ]

    for column in numeric_columns:
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


def find_dataset_csv(folder: str | Path) -> Path:
    folder = Path(folder)

    matches = []

    for csv_file in folder.rglob("*.csv"):
        try:
            columns = pd.read_csv(
                csv_file, nrows=0
            ).columns

            if set(EXPECTED_COLUMNS).issubset(columns):
                matches.append(csv_file)

        except (OSError, UnicodeError, pd.errors.ParserError):
            continue

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one compatible CSV, "
            f"found {len(matches)}: {matches}"
        )

    return matches[0]
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
            f"Missing required columns: {sorted(missing)}"
        )

    df = df[EXPECTED_COLUMNS].copy()

    if df.empty:
        raise ValueError("Dataset is empty")

    if df.isna().any().any():
        raise ValueError(
            "Dataset contains missing values"
        )

    numeric_columns = [
        "year",
        "selling_price",
        "km_driven",
    ]

    for column in numeric_columns:
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


def find_dataset_csv(folder: str | Path) -> Path:
    folder = Path(folder)

    matches = []

    for csv_file in folder.rglob("*.csv"):
        try:
            columns = pd.read_csv(
                csv_file, nrows=0
            ).columns

            if set(EXPECTED_COLUMNS).issubset(columns):
                matches.append(csv_file)

        except (OSError, UnicodeError, pd.errors.ParserError):
            continue

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one compatible CSV, "
            f"found {len(matches)}: {matches}"
        )

    return matches[0]
