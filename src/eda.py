"""Reusable EDA functions."""

import pandas as pd


def describe_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include="all").T


def category_summary(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:

    return (
        df.groupby(column, dropna=False)["selling_price"]
        .agg(["count", "median", "mean"])
        .sort_values("count", ascending=False)
    )
