from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import utils

NA_VALUES = ["", "NA", "N/A", "na", "n/a", "?", "null", "NULL", "NaN", "nan"]


def _read_with_encoding(path: Path, encoding: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding=encoding,
        na_values=NA_VALUES,
        keep_default_na=True,
        engine="c",
    )


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV with a couple of sensible fallbacks for encoding."""
    path = Path(path)
    try:
        return _read_with_encoding(path, "utf-8")
    except UnicodeDecodeError:
        return _read_with_encoding(path, "latin-1")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Trim strings, drop duplicates, and impute missing values."""
    cleaned = df.copy()

    obj_cols = cleaned.select_dtypes(include=["object", "string"]).columns
    for col in obj_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    cleaned.replace("", np.nan, inplace=True)
    cleaned.drop_duplicates(inplace=True)

    num_cols = cleaned.select_dtypes(include=["number", "float", "int"]).columns
    for col in num_cols:
        if cleaned[col].isna().any():
            median = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(median)

    cat_cols = cleaned.select_dtypes(include=["object", "category", "string"]).columns
    for col in cat_cols:
        if cleaned[col].isna().any():
            mode = cleaned[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            cleaned[col] = cleaned[col].fillna(fill_value)

    return cleaned


def split(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.3,
    stratify: bool = True,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataframe into train/test partitions."""
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe")

    X = df.drop(columns=[target])
    y = df[target]

    stratify_y = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y,
    )
    return X_train, X_test, y_train, y_test


def print_data_dictionary(df: pd.DataFrame) -> None:
    """Print a compact data dictionary to stdout."""
    dictionary = utils.describe_dataframe(df)
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print("Data Dictionary:")
        print(dictionary.to_string(index=False))
