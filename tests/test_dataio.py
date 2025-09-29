from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from src import dataio  # noqa: E402


def test_load_csv_returns_dataframe():
    csv_path = ROOT / "data" / "heart.csv"
    df = dataio.load_csv(csv_path)
    assert not df.empty
    assert "target" in df.columns


def test_clean_removes_missing_and_duplicates():
    df = pd.DataFrame(
        {
            "age": [60, 60, np.nan, 70],
            "sex": ["M", "M", "F", None],
            "target": [1, 1, 0, 0],
        }
    )
    df.loc[1] = df.loc[0]
    cleaned = dataio.clean(df)
    assert cleaned.isna().sum().sum() == 0
    assert len(cleaned) < len(df)


def test_split_respects_configuration():
    df = pd.DataFrame(
        {
            "feat": np.arange(100),
            "target": [0] * 50 + [1] * 50,
        }
    )
    X_train, X_test, y_train, y_test = dataio.split(
        df,
        target="target",
        test_size=0.2,
        stratify=True,
        random_state=42,
    )
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert y_train.mean() == pytest.approx(y_test.mean(), rel=0, abs=1e-6)
