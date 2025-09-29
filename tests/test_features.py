from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from src import features  # noqa: E402


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [45, 50, 60, 30],
            "chol": [210, 180, 190, 220],
            "sex": ["M", "F", "M", "F"],
        }
    )


def test_preprocessor_produces_dense_matrix():
    df = _sample_dataframe()
    preprocessor = features.build_preprocessor(df, use_pca=False)
    transformed = features.fit_transform(preprocessor, df)
    assert isinstance(transformed, np.ndarray)
    assert not np.isnan(transformed).any()


def test_preprocessor_with_pca_reduces_dimension():
    df = _sample_dataframe()
    preprocessor = features.build_preprocessor(df, use_pca=True, pca_components=0.95)
    transformed = features.fit_transform(preprocessor, df)
    assert transformed.shape[1] <= features.fit_transform(features.build_preprocessor(df, use_pca=False), df).shape[1]
