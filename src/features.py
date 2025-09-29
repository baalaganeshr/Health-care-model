from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def detect_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return lists of numeric and categorical columns."""
    numeric = df.select_dtypes(include=["number", "float", "int"]).columns.tolist()
    categorical = [col for col in df.columns if col not in numeric]
    return numeric, categorical


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older scikit-learn versions
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    df: pd.DataFrame,
    use_pca: bool = False,
    pca_components: float | int = 0.95,
) -> Pipeline:
    """Create preprocessing pipeline with optional PCA."""
    numeric_features, categorical_features = detect_feature_types(df)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _make_one_hot_encoder()),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )

    steps = [("preprocessor", preprocessor)]
    if use_pca and transformers:
        steps.append(("pca", PCA(n_components=pca_components)))

    return Pipeline(steps)


def transform(preprocessor: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Convenience helper to ensure dense numpy ndarray output."""
    transformed = preprocessor.transform(df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return np.asarray(transformed, dtype=np.float32)


def fit_transform(preprocessor: Pipeline, df: pd.DataFrame) -> np.ndarray:
    transformed = preprocessor.fit_transform(df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return np.asarray(transformed, dtype=np.float32)
