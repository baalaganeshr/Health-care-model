from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from src import models  # noqa: E402


def _sample_data(samples: int = 80, features: int = 10):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(samples, features)).astype(np.float32)
    y = rng.integers(0, 2, size=samples)
    return X, y


def test_logistic_regression_predict_proba_shape():
    X, y = _sample_data()
    model = models.train_logistic_regression(X, y)
    proba = models.predict_proba_logistic(model, X)
    assert proba.shape == (len(X), 2)


def test_random_forest_predict_proba_shape():
    X, y = _sample_data()
    model = models.train_random_forest(X, y)
    proba = models.predict_proba_random_forest(model, X)
    assert proba.shape == (len(X), 2)


def test_dnn_forward_pass_and_probabilities():
    X, y = _sample_data(samples=64, features=8)
    config = models.DNNConfig(
        input_dim=X.shape[1],
        hidden_sizes=[16, 8],
        dropout=0.1,
        epochs=2,
        batch_size=16,
        lr=0.01,
    )
    model, history = models.train_dnn(X, y, config)
    assert history["loss"]
    proba = models.predict_proba_dnn(model, X[:10])
    assert proba.shape == (10, 2)
    assert np.all((proba >= 0) & (proba <= 1))
