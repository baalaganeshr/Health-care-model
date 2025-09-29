from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


@dataclass
class DNNConfig:
    input_dim: int
    hidden_sizes: List[int]
    dropout: float
    epochs: int
    batch_size: int
    lr: float


class HeartMLP(nn.Module):
    """Simple feed-forward network for tabular data."""

    def __init__(self, input_dim: int, hidden_sizes: List[int], dropout: float = 0.2) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = size
        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        return logits.squeeze(-1)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_logistic_regression(X: np.ndarray, y: Iterable[int]) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X, y)
    return model


def predict_proba_logistic(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)


def train_random_forest(X: np.ndarray, y: Iterable[int]) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X, y)
    return model


def predict_proba_random_forest(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)


def _build_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_dnn(
    X: np.ndarray,
    y: Iterable[int],
    config: DNNConfig,
) -> Tuple[HeartMLP, Dict[str, List[float]]]:
    device = get_device()
    model = HeartMLP(config.input_dim, config.hidden_sizes, config.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    X_np = np.asarray(X, dtype=np.float32)
    y_np = np.asarray(list(y), dtype=np.float32)
    dataloader = _build_dataloader(X_np, y_np, config.batch_size)

    history: Dict[str, List[float]] = {"loss": []}
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        model.train()
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss /= len(dataloader.dataset)
        history["loss"].append(epoch_loss)
        tqdm.write(f"Epoch {epoch + 1}/{config.epochs} - loss: {epoch_loss:.4f}")

    return model, history


def predict_proba_dnn(model: HeartMLP, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(X.astype(np.float32)).to(device)
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
    # Convert to shape (n_samples, 2) to keep parity with sklearn .predict_proba
    probs = np.asarray(probs)
    probs = probs.reshape(-1, 1)
    probs = np.hstack([1 - probs, probs])
    return probs


def save_dnn(model: HeartMLP, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_dnn(path: str, input_dim: int, hidden_sizes: List[int], dropout: float) -> HeartMLP:
    model = HeartMLP(input_dim, hidden_sizes, dropout)
    model.load_state_dict(torch.load(path, map_location=get_device()))
    model.to(get_device())
    model.eval()
    return model
