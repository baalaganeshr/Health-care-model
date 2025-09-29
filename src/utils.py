from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_project_root() -> Path:
    """Return repository root assuming this file lives in src/."""
    return PROJECT_ROOT


def resolve_path(*parts: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    return get_project_root().joinpath(*parts)


def ensure_dir(path: str | Path) -> Path:
    """Create directory path if it does not already exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed: int = 42) -> None:
    """Seed python, numpy, and torch (cpu/cuda) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_numpy(array_like: Any) -> np.ndarray:
    """Convert inputs to a 1D numpy array."""
    if isinstance(array_like, np.ndarray):
        return array_like
    if hasattr(array_like, "numpy"):
        return array_like.detach().cpu().numpy()
    if isinstance(array_like, (list, tuple)):
        return np.asarray(array_like)
    raise TypeError(f"Unsupported type for conversion to numpy: {type(array_like)!r}")


def compute_classification_metrics(
    y_true: Iterable[int],
    y_prob: Iterable[float] | np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float | List[List[int]]]:
    """Return standard binary-classification metrics."""
    y_true = to_numpy(list(y_true)).astype(int)
    y_prob_arr = np.asarray(y_prob)
    if y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == 2:
        y_prob_arr = y_prob_arr[:, 1]
    y_pred = (y_prob_arr >= threshold).astype(int)
    metrics: Dict[str, float | List[List[int]]] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob_arr) if len(np.unique(y_true)) > 1 else 0.0),
    }
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def format_metrics_table(metrics_by_model: Mapping[str, Mapping[str, Any]]) -> str:
    """Format metrics into an aligned text table."""
    headers = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = [headers]
    for name, metrics in metrics_by_model.items():
        row = [
            name,
            f"{metrics.get('accuracy', 0):.4f}",
            f"{metrics.get('precision', 0):.4f}",
            f"{metrics.get('recall', 0):.4f}",
            f"{metrics.get('f1', 0):.4f}",
            f"{metrics.get('roc_auc', 0):.4f}",
        ]
        rows.append(row)

    col_widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    lines = []
    for idx, row in enumerate(rows):
        padded = [row[i].ljust(col_widths[i]) for i in range(len(headers))]
        lines.append(" | ".join(padded))
        if idx == 0:
            lines.append("-+-".join("-" * w for w in col_widths))
    return "\n".join(lines)


def save_json(data: Mapping[str, Any], path: str | Path) -> None:
    """Serialize mapping to JSON."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def save_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(obj, path)


def load_pickle(path: str | Path) -> Any:
    path = Path(path)
    return joblib.load(path)


def describe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a data dictionary with basic stats."""
    total = len(df)
    rows = []
    for col in df.columns:
        series = df[col]
        missing = series.isna().sum()
        dtype = str(series.dtype)
        n_unique = series.nunique(dropna=True)
        stats: Dict[str, Any] = {
            "column": col,
            "dtype": dtype,
            "pct_missing": float(missing / total * 100 if total else 0),
            "n_unique": int(n_unique),
        }
        if pd.api.types.is_numeric_dtype(series):
            stats.update(
                {
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                }
            )
        else:
            top = series.mode(dropna=True)
            stats["top"] = None if top.empty else str(top.iloc[0])
        rows.append(stats)
    return pd.DataFrame(rows)
