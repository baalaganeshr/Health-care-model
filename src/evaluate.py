from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from . import dataio, features, models, utils
from .train import _plot_confusion_matrix, _plot_roc_curve


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _load_metadata() -> Dict[str, Any]:
    metadata_path = utils.resolve_path("artifacts/metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError(
            "artifacts/metadata.json not found. Train the models before evaluation."
        )
    return utils.load_json(metadata_path)


def _load_preprocessor(metadata: Dict[str, Any]):
    preprocessor_path = utils.resolve_path(metadata["preprocessor_path"])
    return utils.load_pickle(preprocessor_path)


def _load_model(metadata: Dict[str, Any]):
    best_model_name = metadata["best_model"]
    model_path = utils.resolve_path(metadata["best_model_path"])
    if best_model_name == "dnn":
        dnn_cfg = metadata.get("dnn_config")
        if not dnn_cfg:
            raise KeyError("Missing dnn_config in metadata; cannot load DNN model")
        return models.load_dnn(
            str(model_path),
            input_dim=int(dnn_cfg["input_dim"]),
            hidden_sizes=list(dnn_cfg["hidden_sizes"]),
            dropout=float(dnn_cfg["dropout"]),
        )
    return utils.load_pickle(model_path)


def evaluate_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    metadata = _load_metadata()
    preprocessor = _load_preprocessor(metadata)
    model = _load_model(metadata)
    best_model_name = metadata["best_model"]

    data_cfg = config.get("data", {})
    csv_path = utils.resolve_path(data_cfg.get("csv_path", metadata.get("csv_path", "data/heart.csv")))
    target = data_cfg.get("target", metadata.get("target", "target"))

    df = dataio.load_csv(csv_path)
    df = dataio.clean(df)

    split_cfg = data_cfg.get("split", {})
    X_train, X_test, y_train, y_test = dataio.split(
        df,
        target=target,
        test_size=split_cfg.get("test_size", metadata.get("test_size", 0.3)),
        stratify=split_cfg.get("stratify", True),
        random_state=split_cfg.get("random_state", config.get("seed", metadata.get("seed", 42))),
    )

    X_test_proc = features.transform(preprocessor, X_test)

    if best_model_name == "dnn":
        y_proba = models.predict_proba_dnn(model, X_test_proc)
    else:
        y_proba = model.predict_proba(X_test_proc)

    metrics = utils.compute_classification_metrics(y_test, y_proba)
    print("Evaluation metrics:")
    print(utils.format_metrics_table({best_model_name: metrics}))

    best_proba = y_proba[:, 1] if y_proba.ndim == 2 else np.asarray(y_proba)
    best_pred = (best_proba >= 0.5).astype(int)

    artifacts_dir = utils.ensure_dir(utils.resolve_path("artifacts"))
    roc_path = artifacts_dir / "roc_curve.png"
    cm_path = artifacts_dir / "confusion_matrix.png"
    _plot_roc_curve(y_test.to_numpy(), best_proba, best_model_name, roc_path)
    _plot_confusion_matrix(y_test.to_numpy(), best_pred, cm_path)

    eval_metrics_path = artifacts_dir / "metrics_eval.json"
    utils.save_json({"model": best_model_name, "metrics": metrics}, eval_metrics_path)

    return {
        "model": best_model_name,
        "metrics": metrics,
        "artifact_paths": {
            "roc_curve": str(roc_path.relative_to(utils.get_project_root())),
            "confusion_matrix": str(cm_path.relative_to(utils.get_project_root())),
            "metrics": str(eval_metrics_path.relative_to(utils.get_project_root())),
        },
    }


def run(config_path: str | Path) -> Dict[str, Any]:
    config = load_config(config_path)
    return evaluate_from_config(config)
