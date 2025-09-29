from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve

from . import dataio, features, models, utils


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _prepare_paths() -> Tuple[Path, Path]:
    artifacts_dir = utils.ensure_dir(utils.resolve_path("artifacts"))
    models_dir = utils.ensure_dir(utils.resolve_path("models"))
    return artifacts_dir, models_dir


def train_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    utils.seed_everything(config.get("seed", 42))
    data_cfg = config.get("data", {})
    csv_path = utils.resolve_path(data_cfg.get("csv_path", "data/heart.csv"))
    target = data_cfg.get("target", "target")

    df = dataio.load_csv(csv_path)
    print(f"Loaded dataset from {csv_path} with shape {df.shape}")
    dataio.print_data_dictionary(df)
    df = dataio.clean(df)
    print(f"After cleaning: {df.shape[0]} rows, {df.shape[1]} columns")

    split_cfg = data_cfg.get("split", {})
    X_train, X_test, y_train, y_test = dataio.split(
        df,
        target=target,
        test_size=split_cfg.get("test_size", 0.3),
        stratify=split_cfg.get("stratify", True),
        random_state=split_cfg.get("random_state", config.get("seed", 42)),
    )

    features_cfg = config.get("features", {})
    preprocessor = features.build_preprocessor(
        X_train,
        use_pca=features_cfg.get("pca", False),
        pca_components=features_cfg.get("pca_components", 0.95),
    )

    X_train_proc = features.fit_transform(preprocessor, X_train)
    X_test_proc = features.transform(preprocessor, X_test)

    metrics_by_model: Dict[str, Dict[str, Any]] = {}
    probas: Dict[str, np.ndarray] = {}
    trained_models: Dict[str, Any] = {}
    training_history: Dict[str, Any] = {}

    if config.get("train", {}).get("baseline", True):
        log_model = models.train_logistic_regression(X_train_proc, y_train)
        y_proba = models.predict_proba_logistic(log_model, X_test_proc)
        metrics_by_model["logistic_regression"] = utils.compute_classification_metrics(y_test, y_proba)
        probas["logistic_regression"] = y_proba
        trained_models["logistic_regression"] = log_model

        rf_model = models.train_random_forest(X_train_proc, y_train)
        rf_proba = models.predict_proba_random_forest(rf_model, X_test_proc)
        metrics_by_model["random_forest"] = utils.compute_classification_metrics(y_test, rf_proba)
        probas["random_forest"] = rf_proba
        trained_models["random_forest"] = rf_model

    dnn_cfg = config.get("train", {}).get("dnn")
    if dnn_cfg:
        dnn_config = models.DNNConfig(
            input_dim=X_train_proc.shape[1],
            hidden_sizes=list(dnn_cfg.get("hidden_sizes", [64, 32])),
            dropout=float(dnn_cfg.get("dropout", 0.2)),
            epochs=int(dnn_cfg.get("epochs", 30)),
            batch_size=int(dnn_cfg.get("batch_size", 64)),
            lr=float(dnn_cfg.get("lr", 0.001)),
        )
        dnn_model, history = models.train_dnn(X_train_proc, y_train, dnn_config)
        dnn_proba = models.predict_proba_dnn(dnn_model, X_test_proc)
        metrics_by_model["dnn"] = utils.compute_classification_metrics(y_test, dnn_proba)
        probas["dnn"] = dnn_proba
        trained_models["dnn"] = (dnn_model, dnn_config)
        training_history["dnn"] = history

    if not metrics_by_model:
        raise RuntimeError("No models were trained. Check configuration.")

    # Choose best model by ROC-AUC then F1
    def _ranking(item: Tuple[str, Dict[str, Any]]) -> Tuple[float, float]:
        metrics = item[1]
        return metrics.get("roc_auc", 0.0), metrics.get("f1", 0.0)

    best_model_name, best_metrics = max(metrics_by_model.items(), key=_ranking)
    print("\nModel performance:")
    print(utils.format_metrics_table(metrics_by_model))
    print(f"\nBest model: {best_model_name}")

    artifacts_dir, models_dir = _prepare_paths()
    preprocessor_path = artifacts_dir / "preprocessor.pkl"
    utils.save_pickle(preprocessor, preprocessor_path)

    # Persist best model
    best_model_path: Path
    metadata: Dict[str, Any] = {
        "best_model": best_model_name,
        "metrics": metrics_by_model,
        "target": target,
        "feature_columns": X_train.columns.tolist(),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "preprocessor_path": str(preprocessor_path.relative_to(utils.get_project_root())),
    }

    best_proba = probas[best_model_name]
    if best_proba.ndim == 2:
        best_proba_flat = best_proba[:, 1]
    else:
        best_proba_flat = best_proba

    best_pred = (best_proba_flat >= 0.5).astype(int)

    roc_path = artifacts_dir / "roc_curve.png"
    cm_path = artifacts_dir / "confusion_matrix.png"
    _plot_roc_curve(y_test.to_numpy(), best_proba_flat, best_model_name, roc_path)
    _plot_confusion_matrix(y_test.to_numpy(), best_pred, cm_path)

    metrics_path = artifacts_dir / "metrics.json"
    utils.save_json(
        {
            "models": metrics_by_model,
            "best_model": best_model_name,
        },
        metrics_path,
    )

    metadata.update(
        {
            "roc_curve_path": str(roc_path.relative_to(utils.get_project_root())),
            "confusion_matrix_path": str(cm_path.relative_to(utils.get_project_root())),
            "metrics_path": str(metrics_path.relative_to(utils.get_project_root())),
        }
    )

    if best_model_name == "dnn":
        dnn_model, dnn_config = trained_models["dnn"]
        best_model_path = models_dir / "best_model.pt"
        models.save_dnn(dnn_model, str(best_model_path))
        metadata["dnn_config"] = {
            "hidden_sizes": dnn_config.hidden_sizes,
            "dropout": dnn_config.dropout,
            "epochs": dnn_config.epochs,
            "batch_size": dnn_config.batch_size,
            "lr": dnn_config.lr,
            "input_dim": dnn_config.input_dim,
        }
        metadata["training_history"] = training_history.get("dnn", {})
    else:
        best_model = trained_models[best_model_name]
        best_model_path = models_dir / "best_model.pkl"
        utils.save_pickle(best_model, best_model_path)

    metadata["best_model_path"] = str(best_model_path.relative_to(utils.get_project_root()))
    utils.save_json(metadata, artifacts_dir / "metadata.json")

    results = {
        "metrics": metrics_by_model,
        "best_model": best_model_name,
        "artifact_paths": {
            "preprocessor": str(preprocessor_path.relative_to(utils.get_project_root())),
            "metrics": str(metrics_path.relative_to(utils.get_project_root())),
            "roc_curve": str(roc_path.relative_to(utils.get_project_root())),
            "confusion_matrix": str(cm_path.relative_to(utils.get_project_root())),
            "model": str(best_model_path.relative_to(utils.get_project_root())),
        },
    }
    return results


def run(config_path: str | Path) -> Dict[str, Any]:
    config = load_config(config_path)
    return train_from_config(config)
