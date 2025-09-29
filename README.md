# Heart Disease Classification Pipeline

This project rebuilds a heart-disease classifier from the provided archives, producing a reproducible training and evaluation workflow with saved artifacts, metrics, and tests. The accompanying Healthcare.docx discusses energy-aware IoT deployments with PCA feature reduction and deep neural networks, guiding our inclusion of PCA toggles and a DNN alongside classical baselines.

## Dataset
- Source files: `archive.zip` (`heart.csv`) and `archive (1).zip` (`Heart_disease_cleveland_new.csv`) located in `./data/`.
- Target column: `target` (binary 0/1 heart disease label).
- Cleaning removes whitespace, duplicates, and imputes missing numeric (median) and categorical (mode) values.

## Environment Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
Train and evaluate from the project root:
```powershell
python run.py --mode train --config configs/default.yaml
python run.py --mode eval  --config configs/default.yaml
```
Artifacts land in `artifacts/` and `models/`.

### Latest Metrics (dnn model)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| dnn   | 0.7802   | 0.7843    | 0.8163 | 0.8000 | 0.8639 |

_Baseline scores are stored in `artifacts/metrics.json`._

## Repository Layout
```
configs/        # YAML configs (default uses data/heart.csv)
data/           # Extracted CSVs
src/            # Pipeline code (utils, dataio, features, models, train, evaluate)
tests/          # pytest coverage for data loading, features, and models
artifacts/      # Saved metrics, plots, metadata, preprocessor
models/         # Persisted best model weights/state
```

## Testing
```powershell
pytest
```

## Docker Support
Build and run with Docker:
```powershell
docker build -t healthcare-model .
docker run healthcare-model
```

For CUDA support:
```powershell
docker build -f Dockerfile.cuda -t healthcare-model-cuda .
```

Or use docker-compose:
```powershell
docker-compose up
```

## Results & Reporting
- Metrics JSON: `artifacts/metrics.json`
- Evaluation metrics: `artifacts/metrics_eval.json`
- Best model weights: `models/best_model.pt`
- Confusion matrix: `artifacts/confusion_matrix.png`
- ROC curve: `artifacts/roc_curve.png`