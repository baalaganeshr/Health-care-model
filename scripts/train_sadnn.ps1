#!/usr/bin/env pwsh
# Train SA-DNN with IWOA optimization
docker compose run --rm app python run.py --mode train_sadnn --config configs/default.yaml
