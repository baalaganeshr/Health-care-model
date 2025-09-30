#!/usr/bin/env pwsh
# Evaluate existing trained models
docker compose run --rm app python run.py --mode eval --config configs/default.yaml
