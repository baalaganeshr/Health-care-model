#!/usr/bin/env pwsh
# Reproduce paper results with real data
docker compose run --rm app python run.py --mode reproduce --config configs/default.yaml
