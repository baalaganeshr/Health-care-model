#!/usr/bin/env pwsh
# Generate reports with real metrics only
docker compose run --rm app python run.py --mode report --config configs/default.yaml
