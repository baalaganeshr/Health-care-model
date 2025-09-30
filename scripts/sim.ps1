#!/usr/bin/env pwsh
# Run IoT simulation with ACGA clustering
docker compose run --rm app python run.py --mode sim --config configs/default.yaml
