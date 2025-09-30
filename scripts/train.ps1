#!/usr/bin/env pwsh
# Train baseline models (existing functionality)
docker compose run --rm app python run.py --mode train --config configs/default.yaml
