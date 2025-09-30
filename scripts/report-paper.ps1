#!/usr/bin/env pwsh
# Generate reports including static paper tables
docker compose run --rm -e PAPER_TABLES=1 app python run.py --mode report --config configs/default.yaml --paper_tables true
