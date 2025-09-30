SHELL := /bin/bash
CONFIG ?= configs/default.yaml

.PHONY: build build-gpu train eval report report-paper shell up down
build:
	docker compose build app

build-gpu:
	docker compose --profile gpu build app-gpu

train:
	docker compose run --rm -e CONFIG=$(CONFIG) app python run.py --mode train --config $(CONFIG)

eval:
	docker compose run --rm -e CONFIG=$(CONFIG) app python run.py --mode eval --config $(CONFIG)

report:
	docker compose run --rm -e CONFIG=$(CONFIG) app python run.py --mode report --config $(CONFIG)

report-paper:
	docker compose run --rm -e CONFIG=$(CONFIG) -e PAPER_TABLES=1 app python run.py --mode report --config $(CONFIG) --paper_tables true

shell:
	docker compose run --rm app bash

up:
	docker compose up -d

down:
	docker compose down -v
