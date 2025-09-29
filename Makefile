CONFIG ?= configs/default.yaml

.PHONY: build build-gpu train eval shell up down
build:
	docker compose build app

build-gpu:
	docker compose --profile gpu build app-gpu

train:
	docker compose run --rm app python run.py --mode train --config $(CONFIG)

eval:
	docker compose run --rm app python run.py --mode eval --config $(CONFIG)

shell:
	docker compose run --rm app bash

up:
	docker compose up -d

down:
	docker compose down -v
