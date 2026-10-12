PY ?= .venv/bin/python
BIN = .venv/bin

.PHONY: install lint typecheck test check serve pilot pilot-all-chunks report evaluate docker-build docker-up

install:            ## create venv and install the package with dev tools
	python3 -m venv .venv && $(BIN)/pip install -U pip && $(BIN)/pip install -e ".[dev]"

lint:
	$(BIN)/ruff check src tests scripts

typecheck:
	$(BIN)/mypy src

test:
	$(BIN)/pytest -q

check: lint typecheck test

serve:              ## REST API on :8000 (Swagger UI at /docs)
	$(BIN)/capex serve

pilot:              ## five-organization pilot through the API (server must be running)
	$(PY) scripts/run_pilot.py --out pilot

pilot-all-chunks:   ## same documents, screening disabled (accuracy/cost trade-off)
	$(PY) scripts/run_pilot.py --out pilot/_all_chunks --screening none --force-refresh

report:             ## cost report + screening trade-off
	$(BIN)/capex report --pilot-dir pilot --all-chunks-dir pilot/_all_chunks

evaluate:           ## accuracy evaluation against manually verified samples
	$(BIN)/capex evaluate --gold-dir evaluation/gold --out evaluation/results.json

docker-build:
	docker compose build

docker-up:
	docker compose up -d
