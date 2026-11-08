# Shortcuts for the commands CI runs. Run `make help` to list them. Needs the virtual environment active (source .venv/bin/activate).
.DEFAULT_GOAL := help
PYTHON ?= python

.PHONY: help install up migrate ingest test lint typecheck check eval-retrieval eval-scale

help:  ## List the targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

install:  ## Install the dependencies and the rag package (editable)
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e . --no-deps

up:  ## Start PostgreSQL + pgvector in Docker
	docker compose up -d --wait db

migrate:  ## Create or update the database tables
	$(PYTHON) -m rag.db.migrate

ingest:  ## Load the sample project into the database (needs EMBED_API_KEY)
	$(PYTHON) -m rag.cli.ingest

test:  ## Unit and integration tests (integration tests skip when the database is down)
	$(PYTHON) -W error::ResourceWarning -m unittest discover -s tests -t .

lint:  ## Style and likely bugs
	ruff check .

typecheck:  ## Type hints
	mypy

check: lint typecheck test  ## Everything CI runs on the code

eval-retrieval:  ## Offline retrieval quality gate (no LLM)
	$(PYTHON) -m evals.retrieval_eval --store temp

eval-scale:  ## Offline scale smoke test
	$(PYTHON) -m evals.scale_test --records 3000
