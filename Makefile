.PHONY: up down build lint test logs shell-api shell-worker

COMPOSE := docker compose
BACKEND  := backend

## ── Infrastructure ──────────────────────────────────────────────────────────

up: ## Start all services (detached)
	$(COMPOSE) up -d

down: ## Stop and remove containers
	$(COMPOSE) down

build: ## (Re)build all images
	$(COMPOSE) build --no-cache

logs: ## Tail logs for all services (Ctrl+C to exit)
	$(COMPOSE) logs -f

logs-api: ## Tail API logs only
	$(COMPOSE) logs -f api

logs-worker: ## Tail Worker logs only
	$(COMPOSE) logs -f worker

## ── Development helpers ──────────────────────────────────────────────────────

shell-api: ## Open a shell inside the running API container
	$(COMPOSE) exec api bash

shell-worker: ## Open a shell inside the running Worker container
	$(COMPOSE) exec worker bash

## ── Code quality ─────────────────────────────────────────────────────────────

lint: ## Run ruff linter + mypy type-checker on backend source
	cd $(BACKEND) && uv run ruff check app tests
	cd $(BACKEND) && uv run ruff format --check app tests
	cd $(BACKEND) && uv run mypy app

format: ## Auto-fix linting issues and reformat code
	cd $(BACKEND) && uv run ruff check --fix app tests
	cd $(BACKEND) && uv run ruff format app tests

## ── Tests ────────────────────────────────────────────────────────────────────

test: ## Run the test suite
	cd $(BACKEND) && uv run pytest -v

test-cov: ## Run tests with coverage report
	cd $(BACKEND) && uv run pytest -v --cov=app --cov-report=term-missing

## ── Utilities ────────────────────────────────────────────────────────────────

env: ## Copy .env.example to .env if it doesn't exist yet
	@[ -f .env ] && echo ".env already exists, skipping." || (cp .env && echo "Created .env from .env.example")

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
