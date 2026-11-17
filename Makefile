SHELL := /bin/bash
.DEFAULT_GOAL := help

BACKEND := backend
VENV := $(BACKEND)/.venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTHONPATH := $(BACKEND)/src

.PHONY: help env infra-up infra-local infra-down infra-logs start bootstrap install install-local serve serve-open test lint format config-check summary models-pull clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

env: ## Create .env from the example (never overwrites)
	@test -f .env || (cp .env.example .env && echo "Created .env - set LIVEKIT_API_SECRET and your API keys")

infra-up: env ## Start Redis + LiveKit
	docker compose up -d redis livekit

infra-local: env ## Start Redis + LiveKit + Ollama + Speaches + Kokoro
	docker compose --profile local up -d

infra-down: ## Stop all containers (keeps volumes)
	docker compose --profile local down

infra-logs: ## Follow infrastructure logs
	docker compose logs -f --tail=100

start: ## THE one command: install whatever is missing, download models, open the HUD
	./scripts/install.sh
	$(MAKE) serve-open

bootstrap: ## Install only: uv, Ollama, venv, engines, models
	./scripts/install.sh

install: ## Create the backend venv and install core + dev dependencies
	cd $(BACKEND) && uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev]"

install-local: ## Add local STT/TTS engines (Faster-Whisper, Piper)
	cd $(BACKEND) && uv pip install --python .venv/bin/python -e ".[local]"

serve: ## Run the JARVIS web interface on the configured host/port
	cd $(BACKEND) && PYTHONPATH=src .venv/bin/python -m jarvis.server.cli serve --config ../config/config.yaml

serve-open: ## Run the web interface and open the browser
	cd $(BACKEND) && PYTHONPATH=src .venv/bin/python -m jarvis.server.cli serve --config ../config/config.yaml --open

test: ## Run the backend test suite
	cd $(BACKEND) && PYTHONPATH=src .venv/bin/python -m pytest -q

lint: ## Ruff + mypy
	cd $(BACKEND) && .venv/bin/ruff check src tests
	cd $(BACKEND) && PYTHONPATH=src .venv/bin/mypy src/jarvis

format: ## Format with ruff
	cd $(BACKEND) && .venv/bin/ruff format src tests

config-check: ## Validate config/config.yaml and print a redacted summary
	cd $(BACKEND) && PYTHONPATH=src .venv/bin/python -c "from jarvis.core.config import main; raise SystemExit(main())" ../config/config.yaml

summary: config-check ## Alias for config-check

models-pull: ## Pull the default local models into Ollama
	docker compose exec ollama ollama pull qwen2.5:7b
	docker compose exec ollama ollama pull qwen2.5:3b
	docker compose exec ollama ollama pull nomic-embed-text

clean: ## Remove caches and the venv (not data)
	rm -rf $(VENV) $(BACKEND)/.pytest_cache $(BACKEND)/.ruff_cache $(BACKEND)/.mypy_cache
	find $(BACKEND) -type d -name __pycache__ -prune -exec rm -rf {} +
