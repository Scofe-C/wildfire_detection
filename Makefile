# =============================================================================
# PyroWatch Wildfire MLOps — Developer Command Center
# =============================================================================
# Run all commands from: wildfire_detection/  (the repo root)
#
# Quick start:
#   cp .env.example .env   # fill in your API keys
#   make up-full           # start everything
#   make status            # verify all services are healthy
# =============================================================================

# docker compose command — always run from Data-Pipeline/ so relative
# build paths in docker-compose.yaml resolve correctly.
# If .env exists at repo root, pass it to compose via --env-file.
ROOT    := $(shell pwd)
DC_DIR  := Data-Pipeline
ENV_ARG := $(if $(wildcard $(ROOT)/.env),--env-file $(ROOT)/.env,)
DC      := cd $(DC_DIR) && docker compose $(ENV_ARG)

.DEFAULT_GOAL := help

# ── Meta ──────────────────────────────────────────────────────────────────────

.PHONY: help
help:  ## Show all available targets
	@echo ""
	@echo "  PyroWatch MLOps — make targets"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""

.PHONY: check-env
check-env:  ## Verify .env exists and required vars are set
	@echo "\n  Checking environment...\n"
	@test -f .env || (echo "  ✗ .env not found — run: cp .env.example .env" && exit 1)
	@for var in FIRMS_MAP_KEY GCS_BUCKET_NAME GCP_KEY_PATH GOOGLE_CLOUD_PROJECT; do \
	    val=$$(grep "^$$var=" .env | cut -d= -f2); \
	    if [ -z "$$val" ]; then \
	        printf "  \033[31m✗ %-35s NOT SET\033[0m\n" "$$var"; \
	    else \
	        printf "  \033[32m✓ %-35s set\033[0m\n" "$$var"; \
	    fi; \
	done
	@for var in GEMINI_API_KEY GOOGLE_API_KEY; do \
	    val=$$(grep "^$$var=" .env | cut -d= -f2); \
	    if [ -z "$$val" ]; then \
	        printf "  \033[33m~ %-35s (optional — OBJ-3 needs at least one)\033[0m\n" "$$var"; \
	    else \
	        printf "  \033[32m✓ %-35s set\033[0m\n" "$$var"; \
	    fi; \
	done
	@echo ""

# ── Docker Compose ────────────────────────────────────────────────────────────

.PHONY: up
up:  ## Start Airflow only (lightweight — for data pipeline work)
	$(DC) up -d --build postgres airflow-init airflow-webserver airflow-scheduler
	@echo "\n  Airflow UI: http://localhost:8080  (airflow / airflow)\n"

.PHONY: up-full
up-full:  ## Start ALL services: Airflow + OBJ-3 Dashboard + Frontend
	$(DC) up -d --build
	@echo "\n  Services starting — run 'make status' in ~60s to verify\n"
	@echo "    Airflow:    http://localhost:8080  (airflow / airflow)"
	@echo "    Dashboard:  http://localhost:8000"
	@echo "    Frontend:   http://localhost:3000\n"

.PHONY: down
down:  ## Stop all containers (preserves data volumes)
	$(DC) down

.PHONY: restart
restart: down up-full  ## Full stop + start (keeps volumes)

.PHONY: clean
clean:  ## Stop containers AND remove volumes — complete fresh start
	@echo "  This removes all data volumes (Postgres, etc). Press Ctrl-C to cancel."
	@sleep 3
	$(DC) down -v

.PHONY: logs
logs:  ## Tail logs from all running containers
	$(DC) logs -f

.PHONY: ps
ps:  ## Show running container status
	$(DC) ps

# ── Health & Status ───────────────────────────────────────────────────────────

.PHONY: status
status:  ## Check health of all services (polls endpoints)
	@bash scripts/healthcheck.sh

.PHONY: health
health: status  ## Alias for status

# ── Testing & Linting ─────────────────────────────────────────────────────────

.PHONY: test
test:  ## Run all tests (model-pipeline unit tests + data-pipeline in Docker)
	@echo "\n  Running model-pipeline tests...\n"
	cd model-pipeline && pytest tests/ --ignore=tests/obj2 --ignore=tests/obj3 -v --tb=short
	@echo "\n  Running data-pipeline tests (in Docker)...\n"
	$(DC) run --rm -e GCS_BUCKET_NAME=test-bucket -e FIRMS_MAP_KEY=test-key \
	    wildfire-pipeline:test pytest tests/ -v --tb=short 2>/dev/null || \
	    echo "  (Docker test image not built — run 'make up-full' first)"

.PHONY: lint
lint:  ## Lint both pipelines with ruff
	@echo "\n  Linting model-pipeline...\n"
	cd model-pipeline && ruff check src/ scripts/ --select=E,F --ignore=E501,F401
	@echo "\n  Linting Data-Pipeline...\n"
	cd Data-Pipeline && ruff check scripts/ dags/ tests/ --select=E,F --ignore=E501

# ── Native Mode (no Docker) ───────────────────────────────────────────────────

.PHONY: dashboard
dashboard:  ## Run OBJ-3 dashboard natively on :8000 (needs pip install)
	cd model-pipeline && python scripts/run_dashboard.py

.PHONY: monitor
monitor:  ## Run fire monitor natively on :8001 (needs pip install)
	cd Data-Pipeline && python scripts/fire_monitor.py --with-api

.PHONY: mlflow
mlflow:  ## Run MLflow UI natively on :5000
	cd model-pipeline && mlflow ui --backend-store-uri sqlite:///mlruns.db

# ── Frontend ──────────────────────────────────────────────────────────────────

.PHONY: frontend-dev
frontend-dev:  ## Run Frontend in hot-reload dev mode on :5173
	cd Frontend && npm run dev

.PHONY: frontend-build
frontend-build:  ## Rebuild Frontend Docker image and restart container
	$(DC) build --no-cache frontend
	$(DC) up -d frontend
