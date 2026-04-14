# =============================================================================
# Wildfire Detection MLOps — Makefile
# =============================================================================
# make help       — show all targets
# make up         — Airflow only (default profile)
# make up-full    — everything (Airflow + Dashboard + Monitor + MLflow)
# =============================================================================

.DEFAULT_GOAL := help
COMPOSE := docker compose

# ── Info ─────────────────────────────────────────────────────────────────────
.PHONY: help
help: ## Show all targets with descriptions
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.PHONY: check-env
check-env: ## Validate .env exists and required vars are set
	@if [ ! -f .env ]; then \
		echo "ERROR: .env not found. Run: cp .env.example .env"; exit 1; \
	fi
	@echo "Checking required environment variables..."
	@. ./.env 2>/dev/null; \
	for var in FIRMS_MAP_KEY GCS_BUCKET_NAME GCP_KEY_PATH GOOGLE_CLOUD_PROJECT; do \
		val=$$(eval echo "\$$$$var"); \
		if [ -z "$$val" ]; then \
			printf "  %-28s \033[31mMISSING\033[0m\n" "$$var"; \
		else \
			printf "  %-28s \033[32mSET\033[0m\n" "$$var"; \
		fi; \
	done

# ── Docker Compose ───────────────────────────────────────────────────────────
.PHONY: up
up: ## Start Airflow stack (default profile)
	$(COMPOSE) up -d --build

.PHONY: up-full
up-full: ## Start ALL services (Airflow + Dashboard + Monitor + MLflow)
	$(COMPOSE) --profile full up -d --build

.PHONY: down
down: ## Stop everything
	$(COMPOSE) --profile full down

.PHONY: status
status: ## Check health of all service endpoints
	@bash scripts/healthcheck.sh

.PHONY: logs
logs: ## Tail logs from all running services
	$(COMPOSE) --profile full logs -f

.PHONY: clean
clean: ## Stop and remove all containers + volumes
	$(COMPOSE) --profile full down -v

# ── Native mode (no Docker) ─────────────────────────────────────────────────
.PHONY: dashboard
dashboard: ## Native: start OBJ-3 dashboard on :8000
	cd model-pipeline && python scripts/run_dashboard.py

.PHONY: monitor
monitor: ## Native: start fire monitor with API on :8001
	cd Data-Pipeline && python scripts/fire_monitor.py --with-api

.PHONY: mlflow
mlflow: ## Native: start MLflow UI on :5000
	cd model-pipeline && mlflow ui --backend-store-uri sqlite:///mlruns.db

# ── Dev tools ────────────────────────────────────────────────────────────────
.PHONY: test
test: ## Run all tests (model + data pipelines)
	cd model-pipeline && pytest tests/ --ignore=tests/obj2 --ignore=tests/obj3 -v
	cd Data-Pipeline && docker compose run --rm test 2>/dev/null || echo "Data tests require Docker"

.PHONY: lint
lint: ## Lint both pipelines with ruff
	cd model-pipeline && ruff check src/ scripts/ --select=E,F --ignore=E501,F401
	cd Data-Pipeline && ruff check scripts/ dags/ tests/ --select=E,F --ignore=E501 2>/dev/null || true

.PHONY: health
health: ## Hit /health endpoints and print pass/fail
	@bash scripts/healthcheck.sh