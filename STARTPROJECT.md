# How to Run the Wildfire Detection Project

Quick guide for teammates to clone, set up, run, test, and modify the project locally.

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Docker Desktop | >= 4.x (allocate 12GB RAM) | [docker.com/download](https://www.docker.com/products/docker-desktop/) |
| Git | any | [git-scm.com](https://git-scm.com/) |
| Python | 3.11 | [python.org](https://www.python.org/) (for native mode / tests) |
| Node.js | >= 18 | [nodejs.org](https://nodejs.org/) (for Frontend only) |
| Make | any | Pre-installed on macOS/Linux. Windows: comes with Git Bash or `choco install make` |

---

## 1. Clone and Configure (2 minutes)

```bash
git clone https://github.com/Scofe-C/wildfire_detection.git
cd wildfire_detection
cp .env.example .env
```

Open `.env` and fill in your keys:

```bash
# === Required ===
FIRMS_MAP_KEY=your_nasa_firms_key       # Get free at https://firms.modaps.eosdis.nasa.gov/api/
GCS_BUCKET_NAME=wildfire-mlops-123      # Ask Zhengxin for team bucket name
GCP_KEY_PATH=./gcp-key.json            # Ask Zhengxin for this file
GOOGLE_CLOUD_PROJECT=wildfire-mlops-123

# === LLM (need at least one) ===
GEMINI_API_KEY=your_gemini_key          # Get free at https://aistudio.google.com/app/apikey
LLM_BACKEND=gemini_dev                  # gemini_dev | ollama | vertex_ai

# === Optional ===
SLACK_WEBHOOK_URL=                      # Leave empty if you don't have one
```

Place `gcp-key.json` at the repo root (same level as `docker-compose.yaml`). **Never commit this file.**

> **One `.env`, one `gcp-key.json`, both at the repo root.** No copies in sub-folders needed.

---

## 2. Start All Services (one command)

```bash
make up-full
```

Or if you don't have `make`:
```bash
./start.sh
```

Or raw Docker:
```bash
docker compose --profile full up -d --build
```

Wait 1-2 minutes for first build, then check:

```bash
make status
```

You should see:

```
  Airflow Webserver           UP    http://localhost:8080/health
  OBJ-3 Dashboard            UP    http://localhost:8000/api/status
  Fire Monitor API            UP    http://localhost:8001/status
  MLflow UI                   UP    http://localhost:5000
```

### Service URLs

| Service | URL | What it does |
|---------|-----|-------------|
| Airflow | http://localhost:8080 (airflow / airflow) | DAG orchestration, pipeline runs |
| OBJ-3 Dashboard | http://localhost:8000 | Generate / view / edit disaster reports |
| Fire Monitor | http://localhost:8001 | Real-time fire monitoring control panel |
| MLflow | http://localhost:5000 | ML experiment tracking |
| Frontend SPA | http://localhost:5173 | Fire map + model pipeline UI (see section 6) |

### Lighter startup (Airflow only)

If you're working on the data pipeline and don't need the dashboards:

```bash
make up        # starts only Airflow (postgres + webserver + scheduler)
```

### Stop everything

```bash
make down
```

---

## 3. Project Structure

```
wildfire_detection/
|-- .env                     # Your API keys (git-ignored, never commit)
|-- .env.example             # Template to copy from
|-- gcp-key.json             # GCP credentials (git-ignored, never commit)
|-- docker-compose.yaml      # Root compose — all services with profiles
|-- Makefile                 # Developer command center
|-- start.sh                 # Zero-dep startup wrapper
|
|-- Data-Pipeline/           # Data ingestion + Airflow DAGs
|   |-- dags/                #   wildfire_dag.py, watchdog_sensor_dag.py
|   |-- scripts/             #   ingestion, processing, fusion, validation, fire_monitor
|   |-- configs/             #   schema_config.yaml (single source of truth)
|   |-- docker/              #   Multi-stage Dockerfile (airflow-base, test targets)
|   `-- tests/               #   200+ pytest tests
|
|-- model-pipeline/          # ML models + OBJ-3 dashboard
|   |-- src/
|   |   |-- models/
|   |   |   |-- obj1_xgboost/    # Fire occurrence classifier
|   |   |   |-- obj2_spread/     # Rothermel + Cell2Fire spread simulator
|   |   |   `-- obj3_gemini/     # LLM disaster report orchestrator
|   |   |-- api/                 # FastAPI server (dashboard + inference)
|   |   |-- bias/                # Fairlearn FNR disparity gate
|   |   `-- pipeline/            # Orchestrator + bridge (OBJ-1/2 -> OBJ-3)
|   |-- dashboard/               # OBJ-3 report viewer HTML
|   |-- configs/                 # model_config.yaml, reporting_config.yaml
|   `-- tests/
|
|-- Frontend/                # React + Vite + Tailwind SPA
|   |-- src/components/      #   fire-map, model-pipeline, layout, ui
|   `-- package.json
|
`-- .github/workflows/       # CI/CD (3 workflows)
    |-- ci.yaml              #   Data Pipeline CI
    |-- model_ci.yml         #   Model Pipeline CI/CD (9 stages)
    `-- model_rollback.yml   #   Manual rollback
```

---

## 4. Running Tests

```bash
# All tests (model + data)
make test

# Model pipeline only (from model-pipeline/)
cd model-pipeline
pytest tests/ --ignore=tests/obj2 --ignore=tests/obj3 -v

# Data pipeline only (inside Docker)
cd Data-Pipeline
docker compose run --rm test

# Lint both pipelines
make lint
```

---

## 5. Testing Features Manually

### Generate an OBJ-3 disaster report (fastest test, ~30 seconds)

No Docker needed. Just needs `GEMINI_API_KEY` set in your `.env`.

```bash
cd model-pipeline

# Low risk daily report
python scripts/run_report.py --demo low_risk --backend gemini_dev

# Emergency incident report
python scripts/run_report.py --demo emergency --backend gemini_dev

# Use local Ollama instead (free, no API key)
python scripts/run_report.py --demo emergency --backend ollama
```

### Run the fire monitor (full pipeline loop)

```bash
cd Data-Pipeline

# Single emergency cycle (quickest full E2E test)
python scripts/fire_monitor.py --mode emergency --cycles 1 --region california --backend gemini_dev

# Continuous monitoring with web dashboard
python scripts/fire_monitor.py --with-api --mode emergency --interval 60 --backend gemini_dev
# Dashboard: http://127.0.0.1:8001
```

### Trigger Airflow DAG manually

```bash
# Start Airflow first
make up

# Trigger emergency pipeline for California
docker compose exec airflow-scheduler airflow dags trigger wildfire_data_pipeline \
  --conf '{"resolution_km": 22, "trigger_source": "watchdog_emergency", "fire_cells": ["8e283082ddbffff"], "weather_lookback_hours": 2}'

# Watch logs
docker compose logs -f airflow-scheduler
```

### If APIs are unavailable (offline testing)

```bash
cd Data-Pipeline
python scripts/seed_local_test.py   # generates dummy .env + gcp-key + seed data
cd .. && make up-full
```

---

## 6. Frontend SPA (React)

The Frontend is a separate React + Vite app. Run it alongside the backend services:

```bash
# Start backend services first
make up-full

# Then in a new terminal:
cd Frontend
npm install          # first time only
npm run dev          # starts on http://localhost:5173
```

The frontend calls the OBJ-3 Dashboard API on `:8000` and the Fire Monitor API on `:8001`.

---

## 7. Native Mode (no Docker)

Run individual services directly with Python (useful for debugging):

```bash
# OBJ-3 Dashboard (port 8000)
make dashboard
# Or: cd model-pipeline && python scripts/run_dashboard.py

# Fire Monitor + API (port 8001)
make monitor
# Or: cd Data-Pipeline && python scripts/fire_monitor.py --with-api

# MLflow UI (port 5000)
make mlflow
# Or: cd model-pipeline && mlflow ui --backend-store-uri sqlite:///mlruns.db
```

For native mode, install Python deps first:
```bash
cd model-pipeline && pip install -r requirements.txt
cd ../Data-Pipeline && pip install -r requirements.txt
```

---

## 8. Making Changes

### Which files to modify for each objective

| If you're working on... | Edit files in... | Test with... |
|--------------------------|-----------------|-------------|
| Data ingestion (FIRMS, weather) | `Data-Pipeline/scripts/ingestion/` | `make test` or `pytest tests/` in Data-Pipeline |
| Data processing / fusion | `Data-Pipeline/scripts/processing/`, `fusion/` | Same |
| Airflow DAGs | `Data-Pipeline/dags/` | Restart: `make down && make up` |
| OBJ-1 (XGBoost classifier) | `model-pipeline/src/models/obj1_xgboost/` | `cd model-pipeline && pytest tests/ --ignore=tests/obj2 --ignore=tests/obj3` |
| OBJ-2 (fire spread) | `model-pipeline/src/models/obj2_spread/` | `cd model-pipeline && pytest tests/obj2/` |
| OBJ-3 (LLM reports) | `model-pipeline/src/models/obj3_gemini/` | `python scripts/run_report.py --demo emergency --backend gemini_dev` |
| OBJ-3 dashboard UI | `model-pipeline/dashboard/` | Rebuild: `docker compose --profile full up -d --build obj3-dashboard` |
| Frontend SPA | `Frontend/src/` | `cd Frontend && npm run dev` (hot-reloads) |
| CI/CD workflows | `.github/workflows/` | Push to a branch and check GitHub Actions |

### After modifying Docker-related files

If you changed a Dockerfile, docker-compose.yaml, or requirements.txt:

```bash
make down
make up-full    # rebuilds images automatically (--build flag)
```

### After modifying Python code in model-pipeline

If running in Docker, rebuild the dashboard:
```bash
docker compose --profile full up -d --build obj3-dashboard
```

If running in native mode, just restart the Python process.

---

## 9. CI/CD — What Happens When You Push

### On PR / push to `main` or `develop`

Two workflows run automatically:

**Data Pipeline CI** (`ci.yaml`) — if you changed `Data-Pipeline/` files:
1. Builds Docker test image (cached)
2. Validates Airflow DAG imports
3. Validates `dvc.yaml` syntax
4. Runs pytest (coverage >= 60% required)
5. Lints with ruff + mypy + pip-audit
6. Checks dependency pins

**Model Pipeline CI** (`model_ci.yml`) — if you changed `model-pipeline/` files:
1. Lint (ruff + mypy)
2. Unit tests (coverage >= 35% required)

Both run without any secrets — all external calls are mocked.

### On merge to `master` (full deploy pipeline)

The model pipeline continues with 7 more stages:

```
Stage 3:  Build Docker image -> push to GCR
Stage 4:  Train XGBoost models (CA + TX)
Stage 5:  AUC-PR gate (>= 0.89) -- BLOCKS if model is bad
Stage 6:  Bias gate (FNR <= 5%) -- BLOCKS if model is unfair
Stage 7:  Push to Vertex AI registry
Stage 8:  Deploy to Cloud Run (REQUIRES manual approval)
          + post-deploy smoke test + Slack notification
Stage 9:  Update Cloud Scheduler (6h monitoring job)
```

### Manual rollback

If a deployed model is bad, go to GitHub Actions -> `Model Rollback` -> Run workflow:
- Input the `run_id` of the good model version
- Requires production environment approval
- Swaps Vertex AI labels (no retrain needed)

### GitHub Secrets needed (for full deploy only)

| Secret | Purpose |
|--------|---------|
| `GCP_SA_KEY` | GCP service account credentials |
| `GCP_PROJECT_ID` | Vertex AI project |
| `GCP_SA_EMAIL` | Cloud Scheduler OIDC auth |
| `MLFLOW_TRACKING_URI` | Experiment tracking |
| `SLACK_WEBHOOK_URL` | Deploy notifications |

### GitHub Environments needed

| Environment | Approval? | Used by |
|-------------|-----------|---------|
| `model-training` | Optional | Stages 4-7 |
| `production` | **Required** | Stage 8 (deploy) + rollback |

> PRs only run Stages 1-2 and need **zero secrets**. You can run and test locally without any GCP setup.

---

## 10. Pre-commit Hooks (optional but recommended)

```bash
pip install pre-commit
pre-commit install
```

After this, every `git commit` automatically runs:
- ruff (lint + format)
- mypy (type check)
- trailing whitespace / YAML check
- Secret detection (blocks commits with API keys)
- Large file check (blocks files > 1MB)

---

## Quick Reference — All `make` Commands

```bash
make help           # show all targets
make check-env      # verify .env has required vars
make up             # start Airflow only
make up-full        # start everything (Airflow + Dashboard + Monitor + MLflow)
make down           # stop all containers
make status         # check all service health
make logs           # tail all container logs
make clean          # stop + remove all volumes (fresh start)
make dashboard      # native: OBJ-3 dashboard on :8000
make monitor        # native: fire monitor on :8001
make mlflow         # native: MLflow UI on :5000
make test           # run all tests
make lint           # lint both pipelines
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `make: command not found` (Windows) | Use `./start.sh` instead, or install make via `choco install make` |
| Docker build fails | Make sure Docker Desktop is running with 12GB RAM allocated |
| `FIRMS: Invalid MAP_KEY` | Check `.env` has no smart quotes: `FIRMS_MAP_KEY=your_key_here` |
| `LLM backend not available` | Set `GEMINI_API_KEY` in `.env`. Or install Ollama: `ollama pull qwen3:8b` |
| Port already in use | Stop conflicting service, or change port in docker-compose.yaml |
| Airflow DAG not showing | Wait 30s for scheduler to parse, or check `docker compose logs airflow-scheduler` |
| `gcp-key.json not found` | Place it at repo root (not inside Data-Pipeline/) |
| Tests fail with import errors | Run from the correct directory (`cd model-pipeline` or `cd Data-Pipeline`) |