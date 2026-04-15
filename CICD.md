ma# CI/CD Pipeline — Wildfire Detection MLOps

This document explains how the project's 3 GitHub Actions workflows work, what triggers them, and what each stage does.

---

## Overview

```
                        ┌─────────────────────────────────────┐
                        │         GitHub Actions               │
                        │                                      │
  PR / push             │  ┌──────────┐    ┌───────────────┐  │
  Data-Pipeline/* ──────┼─>│ ci.yaml  │    │ model_ci.yml  │<─┼── PR / push model-pipeline/*
                        │  │ (1 job)  │    │ (9 stages)    │  │
                        │  └──────────┘    └───────┬───────┘  │
                        │                          │ merge     │
                        │                          v           │
                        │                  ┌───────────────┐  │
                        │                  │ Deploy to      │  │
                        │                  │ Cloud Run      │  │
                        │                  └───────┬───────┘  │
                        │                          │ manual    │
                        │                  ┌───────v───────┐  │
                        │                  │ model_rollback │<─┼── manual dispatch only
                        │                  │ .yml           │  │
                        │                  └───────────────┘  │
                        └─────────────────────────────────────┘
```

---

## Workflow 1: `ci.yaml` — Data Pipeline CI

**File:** `.github/workflows/ci.yaml`
**Triggers:** push/PR to `main`/`develop` (path-filtered to `Data-Pipeline/`), or manual `workflow_dispatch`

### What it does

This workflow validates the data pipeline — Airflow DAGs, DVC pipeline, tests, and code quality. Everything runs inside a Docker container built from the same Dockerfile used in production.

### Steps (single job)

```
 Step 1   Build Docker test image (cached with GitHub Actions cache)
   │
 Step 2   Validate wildfire_dag.py imports (Airflow parse check)
   │
 Step 3   Validate watchdog_sensor_dag.py imports
   │
 Step 4   Validate dvc.yaml syntax + dep file existence
   │
 Step 5   Run pytest suite (coverage >= 60% or fail)
   │
 Step 6   Upload test artifacts (pytest XML + coverage XML, 14-day retention)
   │
 Step 7   Lint with ruff (E+F rules, E501 line-length ignored)
   │
 Step 8   Type check with mypy (warnings only, non-blocking)
   │
 Step 9   Vulnerability scan with pip-audit (warnings only, non-blocking)
   │
 Step 10  Check dependency pins (pyarrow, dvc, ruff must be in constraints.txt)
```

### Why each step matters

| Step | What it catches |
|------|----------------|
| DAG import validation | Broken imports crash Airflow at deploy time — catch them in CI |
| DVC validation | Missing dep files break `dvc repro` — catch reference errors early |
| pytest (coverage >= 60%) | Blocks PRs that drop test coverage below threshold |
| ruff lint | Style + error rules (E=pycodestyle errors, F=pyflakes) with GitHub PR annotations |
| mypy | Catches type errors early (non-blocking — runs as warning) |
| pip-audit | Flags known vulnerabilities in dependencies (non-blocking) |
| Dependency pin check | Ensures critical packages stay pinned to avoid cross-machine breakage |

### No secrets needed

This workflow runs entirely in Docker with mock env vars (`GCS_BUCKET_NAME=test-bucket`). No GCP credentials required — all external I/O is mocked in tests.

---

## Workflow 2: `model_ci.yml` — Model Pipeline CI/CD (9 Stages)

**File:** `.github/workflows/model_ci.yml`
**Triggers:** push/PR to `master`/`main`/`develop` (path-filtered to `model-pipeline/`), or manual `workflow_dispatch`

This is the main ML pipeline — it lints, tests, builds, trains, validates, and deploys the model.

### Stage flow

```
  PR opened / push to branch
      │
  ┌───v────────────────────────────┐
  │ Stage 1: Lint (ruff + mypy)    │  ← runs on every PR
  └───┬────────────────────────────┘
      │
  ┌───v────────────────────────────┐
  │ Stage 2: Unit Tests            │  ← coverage >= 35% gate
  │          (OBJ-1 only)          │
  └───┬────────────────────────────┘
      │
      │  ── PR stops here ──────────
      │  ── merge to master continues ↓
      │
  ┌───v────────────────────────────┐
  │ Stage 3: Build + Push Docker   │  ← pushes to gcr.io
  │          image to GCR          │
  └───┬────────────────────────────┘
      │
  ┌───v────────────────────────────┐  environment: model-training
  │ Stage 4: Train CA + TX models  │
  │ Stage 5: AUC-PR gate (>= 0.89)│  ← BLOCKS if model isn't good enough
  │ Stage 6: Bias gate (FNR <= 5%) │  ← BLOCKS if model is unfair
  │ Stage 7: Push to Vertex AI     │
  └───┬────────────────────────────┘
      │
  ┌───v────────────────────────────┐  environment: production
  │ Stage 8: Deploy to Cloud Run   │  ← REQUIRES manual approval
  │          + smoke test          │
  │          + Slack notification  │
  └───┬────────────────────────────┘
      │
  ┌───v────────────────────────────┐
  │ Stage 9: Update Cloud Scheduler│  ← auto-creates monitoring job (every 6h)
  └────────────────────────────────┘
```

### Stage details

#### Stage 1 — Lint
- **ruff**: checks `src/` and `scripts/` for E (pycodestyle) and F (pyflakes) errors
- **mypy**: type checks `src/` (non-blocking — `|| true`)
- No secrets needed

#### Stage 2 — Unit Tests
- Installs `requirements.txt` and runs pytest
- Skips OBJ-2 (fire spread) and OBJ-3 (LLM) integration tests
- **Coverage gate**: fails if coverage drops below 35%
- No secrets needed

#### Stage 3 — Container Build (master only)
- Builds the Docker image from `model-pipeline/Dockerfile`
- Pushes to Google Container Registry: `gcr.io/wildfire-mlops-123/model-pipeline:<sha>`
- **Secrets needed**: `GCP_SA_KEY`

#### Stage 4 — Train
- Trains LightGBM models for California and Texas
- Logs experiment to MLflow
- Outputs `reports/training_result.json` (used by gates below)
- **Secrets needed**: `GCP_SA_KEY`, `MLFLOW_TRACKING_URI`

#### Stage 5 — AUC-PR Gate
- Reads `training_result.json` and checks AUC-PR >= 0.89 for all regions
- **BLOCKS the pipeline** if the model isn't accurate enough
- This prevents deploying a model that performs worse than the current production model

#### Stage 6 — Bias Gate
- Checks FNR (false negative rate) disparity across FEMA NRI vulnerability quartiles
- **BLOCKS the pipeline** if FNR disparity > 5%
- This ensures the model doesn't under-serve vulnerable communities (e.g., missing fires in high-poverty areas)

#### Stage 7 — Registry Push
- Verifies models are registered in Vertex AI Model Registry
- Labels them with `env=staging` + `run_id=<mlflow_run_id>`
- Uploads training report as GitHub artifact (14-day retention)

#### Stage 8 — Deploy (Manual Approval)
- **Requires reviewer approval** via GitHub Environment protection rules
- Deploys the inference API to Cloud Run (`wildfire-inference` service)
- **Smoke test**: hits `/health` endpoint with retry (5 attempts, 10s apart)
- Sends Slack notification on success or failure
- **Secrets needed**: `GCP_SA_KEY`, `MLFLOW_TRACKING_URI`, `SLACK_WEBHOOK_URL`

#### Stage 9 — Monitoring Scheduler
- Creates or updates a Cloud Scheduler job that hits `/monitor` every 6 hours
- This triggers drift detection on the deployed model automatically
- **Secrets needed**: `GCP_SA_KEY`, `GCP_SA_EMAIL`

### Concurrency

```yaml
concurrency:
  group: model-ci-${{ github.ref }}
  cancel-in-progress: true
```

If you push twice quickly to the same branch, the second run cancels the first. This prevents wasting resources on stale builds and avoids conflicting deploys.

---

## Workflow 3: `model_rollback.yml` — Manual Rollback

**File:** `.github/workflows/model_rollback.yml`
**Trigger:** `workflow_dispatch` only (manual — from GitHub Actions UI)

### When to use

If a deployed model is performing badly in production (detected via drift monitoring or user reports), this workflow rolls back to a known-good model version without retraining.

### Inputs

| Input | Required | Description |
|-------|----------|-------------|
| `version` | Yes | The `run_id` of the model to roll back to (from MLflow / Vertex AI) |
| `reason` | No | Why the rollback is needed (logged in Slack notification) |

### What it does

```
  1. Requires production environment approval (reviewer must approve)
  2. Demotes current production model → "archived" label in Vertex AI
  3. Promotes target run_id → "production" label in Vertex AI
  4. Sends Slack notification with result (success or failure)
```

No retraining, no rebuild, no redeploy — just swaps the label. Cloud Run reads the "production" label at inference time.

---

## What triggers what — quick reference

| Event | ci.yaml | model_ci.yml | model_rollback.yml |
|-------|---------|--------------|-------------------|
| PR to `main`/`develop` | Runs (Data-Pipeline paths) | Runs Stages 1-2 (model-pipeline paths) | — |
| Push to `master` | — | Runs all 9 stages | — |
| Manual dispatch | Runs | Runs all stages | Runs (with version input) |
| Drift detected (Cloud Scheduler) | — | Can trigger via dispatch | — |

## GitHub Secrets needed

| Secret | Used by | Purpose |
|--------|---------|---------|
| `GCP_SA_KEY` | model_ci (3,4-7,8,9), rollback | GCP service account credentials |
| `GCP_PROJECT_ID` | rollback | Vertex AI project ID |
| `GCP_SA_EMAIL` | model_ci (9) | OIDC auth for Cloud Scheduler |
| `MLFLOW_TRACKING_URI` | model_ci (4,8) | MLflow experiment tracking backend |
| `SLACK_WEBHOOK_URL` | model_ci (8), rollback | Deploy/rollback notifications |

## GitHub Environments needed

| Environment | Used by | Approval required? |
|-------------|---------|-------------------|
| `model-training` | model_ci Stages 4-7 | Optional (recommended) |
| `production` | model_ci Stage 8, rollback | **Yes** — add at least one reviewer |

---

## Pre-commit hooks (local)

Before code reaches CI, pre-commit hooks catch issues locally:

**File:** `.pre-commit-config.yaml`

| Hook | What it does |
|------|-------------|
| `ruff` | Lint + auto-fix Python errors |
| `ruff-format` | Auto-format Python code |
| `mypy` | Type checking |
| `trailing-whitespace` | Remove trailing whitespace |
| `end-of-file-fixer` | Ensure files end with newline |
| `check-yaml` | Validate YAML syntax |
| `check-added-large-files` | Block files > 1MB (prevents accidental data commits) |
| `detect-private-key` | Block commits containing private keys |
| `detect-secrets` | Block commits containing API keys, passwords, tokens |

### Setup

```bash
pip install pre-commit
pre-commit install
```

After this, hooks run automatically on every `git commit`. First run may be slow (downloads hook repos). To run manually on all files:

```bash
pre-commit run --all-files
```

---

## Architecture diagram (end-to-end)

```
Developer workstation                   GitHub Actions                    GCP
─────────────────────                   ──────────────                    ───
                                        
  git push ──────────────────────────> ci.yaml (Data Pipeline)
  (Data-Pipeline/*)                      │ Build → DAG check → Test
                                         │ → Lint → mypy → pip-audit
                                         └─ pass/fail (no deploy)
  
  git push ──────────────────────────> model_ci.yml (Model Pipeline)
  (model-pipeline/*)                     │ Lint → Test ──── (PR stops here)
                                         │
                                         │ (merge to master)
                                         │ Build image ──────────────────> GCR (container)
                                         │ Train CA + TX ─────────────────> MLflow (metrics)
                                         │ AUC-PR gate ✓
                                         │ Bias gate ✓
                                         │ Registry push ─────────────────> Vertex AI (model)
                                         │ ⏸ Manual approval
                                         │ Deploy ────────────────────────> Cloud Run (API)
                                         │ Smoke test ✓
                                         │ Slack notification ────────────> Slack (#wildfire)
                                         │ Cloud Scheduler ───────────────> Scheduler (6h)
                                         
  Manual dispatch ───────────────────> model_rollback.yml
                                         │ ⏸ Manual approval
                                         │ Swap labels ───────────────────> Vertex AI
                                         │ Slack notification ────────────> Slack (#wildfire)
```