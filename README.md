# Wildfire Detection & Response Platform — MLOps Deployment

A production-grade wildfire risk intelligence platform combining ignition prediction (OBJ-1, XGBoost), fire spread simulation (OBJ-2, Monte Carlo), and LLM-generated disaster reports (OBJ-3, Gemini + RAG), orchestrated through a fully automated CI/CD pipeline on **Google Cloud Platform**.

---

## 1. Deployment type: **Cloud** (GCP)

Following the Deployment PDF's taxonomy, this is a **cloud deployment** on Google Cloud Platform. Every artifact — models, containers, data, inference endpoints, the operator dashboard, and the orchestrator — runs on GCP managed services. No edge components.

### GCP services used

| Service | Role |
|---|---|
| **Cloud Run** | Serves the model inference backend (`wildfire-inference`) and the React dashboard (`wildfire-frontend`) |
| **Cloud Functions (Gen 2)** | `dag-trigger` — HTTP endpoint called by Cloud Scheduler every 30 min to kick off the Airflow DAG |
| **Cloud Scheduler** | Cron jobs: (a) `wildfire-dag-trigger` → fires `dag-trigger` every 30 min; (b) `wildfire-monitor` → fires drift-detection every 6 h |
| **GCE (Compute Engine)** | `wildfire-test-vm` (e2-standard-8) hosts Airflow (postgres + webserver + scheduler via Docker Compose) |
| **Vertex AI Model Registry** | Canonical store for trained models with `env=production` / `env=archived` labels; auto-promote on CI push, auto-archive prior prod |
| **Cloud Storage (GCS)** | `wildfire-mlops-123` bucket — fused feature parquets, inference JSON, simulation JSON, OBJ-3 reports, model artifacts, DVC remote |
| **Container Registry (GCR)** | `gcr.io/wildfire-mlops-123/` — model-pipeline image, wildfire-frontend image |
| **Cloud Build** | Runs `gcloud builds submit` for containerized image builds (triggered by CI/CD) |
| **Cloud Logging** | Structured logs from all Cloud Run services; queryable via `gcloud logging read` |
| **Secret Manager / GitHub Secrets** | `GCP_SA_KEY`, `FIRMS_MAP_KEY`, `GEMINI_API_KEY`, `SLACK_WEBHOOK_URL`, `MLFLOW_TRACKING_URI` |

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  GitHub (Scofe-C/wildfire_detection)                        │
│   ├── .github/workflows/ci.yaml           (Airflow CI/CD)   │
│   ├── .github/workflows/frontend-ci.yml   (Frontend CI/CD)  │
│   ├── .github/workflows/model_ci.yml      (Model CI/CD)     │
│   ├── .github/workflows/deploy-all.yml    (Orchestrator)    │
│   └── .github/workflows/model_rollback.yml                  │
└──────────────┬──────────────────────────────────────────────┘
               │ push / schedule / workflow_dispatch
               ▼
┌─────────────────────────────────────────────────────────────┐
│  GitHub Actions runners                                     │
│   • Cloud Build submit                                      │
│   • gcloud run deploy                                       │
│   • gcloud compute instances reset                          │
│   • Train + bias gate + Vertex AI registry push             │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌───────────────────────────────────────────────────────────────┐
│  GCP (project wildfire-mlops-123, region us-central1)         │
│                                                               │
│  Cloud Scheduler                                              │
│    ├── wildfire-dag-trigger  (*/30 * * * *)                   │
│    └── wildfire-monitor      (0 */6 * * *)                    │
│              │                      │                         │
│              ▼                      ▼                         │
│  Cloud Function             Cloud Run (wildfire-inference)    │
│   dag-trigger                  POST /monitor                  │
│     │                               │                         │
│     │ POST /dagRuns                 │ drift_detector.py       │
│     ▼                               │ → if threshold tripped: │
│  GCE VM (wildfire-test-vm)          │   POST workflow_dispatch│
│   Docker Compose:                   ▼                         │
│     ├── postgres                GitHub Actions                │
│     ├── airflow-webserver          (model_ci.yml)             │
│     └── airflow-scheduler                                     │
│      running wildfire_data_pipeline DAG (21 tasks)            │
│          │                                                    │
│          ▼                                                    │
│   trigger_model_server task POSTs to                          │
│   Cloud Run (wildfire-inference)                              │
│      /api/generate-from-pipeline                              │
│       ├── OBJ-1 XGBoost inference (per region)                │
│       │    uses Vertex AI Model Registry production model     │
│       ├── OBJ-2 Monte Carlo fire spread                       │
│       └── OBJ-3 Gemini 2.5 Flash report                       │
│                                                               │
│   Outputs → GCS wildfire-mlops-123:                           │
│     inference/latest/*.json                                   │
│     simulation/latest/*.json                                  │
│     reports/obj3/{region}/*.json                              │
│                                                               │
│  Cloud Run (wildfire-frontend)                                │
│     React + nginx; /api/* proxied to wildfire-inference       │
└───────────────────────────────────────────────────────────────┘
```

---

## 2. Deployment Automation

Five CI/CD workflows in `.github/workflows/` handle every deploy target. None require manual steps once the repo is set up.

### `ci.yaml` — Data-Pipeline (Airflow on GCE)

**Trigger**: Push to `master` touching `Data-Pipeline/**` or the workflow file.

**Jobs**:
1. `test` — Docker-based pytest (coverage ≥40%), DAG import validation, dvc.yaml syntax, ruff lint, pip-audit
2. `deploy-airflow` — runs on `push` to `master` with `vars.ENABLE_GCP_DEPLOY == 'true'`:
   - Tar repo tree → `/tmp/pipeline.tar.gz`
   - `gsutil cp` → `gs://wildfire-mlops-123/gce-test/pipeline.tar.gz`
   - Generate `.env` from GitHub secrets → upload to `gs://.../gce-test/.env`
   - `gcloud compute instances reset wildfire-test-vm` — the VM's startup script (`Data-Pipeline/cloud/gce_startup.sh`) re-fetches the tarball and runs `docker compose up -d --build`
   - Poll `http://<vm-ip>:8080/health` until Airflow returns 200
   - Slack notify

### `frontend-ci.yml` — React dashboard (Cloud Run)

**Trigger**: Push to `master` touching `Frontend/**` or the workflow file.

**Jobs**:
1. `build` — Node 20, `npm install --legacy-peer-deps`, `npm run build`, upload `dist/` artifact
2. `deploy` — runs on `push` to `master` with `ENABLE_GCP_DEPLOY=true`:
   - `gcloud builds submit Frontend/` → pushes `gcr.io/.../wildfire-frontend:latest`
   - `gcloud run deploy wildfire-frontend` (port 3000, 256Mi, public)
   - Smoke test `/`
   - Slack notify

### `model_ci.yml` — Model pipeline (Cloud Run + Vertex AI)

**Trigger**: (a) push to `master` touching `model-pipeline/**`, (b) nightly cron `0 0 * * *`, (c) `workflow_dispatch` with `triggered_by` input (used by monitor for drift-based retraining).

**Stages**:
| Stage | Purpose | Runs when |
|---|---|---|
| 1 | Unit tests (pytest, coverage ≥35%) | always |
| 3 | Container build + push (`gcr.io/.../model-pipeline:latest`) | `master` + `ENABLE_GCP_DEPLOY=true` |
| 4–7 | Train both regional models, AUC-PR gate (CA ≥ 0.89, TX ≥ 0.78), bias gate (FNR disparity ≤ 0.15 across region/season/fuel), Vertex AI registry push with `env=production` label (auto-archives prior prod) | same gate |
| 8 | `gcloud run deploy wildfire-inference` (port 8000, 4Gi, public), smoke test `/health`, Slack on success/failure | same gate |
| 9 | Create/update `wildfire-monitor` Cloud Scheduler job (drift checks every 6 h) | same gate |

### `deploy-all.yml` — Orchestrator

**Trigger**: `workflow_dispatch` or push touching `DEPLOY_ALL` sentinel file.

Dispatches all three deploy workflows in parallel via the GitHub API:

```bash
gh workflow run "Deploy everything" --ref master
```

### `model_rollback.yml` — One-click rollback

Calls `VertexRegistry.rollback()` — demotes current production to archived and promotes the most-recent archived back to production. Subsequent inference requests pick up the restored model without any redeployment.

### Global kill switch

`vars.ENABLE_GCP_DEPLOY` (repo-level GitHub Actions variable). When set to `false`, every GCP-touching stage in all three workflows skips — only lint/test stages still run. Used during iterative dev to avoid wasting CI minutes + Vertex training compute.

---

## 3. Connection to Repository

Auto-trigger on push is configured per workflow via the `on.push.branches + paths` filter. In addition:

- `model_ci.yml` is callable from code via the GitHub REST API's `workflow_dispatch`, which `monitor_runner._trigger_github_retrain()` uses for drift-driven retrains.
- A repo-level Personal Access Token with `workflow` scope is stored in **GCP Secret Manager** (`github-pat-model-retrain` secret — consumed only at request time by the `/monitor` endpoint).
- `GCP_SA_KEY` GitHub secret authenticates the CI runner to GCP via `google-github-actions/auth@v2`; the `cicd-deployer@wildfire-mlops-123.iam.gserviceaccount.com` service account holds the roles needed for every deploy target (see §9).

---

## 4. Replication steps

### 4.1 Prerequisites

- GCP project with billing enabled (this project uses `wildfire-mlops-123`, region `us-central1`)
- `gcloud` CLI installed and authenticated (`gcloud auth login`)
- `gh` CLI installed and authenticated (`gh auth login`)
- Python 3.11 (for local training/debugging; CI uses its own runner image)
- NASA FIRMS API key (free tier at https://firms.modaps.eosdis.nasa.gov/api/)
- Google Cloud API key with Gemini access (aistudio.google.com) OR Vertex AI enabled on the project
- (Optional) Slack workspace + incoming webhook URL for alerts

### 4.2 One-time GCP setup

```bash
# Set project + region
export PROJECT=wildfire-mlops-123
export REGION=us-central1
gcloud config set project $PROJECT
gcloud config set run/region $REGION

# Enable required APIs
gcloud services enable \
  run.googleapis.com \
  cloudfunctions.googleapis.com \
  cloudscheduler.googleapis.com \
  cloudbuild.googleapis.com \
  compute.googleapis.com \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  logging.googleapis.com

# Create GCS bucket (if not already)
gcloud storage buckets create gs://$PROJECT --location=$REGION

# Create CI/CD service account + grant roles
gcloud iam service-accounts create cicd-deployer \
  --display-name="CI/CD Deployer"

for role in \
  roles/run.admin \
  roles/storage.admin \
  roles/aiplatform.user \
  roles/cloudfunctions.developer \
  roles/cloudscheduler.admin \
  roles/artifactregistry.writer \
  roles/compute.instanceAdmin.v1 \
  roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding $PROJECT \
    --member="serviceAccount:cicd-deployer@${PROJECT}.iam.gserviceaccount.com" \
    --role="$role" --condition=None
done

# Create SA key + upload to GitHub secrets (see 4.3)
gcloud iam service-accounts keys create /tmp/sa-key.json \
  --iam-account=cicd-deployer@${PROJECT}.iam.gserviceaccount.com
```

### 4.3 Clone + set GitHub secrets

```bash
git clone https://github.com/Scofe-C/wildfire_detection.git
cd wildfire_detection

# Authenticate gh CLI (if not already)
gh auth login

# Upload secrets
gh secret set GCP_SA_KEY < /tmp/sa-key.json
gh secret set GCP_SA_EMAIL --body "cicd-deployer@wildfire-mlops-123.iam.gserviceaccount.com"
gh secret set GCP_PROJECT_ID --body "wildfire-mlops-123"
gh secret set FIRMS_MAP_KEY --body "<your NASA FIRMS key>"
gh secret set FIRMS_MAP_KEY_2 --body "<second FIRMS key for failover>"
gh secret set GOOGLE_API_KEY --body "<Google AI Studio key>"
gh secret set GEMINI_API_KEY --body "<Gemini key (can equal GOOGLE_API_KEY)>"
gh secret set SLACK_WEBHOOK_URL --body "<Slack incoming webhook URL>"
gh secret set MLFLOW_TRACKING_URI --body "sqlite:///mlruns.db"

# Set repo variable — master switch for GCP deploys
gh variable set ENABLE_GCP_DEPLOY --body "true"

# Cleanup
rm /tmp/sa-key.json
```

### 4.4 One-time GCE VM provisioning (Airflow)

```bash
cd Data-Pipeline
# Populate .env from .env.example (local dev only; CI regenerates this per deploy)
cp .env.example .env && vi .env    # fill in keys

# Provision the VM (creates instance + attaches startup-script metadata)
./cloud/deploy_gce_test.sh
```

This is only required once. Subsequent updates come automatically from CI — the VM's startup script re-fetches `gs://wildfire-mlops-123/gce-test/pipeline.tar.gz` + `.env` on every boot.

### 4.5 Cloud Function `dag-trigger` provisioning

```bash
cd Data-Pipeline
./cloud/deploy.sh
```

Deploys the `dag-trigger` Cloud Function + `wildfire-dag-trigger` Cloud Scheduler job (every 30 min).

### 4.6 First deploy — fire all three pipelines

```bash
gh workflow run "Deploy everything" --ref master
gh run watch   # follow live progress
```

This triggers (in parallel):
- `ci.yaml` → builds source tarball + generates `.env` → GCS → resets VM → Airflow comes up with fresh code
- `frontend-ci.yml` → Cloud Build → Cloud Run deploys `wildfire-frontend`
- `model_ci.yml` → train both regional models → Vertex AI registry push → Cloud Run deploys `wildfire-inference`

Total end-to-end time: ~10 minutes (Model CI is the slowest).

### 4.7 Verify deployment

```bash
# All three Cloud Run / GCE services should be healthy
gcloud run services list --region=us-central1
gcloud compute instances list

# Backend health
curl -s https://wildfire-inference-987262292513.us-central1.run.app/api/status | jq .
# Expect: reporter_loaded: true, backend: "vertex_ai", gemini.api_key_set: true

# Frontend
curl -sf -o /dev/null -w "%{http_code}\n" https://wildfire-frontend-987262292513.us-central1.run.app/
# Expect: 200

# Frontend proxy to backend
curl -s https://wildfire-frontend-987262292513.us-central1.run.app/api/reports?limit=3 | jq '.[]|.id'

# Airflow webserver
VM_IP=$(gcloud compute instances describe wildfire-test-vm --zone=us-central1-a --format='value(networkInterfaces[0].accessConfigs[0].natIP)')
curl -sf -o /dev/null -w "%{http_code}\n" http://$VM_IP:8080/health
# Expect: 200

# Manually trigger an end-to-end pipeline run
curl -u airflow:airflow -X POST \
  http://$VM_IP:8080/api/v1/dags/wildfire_data_pipeline/dagRuns \
  -H 'Content-Type: application/json' \
  -d '{"dag_run_id":"manual_'$(date +%s)'","conf":{}}'
```

### 4.8 Verifying OBJ-1 → OBJ-2 → OBJ-3 end-to-end

```bash
# Scheduler + Cloud Function will eventually drive this, but to test directly:
curl -s -X POST https://wildfire-inference-987262292513.us-central1.run.app/api/generate-from-pipeline \
  -H 'Content-Type: application/json' \
  -d '{"regions":["california","texas"]}' | jq .

# Inspect GCS outputs
gsutil ls gs://wildfire-mlops-123/inference/latest/
gsutil ls gs://wildfire-mlops-123/simulation/latest/
gsutil ls gs://wildfire-mlops-123/reports/obj3/latest/

# Inspect Vertex AI Model Registry
gcloud ai models list --region=us-central1 \
  --filter='displayName:wildfire-ignition-california OR displayName:wildfire-ignition-texas'
```

---

## 5. Model Monitoring and Retraining

### 5.1 Performance and data drift monitoring

The `/monitor` endpoint on `wildfire-inference` (implemented in `model-pipeline/src/monitoring/monitor_runner.py`, invoked every 6 h by the `wildfire-monitor` Cloud Scheduler job) performs three checks:

1. **Feature distribution drift** (`drift_detector.py`)
   - **PSI** (Population Stability Index) on every numeric feature against the stored training baseline
   - **Jensen-Shannon divergence** as a second-opinion metric
   - Baseline parquets stored at `gs://wildfire-mlops-123/model-artifacts/baselines/{region}/`
   - Threshold: PSI > 0.2 on any single feature

2. **SHAP feature-importance drift**
   - Compares current-batch SHAP importances to training-time SHAP baseline
   - Threshold: relative importance of any top-5 feature drops >0.05 vs. baseline (especially soil moisture, which is the dominant wildfire signal)

3. **Model performance degradation**
   - Computed from recent inference vs. actuals (where FIRMS data acts as near-real-time ground truth)
   - Threshold: rolling PR-AUC drops below the region's deployment gate (CA: 0.89, TX: 0.78)

Thresholds are configured in `model-pipeline/configs/model_config.yaml` under `validation:` and can be tuned without code changes.

### 5.2 Automatic retraining trigger

When any threshold is crossed, `monitor_runner._trigger_github_retrain()` POSTs to the GitHub REST API:

```
POST /repos/Scofe-C/wildfire_detection/actions/workflows/model_ci.yml/dispatches
Body: { "ref": "master", "inputs": { "triggered_by": "drift_detection" } }
```

This kicks off a full `model_ci.yml` run:

```
Stage 1  Unit tests
Stage 3  Container build + push
Stage 4  Train both regional models
Stage 5  AUC-PR gate (PASS required; else rollback)
Stage 6  Bias gate (FNR disparity ≤ 0.15; else rollback)
Stage 7  Vertex AI registry push (new model → env=production, old → env=archived)
Stage 8  gcloud run deploy wildfire-inference
Stage 9  Update wildfire-monitor Cloud Scheduler
```

If any gate fails, `model_ci.yml` aborts, Slack is notified, and the existing production model stays active (no rollback needed since no promotion happened).

If a previous deploy causes regressions, run:
```bash
gh workflow run model_rollback.yml --ref master
```
which invokes `VertexRegistry.rollback()` — atomically re-promotes the most recently archived model. Subsequent inference calls pick it up with no re-deploy needed.

### 5.3 Manually simulate drift → retrain → redeploy (for the video demo)

```bash
gh workflow run model_ci.yml --ref master -f triggered_by=drift_detection
```

Identical to what `monitor_runner` would do when drift is detected.

### 5.4 Notifications

Slack webhook fires on every:
- Airflow task failure after 3 consecutive retries (from `Data-Pipeline/dags/utils/slack_notify.py`)
- **Retraining triggered** (from `monitor_runner`, before the workflow dispatch)
- **Training success + new model deployed** (`alert_success` in `src/notifications/alerter.py`, fired from `model_ci.yml` Stage 8)
- **Training failure / deploy failure** (`alert_validation_failure`, `alert_rollback`)
- CRITICAL fire risk cells detected in inference (`alert_critical_fire_risk`)
- Data drift warning (`alert_data_drift` from the drift detector)

Slack message includes the GitHub commit SHA for traceability.

### 5.5 Pipeline resilience guardrails

- **Weather circuit breaker** — `Data-Pipeline/scripts/fusion/fuse_features.py` aborts fusion if weather null rate > 80% in any region; prevents model poisoning by stale forward-fills.
- **Two-tier weather fallback** — Open-Meteo → NWS → last-known-good; serialized through an Airflow pool to avoid 429 rate limits.
- **Data leakage prevention** — OBJ-1 inference drops 8 FIRMS-derived "pipeline-only" columns before running `full_pipeline`; any leak raises `ValueError` loudly.
- **Idempotent retrain** — `VertexRegistry.push()` atomically demotes prior `env=production` to `env=archived` before promoting the new version, so even interrupted runs leave the registry in a consistent state.

---

## 6. Logging & Observability

| Signal | Where | How to access |
|---|---|---|
| Cloud Run request/response logs | Cloud Logging | `gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="wildfire-inference"' --limit=50` |
| OBJ-1 / OBJ-2 / OBJ-3 execution traces | Cloud Logging (filter `textPayload=~"OBJ-[123]"`) | Same command with added filter |
| Airflow DAG run history, task logs, Gantt | Airflow webserver | `http://<vm-ip>:8080` — credentials `airflow/airflow` |
| Training metrics, SHAP plots, run history | MLflow | `mlflow ui` (local; backing store is `sqlite:///mlruns.db`) or Vertex AI Experiments |
| Cloud Scheduler job state + last-run result | GCP console | `gcloud scheduler jobs describe wildfire-monitor --location=us-central1` |
| Model registry state (which model is prod) | Vertex AI Model Registry | `gcloud ai models list --region=us-central1` |
| Pipeline status, aggregated OBJ-1/2/3 events | Frontend UI | `/api/notifications` endpoint aggregates from GCS → bell icon in dashboard |
| Alert history | Slack | Channel receiving `SLACK_WEBHOOK_URL` notifications |

Cloud Logging retains structured logs for 30 days. For longer retention, export to a BigQuery sink (not configured by default).

---

## 7. Code & Environment Layout

```
wildfire_detection/
├── .github/workflows/
│   ├── ci.yaml                 # Data-Pipeline / Airflow CI/CD
│   ├── frontend-ci.yml         # Frontend CI/CD
│   ├── model_ci.yml            # Model pipeline CI/CD (train + deploy)
│   ├── model_rollback.yml      # One-click rollback workflow
│   └── deploy-all.yml          # Orchestrator — fires all three in parallel
│
├── Data-Pipeline/              # Airflow DAG + data ingestion
│   ├── dags/wildfire_dag.py    # 21-task DAG (CA+TX sharded ingest, fuse, anomaly, DVC, trigger model server)
│   ├── scripts/                # Ingestion (FIRMS, Open-Meteo, NWS, HRRR, GOES), processing, fusion, validation, anomaly
│   ├── configs/                # Pipeline + schema configs
│   ├── docker/Dockerfile       # airflow-base + airflow-init + airflow-webserver + airflow-scheduler multi-stage
│   ├── docker-compose.yaml     # Local dev + VM runtime
│   ├── cloud/
│   │   ├── deploy.sh           # Deploy Cloud Function + Cloud Scheduler
│   │   ├── deploy_gce_test.sh  # One-time VM provisioning
│   │   ├── gce_startup.sh      # VM boot script (fetches tarball from GCS, runs docker compose)
│   │   └── dag_trigger/main.py # Cloud Function — triggers Airflow DAG via REST API
│   └── tests/                  # pytest suite
│
├── model-pipeline/             # Training + inference + OBJ-3 reporting
│   ├── src/
│   │   ├── api/server.py       # FastAPI — OBJ-1 inference, /monitor, /api/generate-from-pipeline, /api/reports
│   │   ├── preprocessing/feature_engineering.py   # full_pipeline (train+inference, returns (X, state) with medians)
│   │   ├── models/
│   │   │   ├── obj1_ignition/  # XGBoost + LightGBM training
│   │   │   ├── obj2_spread/    # Monte Carlo fire spread (Rothermel + Byram)
│   │   │   └── obj3_gemini/    # LLM reporter (Vertex AI Gemini + RAG corpus)
│   │   ├── pipeline/orchestrator.py               # Train → validate → bias gate → registry push → alert chain
│   │   ├── tracking/vertex_registry.py            # push, load_production, rollback
│   │   ├── monitoring/
│   │   │   ├── monitor_runner.py                  # /monitor endpoint implementation + _trigger_github_retrain
│   │   │   ├── drift_detector.py                  # PSI, JS divergence, SHAP drift
│   │   │   └── performance_monitor.py             # Rolling PR-AUC, precision at threshold
│   │   ├── reports/report_manager.py              # Local + GCS persistence for OBJ-3 reports
│   │   ├── bias/                                  # Fairlearn-based bias gate
│   │   ├── validation/model_selector.py           # AUC-PR + F1 gate
│   │   └── notifications/alerter.py               # Slack webhook
│   ├── configs/
│   │   ├── model_config.yaml                      # AUC-PR gates, bias thresholds, decision thresholds
│   │   └── reporting_config.yaml                  # LLM backend, RAG corpus, incident tracker
│   ├── corpus/                                    # RAG reference docs (FEMA NRI, Scott-Burgan fuel types)
│   ├── templates/                                 # Jinja2 templates for rendered OBJ-3 HTML/MD
│   └── Dockerfile                                 # base + dashboard (Cloud Run) targets
│
├── Frontend/                   # React + Vite + Tailwind operator dashboard
│   ├── src/components/         # Overview, DataPipeline, OBJ1/2/3 panels, FireMap, RiskMonitor, IncidentReports
│   ├── src/hooks/useAPI.js     # SWR-style fetch hook with auto-stop polling
│   ├── src/api.js              # apiUrl + normalizeCell helpers
│   ├── nginx.conf              # Reverse proxies /api/* to wildfire-inference
│   └── Dockerfile              # node build → nginx serve multi-stage
│
└── README.md                   # This document
```

---

## 8. Service Account IAM — reference

`cicd-deployer@wildfire-mlops-123.iam.gserviceaccount.com` is the SA whose key is stored as `GCP_SA_KEY` in GitHub secrets. It holds:

| Role | Needed for |
|---|---|
| `roles/run.admin` | Create/update Cloud Run services (backend + frontend) |
| `roles/storage.admin` | Upload tarballs, read/write GCS buckets (model artifacts, reports, inference JSON) |
| `roles/aiplatform.user` | Vertex AI Model Registry push/load/rollback, Gemini API calls |
| `roles/cloudfunctions.developer` | Deploy the `dag-trigger` Cloud Function |
| `roles/cloudscheduler.admin` | Reconcile `wildfire-monitor` job from model_ci Stage 9 |
| `roles/artifactregistry.writer` | Push images to GCR via Cloud Build |
| `roles/compute.instanceAdmin.v1` | Reset/start `wildfire-test-vm` from ci.yaml |
| `roles/iam.serviceAccountUser` | Act as other SAs when needed by gcloud commands |

The GCE VM itself runs as the default Compute SA with `roles/editor` + `cloud-platform` OAuth scope — sufficient for the Airflow containers to read/write GCS and call Vertex AI.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Frontend shows blank data | Backend `wildfire-inference` is down or the Cloud Run URL in `Frontend/nginx.conf` is stale | `gcloud run services describe wildfire-inference --region=us-central1` to confirm, or re-deploy frontend |
| `/api/status` returns 403 | Cloud Run deploy ran with `--no-allow-unauthenticated` (older CI config) | Re-run the updated `model_ci.yml` Stage 8 (current version uses `--allow-unauthenticated`) |
| Cloud Scheduler 400 on DAG trigger | Airflow run_id prefix `scheduled__` is reserved | Cloud Function uses `cloudscheduler__` prefix (committed in this repo) |
| OBJ-1 inference JSON has too many cells | OBJ-1 block reading all parquet snapshots instead of newest | Fixed in server.py — reads only the newest parquet sorted by name descending |
| OBJ-3 reporter fails to load on Cloud Run | `.gitignore` pattern `reports/` excluded `src/reports/` Python package from build context | `.gitignore` narrowed to `reports/disaster_reports/` etc. |
| CI deploy skipped | `vars.ENABLE_GCP_DEPLOY` is `false`, or pushed commit didn't touch any path in the workflow's filter | `gh variable set ENABLE_GCP_DEPLOY --body "true"`; touch a relevant file or use `gh workflow run "Deploy everything"` |
| Airflow VM deploy skipped | `cicd-deployer` missing `compute.instanceAdmin.v1` role | Grant via `gcloud projects add-iam-policy-binding` (see §4.2) |

---

## 10. Evaluation Criteria Coverage (per Deployment PDF §8)

| PDF criterion | Where in this repo |
|---|---|
| Correctness & Completeness | CI/CD fully automated end-to-end — `.github/workflows/` directory |
| Documentation & Replication Steps | §4 of this README; all commands runnable end-to-end |
| Model Optimization (Edge) | N/A — cloud deployment |
| Automated CI/CD Integration | `.github/workflows/` with `on.push` auto-triggers + `workflow_dispatch` API support |
| Logs & Monitoring | §6 — Cloud Logging, Airflow UI, MLflow, Slack, in-app notifications |
| Model Monitoring & Retraining | §5 — `monitor_runner.py` + 6-h Cloud Scheduler + workflow_dispatch |
| Video Demonstration | Separately submitted |

---

## 11. Live endpoints (reference — these may rotate on redeploy)

| Service | URL |
|---|---|
| Backend inference | `https://wildfire-inference-987262292513.us-central1.run.app` |
| Frontend dashboard | `https://wildfire-frontend-987262292513.us-central1.run.app` |
| Airflow webserver | `http://<vm-external-ip>:8080` (get with `gcloud compute instances describe wildfire-test-vm --format='value(networkInterfaces[0].accessConfigs[0].natIP)'`) |
| Cloud Function (dag-trigger) | `https://dag-trigger-axwugrteea-uc.a.run.app` |
| GitHub Actions | `https://github.com/Scofe-C/wildfire_detection/actions` |

---

## 12. Key tunables

Location | Setting | Typical values
---|---|---
`model-pipeline/configs/model_config.yaml` | `validation.auc_pr_threshold` | CA 0.89, TX 0.78 — deployment gate
`model-pipeline/configs/model_config.yaml` | `bias_gate.max_disparity` | 0.15 — FNR disparity across region/season/fuel slices
`model-pipeline/configs/model_config.yaml` | `validation.xgb_decision_threshold` | 0.365 — baseline floor for threshold tuning (per-run tune overrides this)
`model-pipeline/configs/reporting_config.yaml` | `llm_backend` | `vertex_ai` (primary), `gemini_dev` (fallback), `ollama` (local)
`Data-Pipeline/dags/wildfire_dag.py` | `SCHEDULE_INTERVAL` | `None` — triggered externally via Cloud Scheduler → Cloud Function
`Data-Pipeline/scripts/fusion/fuse_features.py` | `null_rate_threshold` | 0.80 — circuit breaker for weather data
GitHub repo variable | `ENABLE_GCP_DEPLOY` | `true` / `false` — master switch for Stage 3+ GCP actions
