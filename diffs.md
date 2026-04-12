# Branch Diff Summary — Wildfire Detection MLOps Project

**Repository:** `wildfire_detection/`  
**Baseline:** `master` (47 commits, last updated 2026-03-30)  
**Analysis date:** 2026-04-10  

---

## Table of Contents

1. [Repository Overview](#1-repository-overview)
2. [Branch Summary Table](#2-branch-summary-table)
3. [master — Baseline State](#3-master--baseline-state)
4. [dev_ack — Model Pipeline (OBJ-1)](#4-dev_ack--model-pipeline-obj-1)
5. [dev-sco — Monitoring, Deployment, OBJ-3 Dashboard](#5-dev-sco--monitoring-deployment-obj-3-dashboard)
6. [ibrahim_dev — OBJ-2 Fire Spread Simulator](#6-ibrahim_dev--obj-2-fire-spread-simulator)
7. [Feature Ownership Matrix](#7-feature-ownership-matrix)
8. [File-Level Diff Inventory](#8-file-level-diff-inventory)
9. [Key Conflicts and Integration Notes](#9-key-conflicts-and-integration-notes)

---

## 1. Repository Overview

The project is an end-to-end MLOps wildfire detection system with three numbered objectives:

| Objective | Name | Description |
|---|---|---|
| **OBJ-1** | Fire Ignition Risk | XGBoost + LightGBM per-region classifiers predicting 6-hour fire probability |
| **OBJ-2** | Fire Spread Simulation | Cell2Fire C++ wrapper + pure-Python Rothermel physics simulator |
| **OBJ-3** | LLM Disaster Reporting | Gemini-powered natural language incident reports via FastAPI dashboard |

The repo has two top-level components:

```
wildfire_detection/
├── Data-Pipeline/     ← Airflow DAGs, data ingestion, fusion, detection (shared infrastructure)
├── model-pipeline/    ← ML models, training, validation, inference, serving
└── .github/workflows/ ← CI/CD
```

---

## 2. Branch Summary Table

| Branch | Owner | Unique Commits | Files Changed vs master | +Insertions | -Deletions | Last Commit |
|---|---|---|---|---|---|---|
| `master` | Shared | 47 (baseline) | — | — | — | 2026-03-30 |
| `dev_ack` | Ackshay | 6 | 56 files | +35,370 | -2,479 | 2026-04-03 |
| `dev-sco` | Scott | 19 | 128 files | +44,210 | -2,710 | 2026-04-09 |
| `ibrahim_dev` | Ibrahim | 9 | 129 files | +48,205 | -2,085 | 2026-04-10 |

**Note on inheritance:** `dev-sco` merged `dev_ack` on 2026-04-03; `ibrahim_dev` merged `dev-sco` on 2026-04-10. So `ibrahim_dev` is a superset of everything.

---

## 3. master — Baseline State

### What master contains

master represents the first stable integration of all three OBJ-1/2/3 implementations as merged from the `dev` branch.

#### Data Pipeline (`Data-Pipeline/`)

| Component | Status |
|---|---|
| Airflow DAG (`dags/wildfire_dag.py`) | 6-hourly, CA+TX region-sharded via `TaskGroups`, resolution escalation 64km → 22km on confirmed fire |
| Watchdog sensor DAG (`watchdog_sensor_dag.py`) | 2-minute `PythonSensor` polls GCS for Cloud Function trigger files |
| 8 data ingestors | FIRMS, GOES, HRRR, LANDFIRE, NDVI, SRTM, weather, field telemetry |
| Feature fusion (`fuse_features.py` + `priority_resolver.py`) | Multi-source fusion with temporal lag |
| Fire detection (`fire_detector.py` + `emergency.py`) | FIRMS-based detection + emergency escalation |
| LLM data pipeline reports (`report_generator.py`) | Separate from OBJ-3 model reports |
| Static parquet data (`data/static/`) | Pre-baked LANDFIRE + static features at 64km |
| Test suite | ~20+ test files for ingestion, fusion, detection, DAGs, DVC, validation, export |
| Cloud Function (`cloud/fire_watchdog/main.py`) | FIRMS polling + GCS trigger file |

#### Model Pipeline (`model-pipeline/`)

| Component | Status in master |
|---|---|
| **OBJ-1** `src/models/obj1_xgboost/model.py` | Basic `XGBoostFireRiskModel` — ERA5 feature names, `max_depth=6`, `n_estimators=100`, no hyperparameter tuning |
| **OBJ-2** `src/models/obj2_spread/` | **Fully refactored** — split into 5 modules: `cell2fire_spread.py`, `weather.py`, `raster.py`, `evaluation.py`, `exceptions.py`; PROPAGATOR demo model; 43 tests |
| **OBJ-3** `src/models/obj3_gemini/` | **Complete** — state machine, 3 LLM adapters (Ollama/Gemini Dev/Vertex), 5 report schemas, FastAPI server, Jinja2 templates, corpus tooling, 43+ tests |
| Bias gate | `src/bias/detector.py` — Fairlearn `MetricFrame`, FNR across NRI vulnerability quartiles, `max_disparity=0.05` |
| MLflow tracking | `src/tracking/mlflow_logger.py` — SQLite backend |
| Validation | `src/validation/model_selector.py`, `metrics.py`, `visualizations.py` |
| FastAPI dashboard | `src/api/server.py` — serves reports, system status |
| Slack notifications | `src/notifications/alerter.py` |
| CI/CD | `.github/workflows/model_ci.yml` (basic), `ci.yaml` (Data Pipeline CI) |

#### Commit history highlights (master)

| Commit | Date | Description |
|---|---|---|
| `bffc6fb` | 2026-03-30 | Merge branch 'dev' — pulled in OBJ-2 Cell2Fire modularization |
| `87e97b9` | 2026-03-30 | Bug fixes + OBJ-3 FastAPI dashboard |
| `0f5faff` | 2026-03-26 | Refactor OBJ-2: monolith → 5 modules + 43 tests |
| `e5e2fd5` | 2026-03-25 | Cell2Fire config + spread logic updates |
| `bf76e06`–`9937293` | 2026-03-21 | 4 commits adding LLM functions (OBJ-3) |
| `da49b67` | 2026-03-18 | Initial Cell2Fire addition |
| `8c694b1`/`69fb0ad` | 2026-03-17 | Bootstrap model-pipeline directory |
| `1f4dd36` | 2026-02-08 | Initial full Data-Pipeline setup (Airflow, Docker, DVC, CI/CD, GCP) |

---

## 4. dev_ack — Model Pipeline (OBJ-1)

**Branch owner:** Ackshay  
**Diverged from:** master  
**6 unique commits** (2026-03-31 → 2026-04-03)  
**56 files changed: +35,370 / -2,479**

### What dev_ack adds

This branch is the primary OBJ-1 implementation: a production-grade per-region XGBoost + LightGBM training pipeline with Vertex AI registry, MLflow tracking, bias gating, SHAP explainability, and an hourly inference loop.

---

### New Directories

| Directory | What was added |
|---|---|
| `model-pipeline/src/preprocessing/` | `feature_engineering.py` — canonical 21-feature transform pipeline |
| `model-pipeline/src/models/obj1_lightgbm/` | LightGBM ignition model (secondary/comparison) |
| `model-pipeline/src/tracking/` (expanded) | `vertex_registry.py`, `vertex_sync.py` added to existing `mlflow_logger.py` |
| `model-pipeline/src/validation/` (expanded) | `bias_check.py` replaces old `src/bias/detector.py` |
| `model-pipeline/scripts/` | `train.py`, `inference.py`, `combine_historical_data.py`, `drop_2026_data.py`, `upload_historical_to_gcs.py` |
| `model-pipeline/historical_data/` | `california_historical.csv` (13,318 rows), `texas_historical.csv` (15,469 rows) |
| `model-pipeline/experimentation/` | `california.ipynb` — experiment notebook, source of truth for validated metrics |

---

### New/Rewritten Files

#### `model-pipeline/src/preprocessing/feature_engineering.py` (405 lines, new)

Single source of truth for all transforms — shared by XGBoost, LightGBM, and inference to prevent training/serving skew.

**Canonical 21-feature set:**
- Weather: `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `precipitation`, `soil_moisture_0_to_7cm`, `vpd`, `fire_weather_index`
- Angular (sin/cos encoded): `wind_direction_10m_{sin,cos}`, `aspect_degrees_{sin,cos}`
- Terrain: `elevation_m`, `slope_degrees`, `dominant_fuel_fraction`, `ndvi`
- Geo: `latitude`, `longitude`
- Categorical: `fuel_model_fbfm40`, `vegetation_type`
- Derived: `cumulative_wind_run_24h`, `drought_index_proxy`

**Preprocessing order** (strictly enforced, must not be reordered):
1. Sentinel fix: `-9999 → NaN`
2. `ffill`/`bfill` time-series derived cols per `grid_id` before it is dropped
3. `drop_non_features()` — removes leakage cols; **raises** at inference if leakage present
4. `impute_before_encoding()` — median-impute angular cols before sin/cos
5. `apply_circular_encoding()` — wind/aspect degrees → sin/cos pairs, drop originals
6. `apply_log1p()` — precipitation, vpd, FWI, soil_moisture clipped to 0 then log1p
7. `apply_median_imputation()` — elevation, slope, ndvi, dominant_fuel_fraction, soil_moisture
8. `apply_categorical_imputation()` — mode-fill fuel_model_fbfm40, vegetation_type
9. `apply_ordinal_encoding()` (XGBoost only) / `cast_category_dtype()` (LightGBM only)
10. `validate_no_nulls()` — raises if any nulls remain; LOG1P_COLS fallback fills 0 for API gaps

#### `model-pipeline/scripts/train.py` (187 lines, new)

CLI entry point with two modes:

| Mode | Behavior |
|---|---|
| `--mode initial` | Trains XGBoost + LightGBM for all specified regions, selects winner by AUC-PR |
| `--mode retrain` | Trains XGBoost only (fast daily CI/CD mode) |
| `--local` flag | Reads local CSVs, skips Vertex AI (dev mode) |

Calls `run_training_pipeline()` from orchestrator per region, writes per-region JSON report used by CI/CD gate checks.

#### `model-pipeline/scripts/inference.py` (484 lines, new)

6-hour fire risk scoring pipeline:

1. Fetches rolling 24h weather from **Open-Meteo API** for all CA + TX grid centroids (~55 cells)
2. Joins static terrain features from `Data-Pipeline/data/static/static_features_64km.parquet` on `grid_id`
3. Loads Production model + decision threshold from **Vertex AI Model Registry**
4. Preprocesses with `feature_engineering.full_pipeline()` using training medians from `model_metadata.json`
5. Scores all cells, assigns risk tiers: `LOW` / `MEDIUM` / `HIGH` / `CRITICAL`
6. Writes partitioned Parquet to GCS: `inference/region=.../year=.../month=.../`
7. Overwrites `inference/latest/{region}_latest.json` (polled by OBJ-3 watchdog)
8. Sends Slack alert when any cell reaches CRITICAL (≥ 0.65 score)

**Risk tier thresholds:** `CRITICAL ≥ 0.65`, `HIGH ≥ 0.365`, `MEDIUM ≥ 0.15`, `LOW < 0.15`

#### `model-pipeline/src/pipeline/orchestrator.py` (590 lines, heavily rewritten +618 net)

12-step training pipeline per region:

| Step | Action |
|---|---|
| 1 | Load region data from GCS (or local CSV in dev mode) with resilient retry/backoff |
| 2 | Temporal split: `train < 2025-01-01`, `test = Jan 2025 LA fires`, `LABEL_CUTOFF = 2025-12-31` |
| 3 | Preprocess with `feature_engineering.full_pipeline()` |
| 4 | Train XGBoost always; train LightGBM in `initial` mode only |
| 5 | Hyperparameter tuning: `RandomizedSearchCV` with `TimeSeriesSplit(5)`, `n_iter=50`, `scoring=roc_auc` |
| 6 | Select winner by AUC-PR; trigger rollback if all candidates fail validation threshold |
| 7 | Tune decision threshold: find highest threshold still achieving ≥ 90% recall (`candidates[-1]` logic from notebook) |
| 8 | SHAP explainability via `TreeExplainer`, `n_background_samples=500` |
| 9 | Generate visualizations: PR curve, confusion matrix, model comparison bar chart |
| 10 | Bias gate — FNR disparity across `region`, `fire_season`, `fuel_model_fbfm40` slices (non-blocking) |
| 11 | Push winning model to Vertex AI Model Registry |
| 12 | Sync experiment run to Vertex AI Experiments (non-blocking) |

**Key design decisions:**
- Non-blocking bias gate: pipeline always proceeds to registry push even if bias gate fails; failure is logged to MLflow and triggers a Slack alert
- Rollback: triggered automatically if AUC-PR < threshold; promotes most recent archived Vertex AI version back to Production
- Threshold re-set after `result.metrics.update()` to prevent default `0.365` overwriting tuned value

#### `model-pipeline/src/models/obj1_xgboost/model.py` (239 lines, rewritten)

Replaces the minimal master version with:
- `tune()`: RandomizedSearchCV (50 iter, 5-fold TimeSeriesSplit, roc_auc), `scale_pos_weight` for class imbalance
- `fit()`: trains `XGBClassifier` with tuned best params
- `predict_proba()`: raw probability output
- `tune_threshold()`: `candidates[-1]` logic — highest threshold ≥ 90% recall
- `explain()`: SHAP TreeExplainer + native XGBoost gain importance
- **Validated metrics:** ROC-AUC 0.9426, PR-AUC 0.8927, decision threshold 0.365

#### `model-pipeline/src/models/obj1_lightgbm/model.py` (239 lines, new)

Secondary model, same interface as XGBoost:
- `is_unbalance=True` (instead of `scale_pos_weight`)
- Natively handles pandas `category` dtype (no OrdinalEncoder)
- Decision threshold 0.239 for ≥ 90% recall
- **Validated metrics:** ROC-AUC 0.9374, PR-AUC 0.8837 (XGBoost wins)

#### `model-pipeline/src/tracking/vertex_registry.py` (270 lines, new)

`VertexRegistry` class:

| Method | Behavior |
|---|---|
| `push(model, region, run_id, metrics, threshold, medians, features, framework)` | Saves `model.bst`/`model.txt` + `model_metadata.json` to `gs://{bucket}/model-artifacts/{run_id}/`; registers with Vertex AI; demotes current Production → archived; promotes new version → Production via `labels.env` |
| `load_production(region)` | Finds model with `labels.env="production"`, reconstructs GCS path from `labels.run_id`, returns `(model, medians, threshold)` |
| `rollback(region)` | Promotes most recently archived version back to Production |

**GCS layout:** `gs://wildfire-mlops-123/model-artifacts/{run_id}/model.bst` + `model_metadata.json`

#### `model-pipeline/src/validation/bias_check.py` (168 lines, new)

Replaces the old Fairlearn-based `src/bias/detector.py`. Uses only pandas + sklearn.

**Three bias slices:**
1. `region` — california vs texas
2. `fire_season` — May–Oct vs Nov–Apr
3. `fuel_model_fbfm40` — per fuel type (high-risk vegetation must not be systematically under-detected)

**Gate thresholds:**
- `max_disparity: 0.15` (raised from 0.05 — statistically unreachable on 713-row test set)
- `min_group_size: 20` — skip groups with fewer than 20 total rows
- `min_fire_count: 5` — skip groups with fewer than 5 actual fire events (FNR unreliable at low counts)

#### `model-pipeline/src/tracking/mlflow_logger.py` (updated)

New capabilities added vs master:
- `log_bias_gate_result()` — logs FNR disparity per slice as MLflow metrics
- `log_shap()` — logs mean absolute SHAP values as `shap_{feature_name}` metrics
- `log_threshold()` — logs operational decision threshold as metric (not param, since MLflow params are immutable)
- `log_visualization()` — uploads visualization artifacts to MLflow run

---

### Config Changes

#### `model-pipeline/configs/model_config.yaml` (substantially rewritten)

Key changes from master default:

| Section | Key Values Added / Changed |
|---|---|
| `validation` | `auc_pr_threshold: 0.89` (real notebook result), `xgb_decision_threshold: 0.365`, `lgbm_decision_threshold: 0.239`, `target_recall: 0.90` |
| `bias_gate` | `metric: false_negative_rate`, `max_disparity: 0.15`, `min_group_size: 20`, `min_fire_count: 5` |
| `tracking.mlflow` | `tracking_uri: sqlite:///mlruns.db`, `experiment_name: wildfire-ignition-v1` |
| `tracking.vertex_ai` | `project_id: wildfire-mlops-123`, `location: us-central1`, `model_artifact_gcs_prefix: model-artifacts` |
| `shap` | `enabled: true`, `n_background_samples: 500`, `min_soil_moisture_importance: 0.05` (drift alert) |
| `notifications` | `alert_on: [bias_gate_failure, validation_failure, pipeline_error, rollback, shap_drift, critical_fire_risk]` |

---

### CI/CD Changes

#### `.github/workflows/model_ci.yml` (substantially rewritten, +418 net lines)

8-stage pipeline:

| Stage | Trigger | Action |
|---|---|---|
| 1. Lint | PR + push | `ruff check src/ scripts/` |
| 2. Unit tests | PR + push | `pytest`, skip obj2/obj3 |
| 3. Container build | PR + push | Docker build + push to `gcr.io/wildfire-mlops-123/model-pipeline:{sha}` |
| 4. Train | merge to master | `python -m scripts.train --mode retrain --regions california texas` |
| 5. AUC-PR gate | merge to master | Read JSON report, fail if any region < 0.89 |
| 6. Bias gate check | merge to master | Fail if any region `bias_gate_passed=False` |
| 7. Vertex AI verify | merge to master | Confirm `registry_version` set for all regions |
| 8. Deploy | manual approval (`environment: production`) | `gcloud run deploy wildfire-inference` to Cloud Run |

#### `.github/workflows/model_rollback.yml` (new, 85 lines)

Manual `workflow_dispatch` rollback with `environment: production` reviewer gate. Demotes current Production → archived, promotes target `run_id` → Production, sends Slack notification.

---

### Files Deleted vs master

| Deleted File | Reason |
|---|---|
| `model-pipeline/src/bias/detector.py` | Replaced by `src/validation/bias_check.py` (no Fairlearn dependency) |
| `model-pipeline/src/bias/mitigation.py` | Not needed for current bias gate approach |
| `model-pipeline/src/bias/nri_loader.py` | NRI no longer used for bias slicing |
| `model-pipeline/src/bias/report.py` | Bias reporting consolidated into bias_check.py |
| `model-pipeline/src/data/smap_cleaner.py` | SMAP data replaced by Open-Meteo soil moisture |
| `model-pipeline/src/models/registry.py` | Replaced by `vertex_registry.py` |
| `Data-Pipeline/scripts/fusion/priority_resolver.py` | Replaced by simplified `fuse_features.py` |
| `Data-Pipeline/scripts/ingestion/ingest_field_telemetry.py` | Replaced by `download_static.py` |
| `Data-Pipeline/scripts/ingestion/ingest_ndvi.py` | NDVI now from static parquet, not live MODIS |

---

## 5. dev-sco — Monitoring, Deployment, OBJ-3 Dashboard

**Branch owner:** Scott  
**Diverged from:** master; **merged dev_ack on 2026-04-03** (contains all dev_ack work)  
**19 unique commits** (2026-03-31 → 2026-04-09)  
**128 files changed: +44,210 / -2,710**

### What dev-sco adds on top of dev_ack

dev-sco contains everything in dev_ack plus: a live monitoring stack, the OBJ-3 operator dashboard UI, a pipeline bridge layer, an operator rerun engine, GCE cloud deployment scripts, and expanded CI/CD.

---

### New Directories

| Directory | What was added |
|---|---|
| `model-pipeline/src/monitoring/` | PSI-based feature drift + prediction distribution monitoring |
| `model-pipeline/dashboard/` | OBJ-3 web dashboard (HTML/CSS) |
| `model-pipeline/src/pipeline/` (expanded) | `bridge.py`, `rerun_engine.py` added to existing `orchestrator.py` |
| `Data-Pipeline/scripts/` (expanded) | `fire_monitor.py`, `fire_monitor_api.py`, `generate_fake_telemetry.py` |
| `Data-Pipeline/cloud/` (expanded) | `deploy_gce_test.sh`, `gce_startup.sh` |

---

### New/Rewritten Files

#### `model-pipeline/src/monitoring/drift_detector.py` (~160 lines, new)

PSI-based feature drift detection:
- Computes Population Stability Index per feature against a GCS-stored baseline
- **Thresholds:** PSI < 0.1 = stable, 0.1–0.25 = warning, > 0.25 = critical (triggers Slack + optional auto-retrain)
- Reads latest inference parquets from GCS; computes per-feature PSI in one pass

#### `model-pipeline/src/monitoring/performance_monitor.py` (~129 lines, new)

Prediction score distribution drift (proxy for model decay since ground-truth labels arrive weeks later):
- Tracks mean prediction score shift vs baseline stats
- Tracks critical-rate ratio (fraction of CRITICAL predictions vs baseline)
- Returns structured `PerformanceReport` dataclass

#### `model-pipeline/src/monitoring/monitor_runner.py` (~254 lines, new)

Orchestrates the full monitoring cycle:
1. Resolves latest GCS baseline + inference outputs
2. Runs `DriftDetector` + `PerformanceMonitor`
3. Sends Slack alerts on WARNING or CRITICAL findings
4. Optionally triggers GitHub Actions `workflow_dispatch` for auto-retraining
5. **Cooldown:** 24-hour default to prevent retrain storms

**Config:** `model-pipeline/configs/monitoring_config.yaml` — PSI thresholds, 6-hour check interval, GCS paths, auto-retrain toggle (off by default)

#### `model-pipeline/src/pipeline/bridge.py` (~268 lines, new)

Formal integration layer connecting OBJ-1 → OBJ-2 → OBJ-3:
- `build_pipeline_result(obj1_result, obj2_result)` — assembles the `pipeline_result` dict that OBJ-3 context builder expects
- `derive_risk_level(max_prob)` — maps max ignition probability to `LOW/MODERATE/HIGH/CRITICAL`
- `extract_telemetry(df)` — handles both pipeline column names (`temperature_2m`, `wind_speed_10m`) and legacy ERA5 names (`t2m`, `u10`) with unit conversions (°C→°F, km/h→mph)

#### `model-pipeline/src/pipeline/rerun_engine.py` (~180 lines, new)

Operator override re-scoring:
- Fire commanders can replace API-sourced weather with their own on-the-ground measurements (`temperature_f`, `wind_speed_mph`, `relative_humidity`, `soil_moisture`, `fire_weather_index`)
- Applies unit conversions, loads XGBoost or LightGBM model from disk, returns updated DataFrame with updated binary predictions and probabilities
- Only the overridden grid cell is re-scored; all others unchanged

#### `model-pipeline/dashboard/` (new, 3 files)

| File | Lines | Purpose |
|---|---|---|
| `index.html` | 333 | Report list dashboard. 4-card summary strip (total reports, critical incidents, high-risk, last run). Reports table with risk-level badge + confidence bar. Modal viewer for full report text. Dark theme, Inter + JetBrains Mono fonts. |
| `generate.html` | 628 | Report generation form. Grid layout: left = incident fields (location, type, coords, description, file upload); right = risk selector (LOW/MODERATE/HIGH/CRITICAL radio cards). |
| `static/style.css` | 212 | Shared CSS variables: `--critical`, `--high`, `--moderate`, `--low`, card/surface/shadow tokens. |

**FastAPI server** `src/api/server.py` rewritten (+218 lines): serves dashboard HTML, handles `POST /api/generate` (multipart form + file uploads), `GET /api/reports`, `GET /api/reports/{id}`, `GET /api/report-file`, `GET /api/status`.

#### `Data-Pipeline/scripts/fire_monitor.py` (~572 lines, new)

Local continuous monitoring loop — runs the full production stack without Airflow/Docker:

| Mode | Resolution | Interval | Trigger |
|---|---|---|---|
| quiet | 64 km | 30 min | No hotspots |
| active | 22 km | 15 min | FIRMS FRP > 50 MW |
| emergency | 22 km | 5 min | FRP > 100 MW or count > 5 |

Auto-escalation and de-escalation logic. Full end-to-end: FIRMS → Weather → Fuse → OBJ-1 → OBJ-2 → OBJ-3 report. User mode override via API or `.mode_override.json`.

#### `Data-Pipeline/scripts/fire_monitor_api.py` (~163 lines, new)

FastAPI app running as background thread inside `fire_monitor.py`:

| Endpoint | Purpose |
|---|---|
| `GET /` | HTML status dashboard |
| `GET /status` | JSON status (mode, cycle count, fire cells detected, next cycle) |
| `POST /mode` | Override monitoring mode |
| `POST /false-alarm` | Trigger de-escalation |
| `POST /field-telemetry` | Submit drone/firefighter observation |

#### `Data-Pipeline/scripts/monitor_dashboard.html` (~260 lines, new)

Dark-themed real-time status dashboard. 4-card status bar, mode control buttons, false-alarm button, field telemetry form, cycle history table with OK/FAIL/SKIP badges. Auto-refreshes by polling `/status` every few seconds.

#### `Data-Pipeline/scripts/generate_fake_telemetry.py` (~190 lines, new)

CLI tool for generating synthetic drone/firefighter/ICS-209 observations for testing. Supports `--source`, `--scenario` (spreading/contained/false_alarm), lat/lon override. Uses realistic CA fire zone coordinates.

#### `Data-Pipeline/cloud/deploy_gce_test.sh` (~306 lines, new)

Provisions ephemeral GCE VM for testing:
- `e2-standard-8`, 50 GB PD-SSD, Debian 12
- 96-hour auto-stop via GCE Resource Policy (estimated cost ~$25.75)
- Uploads pipeline tarball + `.env` to GCS
- Creates firewall rule for Airflow UI (TCP 8080)
- Polls health marker for up to 15 minutes

#### `Data-Pipeline/cloud/gce_startup.sh` (~188 lines, new)

VM startup script:
- Installs Docker Engine on Debian 12
- Downloads pipeline from GCS
- Starts Airflow via `docker compose`
- Writes health marker on success
- Handles reboots gracefully (skips reinstall if Docker already present)

---

### OBJ-3 Schema Refactor

| Change | Details |
|---|---|
| `schemas/__init__.py` | Now exports central `SCHEMA_MAP = {"daily": ..., "high_risk": ..., "incident": ..., "final": ...}` |
| `base_schema.py` | Extracted as shared base with all shared Pydantic types |
| `high_risk_schema.py` | New schema for `HIGH/CRITICAL` risk reports |
| `state_machine.py` | Added `AdminToggle`, `EmergencySubState`, `IncidentTracker`, `mode_to_report_type()`, `resolve_mode()` |
| `context_builder.py` | Added `HumanInput` dataclass (operator text + file uploads), `ContextBundle.uploaded_files` |

---

### New Tests

| Test File | Lines | Coverage |
|---|---|---|
| `tests/test_monitoring.py` | 119 | PSI computation, DriftDetector (no drift / critical / missing features), PerformanceMonitor (mean shift / critical rate) |
| `tests/test_bridge.py` | 214 | `derive_risk_level`, `extract_telemetry` (both column naming conventions), `build_pipeline_result` integration |
| `tests/test_rerun_engine.py` | 96 | Unit conversion correctness (F→C, mph→km/h), override only touches correct grid cell, unknown fields skipped |
| `tests/test_pipeline_integration.py` | 227 | Full OBJ-1 → OBJ-2 → OBJ-3 integration test |

---

### Data Pipeline Changes (vs master)

| File | Change |
|---|---|
| `dags/wildfire_dag.py` | `DEFAULT_RESOLUTION_KM` changed 64 → 22; `task_ingest_goes()` + `task_ingest_hrrr()` + `task_ingest_field_telemetry()` added; watchdog skip optimization for non-fire regions |
| `Data-Pipeline/scripts/ingestion/download_static.py` | New (+301 lines) — downloads static terrain features, replaces deleted ingest_ndvi.py + ingest_field_telemetry.py |
| `Data-Pipeline/scripts/validation/validate_schema.py` | +184-line net update |
| `Data-Pipeline/dvc/processed_64km.dvc` | New — DVC tracking for 64km processed dataset |

---

## 6. ibrahim_dev — OBJ-2 Fire Spread Simulator

**Branch owner:** Ibrahim  
**Diverged from:** master; **merged dev-sco on 2026-04-10** (contains all dev_ack + dev-sco work)  
**9 unique commits** (2026-03-31 → 2026-04-10)  
**129 files changed: +48,205 / -2,085**

### What ibrahim_dev adds on top of dev-sco

ibrahim_dev is the OBJ-2 specialist branch. It contains everything in dev-sco (= everything) plus: a full pure-Python Rothermel fire spread simulator, expanded fire spread evaluation framework, LANDFIRE CBH/CBD data ingestion, FIRMS temporal validation, GeoJSON export, multi-fire physics validation tests, portable path fixes, and real fire case studies (Eaton Fire, Dixie Fire).

---

### Unique Commits

| Commit | Date | Description |
|---|---|---|
| `83db2a7` | 2026-04-10 | Merge dev-sco into ibrahim_dev, preserve OBJ-2 hybrid simulator |
| `5ac9851` | 2026-04-03 | Add `data/` folder for cell2fire static lookup files |
| `25ad216` | 2026-04-03 | Replace all hardcoded absolute paths with portable relative paths |
| `92bfe34` | 2026-04-02 | Fix LANDFIRE ingestion and populate CBH/CBD in static cache |
| `6baa61d` | 2026-04-01 | Update READMEs for LANDFIRE CBH/CBD and Rothermel simulator additions |
| `063e141` | 2026-04-01 | Add pure-Python Rothermel fire spread simulator with physics fixes |
| `3705260` | 2026-03-31 | Bug fix |
| `cac7862` | 2026-03-31 | Deployment data pipeline modification |
| `b70d994` | 2026-03-31 | Deployment data pipeline |

---

### New/Rewritten Files — OBJ-2

#### `model-pipeline/src/models/obj2_spread/fire_spread_simulator.py` (1,411 lines, new)

The primary OBJ-2 deliverable. A pure-Python Rothermel fire spread simulator — no C++ binary required.

**Physics implemented (7 parts):**

| Part | Source | Details |
|---|---|---|
| FBFM40 fuel parameters | Scott & Burgan (2005) RMRS-GTR-153 | 40 fuel models hard-coded in imperial units (lb/ft², ft, 1/ft). Non-burnable: {91,92,93,98,99} → zero spread |
| Dead fuel moisture | Nelson/Simard EMC piecewise | `_estimate_dfmc(rh_pct, temp_c, days_since_precip)` → 1-hr DFMC; `_estimate_fmc(temp_c, vpd_kpa)` → foliar moisture (0.60–1.50) |
| Rothermel (1972) ROS | Rothermel 1972 | Full 11-step: bulk density → packing ratio → moisture damping → mineral damping → net fuel load → reaction velocity → reaction intensity → propagating flux → heating number → heat of preignition → wind+slope → ROS. Andrews (2012) WAF=0.4 (10 m → midflame) |
| Byram (1959) fireline intensity | Byram 1959 | `I_B = H × w_c × R / 60` (H=18,600 kJ/kg) → kW/m |
| Crown fire | Van Wagner (1977) + Scott & Reinhardt (2001) | `I_0 = (0.010 × CBH × (460 + 25.9 × FMC))^1.5`; active crown threshold `R_0 = 3.0 / CBD`; status: surface/passive_crown/active_crown |
| Elliptical fire shape | Anderson (1983) | LB ratio `0.936×exp(0.2566U) + 0.461×exp(-0.1548U) - 0.397` (clamped 1–8); Prometheus/FARSITE-style cos/sin directional interpolation |

**Three simulation modes:**

| Method | Description |
|---|---|
| `simulate()` | Deterministic single run. Outputs per-neighbour: `spread_direction_deg`, `spread_speed_kmh`, `dead_fuel_moisture_pct`, `crown_fire_status`, `byram_intensity_kwm`, `dominant_factor` |
| `simulate_monte_carlo(n_simulations=100, horizon_hours=24)` | N=100 perturbed-weather runs. Perturbations: wind speed (log-normal σ=0.25), wind direction (wrapped-normal σ=25°), RH (normal σ=8%), temp (normal σ=2.5°C). Outputs per-neighbour burn probabilities, p50/p90/p95 speed distribution |
| `simulate_hybrid(det_weight=0.4)` | Blends deterministic (40%) + MC mean (60%); sigmoid ramp for det score; risk levels: `LOW/MEDIUM/HIGH/EXTREME` |

All internal calculations in imperial units; SI only at input/output boundary.

#### `model-pipeline/src/models/obj2_spread/eval_metrics.py` (538 lines, new)

Physics gate evaluation:
- Per-output sanity checks: spread direction validity, speed plausibility, DFMC range, Byram intensity threshold, crown fire status validity
- `GroundTruth` dataclass for historical fire validation
- `compute_physics_gate(result, ground_truth)` — passes/fails based on direction ±30°, speed range, moisture range, crown fire status match

#### `model-pipeline/src/models/obj2_spread/spread_metrics.py` (366 lines, new)

Honest neighbor-level accuracy metrics (no synthetic inflated accuracy):
- `analyze_threatened_cells()` — identifies cells reachable in N hours
- `compute_propagation_honesty()` — measures how well spread direction/speed predicts observed burn area
- `compute_input_quality()` — assesses completeness of input features (CBH/CBD availability, weather coverage)

#### `model-pipeline/src/models/obj2_spread/geojson_export.py` (328 lines, new)

Converts `PythonFireSpreadSimulator` output to GeoJSON `FeatureCollection`:
- Feature status: `ignition / burned / threatened / non_burnable`
- Properties per cell: bearing, Byram intensity, crown fire status, burn probability (MC mode)
- CRS: EPSG:4326 output

#### `model-pipeline/src/models/obj2_spread/firms_validator.py` (350 lines, new)

FIRMS temporal anti-overfitting validation:
- Extracts observed spread events from backfill parquets
- Strict temporal split: `train_end=2024-06-30`, `val_end=2024-12-31`, `test = 2025+`
- Validates that simulator predictions on test fires (Jan 2025 LA fires) were not seen during any calibration

---

### Real Fire Case Studies

#### `model-pipeline/configs/simulations/eaton_fire_config.json` (new)

Cell2Fire configuration for Eaton Fire (Jan 7, 2025):
- AOI: `[-118.15, 34.12, -117.95, 34.28]` (Altadena/Eaton Canyon)
- 50 simulations, 500 m cells, 24 hr weather duration
- Single ignition at `[34.189, -118.103]`

#### `model-pipeline/configs/simulations/eaton_weather.csv` (new)

Hourly weather for Eaton Fire Santa Ana event: RH 5–13%, wind gusts up to 43.5 km/h.

#### `model-pipeline/run_obj2_eaton.py` (107 lines, new)

End-to-end Cell2Fire test on Eaton Fire. Tests: slope/aspect from DEM, wind gust substitution for Santa Ana events, CRS auto-reproject, lat/lon ignition auto-conversion.

#### `model-pipeline/run_obj2_dixie.py` (360 lines, new)

End-to-end test of `PythonFireSpreadSimulator` on Dixie Fire (2021-07-13, Feather River Canyon). Tests both deterministic and Monte Carlo N=100 modes. Outputs JSON to `reports/simulations/`.

---

### Physics Validation Tests

| Test File | Lines | Fire | Key Assertions |
|---|---|---|---|
| `test_physics_all_fires.py` | 278 | Palisades, Camp, Creek, Thomas, Carr | Direction ±30°, speed plausible range, DFMC ≤ 0.12, crown fire expected, Byram > threshold |
| `test_physics_campfire.py` | 167 | Camp Fire (Nov 2018) | Direction 165–225°, speed 4–15 km/h, moisture 3–10%, crown fire confirmed, Byram > 2000 kW/m |
| `test_physics_palisades.py` | 154 | Palisades Fire (Jan 2025) | Santa Ana conditions: low RH, high wind, rapid spread |

---

### LANDFIRE CBH/CBD Additions

#### `Data-Pipeline/scripts/ingestion/ingest_landfire.py` (heavily modified, 279 lines)

New LANDFIRE layers added (optional, graceful degradation if missing):

| Layer | Product | Processing |
|---|---|---|
| CBH | `LF2020_CBH_200_CONUS` | Raw ÷ 10 = `canopy_base_height_m` (source in tenths of meters) |
| CBD | `LF2024_CBD_240_CONUS` | Raw ÷ 100 = `canopy_bulk_density` (source in kg per 100 m³) |
| EVT-CNC | `LF2020_EVT_200_CONUS` | `evt_national_class` (fuel moisture proxy) |

All layers: EPSG:5070 (30 m) → WGS84 at 0.01° via `rasterio.warp.reproject`; categorical layers use `Resampling.nearest`, continuous (CBH, CBD, CC) use `Resampling.bilinear`. Nodata sentinels `{-9999, 0, 32767, 32768, 65535}` excluded.

#### `Data-Pipeline/configs/schema_config.yaml` additions

Three new feature definitions:

| Feature | Type | Bounds | Simulator Use |
|---|---|---|---|
| `canopy_base_height_m` | float32 | 0–40 m | Crown fire initiation (`I_0` in Van Wagner 1977) |
| `canopy_bulk_density` | float32 | 0–0.45 kg/m³ | Crown fire spread rate (`R_0 = 3.0 / CBD`) |
| `evt_national_class` | float64 | — | Fuel moisture proxy |

#### `model-pipeline/src/models/obj2_spread/data/` (new directory)

Placeholder for `spain_lookup_table.csv` required by Cell2Fire C++ wrapper. File not committed (too large); `README.md` documents how to obtain it.

#### Portable path fixes (`25ad216`)

All hardcoded absolute paths (e.g., `/home/ibrahim/C2F-W/...`) replaced with:
- `Path(__file__).resolve().parents[N] / "relative/path"` patterns
- `${CELL2FIRE_BINARY:-cell2fire}` env var override in `model_config.yaml`
- Affects: `cell2fire_spread.py`, `raster.py`, `ingest_landfire.py`, `run_obj2_*.py` scripts

---

### Updated OBJ-2 Tests

| Test File | Changes |
|---|---|
| `tests/obj2/test_cell2fire_spread.py` | Major rewrite: mocked tests for `load_model`, `predict`, `validate`, `explain`; covers binary failure, config loading, param override, prediction shape/binariness |
| `tests/obj2/test_raster.py` | `clip_raster_to_aoi` (smaller output, empty clip error, creates subdirs); `parse_burn_probability` (probability range, shape, burned patch, zero sims) |
| `tests/obj2/test_weather.py` | `format_weather_csv` accepts both pipeline and Cell2Fire column names; `validate_weather_df` warns on nulls and extreme wind |
| `tests/obj2/test_evaluation.py` | Minor update to Dice + IoU + threshold sweep tests |

---

## 7. Feature Ownership Matrix

| Feature | master | dev_ack | dev-sco | ibrahim_dev |
|---|:---:|:---:|:---:|:---:|
| **Data Pipeline (Airflow + DVC)** | ✅ | ✅ | ✅ | ✅ |
| **OBJ-1: XGBoost (basic, no tuning)** | ✅ | — | — | — |
| **OBJ-1: XGBoost (tuned, RandomizedSearchCV)** | — | ✅ | ✅ | ✅ |
| **OBJ-1: LightGBM (secondary)** | — | ✅ | ✅ | ✅ |
| **Feature engineering module (shared transforms)** | — | ✅ | ✅ | ✅ |
| **Temporal train/test split (Jan 2025 LA fires)** | — | ✅ | ✅ | ✅ |
| **Decision threshold tuning (≥ 90% recall)** | — | ✅ | ✅ | ✅ |
| **SHAP explainability** | — | ✅ | ✅ | ✅ |
| **Training visualizations (PR curve, CM, comparison)** | partial | ✅ | ✅ | ✅ |
| **Bias gate (FNR disparity, 3 slices)** | Fairlearn | sklearn | sklearn | sklearn |
| **MLflow tracking (experiment runs)** | basic | ✅ full | ✅ full | ✅ full |
| **Vertex AI Model Registry (push/load/rollback)** | — | ✅ | ✅ | ✅ |
| **Vertex AI Experiments sync** | — | ✅ | ✅ | ✅ |
| **Slack alerting** | basic | ✅ full | ✅ full | ✅ full |
| **6-hour inference loop (Open-Meteo → GCS)** | — | ✅ | ✅ | ✅ |
| **CI/CD: 8-stage model pipeline** | basic | ✅ | ✅ | ✅ |
| **CI/CD: manual rollback workflow** | — | ✅ | ✅ | ✅ |
| **PSI drift detection** | — | — | ✅ | ✅ |
| **Prediction distribution monitoring** | — | — | ✅ | ✅ |
| **Auto-retrain on drift (GitHub Actions)** | — | — | ✅ | ✅ |
| **OBJ-1 → OBJ-2 → OBJ-3 bridge layer** | — | — | ✅ | ✅ |
| **Operator rerun engine** | — | — | ✅ | ✅ |
| **OBJ-3 dashboard UI (index + generate)** | — | — | ✅ | ✅ |
| **OBJ-3 schema consolidation (SCHEMA_MAP)** | — | — | ✅ | ✅ |
| **Local fire monitor loop** | — | — | ✅ | ✅ |
| **Monitor API + dashboard** | — | — | ✅ | ✅ |
| **GCE ephemeral deployment scripts** | — | — | ✅ | ✅ |
| **OBJ-2: Cell2Fire C++ wrapper (refactored)** | ✅ | ✅ | ✅ | ✅ |
| **OBJ-2: Pure-Python Rothermel simulator** | — | — | — | ✅ |
| **OBJ-2: Monte Carlo spread simulation** | — | — | — | ✅ |
| **OBJ-2: Hybrid (deterministic + MC) mode** | — | — | — | ✅ |
| **OBJ-2: Crown fire (Van Wagner 1977)** | — | — | — | ✅ |
| **OBJ-2: Elliptical fire shape (Anderson 1983)** | — | — | — | ✅ |
| **OBJ-2: Physics validation tests (real fires)** | — | — | — | ✅ |
| **OBJ-2: FIRMS temporal anti-overfitting** | — | — | — | ✅ |
| **OBJ-2: GeoJSON export** | — | — | — | ✅ |
| **OBJ-2: Evaluation gate (eval_metrics.py)** | — | — | — | ✅ |
| **LANDFIRE CBH/CBD ingestion** | — | — | — | ✅ |
| **Portable relative paths** | — | — | — | ✅ |
| **Eaton Fire / Dixie Fire case studies** | — | — | — | ✅ |
| **OBJ-3: LLM adapters (Ollama/Gemini/Vertex)** | ✅ | ✅ | ✅ | ✅ |
| **OBJ-3: State machine (QUIET/ACTIVE/EMERGENCY)** | ✅ | ✅ | ✅ | ✅ |
| **OBJ-3: 5 report schemas** | ✅ | ✅ | ✅ | ✅ |

---

## 8. File-Level Diff Inventory

### Files added by dev_ack (not in master)

```
model-pipeline/
  configs/model_config.yaml                    ← substantially rewritten
  historical_data/california_historical.csv    ← 13,318-row training data
  historical_data/texas_historical.csv         ← 15,469-row training data
  experimentation/california.ipynb             ← experiment notebook
  scripts/train.py                             ← 187 lines, training CLI
  scripts/inference.py                         ← 484 lines, 6-hour scoring
  scripts/combine_historical_data.py           ← 139 lines, parquet → CSV
  scripts/drop_2026_data.py                    ← 55 lines, label cleanup
  scripts/upload_historical_to_gcs.py          ← 87 lines, GCS upload
  src/preprocessing/feature_engineering.py    ← 405 lines, canonical transforms
  src/models/obj1_xgboost/model.py             ← rewritten (239 lines)
  src/models/obj1_lightgbm/__init__.py         ← new
  src/models/obj1_lightgbm/model.py            ← 239 lines
  src/pipeline/orchestrator.py                 ← 590 lines (heavily rewritten)
  src/tracking/vertex_registry.py             ← 270 lines, Vertex AI registry
  src/tracking/vertex_sync.py                 ← Vertex Experiments sync
  src/tracking/mlflow_logger.py               ← updated (bias, SHAP, threshold logging)
  src/validation/bias_check.py                ← 168 lines, FNR bias gate
  src/validation/model_selector.py            ← updated
  src/data/loader.py                          ← 108 lines
  Dockerfile                                  ← model pipeline container
  pyproject.toml                              ← Python project config
  .python-version                             ← pins Python 3.11
  OVERVIEW.md                                 ← 297-line overview doc
  MLOPS_DESIGN.md                             ← 138-line design doc
  model_pipeline_summary.md                   ← 340-line professor summary doc
  mlflow.db                                   ← SQLite tracking DB (local dev artifact)
  .cache.sqlite                               ← Open-Meteo request cache

.github/workflows/
  model_ci.yml                                ← rewritten (8-stage pipeline)
  model_rollback.yml                          ← new (manual rollback)

Data-Pipeline/
  dags/wildfire_dag.py                        ← +283 lines (GOES, HRRR tasks)
  scripts/ingestion/download_static.py        ← +301 lines (static terrain download)
  scripts/validation/validate_schema.py       ← +184-line update
  dvc/processed_64km.dvc                      ← DVC tracking file
```

**Files deleted by dev_ack:**
```
model-pipeline/src/bias/detector.py
model-pipeline/src/bias/mitigation.py
model-pipeline/src/bias/nri_loader.py
model-pipeline/src/bias/report.py
model-pipeline/src/data/smap_cleaner.py
model-pipeline/src/models/registry.py
Data-Pipeline/scripts/fusion/priority_resolver.py
Data-Pipeline/scripts/ingestion/ingest_field_telemetry.py
Data-Pipeline/scripts/ingestion/ingest_ndvi.py
```

---

### Files added by dev-sco (beyond dev_ack)

```
model-pipeline/
  src/monitoring/__init__.py
  src/monitoring/drift_detector.py            ← 160 lines, PSI-based drift
  src/monitoring/performance_monitor.py       ← 129 lines, score distribution
  src/monitoring/monitor_runner.py            ← 254 lines, monitoring orchestrator
  configs/monitoring_config.yaml             ← drift check config
  src/pipeline/bridge.py                      ← 268 lines, OBJ-1→2→3 bridge
  src/pipeline/rerun_engine.py               ← 180 lines, operator overrides
  dashboard/index.html                        ← 333 lines, report list dashboard
  dashboard/generate.html                     ← 628 lines, report generation form
  dashboard/static/style.css                  ← 212 lines, dark theme CSS
  src/api/server.py                           ← rewritten (dashboard serving)
  src/models/obj3_gemini/schemas/base_schema.py    ← extracted shared Pydantic base
  src/models/obj3_gemini/schemas/high_risk_schema.py ← new schema
  tests/test_monitoring.py                    ← 119 lines
  tests/test_bridge.py                        ← 214 lines
  tests/test_rerun_engine.py                  ← 96 lines
  tests/test_pipeline_integration.py         ← 227 lines

Data-Pipeline/
  scripts/fire_monitor.py                     ← 572 lines, local monitor loop
  scripts/fire_monitor_api.py                 ← 163 lines, control API
  scripts/monitor_dashboard.html             ← 260 lines, real-time dashboard
  scripts/generate_fake_telemetry.py          ← 190 lines, test data generator
  cloud/deploy_gce_test.sh                   ← 306 lines, GCE provisioning
  cloud/gce_startup.sh                       ← 188 lines, VM startup
  dags/watchdog_sensor_dag.py                ← updated (GCS sensor)
```

---

### Files added by ibrahim_dev (beyond dev-sco)

```
model-pipeline/
  src/models/obj2_spread/fire_spread_simulator.py   ← 1,411 lines, Rothermel physics
  src/models/obj2_spread/eval_metrics.py            ← 538 lines, physics gate
  src/models/obj2_spread/spread_metrics.py          ← 366 lines, honest metrics
  src/models/obj2_spread/geojson_export.py          ← 328 lines, GeoJSON output
  src/models/obj2_spread/firms_validator.py         ← 350 lines, temporal validation
  src/models/obj2_spread/data/.gitkeep
  src/models/obj2_spread/data/README.md             ← spain_lookup_table instructions
  configs/simulations/eaton_fire_config.json        ← Eaton Fire Cell2Fire config
  configs/simulations/eaton_weather.csv             ← Eaton Fire weather data
  run_obj2_eaton.py                                 ← 107 lines, Eaton case study
  run_obj2_dixie.py                                 ← 360 lines, Dixie case study
  evaluate_obj2.py                                  ← 1,553 lines, full eval framework
  test_physics_all_fires.py                         ← 278 lines, 5-fire validation
  test_physics_campfire.py                          ← 167 lines, Camp Fire physics
  test_physics_palisades.py                         ← 154 lines, Palisades physics

Data-Pipeline/
  scripts/ingestion/ingest_landfire.py              ← updated (CBH/CBD/EVT-CNC layers)
  configs/schema_config.yaml                        ← 3 new feature definitions
```

---

## 9. Key Conflicts and Integration Notes

### 1. `model_config.yaml` divergence (resolved in master)

The `obj2.cell2fire.binary_path` was hardcoded to `/home/ibrahim/C2F-W/Cell2Fire/Cell2Fire` in ibrahim_dev. Commit `3bbfed0` on master resolved this conflict keeping the relative-path version. Commit `25ad216` in ibrahim_dev subsequently replaced all hardcoded absolute paths with portable env-var-based alternatives.

### 2. Bias module replacement (dev_ack → all branches)

dev_ack deleted `src/bias/` (Fairlearn-based, 4 files) and replaced it with `src/validation/bias_check.py` (pandas+sklearn only). This is a breaking change — anything importing from `src.bias.*` will fail. All branches that include dev_ack work (dev-sco, ibrahim_dev) carry this change.

### 3. `ingest_ndvi.py` deletion (dev_ack → all branches)

dev_ack deleted `ingest_ndvi.py` (568 lines, MODIS NDVI live fetching). NDVI is now served from static parquet. Any DAG task that called `ingest_ndvi` must be updated to use the static download path.

### 4. `model_ci.yml` conflict risk

Both dev_ack and dev-sco made large independent changes to `.github/workflows/model_ci.yml`. dev-sco merges dev_ack, so the conflict should be resolved within dev-sco. However, merging either branch back to master will require a careful 3-way merge of this file.

### 5. OBJ-2 `__init__.py` exports differ across branches

- **master:** exports `Cell2FireSpread`, `Cell2FireError`, `Cell2FireNotInstalledError`
- **ibrahim_dev:** adds `PythonFireSpreadSimulator` as the primary export; `Cell2FireSpread` becomes legacy

Any code doing `from src.models.obj2_spread import ...` needs updating when merging ibrahim_dev.

### 6. `dags/wildfire_dag.py` `DEFAULT_RESOLUTION_KM` conflict

- **master/dev_ack:** `DEFAULT_RESOLUTION_KM = 64`
- **dev-sco/ibrahim_dev:** `DEFAULT_RESOLUTION_KM = 22`

This affects default data granularity in all DAG runs. Merging will require an intentional choice.

### 7. Merge order for final integration

Recommended order to minimize conflicts:
1. `master ← dev_ack` (OBJ-1 model pipeline — clean addition of model-pipeline)
2. `result ← dev-sco` (monitoring + dashboard — extends dev_ack cleanly)
3. `result ← ibrahim_dev` (OBJ-2 simulator — adds new files; `model_config.yaml` and `__init__.py` need manual review)
