# Wildfire Detection & Response MLOps Platform — Project Summary

A production-grade wildfire risk intelligence platform combining ignition prediction, fire spread simulation, and LLM-generated disaster reports, orchestrated through a fully-automated MLOps pipeline on GCP.

---

## 1. Problem Statement

Wildfires cause billions in losses annually and require rapid, data-driven decision-making by incident commanders. Existing tools are siloed — weather dashboards, fire-detection feeds, and report-writing are separate systems. This project unifies them into a single operator console powered by three ML objectives and an automated data pipeline.

**Target users**: Emergency operations centers, incident commanders, fire-service planners.

**Scope**: California + Texas, H3 grid at resolution 2 (~64 km cells, ~55 cells total), 30-minute update cadence.

---

## 2. The Three ML Objectives

### OBJ-1 — Ignition Probability (XGBoost, per region)
- **Task**: Binary classification — "will this H3 cell ignite in the next observation window?"
- **Model**: Region-specific XGBoost classifiers (`wildfire-ignition-california`, `wildfire-ignition-texas`)
- **Features**: Weather (temperature, RH, wind, soil moisture, VPD), terrain, fuel model (FBFM40), FIRMS aggregates (train only), GOES
- **Validation**: PR-AUC ≥ 0.89 (CA), ≥ 0.78 (TX); F1 ≥ 0.5; threshold 0.365 (XGB) for ≥90% recall
- **Risk tiers** (region-tuned): CA CRITICAL ≥ 0.80 / HIGH ≥ 0.50; TX CRITICAL ≥ 0.75 / HIGH ≥ 0.45
- **Leakage control**: 8 pipeline-only columns (FIRMS aggregates, fire_detected_binary, canopy_base_height_m, canopy_bulk_density, evt_national_class) dropped before inference
- **Explainability**: SHAP TreeExplainer, 500 background samples

### OBJ-2 — Fire Spread Simulation (Monte Carlo)
- **Model**: `PythonFireSpreadSimulator` — Rothermel surface spread + Byram fireline intensity + Van Wagner crown initiation
- **Method**: 100 Monte Carlo runs per cell; mean/p50/p90/p95/max spread_speed_kmh, dominant direction + uncertainty, crown fire probability
- **Output**: GCS `simulation/latest/{region}_latest.json`

### OBJ-3 — Disaster Report Generator (LLM + RAG)
- **LLM**: Vertex AI Gemini 2.5 Flash (primary), Gemini Dev + Ollama qwen3:8b fallback
- **Schema**: Pydantic-validated — IncidentReport / DailyReport / HighRiskReport / FinalReport
- **Mode state machine**: QUIET / ACTIVE / EMERGENCY from `risk_level` × `firms_hotspot_count` × `is_deployable`
- **Grounding**: RAG corpus (FEMA NRI + Scott-Burgan) in Vertex AI context cache
- **Incident tracker**: Persists EMERGENCY sub-states (ACTIVE_FIRE → INTERIM → POST_FIRE → FINAL) with 30-day GC
- **Confidence gate**: `report_confidence ≥ 0.70`, `min_grounding_sources ≥ 3` → else `human_review_required`

---

## 3. Data Pipeline (Airflow DAG: `wildfire_data_pipeline`)

**Schedule**: Every 30 min, triggered externally via Cloud Scheduler → Cloud Function → Airflow REST API. DAG's `schedule_interval=None`.

**21 tasks**, region-sharded:
```
load_static_layers / ingest_field_telemetry
├─ region_california: ingest_firms→process_firms | ingest_weather→process_weather | ingest_hrrr | ingest_goes
├─ region_texas    : same structure
└─ fuse_features → validate_schema → detect_anomalies → export_to_parquet → version_with_dvc → trigger_model_server
```

**Sources**: NASA FIRMS, Open-Meteo (primary weather), NWS (fallback), NOAA HRRR, GOES-16/18, FEMA NRI, LANDFIRE.

**Resilience**:
- Circuit breaker on weather null rate > 80% (prevents forward-fill poisoning model)
- 2-tier fallback: Open-Meteo → NWS → forward-fill
- Airflow pool (`open_meteo_pool = 1`) serializes calls to avoid 429s
- 4 retries with exponential backoff; Slack alert at 3 consecutive failures

---

## 4. Model Pipeline

### Training
- `full_pipeline` — single entrypoint for train + inference; returns `(X, fitted_medians)`
- MLflow tracks metrics/params/SHAP; Vertex AI Model Registry for artifact versioning
- **Promotion**: `registry.push(..., labels={env:production})` auto-demotes prior production
- **Rollback**: `registry.rollback()` one-liner

### Bias gate
- **Metric**: false negative rate
- **Slices**: region, fire_season, fuel_model_fbfm40
- **Threshold**: disparity > 0.15 blocks deployment
- **Min**: 20 samples, 5 fire events per slice

### Inference (Cloud Run FastAPI)
`POST /api/generate-from-pipeline`:
1. OBJ-1: read fused parquet → drop 8 pipeline-only cols → load Vertex AI prod model → model-type-aware predict → write JSON + parquet to GCS
2. OBJ-2: Monte Carlo on top-risk cell → `simulation/latest/{r}_latest.json`
3. OBJ-3: assemble pipeline_result → Gemini → render HTML → `reports/obj3/{region}/{type}_{ts}.json`

---

## 5. GCP Deployment

```
Cloud Scheduler (*/30 * * * *)
  ↓
Cloud Function dag-trigger (Gen 2, Python 3.11)
  ↓ POST /dagRuns
GCE: wildfire-test-vm (e2-standard-8, Docker Compose: postgres + webserver + scheduler)

Cloud Run: wildfire-inference (FastAPI, 4Gi/2CPU, amd64)
Cloud Run: wildfire-frontend (React + nginx, /api/* proxied to backend)
Vertex AI Model Registry
GCS wildfire-mlops-123: data/processed/fused | inference/latest | simulation/latest | reports/obj3 | model-artifacts
```

**CI/CD**: `model_ci.yml` on push to master → Lint → Tests → Build+Push → Train+Validate+Bias+Registry → **Deploy to Cloud Run** → Scheduler update.

**Cost**: ~$55/mo (GCE $48 + Cloud Run ~$5 + GCS ~$2).

---

## 6. Frontend (React + Vite + Tailwind)

- `useAPI` hook: SWR-style with auto-stop after 2 failures
- Pages: Overview, Data Pipeline, OBJ-1/2/3, Fire Map (Mapbox + H3), Risk Monitor, Incident Reports
- Viewer mode (`?mode=viewer`): read-only map + reports
- 60s polling on Overview + RiskMonitor critical-cell counts

---

## 7. Key Engineering Decisions

| Decision | Why |
|---|---|
| H3 res 2 (~64km) | Fits free tier of weather APIs; ~55 cells |
| Per-region models | Different climates + region-specific thresholds |
| Drop FIRMS features at inference | Prevents target leakage |
| `schedule_interval=None` + Cloud Scheduler | Single trigger source, avoids dual-firing |
| GCS as source of truth | Cloud Run ephemeral — local disk wipes on restart |
| Lazy-import lightgbm | xgboost is prod; avoid 50MB image bloat |
| Circuit breaker on weather | Silent degradation → loud abort |
| Vertex AI context cache for corpus | 40K chars × every request blows token budget |
| Pydantic schema on LLM output | Rejects malformed JSON, triggers retry |
| YAML-persisted incident state | Multi-hour incidents survive container restart |
| Docker on GCE vs Composer | Course budget — $48 vs $400/mo |

---

## 8. Interview Talking Points

### "Tell me about a challenging bug"
The `scheduled__` prefix collision: Cloud Scheduler fired every 30 min but all triggers got HTTP 400. Traced via Cloud Function logs — Airflow reserves the `scheduled__` prefix. Changed to `cloudscheduler__`, runs now succeeded and showed `type=manual` in the Airflow API.

### "How do you prevent data leakage?"
Hardcoded list of 8 pipeline-only columns dropped before inference. `full_pipeline` raises `ValueError: Leakage columns present` if any slip through. Caught a real regression when I refactored the DAG→server inference chain.

### "How do you handle drift?"
- Data drift: null-rate tracking + circuit breaker on >80%
- Concept drift: SHAP importance tracked per run; Slack alert on >0.05 drop
- Bias drift: bias gate re-evaluated on retrain; deploy blocked if FNR disparity > 0.15

### "Design for failure"
- Weather: 2-tier fallback + circuit breaker
- LLM: Vertex AI → Gemini Dev → Ollama fallback; Pydantic + parse retry
- State: incident tracker in YAML, GCS as source of truth
- Trigger: single cron path; DAG's internal schedule disabled

### "End-to-end walkthrough"
Every 30 min: Cloud Scheduler → Cloud Function → Airflow REST API → 21-task DAG → Cloud Run inference (OBJ-1 XGBoost → OBJ-2 Monte Carlo → OBJ-3 Gemini). All outputs → GCS. React polls every 60s. Model promotion gated by PR-AUC + bias + human approval.

### "Tool choices"
- **Airflow** over Prefect/Dagster: industry standard, rich UI, task-level retries
- **XGBoost** over LightGBM: better PR-AUC on CA (0.9426 vs 0.9374)
- **Vertex AI Gemini** over OpenAI: GCP-native service-account auth
- **H3 over S2**: hexagons have uniform 6-neighbor topology (critical for fire spread sim)
- **DVC over Git-LFS**: first-class GCS remote

### "What would you change with more time?"
- Cloud Composer when budget allows
- Online feature store (Feast on Redis) for sub-second inference
- Evidently.ai dashboards beyond SHAP drift
- Active learning from operator PATCH edits → labeled dataset for LLM fine-tuning
- Higher H3 resolution (level 5, ~8.5km) with paid weather API

---

## 9. Key Metrics

- **Model**: PR-AUC 0.94 (CA), 0.88 (TX); precision ≥ 0.70, recall ≥ 0.90
- **Throughput**: 55 cells × 2 regions × 48 runs/day ≈ 5.3K inferences/day
- **LLM cost**: <$0.01/report with cached corpus
- **Infra cost**: ~$55/mo (GCE $48 + Cloud Run ~$5 + GCS ~$2)
