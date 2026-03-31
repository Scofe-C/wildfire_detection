# Wildfire Detection MLOps Platform

A production-grade MLOps platform for wildfire detection and disaster response in California and Texas, built for Northeastern University's MLOps course (February 2026).

---

## What It Does

The system monitors active wildfires using multi-source satellite and weather data, runs ML models to classify fire risk and simulate spread, and generates automated disaster response reports using LLMs. It operates continuously, escalating from quiet background scans to real-time emergency mode when a fire is confirmed.

---

## System Architecture

The platform has two main layers:

```
┌─────────────────────────────────────────────────────────────┐
│ GCP CLOUD LAYER (fire_watchdog Cloud Function)              │
│                                                             │
│  Cloud Scheduler (every 15–30 min)                         │
│    → Poll FIRMS/GOES for candidate fires                    │
│    → 4-gate false alarm filter                              │
│    → On confirmation: write trigger to GCS                  │
└───────────────────────────┬─────────────────────────────────┘
                            ↓ (within 2 min)
┌───────────────────────────┴─────────────────────────────────┐
│ LOCAL AIRFLOW LAYER (Data + Model Pipeline)                 │
│                                                             │
│  wildfire_data_pipeline DAG (every 6h)                     │
│    → Ingest → Process → Fuse → Validate → Export           │
│                                                             │
│  model-pipeline orchestrator                               │
│    → Load data → Train/Predict → Gate → Deploy             │
└─────────────────────────────────────────────────────────────┘
```

The project is split into two subdirectories:

| Directory | Role |
|-----------|------|
| `Data-Pipeline/` | Data ingestion, feature engineering, Airflow orchestration |
| `model-pipeline/` | ML model training, validation, bias gating, deployment |

---

## Data Sources

| Source | Type | Frequency | Role |
|--------|------|-----------|------|
| NASA FIRMS (VIIRS, MODIS) | Active fire detections | ~3h latency | Primary fire label |
| GOES-R ABI FDC | Geostationary fire pixels | Every 10 min | Watchdog quick-check |
| Open-Meteo | Hourly weather | Hourly | Primary weather |
| NWS API | Forecast weather | Hourly | Weather fallback |
| NOAA HRRR | Rapid-refresh weather | 15 min | Emergency/active mode |
| LANDFIRE 2022 | Fuel model (FBFM40), canopy | Static (cached) | Fuel features |
| USGS SRTM 30m | Elevation, slope, aspect | Static | Terrain features |

All features are fused onto an **H3 hexagonal grid** (22 km resolution by default, escalating to finer resolution on fire confirmation). The full feature set is defined as a single source of truth in `configs/schema_config.yaml` — 28 features across fire, weather, terrain, and fuel categories.

---

## Data Pipeline (`Data-Pipeline/`)

The Airflow DAG runs on 6-hour UTC boundaries and processes California and Texas in parallel via TaskGroups.

### Stages

```
INGEST
  ├─ FIRMS: Active fire detections (24h lookback)
  ├─ Weather: Open-Meteo → NWS fallback → HRRR (emergency)
  └─ Static: LANDFIRE/SRTM (cached)
        ↓
PROCESS
  ├─ FIRMS: Spatial join to H3, FRP clipping, fire feature aggregation
  ├─ Weather: 6h rolling aggregation, derived indices (VPD, drought proxy, wind run)
  └─ Static: Zonal statistics over H3 cells
        ↓
FUSE
  • Left-join all sources onto master H3 grid
  • Priority hierarchy: ground_truth > satellite > model
  • Gap-fill: forward-fill, NWS fallback, HRRR substitution
  • Data quality flags (0–5) per cell
        ↓
VALIDATE
  • Great Expectations schema checks (28 features, null rates, value ranges)
  • Seasonal z-score anomaly detection (Welford online updates)
  • Slack alert on anomalies
        ↓
EXPORT
  • Parquet: partitioned by region/year/month
  • Spatial: H3 grid arrays (.npz) + adjacency matrix
        ↓
VERSION
  • DVC tracks Parquet blobs in GCS
  • Git tracks code, configs, .dvc lock files
```

### Data Quality Flags

| Flag | Meaning |
|------|---------|
| 0 | All sources present |
| 1 | Weather gap-filled via NWS |
| 2 | Weather forward-filled from previous window |
| 3 | HRRR substituted for Open-Meteo |
| 4 | FIRMS absent (fire features set to 0) |
| 5 | Multiple sources missing — excluded from training |

---

## Real-Time Monitoring (`cloud/fire_watchdog/`)

A GCP Cloud Function polls FIRMS and GOES-R every 15–30 minutes. Before triggering the main pipeline, it applies a **4-gate false alarm filter**:

1. **Spatial clustering** — ≥3 detections within 50 km
2. **Temporal persistence** — ≥2 consecutive GOES windows
3. **VIIRS cross-reference** — MODIS-only detections require VIIRS confirmation
4. **Industrial exclusion** — 2 km buffer from known industrial sources

On confirmation, it writes a trigger JSON to GCS. The local `watchdog_sensor_dag` polls GCS every 60 seconds and triggers the main DAG with escalated parameters (finer grid, HRRR ingestion).

### Operating Modes

| Mode | Grid Resolution | Trigger |
|------|----------------|---------|
| Quiet | 64 km | Scheduled (6h) |
| Active | 22 km | Watchdog fire candidate |
| Emergency | 22 km + HRRR | FRP > 500 MW confirmed |

---

## ML Models (`model-pipeline/`)

### OBJ-1 — XGBoost Fire Occurrence Classifier

Classifies each H3 cell as fire / no-fire given current conditions.

- **Input**: 7 core features — temperature, relative humidity, wind speed/direction, precipitation, lagged fire detection, lagged active fire count
- **Output**: Binary prediction + fire probability [0, 1]
- **Hyperparameters**: `max_depth=6`, `n_estimators=100`, `scale_pos_weight` auto-computed for class imbalance
- **Validation gate**: AUC-PR > 0.75 required before deployment

### OBJ-2 — Cell2Fire Physics-Based Spread Simulator

Wraps the Cell2Fire C++ simulator to model how a confirmed fire spreads over terrain.

- **Input**: DEM (elevation), fuel model (FBFM40), canopy data, weather CSV, ignition point coordinates
- **Output**: Burn probability grid per timestep
- **Method**: 100 Monte Carlo simulations at 30 m cell resolution, 1-hour timesteps
- **Validation**: Dice coefficient ≥ 0.50 vs. historical CAL FIRE perimeters
- **Config**: Parameter sweep over `n_simulations` (50–500), `fire_period` (0.5–4h), grid resolution (30–90 m)

### OBJ-3 — Gemini LLM Disaster Reporting

Generates structured incident reports using a state-machine-driven LLM orchestrator.

- **Backends**: Ollama (local) → Gemini Developer API → Vertex AI (swappable without code change)
- **Report types**: `incident_brief`, `tactical_operations`, `strategic_impact`, `lessons_learned`
- **Context**: Fire metadata, incident history, operational mode, pre-computed disaster knowledge corpus
- **Output**: JSON + Markdown + HTML (Jinja2 rendered)
- **State machine**: quiet → active → emergency (with admin overrides)
- **Validation**: Schema validity, section completeness, confidence scoring

---

## Training & Deployment Pipeline

```
1. dvc pull               # Get latest versioned Parquet backfill from GCS
2. Bias analysis gate     # 4-dimension slicing: geography, fuel tier, season, data quality
                          #   → KL divergence must be < 0.1 nats
                          #   → Fire rate disparity must be < 5%
                          #   → DVC enforces bias_report.json before training proceeds
3. Train OBJ-1 (XGBoost)  # MLflow experiment tracking; artifacts saved to models/ignition/
4. Validate               # AUC-PR > 0.75 gate
5. Bias gate (post-train) # Fairlearn FNR disparity < 5% (FEMA NRI vulnerability stratification)
6. Train OBJ-2 (Cell2Fire) parameter sweep
7. Build OBJ-3 (Gemini) corpus embeddings
8. Push to GCS model registry
9. Slack notification     # PASS / FAIL with root cause analysis
```

Model rollback is automatic if any validation or bias gate fails.

---

## Fairness & Bias Detection

Bias is checked at two points:

- **Pre-training** (data level): 4-dimensional slicing across geography, fuel tier, season, and data quality. KL divergence and fire rate disparity are computed per slice.
- **Pre-deployment** (model level): Fairlearn-based false negative rate (FNR) disparity check, stratified by FEMA National Risk Index (NRI) vulnerability zones. Disparity must be < 5%.

DVC enforces both gates — training cannot proceed if the pre-training bias report fails, and deployment cannot proceed if the model-level FNR gate fails.

---

## Versioning & Reproducibility

| Tool | What It Tracks |
|------|---------------|
| DVC | Parquet data blobs in GCS (immutable, content-hashed) |
| Git | Code, configs, `.dvc` lock files |
| MLflow / Vertex AI | Model hyperparameters, metrics, artifacts per run |

To reproduce any historical state:
```bash
git checkout <commit>
dvc pull          # Restores exact data state for that commit
dvc repro         # Re-runs pipeline from that state
```

---

## Configuration Files

| File | Purpose |
|------|---------|
| `Data-Pipeline/configs/schema_config.yaml` | Single source of truth: 28 features, H3 resolution maps, region bboxes, source URLs, data quality flags |
| `model-pipeline/configs/feature_schema.yaml` | Data contract: column dtypes, bounds, required flags |
| `model-pipeline/configs/model_config.yaml` | Validation thresholds (AUC-PR, FNR), MLflow/GCS paths, Slack config, OBJ-2/OBJ-3 params |
| `model-pipeline/configs/reporting_config.yaml` | LLM backends, corpus paths, report schemas, Jinja2 templates |
| `Data-Pipeline/dvc.yaml` | Data pipeline stages + dependencies |
| `model-pipeline/dvc.yaml` | Model pipeline stages: `validate_model`, `bias_gate` |
| `Data-Pipeline/docker-compose.yaml` | Local Airflow + Postgres + Redis dev environment |

### Required Environment Variables

```bash
FIRMS_MAP_KEY        # NASA FIRMS API key
GCS_BUCKET_NAME      # GCS bucket (e.g. wildfire-mlops-dev)
GCP_KEY_PATH         # Path to GCP service account JSON
SLACK_WEBHOOK_URL    # Optional: Slack alerts
GOOGLE_CLOUD_PROJECT # GCP project ID
```

---

## Testing & CI/CD

The test suite has 200+ pytest tests covering ingestion, processing, fusion, validation, bias analysis, export, utilities, DAG structure, and end-to-end integration (with mocked APIs).

GitHub Actions runs on every push:
1. DAG parse-time import validation
2. Full pytest suite inside Docker
3. `ruff` linting (rules E and F, zero tolerance)
4. Dependency pin check (`pyarrow` constraint)
5. Docker layer build with cache

---

## Directory Structure

```
wildfire_detection/
├── Data-Pipeline/
│   ├── dags/                    # Airflow DAGs
│   │   ├── wildfire_dag.py      # Main 6-hourly pipeline DAG
│   │   └── watchdog_sensor_dag.py  # GCS trigger sensor
│   ├── scripts/
│   │   ├── ingestion/           # FIRMS, weather, GOES, HRRR, field telemetry
│   │   ├── processing/          # Per-source aggregation and cleaning
│   │   ├── fusion/              # Feature joining onto H3 grid
│   │   ├── validation/          # Schema, anomaly detection, bias analysis
│   │   ├── export/              # Parquet + spatial .npz output
│   │   └── utils/               # H3 grid, rate limiting, GCS state, schema loading
│   ├── cloud/
│   │   └── fire_watchdog/       # GCP Cloud Function (real-time monitor)
│   ├── tests/                   # 200+ pytest tests
│   ├── configs/
│   │   └── schema_config.yaml   # 28-feature schema (single source of truth)
│   ├── docker/                  # Multi-stage Dockerfile
│   ├── docker-compose.yaml
│   └── dvc.yaml
│
└── model-pipeline/
    ├── src/
    │   ├── data/                # Parquet loader + schema validation
    │   ├── models/
    │   │   ├── obj1_xgboost/    # Fire occurrence classifier
    │   │   ├── obj2_spread/     # Cell2Fire C++ wrapper
    │   │   └── obj3_gemini/     # LLM disaster report orchestrator
    │   ├── validation/          # AUC-PR, F1, FNR metrics
    │   ├── bias/                # Fairlearn FNR disparity gate
    │   ├── tracking/            # MLflow + Vertex AI
    │   ├── notifications/       # Slack webhook alerts
    │   └── pipeline/            # Orchestrator (wires all stages)
    ├── configs/
    │   ├── feature_schema.yaml
    │   ├── model_config.yaml
    │   └── reporting_config.yaml
    ├── models/ignition/         # Trained model artifacts (DVC-tracked)
    ├── reports/                 # Validation + bias reports
    └── dvc.yaml
```
