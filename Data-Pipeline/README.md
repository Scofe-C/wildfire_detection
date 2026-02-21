# 🔥 Wildfire Detection & Response System — Data Pipeline

> **MLOps Course · Phase 1 Deliverable · February 2026**
> 
> An end-to-end data pipeline that ingests wildfire and environmental data from six public sources, fuses them into a unified H3-indexed feature table, and exports validated, versioned Parquet files ready for model training — with an adaptive GCP-based fire detection watchdog that escalates resolution and polling frequency automatically when a fire is confirmed.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Adaptive Watchdog](#adaptive-watchdog)
- [False Alarm Prevention](#false-alarm-prevention)
- [Feature Schema](#feature-schema)
- [Project Structure](#project-structure)
- [GCP Setup](#gcp-setup)
- [Running Tests](#running-tests)
- [Team Ownership](#team-ownership)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

---

## Overview

The pipeline fuses **six public data sources** into a single Parquet-based feature table indexed by [H3 hexagonal grid cells](https://h3geo.org/), covering California and Texas.

| Source | Data Type | Frequency | Role |
|---|---|---|---|
| NASA FIRMS (VIIRS + MODIS) | Active fire detections | Near real-time | Primary fire label |
| GOES-R ABI FDC (GOES NRT) | Geostationary fire pixels | Every 10 min | Watchdog quick-check |
| Open-Meteo | Hourly weather (8 variables) | Hourly | Weather features |
| NWS API | Forecast weather | Hourly | Weather fallback |
| LANDFIRE | Fuel model, canopy, vegetation | Static (2022) | Fuel features |
| USGS SRTM | 30m DEM elevation | Static | Terrain features |

**Key design decisions:**
- 🗺️ **Regional sharding** — California and Texas process in parallel via Airflow TaskGroups
- 📐 **Adaptive resolution** — coarse 64 km scans by default, escalates to 22 km on confirmed fire
- 🛡️ **Four-gate false alarm prevention** — spatial + temporal + VIIRS cross-reference + industrial exclusion
- 🔔 **GCP watchdog** — Cloud Function polls GOES NRT every 15–30 min and triggers the pipeline within ~20 min of a real fire
- 📦 **DVC versioning** — all processed data tracked in GCS with region/year/month partitioning

---

## Architecture

### Two-Layer Design

```
╔══════════════════════════════════════════════════════════════════════╗
║  GCP LAYER                                                           ║
║                                                                      ║
║  Cloud Scheduler ──15–30 min──► fire_watchdog (Cloud Function)       ║
║                                        │                             ║
║                                Poll FIRMS GOES_NRT                   ║
║                                (CA bbox + TX bbox)                   ║
║                                        │                             ║
║                           ┌────────────┴────────────┐               ║
║                      False alarm               Confirmed fire        ║
║                           │                         │               ║
║                    Write FA record          Write trigger JSON       ║
║                    Revert in 30 min    gs://{bucket}/watchdog/       ║
║                                             triggers/               ║
╚══════════════════════════════════════════════════════════════════════╝
                                             │
                                             ▼ (within 2 min)
╔══════════════════════════════════════════════════════════════════════╗
║  LOCAL DOCKER AIRFLOW                                                ║
║                                                                      ║
║  watchdog_sensor_dag  ── polls GCS every 60s ──► trigger_dag()       ║
║                                                                      ║
║  wildfire_data_pipeline:                                             ║
║                                                                      ║
║  ┌─ [region_california] ──────────────────────────────────────┐     ║
║  │  ingest_firms_ca   →  process_firms_ca                     │     ║
║  │  ingest_weather_ca →  process_weather_ca                   │──┐  ║
║  └─────────────────────────────────────────────────────────────┘  │  ║
║  ┌─ [region_texas] ───────────────────────────────────────────┐  ├─►║
║  │  ingest_firms_tx   →  process_firms_tx                     │  │  ║
║  │  ingest_weather_tx →  process_weather_tx                   │──┘  ║
║  └─────────────────────────────────────────────────────────────┘     ║
║  check_static ──► load_static_layers (cached per resolution) ────────┘
║                                                                      ║
║       fuse_features  (left-join from master grid — all cells kept)   ║
║            │                                                         ║
║       validate_schema  (Great Expectations, dynamic from config)     ║
║            │                                                         ║
║       detect_anomalies  (seasonal baseline z-score, JSON storage)    ║
║            │                                                         ║
║       export_to_parquet                                              ║
║         data/processed/22km/region=california/year=2026/month=02/   ║
║         data/processed/22km/region=texas/year=2026/month=02/        ║
║            │                                                         ║
║       version_with_dvc  (dvc add + dvc push → GCS)                  ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Adaptive Scheduling Modes

| Mode | Poll Interval | Pipeline Run | Resolution | Trigger |
|---|---|---|---|---|
| `quiet` | Every 30 min | Every 6 hours | 64 km | Off-season default |
| `active` | Every 15 min | Every 2 hours | 64 km | Fire season (Jun–Nov) |
| `emergency` | Every 5 min | Every 30 min | 22 km | FRP > 200 MW + expanding footprint |

---

## Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (WSL2 backend on Windows)
- Python 3.11
- NASA FIRMS API key — [register free here](https://firms.modaps.eosdis.nasa.gov/api/)
- GCP service account key JSON — provided by Zhengxin

### Setup

```bash
# 1. Clone
git clone <repo-url>
cd wildfire-pipeline-merged

# 2. Configure environment
cp .env.example .env
# Edit .env — fill in FIRMS_MAP_KEY, GCS_BUCKET_NAME, GCP_KEY_PATH

# 3. Start Airflow
docker compose up -d --build

# 4. Open http://localhost:8080
#    Username: airflow  |  Password: airflow
```

### DAGs to Unpause

> ⚠️ **Unpause BOTH DAGs** — the pipeline will not respond to fire alerts unless `watchdog_sensor_dag` is running.

| DAG | Purpose | Schedule |
|---|---|---|
| `wildfire_data_pipeline` | Main pipeline | Every 6 hrs (fallback cron) |
| `watchdog_sensor_dag` | Polls GCS for fire triggers | Every 2 min, **always-on** |
| `wildfire_local_test` | Local testing — no API keys | Manual trigger only |

### Local Test (no API keys needed)

```bash
# Generate synthetic seed data
python scripts/seed_local_test.py

# Then trigger 'wildfire_local_test' manually in the Airflow UI
# Runs the full pipeline end-to-end using fake but schema-valid data
```

---

## Adaptive Watchdog

### How It Works

The `fire_watchdog` Cloud Function runs on Google Cloud Scheduler and acts as a lightweight fire sentinel — far cheaper and faster than running the full Airflow pipeline every 15 minutes.

```
Every 15–30 min:
  1. Poll FIRMS GOES_NRT API for CA + TX bounding boxes (2 API calls)
  2. Filter to detections in last 60 min with FRP ≥ 10 MW
  3. Run four-gate false alarm check (see below)
  4. If confirmed → write trigger JSON to GCS
  5. Local Airflow sensor finds it within 2 min → triggers full pipeline
  6. Update watchdog state in GCS (race-safe conditional write)
```

### GCS State Structure

```
gs://{bucket}/watchdog/
├── state/current.json          ← Live mode, fire cells, consecutive scan counts
├── triggers/                   ← Trigger files read by Airflow sensor
├── false_alarms/               ← Audit trail (every false positive logged)
├── emergency/                  ← Emergency activation/deactivation log
└── config/
    ├── schema_config.yaml      ← Runtime config (watchdog reads this at startup)
    └── industrial_sources.json ← 15 CA/TX refineries — updateable without redeploy
```

### Fire Detection Range

At H3 resolution 5 (~5.1 km edge length):

| Ring | Approximate Distance | Purpose |
|---|---|---|
| 0 | Fire cell itself | Confirmed fire location |
| 1 | ~5 km | Spatial corroboration check (G1) |
| 3 | ~15 km | Mid-range detection zone |
| 5 | ~25 km | Outer detection boundary |

---

## False Alarm Prevention

Every GOES-R detection passes four sequential gates. **All four must pass** to trigger the pipeline.

```
Detection detected
       │
  ┌────▼────┐
  │   G1    │  Spatial: ≥2 neighbouring cells also fire in same scan?
  │         │  → FAIL: isolated pixel (sun glint, sensor artifact) → discard
  └────┬────┘
       │ PASS
  ┌────▼────┐
  │   G2    │  Temporal: detection in ≥2 consecutive GOES scans (~20 min)?
  │         │  → FAIL: transient → wait 1 more scan
  └────┬────┘
       │ PASS
  ┌────▼────┐
  │   G3    │  Multi-source: VIIRS SNPP or NOAA20 confirms within 3 hours?
  │         │  → BYPASS if FRP > 50 MW (large fire, VIIRS hasn't passed yet)
  │         │  → FAIL: no VIIRS + low FRP → discard
  └────┬────┘
       │ PASS
  ┌────▼────┐
  │   G4    │  Industrial: cell NOT within radius of known heat source?
  │         │  (15 CA/TX refineries loaded from GCS — updateable without redeploy)
  │         │  → FAIL: hard block regardless of FRP
  └────┬────┘
       │ ALL PASS
  ✅ CONFIRMED — write GCS trigger → Airflow pipeline triggered
```

**False alarm revert:** 30-minute timer after any failure — automatically reverts polling back to prior mode.

---

## Feature Schema

All features are defined in `configs/schema_config.yaml` — the single source of truth. Never hardcode feature names; always read from `FeatureRegistry` via `schema_loader.py`.

### Feature Groups (28 total)

| Group | Features | Source |
|---|---|---|
| `identifiers` | grid_id, region, timestamp, resolution_km, lat, lon | computed |
| `weather` | temperature_2m, relative_humidity_2m, wind_speed_10m, wind_direction_10m, precipitation, soil_moisture, vpd, fire_weather_index | Open-Meteo / NWS |
| `vegetation` | fuel_model_fbfm40, canopy_cover_pct, vegetation_type, ndvi | LANDFIRE / MODIS |
| `topography` | elevation_m, slope_degrees, aspect_degrees | USGS SRTM |
| `fire_context` | active_fire_count, mean_frp, median_frp, max_confidence, nearest_fire_distance_km, fire_detected_binary | NASA FIRMS |
| `derived` | days_since_last_precipitation, cumulative_wind_run_24h, drought_index_proxy | computed |
| `metadata` | data_quality_flag, dominant_fuel_fraction | computed |

### Data Quality Flag

| Flag | Meaning |
|---|---|
| `0` | All sources fresh — Open-Meteo primary, all features present |
| `1` | Open-Meteo primary used (explicit lineage) |
| `2` | NWS fallback used (Open-Meteo unavailable) |
| `3` | HRRR rapid weather used (future streaming phase) |
| `4` | Weather interpolated / forward-filled (gap > 1 window) |
| `5` | Partial data — >30% features missing |

### Regional Feature Priority

Features in `schema_config.yaml` carry `regional_priority` metadata consumed by the model training phase:

| Feature | California | Texas | Rationale |
|---|---|---|---|
| `slope_degrees` | critical | standard | CA canyon/forest fires; TX is flat |
| `ndvi` | standard | critical | TX grass fire detection and fuel curing state |
| `wind_direction_10m` | critical | standard | Santa Ana / Diablo wind events |
| `days_since_last_precipitation` | standard | critical | TX drought-driven fires |
| `drought_index_proxy` | standard | critical | TX soil moisture deficit |

---

## Project Structure

```
wildfire-pipeline-merged/
│
├── .github/
│   └── workflows/ci.yml          # GitHub Actions CI
│
├── cloud/
│   ├── deploy.sh                 # One-command GCP deployment
│   └── fire_watchdog/
│       ├── main.py               # Cloud Function entry point
│       └── requirements.txt      # Minimal CF deps (6 packages, fast cold start)
│
├── configs/
│   └── schema_config.yaml        # ← SINGLE SOURCE OF TRUTH
│                                 #   features, validation rules, watchdog config,
│                                 #   resolution tiers, anomaly thresholds,
│                                 #   training sufficiency thresholds
│
├── dags/
│   ├── wildfire_dag.py           # Production DAG (CA + TX TaskGroups)
│   ├── watchdog_sensor_dag.py    # GCS trigger poller (always-on, reschedule mode)
│   └── wildfire_local_test_dag.py
│
├── scripts/
│   ├── detection/
│   │   ├── fire_detector.py      # Four-gate false alarm prevention
│   │   └── emergency.py          # Emergency state machine
│   │
│   ├── fusion/
│   │   ├── fuse_features.py      # Left-join from master grid (all cells preserved)
│   │   └── priority_resolver.py  # Priority Hierarchy Engine (ground truth > satellite)
│   │
│   ├── ingestion/
│   │   ├── ingest_firms.py       # NASA FIRMS (VIIRS + MODIS, per-region)
│   │   ├── ingest_goes.py        # GOES NRT quick-check + S3 direct access
│   │   ├── ingest_weather.py     # Open-Meteo + NWS fallback
│   │   └── ingest_field_telemetry.py  # Field telemetry schema validation [NEW]
│   │
│   ├── processing/
│   │   ├── process_firms.py      # Spatial join, FRP clipping, MODIS confidence norm
│   │   ├── process_static.py     # LANDFIRE + SRTM download → H3 zonal statistics
│   │   └── process_weather.py    # 6h aggregation + derived features:
│   │                             #   days_since_precipitation
│   │                             #   cumulative_wind_run_24h
│   │                             #   drought_index_proxy
│   │
│   ├── utils/
│   │   ├── gcs_state.py          # GCS watchdog state I/O (race-safe conditional writes)
│   │   ├── grid_utils.py         # H3 grid, h3 v3/v4 compat, spatial pruning, focal grid
│   │   ├── rate_limiter.py       # FIRMS + weather API rate limiting
│   │   └── schema_loader.py      # FeatureRegistry from schema_config.yaml
│   │
│   ├── export/
│   │   └── export_spatial.py     # Track B: spatial grid + adjacency matrix (.npz)
│   │
│   └── validation/
│       ├── detect_anomalies.py   # Seasonal baseline z-score (JSON files, Welford update)
│       └── validate_schema.py    # Great Expectations (dynamic from schema)
│
├── tests/
│   ├── test_export/              # Dual-track (tabular + spatial) consistency
│   ├── test_fusion/              # Fusion left-join completeness, priority resolver,
│   │                             #   temporal lag, weather fallback
│   ├── test_ingestion/           # FIRMS: API timeout, malformed CSV, multi-resolution;
│   │                             #   SLA alerting, retry logic, field telemetry
│   ├── test_processing/          # Weather (7 scenarios) + static layer + caching
│   ├── test_utils/               # Grid: cell counts, reprojection, circular mean aspect,
│   │                             #        h3 compat, focal grid, spatial pruning
│   └── test_validation/          # Anomaly: baseline storage, Welford, per-season files,
│                                 #           fire/off-season threshold verification
│
├── docker-compose.yaml           # Multi-stage, multi-service Airflow setup
├── docker/Dockerfile             # airflow-base + test target for CI
├── requirements.txt              # Unpinned (works with Airflow constraint resolver)
├── constraints.txt               # Hard pins — pyarrow MUST match across all machines
├── environment.yml               # Conda environment for macOS teammates
└── .env.example                  # All required variables with documentation
```

---

## GCP Setup

### DVC Remote (one-time per developer)

```bash
dvc remote add -d gcs_remote gs://<bucket>/dvc-store
dvc remote modify gcs_remote credentialpath gcp-key.json

# Pull existing team data
dvc pull

# Push after a pipeline run
dvc push
```

### Cloud Function Deployment (Zhengxin only)

```bash
chmod +x cloud/deploy.sh
./cloud/deploy.sh

# Verify deployment
gcloud scheduler jobs run watchdog-quiet --location=us-central1

# Watch for trigger files
gcloud storage ls gs://<bucket>/watchdog/triggers/

# Check live watchdog state
gcloud storage cat gs://<bucket>/watchdog/state/current.json
```

> **Cost estimate: ~$0/month** — Cloud Functions (2M free invocations/month) + Cloud Scheduler (3 free jobs/month).

### Update Industrial Exclusion List

No redeployment needed — just upload a new JSON to GCS:

```bash
# Edit the file, then:
gcloud storage cp industrial_sources.json \
  gs://<bucket>/watchdog/config/industrial_sources.json
```

---

## Running Tests

```bash
# Local (requires pip install -r requirements.txt -c constraints.txt)
export PYTHONPATH=.
pytest tests/ -v --tb=short

# Inside Docker (recommended — matches CI exactly)
docker run --rm \
  -e GCS_BUCKET_NAME=test-bucket \
  -e FIRMS_MAP_KEY=test-key \
  wildfire-pipeline:test \
  pytest tests/ -v --tb=short

# With coverage
pytest tests/ -v --cov=scripts --cov-report=term-missing
```

### CI Pipeline

GitHub Actions runs on every push/PR to `main` or `develop`:

1. **Build** Docker image to `test` target (caches layers via GHA cache)
2. **DAG validation** — `python dags/wildfire_dag.py` + `watchdog_sensor_dag.py`
3. **pytest** — full test suite inside Docker
4. **ruff** — lint `scripts/`, `dags/`, `tests/` (E and F rules)
5. **Dependency check** — verifies `pyarrow` is pinned in `constraints.txt`

> `pyarrow` is hard-pinned in `constraints.txt` because Parquet encoding changed between major versions — all team members (Windows 11, Windows 10, macOS) must use the same version to avoid silent schema divergence.

---

### Code Standards

- **Linting:** `ruff check scripts/ dags/ tests/` — CI blocks on E/F failures
- **Docstrings:** Google-style with `Args`, `Returns`, `Raises` on every function
- **Logging:** `logging` only — never `print()`. `INFO` for normal ops, `WARNING` for recoverable issues
- **Type hints:** required on all function signatures
- **Schema:** never hardcode feature names — always read from `schema_config.yaml` via `FeatureRegistry`
- **Branches:** `feature/person-name-description` → PR to `develop` → Zhengxin reviews

---

## Environment Variables

Copy `.env.example` to `.env` — never commit `.env` to Git.

| Variable | Required | Description |
|---|---|---|
| `FIRMS_MAP_KEY` | ✅ Yes | NASA FIRMS API key |
| `GCS_BUCKET_NAME` | ✅ Yes | GCS bucket for DVC + watchdog state |
| `GCP_KEY_PATH` | ✅ Yes | Path to service account JSON (`./gcp-key.json`) |
| `GOOGLE_CLOUD_PROJECT` | GCP deploy only | GCP project ID |
| `GCP_REGION` | GCP deploy only | Cloud Function region (default: `us-central1`) |
| `WATCHDOG_TRIGGER_PREFIX` | Auto-set | Set by `deploy.sh` — do not edit manually |
| `SLACK_WEBHOOK_URL` | Optional | Slack alerts for anomalies and emergencies |

### Windows Notes

- Use forward slashes in `.env` — backslashes break Docker volume mounts
- Increase Docker Desktop memory to **12 GB** before the first run (LANDFIRE static download is 3–5 GB)
- Run `docker compose up -d --build` after replacing files (not just `up`)

---

## Troubleshooting

| Problem | Fix |
|---|---|
| DAG not appearing in Airflow UI | Check for import errors: `docker run wildfire-pipeline:test python3 dags/wildfire_dag.py` |
| `watchdog_sensor_dag` missing | Ensure it's in `dags/` folder and not paused in the UI |
| No fire triggers arriving | Run `gcloud storage ls gs://{bucket}/watchdog/triggers/` and confirm `watchdog_sensor_dag` is unpaused |
| DVC push fails | Run `dvc remote list` — if empty: `dvc remote add -d gcs_remote gs://<bucket>/dvc-store` |
| Static layer download OOM | Increase Docker Desktop memory to 12 GB. Static only downloads once, then caches |
| Tests fail with GCS errors | Pass `-e GCS_BUCKET_NAME=test-bucket` — tests are designed to run without real GCS |
| `docker compose up` build fails | Run `docker compose build --no-cache` first |
| PyCharm shows `{data}` folders | Cosmetic only — disable **Compact Middle Packages** in the Project tree gear icon ⚙ |

---

## License

University course project — for academic and educational use.

---

<div align="center">

Built with [Apache Airflow](https://airflow.apache.org/) · [DVC](https://dvc.org/) · [Great Expectations](https://greatexpectations.io/) · [H3](https://h3geo.org/) · [Google Cloud](https://cloud.google.com/)

</div>
