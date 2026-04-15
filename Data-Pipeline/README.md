# 🔥 Wildfire Detection & Response System — Data Pipeline

> **MLOps Course · Northeastern University · February 2026**
>
> An end-to-end, production-grade data pipeline that ingests wildfire and environmental data from six public sources, fuses them into a unified H3-indexed feature table, validates data quality, detects anomalies, performs bias analysis across geographic and seasonal subgroups, and exports versioned Parquet files ready for ML model training.


![Python](https://img.shields.io/badge/python-3.10%20|%203.11-blue)
![Airflow](https://img.shields.io/badge/Airflow-2.8.1-green)
![DVC](https://img.shields.io/badge/DVC-3.x-orange)
![License](https://img.shields.io/badge/license-Academic-lightgrey)

---

## Table of Contents

1. [What This Pipeline Does](#1-what-this-pipeline-does)
2. [Architecture](#2-architecture)
3. [Quick Start](#3-quick-start)
4. [Project Structure](#4-project-structure)
5. [Data Sources](#5-data-sources)
6. [Static Data Setup (One-Time)](#6-static-data-setup-one-time)
7. [Feature Schema](#7-feature-schema)
8. [Pipeline Orchestration (Airflow DAGs)](#8-pipeline-orchestration-airflow-dags)
9. [Data Versioning with DVC](#9-data-versioning-with-dvc)
10. [Schema Validation & Statistics](#10-schema-validation--statistics)
11. [Anomaly Detection & Alerts](#11-anomaly-detection--alerts)
12. [Data Bias Detection & Mitigation](#12-data-bias-detection--mitigation)
13. [Historical Backfill](#13-historical-backfill)
14. [Pipeline Flow Optimization](#14-pipeline-flow-optimization)
15. [Running Tests](#15-running-tests)
16. [CI/CD Pipeline](#16-cicd-pipeline)
17. [Code Style & Standards](#17-code-style--standards)
18. [GCP Setup](#18-gcp-setup)
19. [Environment Variables](#19-environment-variables)
20. [Reproducibility](#20-reproducibility)
21. [Error Handling & Logging](#21-error-handling--logging)
22. [Adaptive Watchdog](#22-adaptive-watchdog)
23. [Troubleshooting](#23-troubleshooting)

---

## 1. What This Pipeline Does

The pipeline processes wildfire and environmental data across California and Texas through the following stages:

1. **Ingest** satellite fire detections (NASA FIRMS), weather data (Open-Meteo + NWS fallback + NOAA HRRR), terrain features (USGS SRTM), vegetation/fuel data (LANDFIRE), and GOES-R geostationary observations
2. **Process** raw data into H3 hexagonal grid cells, running California and Texas in parallel
3. **Fuse** all feature layers into a unified 28-feature table per grid cell per 6-hour window
4. **Validate** the fused dataset against a dynamic schema defined in `configs/schema_config.yaml`
5. **Detect anomalies** using seasonal z-score baselines and trigger Slack alerts
6. **Analyze bias** by slicing features across geographic regions, land cover classes, and fire seasons
7. **Export** partitioned Parquet files and spatial grid arrays for ML training
8. **Version** all processed data with DVC backed by Google Cloud Storage

---

## 2. Architecture

### DAG Flow

```
check_static ───────────────────────────────────────────┐
                                                        │
[region_california TaskGroup]                           │
  ingest_firms_ca  → process_firms_ca ──────────────────┤
  ingest_weather_ca → process_weather_ca ───────────────┤──────┐
                                                        │      ▼
[region_texas TaskGroup]                                │ fuse_features
  ingest_firms_tx  → process_firms_tx ──────────────────┤      │
  ingest_weather_tx → process_weather_tx ───────────────┘      │
                                                               │
                                                       validate_schema
                                                               │
                                                      detect_anomalies
                                                               │
                                          ┌────────────────────┴────────────────────┐
                                   export_to_parquet                      export_spatial
                                          └────────────────────┬────────────────────┘
                                                       version_with_dvc
```

### Two-Layer Design

```
╔════════════════════════════════════════════════════════════╗
║  GCP LAYER                                                 ║
║                                                            ║
║  Cloud Scheduler ──15–30 min──► fire_watchdog (CF)         ║
║                                     │                      ║
║                             Poll FIRMS / GOES_NRT          ║
║                                     │                      ║
║                      ┌──────────────┴──────────────┐       ║
║                 False alarm                 Confirmed fire ║
║                      │                         │           ║
║               FA record + revert      trigger JSON → GCS   ║
╚════════════════════════════════════════════════════════════╝
                                         │ (within 2 min)
╔════════════════════════════════════════════════════════════╗
║  LOCAL DOCKER / AIRFLOW                                    ║
║                                                            ║
║  watchdog_sensor_dag polls GCS every 60s                   ║
║  → triggers wildfire_data_pipeline with escalated params   ║
╚════════════════════════════════════════════════════════════╝
```

### Adaptive Operating Modes

| Mode | Trigger | Resolution | Poll Interval |
|---|---|---|---|
| `quiet` | Scheduled cron (every 6h) | 64 km (H3 res 2) | — |
| `active` | Watchdog detects candidate fire | 22 km (H3 res 5) | 15 min |
| `emergency` | FRP > 500 MW confirmed | 22 km + HRRR wind | 15 min |

---

## 3. Quick Start

### Prerequisites

- Docker Desktop (≥ 4.x) with **12 GB** RAM allocated
- `git` and `dvc` installed on host
- A NASA FIRMS API key — free at [firms.modaps.eosdis.nasa.gov](https://firms.modaps.eosdis.nasa.gov/api/area/)
- GCS bucket + GCP service account JSON (for DVC remote and watchdog)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Scofe-C/wildfire_detection/data-pipeline.git
cd wildfire-pipeline

# 2. Configure environment variables
cp ../.env.example ../.env
# Edit ../.env: set FIRMS_MAP_KEY, GCS_BUCKET_NAME, GCP_KEY_PATH (all at repo root)

# 3. Configure DVC remote (one-time per developer)
dvc remote add -d gcs_remote gs://<your-bucket>/dvc-store
dvc remote modify gcs_remote credentialpath gcp-key.json

# 4. Pull versioned data
dvc pull

# 5. Start all services (from repo root)
cd .. && make up-full
# Or Airflow only: make up

# 6. Open services:
#    Airflow:   http://localhost:8080  (airflow / airflow)
#    Dashboard: http://localhost:8000
#    Monitor:   http://localhost:8001
#    MLflow:    http://localhost:5000
```

### Unpause DAGs

```bash
docker exec airflow-webserver airflow dags unpause wildfire_data_pipeline
docker exec airflow-webserver airflow dags unpause watchdog_sensor_dag
```

### Trigger a Manual Run

```bash
docker exec airflow-webserver airflow dags trigger wildfire_data_pipeline \
  --conf '{"resolution_km": 22, "trigger_source": "manual"}'
```

---

## 4. Project Structure

```
data-pipeline/
│
├── dags/
│   ├── wildfire_dag.py              # Main pipeline DAG (CA + TX parallel TaskGroups)
│   ├── watchdog_sensor_dag.py       # GCS-polling sensor DAG
│   └── utils/
│       └── slack_notify.py          # Slack alert helpers
│
├── scripts/
│   ├── ingestion/
│   │   ├── ingest_firms.py          # NASA FIRMS (VIIRS + MODIS, per-region)
│   │   ├── ingest_goes.py           # GOES NRT quick-check + S3 direct access
│   │   ├── ingest_weather.py        # Open-Meteo + NWS fallback (rate-limited)
│   │   ├── ingest_hrrr.py           # NOAA HRRR rapid weather (emergency mode)
│   │   ├── ingest_landfire.py       # LANDFIRE 2022 raster → H3 zonal stats (one-time). --resolution-km accepts multiple values; --raw-dir for custom download path
│   │   ├── ingest_srtm.py           # USGS SRTM 30m tiles → elevation/slope/aspect (one-time)
│   │   ├── ingest_ndvi.py           # MODIS MOD13A2 NDVI via AppEEARS GeoTIFF (one-time)
│   │   └── ingest_field_telemetry.py
│   │
│   ├── processing/
│   │   ├── process_firms.py         # Spatial join, FRP clipping, confidence norm
│   │   ├── process_static.py        # Merge LANDFIRE + SRTM + NDVI parquets into unified cache. Full CLI: --resolution-km, --static-dir, --force-rebuild
│   │   └── process_weather.py       # 6h aggregation + Canadian FWI + derived fire weather features
│   │
│   ├── fusion/
│   │   ├── fuse_features.py         # Left-join from master grid (all cells preserved). Includes LANDFIRE spread-simulation stubs: canopy_base_height_m, canopy_bulk_density, evt_national_class
│   │   └── priority_resolver.py     # Ground truth > satellite priority hierarchy
│   │
│   ├── validation/
│   │   ├── validate_schema.py       # Great Expectations validation + stats output
│   │   │                            #   CLI: python scripts/validation/validate_schema.py
│   │   │                            #        --input data/processed/fused
│   │   │                            #        --resolution-km 64
│   │   │                            #        --output-dir data/processed/baselines
│   │   ├── detect_anomalies.py      # Seasonal z-score anomaly detection + Slack alerts
│   │   │                            #   CLI: python scripts/validation/detect_anomalies.py
│   │   │                            #        --input data/processed/fused
│   │   │                            #        --baseline-dir data/processed/baselines
│   │   │                            #        --output data/processed/baselines/anomaly_last_run.json
│   │   └── bias_analysis.py         # Subgroup bias slicing across 4 dimensions
│   │                                #   CLI: python scripts/validation/bias_analysis.py
│   │                                #        --input data/processed/backfill
│   │                                #        --output data/processed/baselines/bias_report.json
│   ├── export/
│   │   └── export_spatial.py        # Spatial grid (H×W×C .npz) + adjacency matrix
│   │                                #   CLI: python scripts/export/export_spatial.py
│   │                                #        --input data/processed/fused
│   │                                #        --output-dir data/processed/spatial
│   │                                #        --resolution-km 64
│   ├── backfill/
│   │   └── historical_backfill.py   # Replay pipeline over any date range (resume-safe)
│   │
│   ├── reporting/
│   │   ├── report_generator.py      # LLM-based pipeline reports (Phase 2)
│   │   └── prompts.py               # Prompt templates for Gemini
│   │
│   └── utils/
│       ├── datetime_utils.py        # Shared UTC datetime coercion
│       ├── export_appeears_points.py# Generate AppEEARS upload CSV/GeoJSON from H3 grid
│       ├── gcs_state.py             # GCS watchdog state I/O (race-safe writes)
│       ├── grid_utils.py            # H3 grid generation, spatial pruning, haversine
│       ├── rate_limiter.py          # Token-bucket rate limiter with jitter
│       ├── run_pipeline_once.py     # Local end-to-end smoke test (no Airflow)
│       └── schema_loader.py         # FeatureRegistry from schema_config.yaml
│
├── tests/
│   ├── conftest.py                  # Shared fixtures
│   ├── test_ingestion/              # FIRMS: API timeout, malformed CSV, retry, SLA
│   ├── test_processing/             # Weather (7 scenarios), static layer, caching
│   ├── test_fusion/                 # Left-join, temporal lag, weather fallback
│   ├── test_validation/
│   │   ├── test_anomalies.py        # Welford, per-season baselines, thresholds
│   │   ├── test_bias_analysis.py    # 38 tests across all 4 slicing dimensions
│   │   └── test_validation.py       # Great Expectations schema checks
│   ├── test_export/                 # Tabular + spatial consistency
│   ├── test_utils/                  # H3 grid, focal grid, spatial pruning
│   ├── test_dags/                   # DAG structure, task count, dependencies
│   ├── test_dvc/                    # DVC stage hashes, lock file integrity
│   ├── test_backfill/               # Historical replay correctness
│   └── test_integration/           # End-to-end pipeline integration
│
├── configs/
│   └── schema_config.yaml           # ← SINGLE SOURCE OF TRUTH for all features
│
├── cloud/
│   ├── fire_watchdog/main.py        # GCP Cloud Function — GOES/FIRMS polling
│   ├── deploy.sh                    # One-command Cloud Function deploy
│   └── gce_startup.sh               # GCE startup script
│
├── docker/Dockerfile                # Multi-stage: airflow-base + test target
├── docker-compose.yaml              # Airflow + Postgres + Redis
├── dvc.yaml                         # DVC pipeline stages (mirrors Airflow DAG)
├── dvc.lock                         # Locked stage hashes — commit after every run
├── .gitignore                       # Data dirs excluded; only .dvc pointer files tracked
├── requirements.txt                 # Unpinned (use with constraints.txt)
├── constraints.txt                  # Hard pins — pyarrow version must match team-wide
├── environment.yml                  # Conda environment for macOS
└── .env.example                     # All required variables with descriptions
```

---

## 5. Data Sources

| Source | Type | Frequency | Role | Status |
|---|---|---|---|---|
| NASA FIRMS (VIIRS SNPP, NOAA-20, MODIS) | Active fire detections | Near real-time (~3h) | Primary fire label | ✅ Live |
| GOES-R ABI FDC | Geostationary fire pixels | Every 10 min | Watchdog quick-check | ✅ Live |
| Open-Meteo | Hourly weather (7 variables) | Hourly | Weather features | ✅ Live |
| NWS API | Forecast weather | Hourly | Weather fallback | ✅ Live |
| NOAA HRRR | Rapid-refresh weather | Hourly | Emergency-mode weather | ✅ Live |
| LANDFIRE 2022 (FBFM40, CC, EVT, CBH, CBD) | Fuel model, canopy cover, vegetation type, canopy base height, canopy bulk density | Static (one-time) | Fuel/veg features + OBJ-2 crown fire inputs | ⚠️ Manual download — see §6a |
| USGS SRTM 30m (OpenTopography) | Elevation, slope, aspect | Static (one-time) | Terrain features | ⚠️ Manual download — see §6b |
| MODIS MOD13A2 v061 (NASA AppEEARS) | 1 km 16-day NDVI composite | Static (seasonal) | Vegetation greenness | ⚠️ Manual download — see §6c |

All data is indexed to an **H3 hexagonal grid** at configurable resolution (default 64 km, escalates to 22 km on watchdog). Static layers are computed once and cached as Parquet — all subsequent pipeline runs load the cache directly. Without static caches, static columns output `NaN` and `data_quality_flag=4/5` but the pipeline still runs.

---

## 6. Static Data Setup (One-Time)

Static layers must be downloaded once. Without them the pipeline still runs — static columns output `NaN` with `data_quality_flag=4` (all static missing) or `5` (partial). Rebuild the cache any time a new source is added.

### 6a. LANDFIRE 2022 — Fuel Model, Canopy Cover, Vegetation Type, Canopy Structure

**Download:** https://landfire.gov/data/FullExtentDownloads → LF 2022 → CONUS

Download and extract these five ZIPs:

| Layer | ZIP | Column produced | Used by |
|---|---|---|---|
| Fuel model (FBFM40) | `LF2022_FBFM40_230_CONUS.zip` | `fuel_model_fbfm40` | OBJ-1, OBJ-2 |
| Canopy cover (CC) | `LF2022_CC_230_CONUS.zip` | `canopy_cover_pct` | OBJ-1 |
| Vegetation type (EVT) | `LF2022_EVT_230_CONUS.zip` | `vegetation_type`, `dominant_fuel_fraction` | OBJ-1 |
| Canopy base height (CBH) | `LF2022_CBH_230_CONUS.zip` | `canopy_base_height_m` | OBJ-2 crown fire |
| Canopy bulk density (CBD) | `LF2022_CBD_230_CONUS.zip` | `canopy_bulk_density` | OBJ-2 crown fire |

`canopy_base_height_m` and `canopy_bulk_density` are required by the OBJ-2 fire spread simulator (Van Wagner 1977 crown fire initiation and Scott & Reinhardt 2001 active crown fire threshold). Without them the simulator runs surface-fire-only mode — no crown fire classification.

Place the extracted `.tif` files in:
```
data/static/landfire_raw/
    LC22_F40_230.tif
    LC22_CC_230.tif
    LC22_EVT_230.tif
    LC22_CBH_230.tif
    LC22_CBD_230.tif
```

Run for both resolutions at once (recommended):
```bash
python -m scripts.ingestion.ingest_landfire \
    --resolution-km 64 22 \
    --raw-dir data/static/landfire_raw \
    --output-dir data/static
```

Or one resolution at a time:
```bash
python -m scripts.ingestion.ingest_landfire --resolution-km 64 --output-dir data/static
python -m scripts.ingestion.ingest_landfire --resolution-km 22 --output-dir data/static
```

**New CLI arguments:**

| Argument | Default | Description |
|---|---|---|
| `--resolution-km` | `64 22` | One or more resolutions (space-separated). Loops through each in a single run |
| `--raw-dir` | `data/static/landfire_raw` | Path to the folder containing downloaded LANDFIRE `.tif` files |
| `--output-dir` | `data/static` | Where to write output Parquet files |

The script clips the CONUS rasters to CA + TX, resamples to 0.01° (~1 km) to avoid OOM, reprojects to WGS84, and runs `rasterstats.zonal_stats()` on each H3 cell.

Outputs:
```
data/static/landfire_features_64km.parquet   ← quiet-mode (H3 res-2)
data/static/landfire_features_22km.parquet   ← active/emergency mode (H3 res-5)
```

---

### 6b. SRTM 30m — Elevation, Slope, Aspect

**Source:** OpenTopography (https://opentopography.org/developers) — free API key required.

The 450,000 km² per-request area limit requires 7 tiles total. Replace `YOUR_KEY` below:

**California — 3 tiles (south/mid/north):**
```
# South (32.53–36.0)
https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1&south=32.53&north=36.0&west=-124.48&east=-114.13&outputFormat=GTiff&API_Key=YOUR_KEY

# Mid (36.0–39.0)
https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1&south=36.0&north=39.0&west=-124.48&east=-114.13&outputFormat=GTiff&API_Key=YOUR_KEY

# North (39.0–42.01)
https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1&south=39.0&north=42.01&west=-124.48&east=-114.13&outputFormat=GTiff&API_Key=YOUR_KEY
```

**Texas — 4 tiles (2×2 grid):**
```
# NW
https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1&south=31.17&north=36.50&west=-106.65&east=-100.08&outputFormat=GTiff&API_Key=YOUR_KEY
# NE
https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1&south=31.17&north=36.50&west=-100.08&east=-93.51&outputFormat=GTiff&API_Key=YOUR_KEY
# SW
https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1&south=25.84&north=31.17&west=-106.65&east=-100.08&outputFormat=GTiff&API_Key=YOUR_KEY
# SE
https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1&south=25.84&north=31.17&west=-100.08&east=-93.51&outputFormat=GTiff&API_Key=YOUR_KEY
```

Save files in `data/static/srtm_raw/`. Filenames must contain `california` or `texas` (prefix variants `srtm_`, `strm_`, `strn_` are all accepted by the script):
```
data/static/srtm_raw/
    srtm_california_south.tif   (or strm_california_south.tif)
    srtm_california_mid.tif
    srtm_california_north.tif
    srtm_texas_nw.tif
    srtm_texas_ne.tif
    srtm_texas_sw.tif
    srtm_texas_se.tif
```

Run:
```bash
python -m scripts.ingestion.ingest_srtm --resolution-km 64 --output-dir data/static
```

The script mosaics tiles per region (resampled to 0.01° during merge to avoid OOM), derives slope and aspect via `numpy.gradient`, and computes circular-mean aspect per H3 cell.

Output: `data/static/srtm_features_64km.parquet` — columns: `elevation_m`, `slope_degrees`, `aspect_degrees`

---

### 6c. MODIS NDVI (MOD13A2 v061) via NASA AppEEARS

**Portal:** https://appeears.earthdatacloud.nasa.gov/ — free NASA Earthdata account required.

**Step 1 — generate the area polygon:**
```bash
python -m scripts.utils.export_appeears_points --geojson
# writes: data/static/appeears_area.geojson
```

**Step 2 — submit an AppEEARS Area Sample task:**
1. Sign in → Start → **Area Sample**
2. Upload `data/static/appeears_area.geojson`
3. Product: `MOD13A2.061`, Layer: `1 km 16 days NDVI`
4. Date range: `01-01-2024` → `08-31-2024` (covers fire season)
5. Projection: **Geographic (WGS84)**
6. Submit → wait for email → download the GeoTIFF zip

**Step 3 — place GeoTIFFs and process:**
```
data/static/ndvi_raw/
    MOD13A2.061__1_km_16_days_NDVI_20240525T000000_aid0001.tif
    MOD13A2.061__1_km_16_days_NDVI_20240610T000000_aid0001.tif
    ...  (all downloaded .tif files)
```

```bash
python -m scripts.ingestion.ingest_ndvi --resolution-km 64 --output-dir data/static --force-rebuild
```

The script auto-detects local GeoTIFFs in `ndvi_raw/` and skips earthaccess download. It averages all 16-day composites to produce a single seasonal-mean NDVI per H3 cell.

Output: `data/static/ndvi_features_64km.parquet` — column: `ndvi` (scaled 0.0–1.0)

> **Note:** `soil_moisture_0_to_7cm` is requested from Open-Meteo but returns null on the free tier — this is expected and does not indicate a pipeline error.

---

### 6d. Rebuild the static cache

After adding any new source parquet, regenerate the unified static cache. `process_static.py` now has a full CLI:

```bash
# Rebuild both resolutions in one command (recommended)
python -m scripts.processing.process_static \
    --resolution-km 64 22 \
    --static-dir data/static \
    --force-rebuild

# Or one resolution at a time
python -m scripts.processing.process_static --resolution-km 64 --static-dir data/static
```

**CLI arguments:**

| Argument | Default | Description |
|---|---|---|
| `--resolution-km` | `64 22` | One or more resolutions to rebuild (space-separated) |
| `--static-dir` | `data/static` | Directory containing individual source Parquet files |
| `--force-rebuild` | `False` | Re-run even if the cache file already exists |

Missing sources fall back to `NaN` gracefully — partial static data is flagged `data_quality_flag=5` and still used for training.

---

## 7. Feature Schema

All 28 features are defined in `configs/schema_config.yaml` — the single source of truth. Feature names are **never** hardcoded in scripts; they are always read via `schema_loader.FeatureRegistry`.

### Feature Groups

| Group | Features | Source |
|---|---|---|
| Fire detections | `fire_radiative_power`, `fire_confidence`, `fire_pixel_count`, `active_fire_count`, `mean_frp` | FIRMS |
| Weather | `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `wind_direction_10m`, `precipitation`, `days_since_precipitation`, `drought_index_proxy`, `cumulative_wind_run_24h` | Open-Meteo / NWS |
| Terrain | `elevation_m`, `slope_deg`, `aspect_deg`, `terrain_roughness` | USGS SRTM |
| Fuel / Vegetation | `fuel_model_fbfm40`, `canopy_cover_pct`, `canopy_height_m`, `stand_age`, `vegetation_type`, `dominant_fuel_fraction` | LANDFIRE |
| Fire spread (OBJ-2) | `canopy_base_height_m`, `canopy_bulk_density`, `evt_national_class` | LANDFIRE CBH/CBD/EVT |
| Grid metadata | `grid_id`, `latitude`, `longitude`, `region`, `resolution_km`, `data_quality_flag`, `timestamp_utc` | Computed |

### Data Quality Flag Codes

| Code | Meaning | Action |
|---|---|---|
| 0 | All sources present — full confidence | Use for training |
| 2 | Weather from NWS fallback (Open-Meteo unavailable) | Use with caution |
| 3 | >30 % of dynamic feature columns are null | Inspect before use |
| 4 | All static columns null — no LANDFIRE/SRTM cache for this cell | Exclude from training |
| 5 | Some static columns null — boundary cell or missing tile | Use for training (partial static) |

Phase 2 placeholder columns (`fire_weather_index` is now computed; `ndvi` requires AppEEARS download) are excluded from quality flag null-rate calculations so they do not inflate flag=3 counts.

**Fire spread columns (OBJ-2 inputs):** `canopy_base_height_m`, `canopy_bulk_density`, and `evt_national_class` are optional spread-simulation columns added via `fuse_features.py`. They are `NaN` until the LANDFIRE CBH/CBD download is complete (see §6a). When `NaN`, the OBJ-2 simulator falls back to surface-fire-only mode — no crown fire classification — which is still valid. These columns are excluded from the `data_quality_flag=3` null-rate threshold.

---

## 8. Pipeline Orchestration (Airflow DAGs)

### `wildfire_data_pipeline`

Scheduled at `0 */6 * * *` (00:00, 06:00, 12:00, 18:00 UTC). Also triggered dynamically by `watchdog_sensor_dag` with escalated parameters when a fire is confirmed.

**Key design decisions:**

- `max_active_runs=1` — prevents concurrent DVC lock conflicts
- `trigger_rule='none_failed'` on `fuse_features` — handles the `check_static_cache` ShortCircuit skip without propagating downstream
- `retry_exponential_backoff=True` with `max_retry_delay=30min` — handles transient API failures gracefully
- `execution_timeout=1h` per task — prevents zombie tasks from holding slots
- `detect_anomalies` uses `trigger_rule='all_done'` — runs and sends alerts even when `validate_schema` raises warnings

**Task dependency summary:**

```python
check_static_cache >> load_static_layers >> fuse_features
region_ca_taskgroup >> fuse_features
region_tx_taskgroup >> fuse_features
fuse_features >> validate_schema >> detect_anomalies
detect_anomalies >> [export_to_parquet, export_spatial] >> version_with_dvc
```

### `watchdog_sensor_dag`

Polls GCS for new trigger files every 60 seconds. On detection, calls `trigger_dag()` on `wildfire_data_pipeline` passing fire cell IDs, FRP values, and resolution escalation parameters.

---

## 9. Data Versioning with DVC

DVC tracks all processed data blobs in GCS. Git tracks code, configs, and `.dvc` lock files.

### DVC pipeline stages

```
ingest_firms  ─┐
ingest_weather ┤→ process_* → fuse_features → validate_schema → detect_anomalies
                                          └──────────────────→ export_spatial
historical_backfill → bias_analysis → train_ignition_model
```

All stages have CLI entry points and can be run individually or as a full pipeline via `dvc repro`.

### Developer workflow

```bash
# After a pipeline run — add changed outputs and push
dvc add data/processed/fused
dvc add data/processed/spatial
dvc push

# Commit the updated lock files
git add dvc.lock data/processed/fused.dvc data/processed/spatial.dvc
git commit -m "data: update fused features 2026-02-22"
git push

# On another machine — pull the exact same data
git pull && dvc pull
```

### Offline reproduction without Airflow

```bash
# Reproduce the full pipeline
dvc repro

# Reproduce from a specific stage onwards
dvc repro validate_schema

# Check what would re-run without executing
dvc status
```

### Important: `cache: false` outputs

Two outputs use `cache: false` — they are committed directly to Git rather than stored in GCS:

- `data/processed/baselines/anomaly_last_run.json` — small run metadata, useful for graders to read without `dvc pull`
- `data/processed/baselines/bias_report.json` — human-readable bias analysis report; reviewers should be able to read it in the repo

---

## 10. Schema Validation & Statistics

`validate_schema.py` runs Great Expectations validation driven dynamically from `configs/schema_config.yaml`. No expectations are hardcoded in Python.

### What is validated

- Column presence and dtype for all 28 features
- Null rate thresholds per column (configurable per feature in schema YAML)
- Value range checks (`relative_humidity_2m` ∈ [0, 100], etc.)
- `grid_id` uniqueness per `(timestamp_utc, region)` partition
- `data_quality_flag` only takes values 0–5
- Row count within expected bounds for each region/resolution

### Statistics generation

On each run, `validate_schema.py` writes `data/processed/baselines/stats_latest.json` containing row counts, null rates, and value distributions per feature. These baselines feed `detect_anomalies` in the next task.

```bash
# Run standalone (also called by dvc repro validate_schema)
python scripts/validation/validate_schema.py \
  --input data/processed/fused \
  --resolution-km 64 \
  --output-dir data/processed/baselines

# Inspect the latest statistics
cat data/processed/baselines/stats_latest.json | python -m json.tool
```

---

## 11. Anomaly Detection & Alerts

`detect_anomalies.py` applies a **seasonal z-score** test using Welford online updates per feature per season.

### How it works

1. Load seasonal baseline JSON from `data/processed/baselines/`
2. Compute z-score for each monitored feature against the seasonal mean/std
3. Apply z > 4.0 threshold during fire season (Jun–Nov), z > 3.5 off-season
4. Features exceeding the threshold are flagged
5. Update the baseline with the current window's values (Welford — no full recompute)
6. Write `anomaly_last_run.json` summary

### Slack Notifications

All Slack notification logic lives in `dags/utils/slack_notify.py`:

| Callback | Trigger | Behavior |
|----------|---------|----------|
| `sla_on_failure_callback` | Any task failure | Tracks consecutive failures per task via XCom. Escalates to SLA BREACH after 3 consecutive failures. |
| `sla_on_success_callback` | Any task success | Resets the consecutive failure counter to 0. |
| `notify_slack` | DAG-level failure | Basic failure alert with DAG/task/run info. |
| `notify_anomaly_alert` | Anomalies detected | Posts outlier details (feature, z-score, season). |

Both `wildfire_data_pipeline` and `watchdog_sensor` DAGs have `on_failure_callback` and `on_success_callback` wired in `default_args`, plus a DAG-level `on_failure_callback` for catastrophic failures.

If `SLACK_WEBHOOK_URL` is not set, all notifications are silently skipped and the pipeline does not fail.

```bash
# Run standalone (also called by dvc repro detect_anomalies)
python scripts/validation/detect_anomalies.py \
  --input data/processed/fused \
  --baseline-dir data/processed/baselines \
  --output data/processed/baselines/anomaly_last_run.json
```

---

## 12. Data Bias Detection & Mitigation

`scripts/validation/bias_analysis.py` evaluates whether feature distributions and fire detection rates differ systematically across subgroups — a prerequisite before any model training.

### Slicing dimensions

| Dimension | Values | Rationale |
|---|---|---|
| Geographic region | `california`, `texas` | Different fire regimes, terrain, data density |
| Fuel model tier | grass, shrub, timber, slash, non-burnable | Fuel-type imbalance affects fire risk scores |
| Fire season | fire season (Jun–Nov) vs. off-season (Dec–May) | Seasonal sensor sensitivity differs |
| Data quality tier | tier A (flag 0–2) vs. tier B (flag 3–5) | Degraded data disproportionately affects rural cells |

### How bias is measured

For each slice, the analysis computes:

- Feature mean, std, null rate, and percentiles
- **KL divergence** (approximated via histogram) between the slice and overall population — flags distributional shift not just mean differences
- **Fire detection rate disparity** — absolute difference between slice fire rate and overall fire rate

A slice is flagged as biased when KL divergence > 0.1 nats for any feature, or fire rate disparity > 5 percentage points.

### Running the analysis

```bash
# Run on the historical backfill (also called by dvc repro bias_analysis)
python scripts/validation/bias_analysis.py \
  --input  data/processed/backfill \
  --output data/processed/baselines/bias_report.json

# Review findings
cat data/processed/baselines/bias_report.json | python -m json.tool | grep -A3 '"findings"'
```

### Findings (February 2026 backfill)

| Finding | Severity |
|---|---|
| Texas rural cells have 3× higher data_quality_flag ≥ 3 rate vs. California | Medium |
| Off-season fire detection rate near zero (class imbalance) | Low |
| MODIS vs. VIIRS confidence score split before normalization | Low |

### Mitigations applied

- **Texas rural cells**: `data_quality_flag` codes 1–3 are now populated with specific gap-fill codes so downstream models can weight or exclude degraded rows
- **Class imbalance**: `historical_backfill.py` supports `--oversample-minority` to apply SMOTE at the data layer before training
- **Sensor normalization**: MODIS confidence scores are normalized to the VIIRS scale in `process_firms.py:normalize_confidence()`

### DVC enforced gate

`train_ignition_model` in `dvc.yaml` explicitly depends on `bias_report.json`. This means `dvc repro train_ignition_model` will fail if the bias analysis has not been run, enforcing a reviewable audit trail before any model training.

---

## 13. Historical Backfill

The backfill script replays the full pipeline over an arbitrary date range to generate training data for ML models.

### Usage

```bash
# Dry-run: list windows without fetching data
python -m scripts.backfill.historical_backfill \
    --start 2024-01-01 --end 2024-01-07 --dry-run

# Full backfill (resumes automatically from last completed window)
python -m scripts.backfill.historical_backfill \
    --start 2020-01-01 --end 2025-12-31 \
    --frequency-hours 6 \
    --resolution-km 64 \
    --output-dir data/processed/backfill
```

### Key design decisions

| Feature | Implementation |
|---|---|
| Resume support | Skips windows where all region output files already exist (`--resume` on by default) |
| Partitioned output | `region={r}/year={yyyy}/month={mm}/features_{timestamp}.parquet` — DVC-friendly |
| Reuses live pipeline | Calls the same `fetch_*`, `process_*`, `fuse_features` functions as the Airflow DAG |
| Static data shared | Reads from `data/static/` (same LANDFIRE/SRTM/NDVI caches as live pipeline) |

### Output structure

```
data/processed/backfill/
  region=california/
    year=2024/month=01/features_20240101T0000.parquet
    year=2024/month=01/features_20240101T0600.parquet
    ...
  region=texas/
    ...
```

---

## 14. Pipeline Flow Optimization

### Parallelism

| Optimization | Implementation | Impact |
|---|---|---|
| Regional sharding | CA and TX in separate Airflow `TaskGroup` blocks | ~40% reduction in ingestion wall time |
| Static layer caching | `ShortCircuitOperator` skips 15–20 min LANDFIRE download after first run | Eliminated on every subsequent run |
| Focal grid on watchdog trigger | Only fire-zone cells processed at 22 km | Reduces active-mode cell count from ~1000 to ~50-100 |
| Weather rate limit backoff | Token bucket with jitter in `rate_limiter.py` | Prevents cascading API failures |
| `max_active_runs=1` | Prevents concurrent DVC lock contention | Eliminates race conditions on `dvc push` |

### Reading the Gantt chart

1. Airflow UI → `wildfire_data_pipeline` → select a completed run → **Gantt** tab
2. The longest bars in a typical run: `load_static_layers` (first run only, ~15–20 min)
3. Both CA and TX `TaskGroup` bars should overlap — if they are sequential, the parallel execution broke and the DAG definition needs checking

---

## 15. Running Tests

### Local (Linux / macOS)

```bash
export PYTHONPATH=.
pytest tests/ -v --tb=short

# With coverage
pytest tests/ -v --cov=scripts --cov-report=term-missing

# Bias analysis tests only
pytest tests/test_validation/test_bias_analysis.py -v
```

### Local (Windows)

```powershell
$env:PYTHONPATH="."
python -m pytest tests/ -v --tb=short
```

> `test_dags/test_dag_structure.py` requires Airflow's `fcntl` module (Unix-only). These tests auto-skip on Windows. Run in Docker for full coverage.

### Inside Docker (matches CI exactly)

```bash
docker run --rm \
  -e GCS_BUCKET_NAME=test-bucket \
  -e FIRMS_MAP_KEY=test-key \
  wildfire-pipeline:test \
  pytest tests/ -v --tb=short
```

### Test coverage by module

| Module | What it covers |
|---|---|
| `test_ingestion/` | FIRMS API timeouts, malformed CSV, retry logic, SLA alerting |
| `test_processing/` | Weather aggregation (7 scenarios), static layer caching |
| `test_fusion/` | Left-join completeness, temporal lag rotation, weather fallback |
| `test_validation/test_anomalies.py` | Welford updates, per-season files, fire/off-season thresholds |
| `test_validation/test_bias_analysis.py` | 38 tests: KL divergence, slice grouping, fire rate disparity, findings synthesis |
| `test_export/` | Tabular Parquet + spatial .npz consistency |
| `test_utils/` | H3 grid cell counts, focal grid, spatial pruning |
| `test_dags/` | DAG structure, task count, dependency edges |
| `test_integration/` | End-to-end pipeline with mocked APIs |

---

## 16. CI/CD Pipeline

GitHub Actions (`.github/workflows/ci.yaml`) runs on every push and PR to `main` or `develop`:

1. **Build** Docker image to `test` target (layer cache via GHA cache)
2. **DAG validation** — parse-time import check on both DAGs
3. **pytest** — full test suite inside Docker with `GCS_BUCKET_NAME=test-bucket`
4. **ruff** — lint `scripts/`, `dags/`, `tests/` (rules E and F; zero tolerance)
5. **Dependency check** — verifies `pyarrow` is hard-pinned in `constraints.txt`

Merges to `main` are blocked if any step fails.

---

## 17. Code Style & Standards

- **Linting**: `ruff check scripts/ dags/ tests/` — run before every commit
- **Formatting**: PEP 8 compliant; `ruff format` is the formatter
- **Docstrings**: Google-style with `Args`, `Returns`, `Raises` on every public function
- **Type hints**: required on all function signatures
- **Logging**: `logging` only — never `print()`. `INFO` for normal ops, `WARNING` for recoverable issues
- **Schema**: never hardcode feature names — always read from `schema_config.yaml` via `FeatureRegistry`
- **Branching**: `feature/your-name-description` → PR to `develop` → review → merge

```bash
ruff check scripts/ dags/ tests/
ruff check --fix scripts/ dags/ tests/   # auto-fix safe issues
```

---

## 18. GCP Setup

### DVC Remote (one-time per developer)

```bash
dvc remote add -d gcs_remote gs://<your-bucket>/dvc-store
dvc remote modify gcs_remote credentialpath gcp-key.json
dvc remote list   # verify
dvc pull
```

### Cloud Function Deployment

```bash
chmod +x cloud/deploy.sh && ./cloud/deploy.sh

# Verify
gcloud scheduler jobs run watchdog-quiet --location=us-central1
gcloud storage ls gs://<bucket>/watchdog/triggers/
gcloud storage cat gs://<bucket>/watchdog/state/current.json
```

### Update Industrial Exclusion List (no redeployment needed)

```bash
gcloud storage cp industrial_sources.json \
  gs://<bucket>/watchdog/config/industrial_sources.json
```

---

## 19. Environment Variables

Environment variables are set in the **repo root `.env`** file (`wildfire_detection/.env`). See [SETUP.md](../SETUP.md) for the full template. **Never commit `.env` to Git**.

| Variable | Required | Description |
|---|---|---|
| `FIRMS_MAP_KEY` | ✅ Yes | NASA FIRMS API key |
| `GCS_BUCKET_NAME` | ✅ Yes | GCS bucket for DVC and watchdog state |
| `GCP_KEY_PATH` | ✅ Yes | Path to GCP service account JSON |
| `GOOGLE_CLOUD_PROJECT` | GCP deploy only | GCP project ID |
| `GCP_REGION` | GCP deploy only | Cloud Function region (default: `us-central1`) |
| `SLACK_WEBHOOK_URL` | Optional | Slack webhook for anomaly and emergency alerts |
| `WATCHDOG_TRIGGER_PREFIX` | Auto-set | Set by `deploy.sh` — do not edit manually |

---

## 20. Reproducibility

Any team member should be able to reproduce the pipeline from scratch:

```bash
# 1. Clone
git clone https://github.com/Scofe-C/wildfire_detection/data-pipeline.git && cd data-pipeline

# 2. Install dependencies (pyarrow version pinned in constraints.txt — do not skip -c)
pip install -r requirements.txt -c constraints.txt

# 3. Configure DVC and pull data
dvc remote add -d gcs_remote gs://<bucket>/dvc-store
dvc remote modify gcs_remote credentialpath gcp-key.json
dvc pull

# 4a. Reproduce full pipeline offline
dvc repro

# 4b. Or run in Docker (from repo root — no local Python install needed)
cd .. && cp .env.example .env   # fill in your keys
make up-full                    # starts all services
```

`pyarrow` is hard-pinned in `constraints.txt` because Parquet encoding changed between major versions. All team members across Windows 11, Windows 10, and macOS must use the identical version to prevent silent schema divergence when reading each other's Parquet files.

The `dvc.lock` file commits the exact MD5 hash of every stage dependency and output. Checking out any historical Git commit and running `dvc checkout` restores the exact data state for that commit.

---

## 21. Error Handling & Logging

### Logging conventions

All modules use Python's `logging` library with `python-json-logger` for structured JSON output. Never use `print()`.

```python
import logging
logger = logging.getLogger(__name__)
logger.info(f"[{region}] FIRMS ingestion complete: {len(df)} rows → {output_path}")
logger.warning(f"Anomaly: '{col}' has {outlier_count} outliers (z>{z_threshold})")
```

### Key failure points and handling

| Failure Point | Handling |
|---|---|
| FIRMS API unavailable | `max_retries=3` with exponential backoff; task fails after exhaustion |
| Open-Meteo rate limit | Token bucket with jitter; automatic NWS fallback on consecutive failures |
| Static layer OOM | ShortCircuit cache check avoids re-download; Docker memory limit enforced |
| Schema validation failure | Non-fatal warning; `detect_anomalies` still runs via `trigger_rule='all_done'` |
| DVC remote unreachable | `version_with_dvc` task fails; exported data is still on disk |
| Malformed GCS trigger file | `watchdog_sensor_dag` logs error, skips file, does not trigger pipeline |
| Slack webhook failure | Caught, logged as WARNING — never raises or fails a task |
| Any task failure | SLA-aware Slack callback tracks consecutive failures; escalates after 3 |

---

## 22. Adaptive Watchdog

### Four-Gate False Alarm Prevention

| Gate | Method |
|---|---|
| Spatial clustering | Require ≥ 3 FIRMS detections within a 50 km radius |
| Temporal persistence | Detection in ≥ 2 consecutive 15-min GOES windows |
| VIIRS cross-reference | MODIS-only detections require VIIRS confirmation before escalation |
| Industrial exclusion | Candidate cells within 2 km of known industrial sources are suppressed |

### Trigger flow

1. Cloud Function detects confirmed fire → writes trigger JSON to GCS
2. `watchdog_sensor_dag` polls GCS every 60s → detects trigger file
3. Calls `trigger_dag()` on `wildfire_data_pipeline` with escalated `resolution_km=22` and `fire_cells` list
4. Pipeline runs at 22 km resolution over the focal grid only (~50–100 cells vs ~1000 full-region)
5. On false alarm: FA record written, state reverts after 30 minutes

---

## 23. Troubleshooting

| Problem | Fix |
|---|---|
| DAG not appearing in Airflow UI | `docker run wildfire-pipeline:test python3 dags/wildfire_dag.py` — check for import errors |
| `watchdog_sensor_dag` missing | Ensure it is in `dags/` and not paused in the UI |
| DVC push fails | `dvc remote list` — if empty, re-add the GCS remote |
| Static layer download OOM | Increase Docker Desktop RAM to 12 GB |
| Tests fail with GCS errors | Pass `-e GCS_BUCKET_NAME=test-bucket` — all tests mock GCS by default |
| `docker compose up` build fails | Run `docker compose build --no-cache` first |
| `fcntl` import error on Windows | DAG structure tests auto-skip on Windows; run in Docker for full suite |
| `dvc repro` shows all stages changed | Expected after adding `__main__` blocks — run `dvc repro` once to re-lock |
| PyArrow schema mismatch across machines | Ensure `pip install -r requirements.txt -c constraints.txt` was used — not plain `pip install -r requirements.txt` |
 | WinError 1337| Make sure pytest.ini is in Data-Pipeline/. It sets rootdir=., importmode=importlib, and norecursedirs which stops pytest walking into protected Windows system folders.|

---

<div align="center">

Built with [Apache Airflow](https://airflow.apache.org/) · [DVC](https://dvc.org/) · [Great Expectations](https://greatexpectations.io/) · [H3](https://h3geo.org/) · [Google Cloud](https://cloud.google.com/)

</div>
