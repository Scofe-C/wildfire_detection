# Wildfire Detection & Response System

A production-grade MLOps pipeline for wildfire risk prediction across California and Texas. Built with Apache Airflow, DVC, and Docker.

## What It Does

- Ingests near-real-time fire detections from NASA FIRMS and weather data from Open-Meteo
- Processes and fuses data into a unified feature table on an H3 hexagonal grid
- Validates data quality and detects anomalies with seasonal baselines
- Versions all data with DVC on Google Cloud Storage
- Runs on a 6-hour schedule via Airflow

## Architecture

```
[NASA FIRMS]   → ingest_firms   → process_firms   ─┐
[Open-Meteo]   → ingest_weather → process_weather  ┼→ fuse → validate → detect_anomalies → export → dvc_version
[LANDFIRE/SRTM] → load_static_layers ──────────────┘
```

## Quick Start

```bash
cd Data-Pipeline

# 1. Set up environment
cp .env.example .env        # Edit with your API keys
# Place gcp-key.json in this directory (see team_setup_guide.md)

# 2. Build and run
docker compose build
docker compose up -d

# 3. Open Airflow
# http://localhost:8080  (airflow / airflow)
```

For detailed setup instructions (GCP, Docker tuning, local testing), see **[team_setup_guide.md](Data-Pipeline/team_setup_guide.md)**.

## Project Structure

```
wildfire_detection/
├── .github/workflows/ci.yaml        # CI: build, DAG validation, pytest, ruff
├── .gitignore
└── Data-Pipeline/
    ├── configs/schema_config.yaml    # Feature registry (28 columns)
    ├── dags/wildfire_dag.py          # Airflow DAG (10 tasks)
    ├── scripts/
    │   ├── ingestion/                # FIRMS + weather API clients
    │   ├── processing/               # Spatial aggregation, feature engineering
    │   ├── fusion/                   # Multi-source join
    │   ├── validation/               # Schema checks, anomaly detection
    │   └── utils/                    # Grid, schema loader, rate limiter
    ├── tests/                        # pytest suite
    ├── docker/Dockerfile             # Airflow 2.8.1 + geospatial deps
    ├── docker-compose.yaml           # Local dev: Postgres + Airflow
    ├── dvc.yaml                      # DVC pipeline (9 stages)
    ├── requirements.txt              # Pinned Python dependencies
    ├── .env.example                  # Environment variable template
    ├── team_setup_guide.md           # Full setup instructions
    └── README.md                     # Pipeline-specific docs
```

## Data Sources

| Source | Type | Access |
|--------|------|--------|
| NASA FIRMS | Active fire detections | Free API key |
| Open-Meteo | Weather (8 variables + FWI) | Free, no key |
| NWS | Weather fallback | Free, no key |
| LANDFIRE | Fuel models, vegetation | Static download |
| USGS SRTM | Elevation, slope, aspect | Static download |

## Tech Stack

- **Orchestration:** Apache Airflow 2.8.1
- **Containerization:** Docker + Docker Compose
- **Data Versioning:** DVC with GCS backend
- **Spatial Indexing:** H3 hexagonal grid (64km / 10km / 1km)
- **CI/CD:** GitHub Actions
- **Cloud:** Google Cloud Platform (GCS, future: Cloud Run, Spot VMs)

