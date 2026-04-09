# Wildfire Detection — MLOps Dashboard (Frontend)

A demo frontend that visually represents the **actual** wildfire detection pipeline implemented in this repository. Every stage name, metric value, field name, model identifier, and threshold is sourced directly from the codebase — nothing is invented.

## Quick Start

```bash
cd Frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

**Requirements:** Node 18+

---

## How the UI Maps to the Repository

### 1. Overview (`/overview`)
**Source:** `Data-Pipeline/dags/wildfire_dag.py`, `model-pipeline/configs/model_config.yaml`

- Operational mode banner: QUIET / ACTIVE / EMERGENCY (from `schema_config.yaml` watchdog modes)
- Pipeline run history bar chart: last 12 × 6-hour runs
- Data quality flag distribution: flags 0–5 from `Data-Pipeline/configs/schema_config.yaml`
- Model registry strip: run IDs, AUC-PR, FNR, thresholds from actual training runs
- Top risk cells table with `fire_risk_score`, `vpd`, `fire_weather_index`, `fuel_model_fbfm40`

### 2. Data Pipeline (`/data-pipeline`)
**Source:** `Data-Pipeline/scripts/`, `Data-Pipeline/dags/wildfire_dag.py`

Mirrors the 5 stages of `wildfire_data_pipeline`:

| UI Stage | Code Module |
|----------|-------------|
| Ingestion | `scripts/ingestion/ingest_firms.py`, `ingest_weather.py`, `ingest_landfire.py`, `ingest_srtm.py`, `ingest_goes.py` |
| Processing | `scripts/processing/process_firms.py`, `process_weather.py`, `process_static.py` |
| Feature Fusion | `scripts/fusion/fuse_features.py` |
| Validation | `scripts/validation/validate_schema.py`, `detect_anomalies.py` |
| Export & Version | `scripts/export/export_spatial.py` |

- Shows exact column names output by each stage
- Fill strategies (forward_fill, zero_fill, default_zero) from `fuse_features.py`
- Schema checks and anomaly detection z-score thresholds
- Data quality flag legend (F0–F5) from `schema_config.yaml`

### 3. Model Pipeline — OBJ-1 Ignition Classifier (`/obj1`)
**Source:** `model-pipeline/src/models/obj1_xgboost/model.py`, `src/validation/metrics.py`, `src/validation/bias_check.py`

- Three training runs: XGBoost CA (970bb676, production), LightGBM CA (a3f1c291, staging), XGBoost TX (b7e52d18, production)
- Real metrics: AUC-PR 0.9051 (CA), 0.9124 (TX)
- Tuned thresholds: 0.4596 (CA), 0.4201 (TX) from threshold tuning for ≥90% recall
- Gates: AUC-PR ≥ 0.89, FNR disparity ≤ 0.15, recall ≥ 0.90
- SHAP feature importance ranked by mean |SHAP| (vpd #1, fire_weather_index #2)
- Confusion matrix, Precision-Recall curve with tuned threshold marked
- Bias analysis: FNR across region/season/fuel slices

### 4. Model Pipeline — OBJ-2 Fire Spread Simulator (`/obj2`)
**Source:** `model-pipeline/src/models/obj2_spread/fire_spread_simulator.py`, `evaluation.py`

- Physics model reference: Rothermel (1972), FBFM40, Van Wagner (1977), Scott & Reinhardt (2001), Byram (1959), etc.
- Spread direction compass, spread_speed_kmh, dead_fuel_moisture_pct, crown_fire_status
- Validation: buffered_iou ≥ 0.35, dice_coefficient ≥ 0.50 vs. CAL FIRE FRAP perimeters
- Configuration from `model_config.yaml`: n_simulations, fire_period_length_hr, grid_resolution_m

### 5. Model Pipeline — OBJ-3 AI Disaster Reporter (`/obj3`)
**Source:** `model-pipeline/src/models/obj3_gemini/reporter.py`, `state_machine.py`, `schemas/`

- Current operational mode and state machine decision matrix
- 9-cell mode matrix (risk_level × firms_hotspot_count × is_deployable → QUIET/ACTIVE/EMERGENCY)
- LLM backend switcher: Ollama (Phase 1), Gemini Dev API (Phase 2), Vertex AI (Phase 3)
- Watchdog configuration: poll intervals, resolution, false alarm gates, emergency triggers
- Report schema hierarchy: BaseReport → IncidentReport / DailyReport / HighRiskReport / FinalReport

### 6. Risk Monitor (`/risk-monitor`)
**Source:** `model-pipeline/configs/model_config.yaml`, `Data-Pipeline/scripts/fusion/fuse_features.py`

- 55 H3 grid cells at 64 km resolution: ~35 California + ~20 Texas
- Risk tiers from `model_config.yaml`:
  - CRITICAL: score ≥ 0.65
  - HIGH: score ≥ 0.365 (= default XGBoost decision threshold)
  - MEDIUM: score ≥ 0.15
  - LOW: score < 0.15
- Cell detail shows: `fire_risk_score`, `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `vpd`, `fire_weather_index`, `fuel_model_fbfm40`, `elevation_m`, `active_fire_count`, `fire_detected_binary`
- Filter by region (California / Texas) or risk tier

### 7. Incident Reports (`/reports`)
**Source:** `model-pipeline/src/models/obj3_gemini/schemas/`, `src/models/obj3_gemini/reporter.py`

- Mock AI-generated reports in QUIET, ACTIVE, and EMERGENCY modes
- Schema: ICS-209 aligned Pydantic models
- Shows: situation_summary, weather_outlook, risk distribution, key driving features, recommended actions
- Validation requirements: confidence ≥ 0.70, grounding_sources ≥ 3

---

## Technology Stack

| Tool | Version | Purpose |
|------|---------|---------|
| React | 18 | UI framework |
| Vite | 5 | Build tool |
| Tailwind CSS | 3 | Utility-first styling |
| Recharts | 2 | AUC-PR curve, SHAP bar chart, run history |
| Lucide React | latest | Icons |

## Project Structure

```
Frontend/
├── src/
│   ├── App.jsx                         # Root + view routing
│   ├── main.jsx                        # Entry point
│   ├── index.css                       # Tailwind + global styles
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Sidebar.jsx             # Navigation
│   │   │   └── Header.jsx             # Page header + alerts
│   │   ├── overview/
│   │   │   └── Overview.jsx
│   │   ├── data-pipeline/
│   │   │   └── DataPipeline.jsx
│   │   ├── model-pipeline/
│   │   │   ├── OBJ1Ignition.jsx
│   │   │   ├── OBJ2Spread.jsx
│   │   │   └── OBJ3Reporter.jsx
│   │   ├── risk-monitor/
│   │   │   └── RiskMonitor.jsx
│   │   └── reports/
│   │       └── IncidentReports.jsx
│   └── data/
│       ├── mockPipelineData.js         # Data-Pipeline stage mock data
│       ├── mockModelData.js            # model-pipeline metrics/config
│       ├── mockGridData.js             # H3 grid cells for CA + TX
│       └── mockReports.js             # OBJ-3 mock AI reports
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
├── postcss.config.js
└── README.md
```

## Design Decisions

- **Dark MLOps theme**: `#080c14` background, surface layers at `#0d1117` / `#131b2e` / `#1a2540`
- **Monospace font (JetBrains Mono)**: all IDs, metric values, column names, paths
- **Color encoding**: green=healthy/low risk, orange=warning/active, red=critical/failure
- **All mock data is grounded**: no invented feature names — everything traces to a specific file in the repo

## No Backend Required

All data is mock/static. To connect to a real backend:
- Replace imports in `src/data/` files with API fetch calls
- The FastAPI server is at `model-pipeline/src/api/server.py`
