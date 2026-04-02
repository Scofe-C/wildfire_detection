# Wildfire Detection Data Pipeline Summary

## Overview

The pipeline ingests data from 4 external sources every 6 hours, processes each independently, then fuses them into a single row per H3 hexagonal grid cell. The final output is used for both real-time inference and ML model training.

---

## Pipeline Stages

```
[NASA FIRMS API]    [Open-Meteo API]    [LANDFIRE/SRTM]    [H3 Math]
      |                   |                    |                 |
      v                   v                    v                 v
  raw/firms/          raw/weather/        static/           (in memory)
  *.csv               *.csv               *.parquet
      |                   |                    |
      v                   v                    |
processed/firms/    processed/weather/         |
  *.parquet           *.parquet                |
      |                   |                    |
      +-------------------+--------------------+
                          |
                          v
              processed/fused/fused_features_latest.parquet
              processed/fused/fused_features_ml_latest.parquet
                          |
                          v
              processed/64km/region=*/year=*/month=*/features_*.parquet
```

---

## Stage 1: Raw Ingestion

### 1a. FIRMS (NASA Fire Detections)
**File:** `data/raw/firms/firms_raw_<timestamp>.csv`
**Source:** NASA FIRMS API — returns only pixels NASA has already classified as fire
**Sensors:** MODIS (Terra/Aqua satellites) and VIIRS (Suomi NPP/NOAA-20)
**One row = one satellite pixel flagged as fire**

| Column | Sensor | Description |
|--------|--------|-------------|
| `latitude`, `longitude` | Both | Center coordinates of the detected pixel |
| `bright_ti4` | VIIRS only | Mid-infrared brightness temp (K) — primary fire detection band |
| `bright_ti5` | VIIRS only | Thermal IR background temp (K) — used as reference |
| `brightness` | MODIS only | Equivalent of bright_ti4 for MODIS |
| `bright_t31` | MODIS only | Equivalent of bright_ti5 for MODIS |
| `frp` | Both | Fire Radiative Power (megawatts) — fire intensity |
| `confidence` | Both | VIIRS: l/n/h (low/nominal/high), MODIS: 0–100 |
| `scan`, `track` | Both | Pixel size in km (varies with viewing angle) |
| `acq_date`, `acq_time` | Both | Date and UTC time of satellite overpass |
| `satellite` | Both | N=NOAA-20, T=Terra, A=Aqua |
| `instrument` | Both | VIIRS or MODIS |
| `version` | Both | Algorithm version (e.g. 2.0NRT = near real-time) |
| `daynight` | Both | D=day pass, N=night pass |
| `region` | Pipeline-added | california or texas (tagged by ingest_firms.py) |
| `sensor_source` | Pipeline-added | VIIRS_SNPP_NRT or MODIS_NRT |

### 1b. Weather
**File:** `data/raw/weather/weather_raw_<region>_<timestamp>.csv`
**Source:** Open-Meteo API — provides weather for every grid cell regardless of fire
**One row = one grid cell at one timestamp**

| Column | Description |
|--------|-------------|
| `grid_id` | H3 hex cell identifier |
| `timestamp` | Observation timestamp (UTC) |
| `temperature_2m` | Air temperature at 2m (°C) |
| `relative_humidity_2m` | Relative humidity at 2m (%) |
| `wind_speed_10m` | Wind speed at 10m (km/h) |
| `wind_direction_10m` | Wind direction at 10m (degrees, 0=N) |
| `precipitation` | Total precipitation in window (mm) |
| `soil_moisture_0_to_7cm` | Volumetric soil moisture 0–7cm (m³/m³) |
| `vpd` | Vapor Pressure Deficit (kPa) |
| `fire_weather_index` | Canadian Fire Weather Index |
| `data_quality_flag` | Quality indicator from Open-Meteo |

### 1c. Static Data (loaded once, not re-ingested every run)
**Files:** `data/static/static_features_64km.parquet`, `data/static/landfire_features_64km.parquet`
**Sources:** LANDFIRE (vegetation/fuel) + USGS SRTM 30m (terrain)
**One row = one grid cell (static, does not change over time)**

| Column | Source | Description |
|--------|--------|-------------|
| `grid_id` | Computed | H3 hex cell identifier |
| `latitude`, `longitude` | Computed | Cell centroid coordinates |
| `fuel_model_fbfm40` | LANDFIRE | Scott & Burgan 40 fuel model code (categorical) |
| `canopy_cover_pct` | LANDFIRE | Forest canopy cover percentage |
| `vegetation_type` | LANDFIRE | Existing Vegetation Type code (categorical) |
| `dominant_fuel_fraction` | LANDFIRE | Fraction of dominant fuel type in cell |
| `ndvi` | MODIS | Normalized Difference Vegetation Index (–1 to 1) |
| `elevation_m` | SRTM | Mean elevation within grid cell (meters) |
| `slope_degrees` | SRTM | Mean terrain slope (degrees) |
| `aspect_degrees` | SRTM | Dominant terrain aspect (degrees, 0=N) |

### 1d. GOES
**File:** `data/raw/goes/goes_<region>_<timestamp>.json`
**Status:** Ingested but currently empty — not yet contributing features to the pipeline.

---

## Stage 2: Processing

### 2a. FIRMS Processing → Grid Aggregation
**Script:** `scripts/processing/process_firms.py`
**File:** `data/processed/firms/firms_features_<region>_latest.parquet`

What happens:
1. Drop rows with missing lat/lon
2. Normalize MODIS confidence (l/n/h → 30/60/90)
3. Clip FRP outliers at 99.5th percentile; clip negatives to 0
4. Snap each pixel lat/lon to H3 64km hex cell using `points_to_grid_ids()`
5. `groupby("grid_id")` aggregate: count pixels, mean/median FRP, max confidence
6. Every grid_id that appears gets `fire_detected_binary = 1`
7. Compute nearest fire distance between fire cells using H3 grid distance

**Output columns (one row per fire cell, cells with NO fire are absent):**

| Column | Description |
|--------|-------------|
| `grid_id` | H3 hex cell |
| `active_fire_count` | Number of satellite pixels flagged as fire in this cell |
| `mean_frp` | Mean Fire Radiative Power (MW) |
| `median_frp` | Median Fire Radiative Power (MW) |
| `max_confidence` | Highest confidence detection (0–100) |
| `fire_detected_binary` | Always 1 in this file (absent cells will get 0 in fusion) |
| `nearest_fire_distance_km` | Distance to nearest other fire cell (km), –1 if only fire cell |

### 2b. Weather Processing → Grid Aggregation
**Script:** `scripts/processing/process_weather.py`
**File:** `data/processed/weather/weather_features_<region>_latest.parquet`

What happens:
1. Temporally aggregate multiple raw observations into one row per grid cell per window
2. Compute derived features: `days_since_last_precipitation`, `cumulative_wind_run_24h`, `drought_index_proxy`
3. One row per grid cell covering all cells (not just fire cells)

**Output columns:**

| Column | Description |
|--------|-------------|
| `grid_id` | H3 hex cell |
| `temperature_2m` | Aggregated air temperature (°C) |
| `relative_humidity_2m` | Aggregated relative humidity (%) |
| `wind_speed_10m` | Aggregated wind speed (km/h) |
| `wind_direction_10m` | Aggregated wind direction (degrees) |
| `precipitation` | Total precipitation (mm) |
| `soil_moisture_0_to_7cm` | Aggregated soil moisture (m³/m³) |
| `vpd` | Aggregated Vapor Pressure Deficit (kPa) |
| `fire_weather_index` | Canadian Fire Weather Index |
| `data_quality_flag` | Data quality indicator |
| `days_since_last_precipitation` | Days since last rain > 1mm |
| `cumulative_wind_run_24h` | Total wind distance over 24h (km) |
| `drought_index_proxy` | Cumulative dryness indicator |

---

## Stage 3: Fusion
**Script:** `scripts/fusion/fuse_features.py`
**Files:**
- `data/processed/fused/fused_features_latest.parquet` — all 55 CA+TX grid cells
- `data/processed/fused/fused_features_ml_latest.parquet` — ML-ready variant with temporal lag applied
- `data/processed/fused/64km/region=*/year=*/month=*/fused_*.parquet` — partitioned archive

How fusion works:
1. **Generate master grid** — H3 mathematically fills CA+TX bounding boxes with ~55 hex cells at 64km resolution. This is the backbone — every cell exists here regardless of fire.
2. **LEFT JOIN FIRMS** — fire cells get fire features; non-fire cells get hardcoded defaults: `active_fire_count=0`, `mean_frp=0`, `max_confidence=0`, `nearest_fire_distance_km=-1`, `fire_detected_binary=0`
3. **LEFT JOIN Weather** — all cells get weather features from Open-Meteo (covers entire grid)
4. **LEFT JOIN Static** — all cells get terrain/vegetation features from LANDFIRE/SRTM
5. **Apply fill strategies** for any remaining nulls:
   - `forward_fill` — use previous window's value (temperature, humidity, wind, soil moisture, vpd, FWI, ndvi)
   - `zero` — assume 0 (precipitation)
6. **ML variant** — applies temporal lag: fire context columns (active_fire_count, frp, etc.) shifted to T-1 to prevent leakage; `fire_detected_binary` stays at T as the label

**Output columns (32 total, one row per grid cell):**

| Column | Source | Type |
|--------|--------|------|
| `grid_id` | Computed (H3) | ID |
| `region` | Computed | Metadata |
| `latitude`, `longitude` | Computed (H3) | Metadata |
| `timestamp` | Computed | Metadata |
| `resolution_km` | Computed | Metadata |
| `temperature_2m` | Open-Meteo | Weather |
| `relative_humidity_2m` | Open-Meteo | Weather |
| `wind_speed_10m` | Open-Meteo | Weather |
| `wind_direction_10m` | Open-Meteo | Weather |
| `precipitation` | Open-Meteo | Weather |
| `soil_moisture_0_to_7cm` | Open-Meteo | Weather |
| `vpd` | Open-Meteo | Weather |
| `fire_weather_index` | Open-Meteo | Weather |
| `days_since_last_precipitation` | Computed from weather | Weather derived |
| `cumulative_wind_run_24h` | Computed from weather | Weather derived |
| `drought_index_proxy` | Computed from weather | Weather derived |
| `fuel_model_fbfm40` | LANDFIRE | Static |
| `canopy_cover_pct` | LANDFIRE | Static |
| `vegetation_type` | LANDFIRE | Static |
| `dominant_fuel_fraction` | LANDFIRE | Static |
| `ndvi` | MODIS (via LANDFIRE) | Static |
| `elevation_m` | SRTM | Static |
| `slope_degrees` | SRTM | Static |
| `aspect_degrees` | SRTM | Static |
| `active_fire_count` | FIRMS | Fire |
| `mean_frp` | FIRMS | Fire |
| `median_frp` | FIRMS | Fire |
| `max_confidence` | FIRMS | Fire |
| `nearest_fire_distance_km` | FIRMS | Fire |
| `fire_detected_binary` | FIRMS | **Label** |
| `data_quality_flag` | Computed | Quality |

---

## Stage 4: Final Export
**Script:** `scripts/export/export_spatial.py`
**File:** `data/processed/64km/region=<region>/year=<year>/month=<month>/features_<date>.parquet`

Same 32 columns as fused output plus one added column:

| Column | Description |
|--------|-------------|
| `date` | Human-readable date string (e.g. `3/31/26`) added for export |

Partitioned by `region`, `year`, `month` for efficient querying.

---

## Key Design Points

| Point | Detail |
|-------|--------|
| Grid system | H3 hexagonal grid, 64km resolution (~35–40 cells for CA, ~20 for TX) |
| Fire label source | NASA FIRMS API — NASA does the thermal thresholding, pipeline just consumes already-classified detections |
| Non-fire zeros | Grid cells absent from FIRMS → `fire_detected_binary=0` filled during fusion |
| Static data | Loaded once from disk, not re-ingested every run |
| Leakage prevention | ML variant uses T-1 fire context features; `fire_detected_binary` stays at T |
| Missing data strategy | Weather gaps → forward-fill from previous window; precipitation → zero; static gaps → NaN (ocean/urban cells) |
| Historical training data | `model-pipeline/historical_data/california_historical.csv` — 5,635 rows, June 2024–Jan 2025, same 33 columns |
