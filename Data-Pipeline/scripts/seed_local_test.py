"""
Local Test Setup — Synthetic Data Seed
=======================================
Generates minimal but schema-valid test data so you can run the Airflow
DAG end-to-end without real API keys, GCP credentials, or network access.

Usage:
    cd Data-Pipeline
    python scripts/seed_local_test.py

What it creates:
    1. data/raw/firms/firms_raw_YYYY-MM-DD.csv — fake FIRMS fire detections
    2. data/raw/weather/weather_raw_YYYY-MM-DD.csv — fake weather observations
    3. data/static/static_features_64km.parquet — placeholder static features
    4. gcp-key.json — dummy service account file (Docker mount won't fail)
    5. .env — populated with dummy values so docker compose starts

After running this script:
    docker compose up -d
    # Wait for airflow-init to complete, then open http://localhost:8080
    # The DAG will still fail at ingest_firms (API call), but you can:
    #   Option A: Mark ingest tasks as "success" in the UI, let the rest run
    #   Option B: Use the Airflow CLI to test individual tasks (see below)
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXECUTION_DATE = datetime.utcnow() - timedelta(days=1)
DATE_STR = EXECUTION_DATE.strftime("%Y-%m-%d")
RESOLUTION_KM = 64

# California and Texas bounding boxes (same as ingest_firms.py)
# We'll scatter some fake detections in these regions
CA_BOUNDS = {"lat_min": 32.5, "lat_max": 42.0, "lon_min": -124.5, "lon_max": -114.0}
TX_BOUNDS = {"lat_min": 25.8, "lat_max": 36.5, "lon_min": -106.6, "lon_max": -93.5}


def seed_firms_raw():
    """Generate a small but valid FIRMS-format CSV.

    Columns match the NASA FIRMS CSV API response:
    latitude, longitude, brightness, scan, track, acq_date, acq_time,
    satellite, confidence, version, bright_t31, frp, daynight, type
    """
    n_detections = 50
    rng = np.random.default_rng(42)

    # Split detections between CA and TX
    n_ca = n_detections // 2
    n_tx = n_detections - n_ca

    lats = np.concatenate([
        rng.uniform(CA_BOUNDS["lat_min"], CA_BOUNDS["lat_max"], n_ca),
        rng.uniform(TX_BOUNDS["lat_min"], TX_BOUNDS["lat_max"], n_tx),
    ])
    lons = np.concatenate([
        rng.uniform(CA_BOUNDS["lon_min"], CA_BOUNDS["lon_max"], n_ca),
        rng.uniform(TX_BOUNDS["lon_min"], TX_BOUNDS["lon_max"], n_tx),
    ])

    df = pd.DataFrame({
        "latitude": np.round(lats, 5),
        "longitude": np.round(lons, 5),
        "brightness": rng.uniform(300, 400, n_detections),
        "scan": rng.uniform(1.0, 2.0, n_detections),
        "track": rng.uniform(1.0, 2.0, n_detections),
        "acq_date": DATE_STR,
        "acq_time": [f"{h:02d}{m:02d}" for h, m in zip(
            rng.integers(0, 24, n_detections),
            rng.integers(0, 60, n_detections),
        )],
        "satellite": rng.choice(["N", "1"], n_detections),  # VIIRS SNPP / NOAA-20
        "confidence": rng.choice(["l", "n", "h"], n_detections, p=[0.2, 0.5, 0.3]),
        "version": "2.0NRT",
        "bright_t31": rng.uniform(280, 320, n_detections),
        "frp": np.round(rng.exponential(15, n_detections), 1),  # Realistic FRP distribution
        "daynight": rng.choice(["D", "N"], n_detections),
        "type": 0,
    })

    output_dir = DATA_DIR / "raw" / "firms"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"firms_raw_{DATE_STR}.csv"
    df.to_csv(output_path, index=False)
    print(f"  FIRMS raw: {output_path} ({len(df)} rows)")
    return output_path


def seed_weather_raw():
    """Generate a small but valid weather CSV matching ingest_weather output.

    The weather ingestion task outputs a CSV with grid_id + weather columns.
    Since generate_full_grid requires h3/geopandas (heavy), we generate a
    small set of grid_ids that are plausible H3 hex strings.
    """
    rng = np.random.default_rng(42)

    # Generate 30 fake but plausible H3 index strings (resolution 2 = 15-char hex)
    # Real H3 indices are 15 hex chars. We'll use deterministic fake ones.
    n_cells = 30
    grid_ids = [f"822{i:03d}fffffffff"[:15] for i in range(n_cells)]

    lats = np.concatenate([
        rng.uniform(CA_BOUNDS["lat_min"], CA_BOUNDS["lat_max"], n_cells // 2),
        rng.uniform(TX_BOUNDS["lat_min"], TX_BOUNDS["lat_max"], n_cells - n_cells // 2),
    ])
    lons = np.concatenate([
        rng.uniform(CA_BOUNDS["lon_min"], CA_BOUNDS["lon_max"], n_cells // 2),
        rng.uniform(TX_BOUNDS["lon_min"], TX_BOUNDS["lon_max"], n_cells - n_cells // 2),
    ])

    df = pd.DataFrame({
        "grid_id": grid_ids,
        "latitude": np.round(lats, 3),
        "longitude": np.round(lons, 3),
        "temperature_2m": rng.uniform(5, 40, n_cells),            # °C
        "relative_humidity_2m": rng.uniform(10, 95, n_cells),      # %
        "wind_speed_10m": rng.uniform(0, 25, n_cells),             # m/s
        "wind_direction_10m": rng.uniform(0, 360, n_cells),        # degrees
        "precipitation": rng.exponential(2, n_cells),              # mm
        "soil_moisture_0_to_7cm": rng.uniform(0.05, 0.45, n_cells),# m³/m³
        "vpd": rng.uniform(0, 5, n_cells),                        # kPa
        "fire_weather_index": rng.uniform(0, 50, n_cells),         # unitless
        "data_quality_flag": 0,  # 0 = fresh data
    })

    output_dir = DATA_DIR / "raw" / "weather"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"weather_raw_{DATE_STR}.csv"
    df.to_csv(output_path, index=False)
    print(f"  Weather raw: {output_path} ({len(df)} rows)")
    return output_path


def seed_static_features():
    """Generate placeholder static features parquet.

    This mirrors what task_load_static_layers produces (the stub version).
    """
    rng = np.random.default_rng(42)
    n_cells = 30
    grid_ids = [f"822{i:03d}fffffffff"[:15] for i in range(n_cells)]

    df = pd.DataFrame({
        "grid_id": grid_ids,
        "fuel_model_fbfm40": rng.integers(90, 205, n_cells),
        "canopy_cover_pct": rng.uniform(0, 80, n_cells),
        "vegetation_type": rng.integers(1, 15, n_cells),
        "ndvi": rng.uniform(0.1, 0.9, n_cells),
        "elevation_m": rng.uniform(50, 3000, n_cells),
        "slope_degrees": rng.uniform(0, 45, n_cells),
        "aspect_degrees": rng.uniform(0, 360, n_cells),
        "dominant_fuel_fraction": rng.uniform(0.3, 1.0, n_cells),
    })

    output_dir = DATA_DIR / "static"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"static_features_{RESOLUTION_KM}km.parquet"

    # Try parquet first, fall back to CSV if pyarrow not installed locally
    try:
        df.to_parquet(output_path, index=False)
        print(f"  Static features: {output_path} ({len(df)} rows)")
    except ImportError:
        output_path = output_dir / f"static_features_{RESOLUTION_KM}km.csv"
        df.to_csv(output_path, index=False)
        print(f"  Static features: {output_path} ({len(df)} rows, CSV fallback — pyarrow not installed)")
        print(f"    NOTE: Install pyarrow ('pip install pyarrow') for parquet support,")
        print(f"    or ignore this — the Docker container has pyarrow and will handle it.")

    return output_path


def create_dummy_gcp_key():
    """Create a dummy gcp-key.json so the Docker volume mount doesn't fail.

    This is NOT a real service account key. It exists only to prevent
    docker compose from failing on the bind mount.
    """
    dummy_key = {
        "type": "service_account",
        "project_id": "wildfire-mlops-dev-local",
        "private_key_id": "dummy-key-for-local-testing",
        "private_key": "-----BEGIN RSA PRIVATE KEY-----\nDUMMY\n-----END RSA PRIVATE KEY-----\n",
        "client_email": "local-test@wildfire-mlops-dev-local.iam.gserviceaccount.com",
        "client_id": "000000000000000000000",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }

    output_path = PROJECT_ROOT / "gcp-key.json"
    if output_path.exists():
        print(f"  gcp-key.json: already exists, skipping (remove manually to regenerate)")
        return output_path

    with open(output_path, "w",encoding="utf-8") as f:
        json.dump(dummy_key, f, indent=2)
    print(f"  gcp-key.json: {output_path} (dummy — local testing only)")
    return output_path


def create_dot_env():
    """Create .env with dummy values so docker compose starts."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        print(f"  .env: already exists, skipping (remove manually to regenerate)")
        return env_path

    env_content = """# Auto-generated for local testing — NOT for production
FIRMS_MAP_KEY=DUMMY_KEY_FOR_LOCAL_TESTING
GCS_BUCKET_NAME=wildfire-mlops-dev
GCP_KEY_PATH=./gcp-key.json
SLACK_WEBHOOK_URL=
"""
    with open(env_path, "w",encoding="utf-8") as f:
        f.write(env_content)
    print(f"  .env: {env_path} (dummy values for local testing)")
    return env_path


def main():
    print("=" * 60)
    print("Wildfire Pipeline — Local Test Seed")
    print("=" * 60)
    print(f"Execution date: {DATE_STR}")
    print(f"Resolution: {RESOLUTION_KM} km")
    print()

    print("Creating seed data...")
    firms_path = seed_firms_raw()
    weather_path = seed_weather_raw()
    static_path = seed_static_features()
    print()

    print("Creating dummy credential files...")
    create_dummy_gcp_key()
    create_dot_env()
    print()

    # Create empty dirs that Airflow expects
    for subdir in ["processed/firms", "processed/weather", "processed/fused", "processed/64km"]:
        (DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    print("Created empty output directories.")
    print()

    print("=" * 60)
    print("READY. Next steps:")
    print("=" * 60)
    print()
    print("1. Start the stack:")
    print("     docker compose up -d")
    print()
    print("2. Wait for init to finish:")
    print("     docker compose logs -f airflow-init")
    print()
    print("3. Open the Airflow UI:")
    print("     http://localhost:8080  (airflow / airflow)")
    print()
    print("4. Test individual tasks via CLI (recommended):")
    print(f"     docker compose exec airflow-scheduler airflow tasks test \\")
    print(f"       wildfire_data_pipeline process_firms {DATE_STR}")
    print()
    print("   This bypasses ingest and feeds the seed data directly.")
    print("   Test in order: process_firms → process_weather → fuse_features")
    print("                → validate_schema → detect_anomalies → export_to_parquet")
    print()
    print("5. Or trigger the full DAG and mark ingestion tasks as success:")
    print("     - Unpause 'wildfire_data_pipeline' in the UI")
    print("     - Trigger a manual run")
    print("     - ingest_firms and ingest_weather will fail (no API access)")
    print("     - Click each failed task → 'Mark Success'")
    print("     - Downstream tasks will pick up the seed data and run")
    print()
    print(f"Seed data paths (for XCom overrides if needed):")
    print(f"  firms_raw_path:         {firms_path}")
    print(f"  weather_raw_path:       {weather_path}")
    print(f"  static_features_path:   {static_path}")


if __name__ == "__main__":
    main()