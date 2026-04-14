"""
Wildfire Data Pipeline DAG
==========================
Main Airflow DAG: ingest → process → fuse → validate → detect anomalies → export → version

Schedule: Every 6 hours (00:00, 06:00, 12:00, 18:00 UTC)

Improvements applied (pipeline_improvements_guide.md):
  1a. DEFAULT_RESOLUTION_KM = 22 (H3 res 5)
  1b. Regional sharding via Airflow TaskGroups (CA + TX run in parallel)
  4c. task_export_to_parquet partitions by region/year/month

Architecture:
  Static layers are a shared pre-fusion task (single LANDFIRE/SRTM download).
  Firms and weather are sharded per region inside TaskGroups.
  Fusion waits for: CA TaskGroup + TX TaskGroup + shared static.

  check_static ─────────────────────────────────────────────┐
  [region_ca TaskGroup]                                      │
    ingest_firms_ca → process_firms_ca ────────────────────┤→ fuse → validate → detect → export → version
    ingest_weather_ca → process_weather_ca ────────────────┤
  [region_tx TaskGroup]                                      │
    ingest_firms_tx → process_firms_tx ────────────────────┤
    ingest_weather_tx → process_weather_tx ─────────────────┘

XCom key convention:
  Region-scoped keys: firms_raw_path_{region}, weather_raw_path_{region},
  firms_features_path_{region}, weather_features_path_{region}.
  Shared keys (no suffix): static_features_path, fused_features_path, export_path.

Cross-platform:
  - ShortCircuitOperator: ignore_downstream_trigger_rules=False is explicit.
  - DVC BashOperator: set -euo pipefail + explicit /bin/bash. Works on WSL2,
    macOS Docker, Windows 10 Docker Desktop.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path


from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
import pandas as pd

from dags.utils.slack_notify import (
    notify_slack,
    sla_on_failure_callback,
    sla_on_success_callback,
    notify_anomaly_alert,
)

# ---------------------------------------------------------------------------
# DAG-level configuration
# ---------------------------------------------------------------------------
DAG_ID = "wildfire_data_pipeline"
SCHEDULE_INTERVAL = "0 */6 * * *"  # Fallback cron; watchdog_sensor_dag overrides

# Resolution tiers (watchdog escalation):
#   quiet mode:  64 km (H3 res 2) — coarse default scan, ~200 cells CA+TX
#   fire mode:   22 km (H3 res 5) — fire-confirmed detailed scan, ~800-1000 cells CA
DEFAULT_RESOLUTION_KM = 64  # Matches schema_config.yaml default_resolution_km

# Region definitions — mirrors schema_config.yaml geographic_scope
# Defined here so the DAG can build TaskGroups without reading the config at
# parse time (Airflow parses DAGs frequently; keep parse-time work minimal).
REGIONS = {
    "california": {"bbox": [-124.48, 32.53, -114.13, 42.01]},
    "texas":      {"bbox": [-106.65, 25.84,  -93.51, 36.50]},
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
STATIC_DIR   = DATA_DIR / "static"
LOGS_DIR     = PROJECT_ROOT / "logs"

sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parquet compatibility helper
# ---------------------------------------------------------------------------

def _cast_parquet_compatible(df: "pd.DataFrame") -> "pd.DataFrame":
    """Cast columns to types that older parquet-mr / Groovy RowMaterializer can read.

    pyarrow writes int8 and int64 logical types that pre-1.12 parquet-mr
    does not recognise, causing ExceptionInInitializerError on the Java side.
    Casting to float64 / int32 and writing with version="1.0" keeps the data
    intact while using only encodings the Groovy loader understands.
    """
    import pandas as pd

    df = df.copy()
    safe_casts = {
        "fuel_model_fbfm40":            "float64",  # was int64 from static
        "vegetation_type":              "float64",  # was int64 from static
        "active_fire_count":            "int32",
        "max_confidence":               "int32",
        "fire_detected_binary":         "int32",
        "data_quality_flag":            "int32",    # was int8 — parquet-mr rejects INT8
        "days_since_last_precipitation":"float64",  # was int16 from weather
    }
    for col, dtype in safe_casts.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype, errors="ignore")
    return df

# ---------------------------------------------------------------------------
# Default DAG arguments
# ---------------------------------------------------------------------------
default_args = {
    "owner": "wildfire-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "execution_timeout": timedelta(hours=1),
    "on_failure_callback": sla_on_failure_callback,
    "on_success_callback": sla_on_success_callback,
}

# ---------------------------------------------------------------------------
# Shared static layer tasks (pre-fusion, not inside any TaskGroup)
# ---------------------------------------------------------------------------

def task_check_static_cache(**context):
    """Check if the full-grid static cache exists.

    Returns False (skip load_static_layers) if cache is hot.
    Returns True  (run  load_static_layers) if cache is missing.

    ignore_downstream_trigger_rules=False ensures the skip stays contained —
    it must not propagate past fuse_features (handled by trigger_rule='none_failed').
    """
    resolution_km = context["params"].get("resolution_km", DEFAULT_RESOLUTION_KM)
    cache_path = STATIC_DIR / f"static_features_{resolution_km}km.parquet"

    if cache_path.exists():
        logger.info(f"Static cache found: {cache_path}")
        context["ti"].xcom_push(key="static_features_path", value=str(cache_path))
        return False
    logger.info(f"No static cache at {cache_path} — download needed.")
    return True


def task_load_static_layers(**context):
    """Download and process LANDFIRE + SRTM. Expensive; runs once per resolution."""
    from scripts.processing.process_static import load_and_process_static

    resolution_km = context["params"].get("resolution_km", DEFAULT_RESOLUTION_KM)
    output_path = load_and_process_static(
        resolution_km=resolution_km,
        output_dir=str(STATIC_DIR),
    )
    context["ti"].xcom_push(key="static_features_path", value=str(output_path))
    logger.info(f"Static layers processed → {output_path}")


# ---------------------------------------------------------------------------
# Per-region task callables (Improvement 1b)
# Each callable accepts a `region` kwarg injected via op_kwargs in the TaskGroup.
# ---------------------------------------------------------------------------

def task_ingest_firms(region: str, **context):
    """Fetch FIRMS for a single region (scoped via region kwarg)."""
    from scripts.ingestion.ingest_firms import fetch_firms_data

    execution_date = context["execution_date"]
    resolution_km  = context["params"].get("resolution_km", DEFAULT_RESOLUTION_KM)

    output_path = fetch_firms_data(
        execution_date=execution_date,
        resolution_km=resolution_km,
        lookback_hours=24,
        output_dir=str(RAW_DIR / "firms"),
        region=region,                   # ← scopes to this region's bbox only
    )

    context["ti"].xcom_push(key=f"firms_raw_path_{region}", value=str(output_path))
    logger.info(f"[{region}] FIRMS ingestion complete → {output_path}")


def task_ingest_weather(region: str, **context):
    from scripts.ingestion.ingest_weather import fetch_weather_data
    from scripts.utils.grid_utils import generate_full_grid

    execution_date = context["execution_date"]
    params         = context["params"]
    resolution_km  = params.get("resolution_km", DEFAULT_RESOLUTION_KM)
    default_lookback = 6 if resolution_km == 22 else 24
    lookback_hours = params.get("weather_lookback_hours", default_lookback)
    trigger_source = params.get("trigger_source", "cron")
    fire_cells     = params.get("fire_cells", None)
    h3_ring_max    = params.get("h3_ring_max", 5)

    # At 22km (H3 res 5), full grid = ~800-1000 cells per region.
    # On watchdog triggers: only fetch fire-detected region + focal cells.
    # On cron: fetch only the active region (not the entire grid).
    full_grid = generate_full_grid(resolution_km)
    grid = full_grid[full_grid["region"] == region].copy()

    # If watchdog trigger has fire_cells, filter to fire-active region only
    # to avoid unnecessary API calls for the other region.
    is_watchdog = trigger_source in ("watchdog_emergency", "watchdog_active")
    if is_watchdog and fire_cells:
        fire_regions = full_grid[
            full_grid["grid_id"].isin(fire_cells)
        ]["region"].unique().tolist()
        if fire_regions and region not in fire_regions:
            # This region has no detected fire — skip expensive weather fetch
            # The fuse step will forward-fill from previous cron data.
            logger.info(
                "[%s] No fire detected in this region (fire in %s). "
                "Skipping weather fetch — will forward-fill from last cron run.",
                region, fire_regions,
            )
            # Write empty CSV so downstream tasks don't fail on missing XCom
            out_dir = RAW_DIR / "weather"
            out_dir.mkdir(parents=True, exist_ok=True)
            empty_path = out_dir / f"weather_raw_{region}_skip.csv"
            import pandas as _pd
            _pd.DataFrame(columns=["grid_id", "timestamp"]).to_csv(
                empty_path, index=False,
            )
            context["ti"].xcom_push(
                key=f"weather_raw_path_{region}", value=str(empty_path),
            )
            logger.info(f"[{region}] Weather skip (no fire) → {empty_path}")
            return

    grid_centroids = grid[["grid_id", "latitude", "longitude"]]

    output_path = fetch_weather_data(
        grid_centroids=grid_centroids,
        execution_date=execution_date,
        lookback_hours=lookback_hours,
        output_dir=str(RAW_DIR / "weather"),
        trigger_source=trigger_source,
        fire_cells=fire_cells,
        h3_ring_max=h3_ring_max,
        region=region,
    )

    context["ti"].xcom_push(key=f"weather_raw_path_{region}", value=str(output_path))
    logger.info(f"[{region}] Weather ingestion complete → {output_path}")



def task_process_firms(region: str, **context):
    """Aggregate FIRMS point data to grid features for one region."""
    from scripts.processing.process_firms import process_firms_data
    import shutil

    raw_path      = context["ti"].xcom_pull(key=f"firms_raw_path_{region}")
    resolution_km = context["params"].get("resolution_km", DEFAULT_RESOLUTION_KM)

    firms_features = process_firms_data(
        raw_csv_path=raw_path,
        resolution_km=resolution_km,
    )

    latest_path = PROCESSED_DIR / "firms" / f"firms_features_{region}_latest.parquet"
    previous_path = PROCESSED_DIR / "firms" / f"firms_features_{region}_previous.parquet"
    latest_path.parent.mkdir(parents=True, exist_ok=True)

    # Bug #1 fix: rotate _latest → _previous BEFORE overwriting _latest,
    # so the fusion step can read genuine T-1 data from _previous.
    if latest_path.exists():
        shutil.copy2(str(latest_path), str(previous_path))
        logger.info(f"[{region}] Rotated _latest → _previous for T-1 lag")

    firms_features.to_parquet(latest_path, index=False)

    context["ti"].xcom_push(key=f"firms_features_path_{region}", value=str(latest_path))
    logger.info(f"[{region}] FIRMS processing complete: {len(firms_features)} rows")


def task_process_weather(region: str, **context):
    """Process raw weather CSV into grid-aligned features for one region."""
    from scripts.processing.process_weather import process_weather_data

    raw_path      = context["ti"].xcom_pull(key=f"weather_raw_path_{region}")
    resolution_km = context["params"].get("resolution_km", DEFAULT_RESOLUTION_KM)

    weather_features = process_weather_data(
        raw_csv_path=raw_path,
        resolution_km=resolution_km,
    )

    output_path = PROCESSED_DIR / "weather" / f"weather_features_{region}_latest.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weather_features.to_parquet(output_path, index=False)

    context["ti"].xcom_push(key=f"weather_features_path_{region}", value=str(output_path))
    logger.info(f"[{region}] Weather processing complete: {len(weather_features)} rows")


def task_ingest_goes(region: str, **context):
    """Collect raw GOES NRT fire detections for one region. Non-blocking — failures are logged only."""
    import json
    try:
        from scripts.ingestion.ingest_goes import fetch_goes_nrt_detections

        bbox = REGIONS[region]["bbox"]  # [west, south, east, north]
        detections = fetch_goes_nrt_detections(bbox=bbox)

        output_dir = RAW_DIR / "goes"
        output_dir.mkdir(parents=True, exist_ok=True)
        date_str = context["execution_date"].strftime("%Y%m%dT%H%M%S")
        output_path = output_dir / f"goes_{region}_{date_str}.json"

        with open(output_path, "w") as f:
            json.dump(detections, f)

        context["ti"].xcom_push(key=f"goes_raw_path_{region}", value=str(output_path))
        logger.info(f"[{region}] GOES NRT: {len(detections)} detections → {output_path}")
    except Exception as e:
        logger.warning(f"[{region}] GOES NRT collection failed (non-blocking): {e}")


def task_ingest_hrrr(region: str, **context):
    """Collect raw HRRR wind/weather for one region. Non-blocking — failures are logged only."""
    try:
        from scripts.ingestion.ingest_hrrr import fetch_hrrr_for_focal_grid
        from scripts.utils.grid_utils import generate_full_grid

        execution_date = context["execution_date"]
        resolution_km  = context["params"].get("resolution_km", DEFAULT_RESOLUTION_KM)

        full_grid = generate_full_grid(resolution_km)
        grid = full_grid[full_grid["region"] == region][["grid_id", "latitude", "longitude"]]

        output_path = fetch_hrrr_for_focal_grid(
            focal_grid=grid,
            execution_date=execution_date,
            output_dir=str(RAW_DIR / "hrrr"),
        )
        if output_path:
            context["ti"].xcom_push(key=f"hrrr_raw_path_{region}", value=str(output_path))
            logger.info(f"[{region}] HRRR collection complete → {output_path}")
        else:
            logger.warning(f"[{region}] HRRR returned no data")
    except Exception as e:
        logger.warning(f"[{region}] HRRR collection failed (non-blocking): {e}")


def task_ingest_field_telemetry(**context):
    """Ingest pending field telemetry (drone/firefighter/ICS-209) observations.

    Reads JSON files from ``data/raw/field_telemetry/``, validates, converts
    to DataFrame, and pushes via XCom for the fusion step. Non-blocking.
    """
    try:
        from scripts.ingestion.ingest_field_telemetry import (
            batch_field_telemetry_to_dataframe,
            load_pending_field_telemetry,
        )

        input_dir = DATA_DIR / "raw" / "field_telemetry"
        payloads = load_pending_field_telemetry(input_dir)

        if payloads:
            df = batch_field_telemetry_to_dataframe(payloads)
            output_path = PROCESSED_DIR / "field_telemetry" / "field_telemetry_latest.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, index=False)
            context["ti"].xcom_push(key="field_telemetry_path", value=str(output_path))
            logger.info("Field telemetry: %d observations processed → %s", len(df), output_path)
        else:
            logger.info("No pending field telemetry observations")
    except Exception as e:
        logger.warning("Field telemetry ingestion failed (non-blocking): %s", e)


# ---------------------------------------------------------------------------
# Fusion and downstream tasks (shared — wait for all regions)
# ---------------------------------------------------------------------------

def task_fuse_features(**context):
    """Join all regions data into the unified feature table."""
    from scripts.fusion.fuse_features import fuse_features
    import pandas as pd

    execution_date = context["execution_date"]
    resolution_km  = context["params"].get("resolution_km", DEFAULT_RESOLUTION_KM)

    firms_dfs, weather_dfs = [], []

    for region in REGIONS:
        firms_path   = context["ti"].xcom_pull(key=f"firms_features_path_{region}")
        weather_path = context["ti"].xcom_pull(key=f"weather_features_path_{region}")
        if firms_path:
            df = pd.read_parquet(firms_path)
            df["region"] = region
            firms_dfs.append(df)
        if weather_path:
            df = pd.read_parquet(weather_path)
            df["region"] = region
            weather_dfs.append(df)

    firms_df   = pd.concat(firms_dfs,   ignore_index=True) if firms_dfs   else pd.DataFrame()
    weather_df = pd.concat(weather_dfs, ignore_index=True) if weather_dfs else pd.DataFrame()
    static_path = (
            context["ti"].xcom_pull(task_ids="check_static_cache", key="static_features_path")
            or context["ti"].xcom_pull(task_ids="load_static_layers", key="static_features_path")
    )
    static_df   = pd.read_parquet(static_path) if static_path else pd.DataFrame()

    # --- Load field telemetry (if any) ---
    field_telemetry_path = context["ti"].xcom_pull(
        task_ids="ingest_field_telemetry", key="field_telemetry_path",
    )
    field_telemetry_df = (
        pd.read_parquet(field_telemetry_path)
        if field_telemetry_path and Path(field_telemetry_path).exists()
        else None
    )

    # --- Forward-fill: load previous window's fused output ---
    prev_fused_path = str(PROCESSED_DIR / "fused" / "fused_features_previous.parquet")

    fused = fuse_features(
        firms_features=firms_df,
        weather_features=weather_df,
        static_features=static_df,
        execution_date=pd.Timestamp(str(execution_date)),
        resolution_km=resolution_km,
        previous_fused_path=prev_fused_path,
        field_telemetry=field_telemetry_df,
    )

    # --- Circuit breaker: fail loudly on >80% weather nulls (Item 6) ---
    from scripts.fusion.fuse_features import check_weather_circuit_breaker
    from airflow.exceptions import AirflowFailException
    try:
        check_weather_circuit_breaker(fused, threshold=0.80)
    except ValueError as exc:
        raise AirflowFailException(str(exc)) from exc

    output_path = PROCESSED_DIR / "fused" / "fused_features_latest.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Rotate _latest → _previous before overwriting (Item 5)
    if output_path.exists():
        import shutil
        shutil.copy2(str(output_path), prev_fused_path)

    fused = _cast_parquet_compatible(fused)
    fused.to_parquet(output_path, index=False, version="1.0")
    context["ti"].xcom_push(key="fused_features_path", value=str(output_path))

    # --- ML-ready variant with temporal lag (Plan §Problem 2) ---
    # Bug #1 fix: Load _previous.parquet (genuine T-1 data) instead of
    # _latest.parquet (which was just overwritten with current-window data).
    from scripts.fusion.fuse_features import apply_temporal_lag

    prev_fire_dfs = []
    for region in REGIONS:
        prev_path = (
            PROCESSED_DIR / "firms"
            / f"firms_features_{region}_previous.parquet"
        )
        if prev_path.exists():
            prev_fire_dfs.append(pd.read_parquet(prev_path))
        else:
            logger.info(f"[{region}] No _previous file — first run, T-1 will use defaults")

    prev_fire_df = (
        pd.concat(prev_fire_dfs, ignore_index=True)
        if prev_fire_dfs else None
    )

    has_genuine_lag = prev_fire_df is not None and len(prev_fire_df) > 0
    context["ti"].xcom_push(key="has_genuine_temporal_lag", value=has_genuine_lag)
    if not has_genuine_lag:
        logger.warning(
            "No genuine T-1 fire data available — ML-ready variant will use "
            "default fills for fire context columns. Models trained on this "
            "window should treat it as a cold-start sample."
        )

    ml_fused = apply_temporal_lag(fused, prev_fire_df)

    ml_output_path = PROCESSED_DIR / "fused" / "fused_features_ml_latest.parquet"
    ml_fused = _cast_parquet_compatible(ml_fused)
    ml_fused.to_parquet(ml_output_path, index=False, version="1.0")
    context["ti"].xcom_push(key="fused_ml_features_path", value=str(ml_output_path))

    region_counts = fused["region"].value_counts().to_dict() if "region" in fused.columns else {}
    logger.info(
        f"Fusion: {len(fused)} rows (regions: {region_counts}, res: {resolution_km}km) -> {output_path}"
    )
    logger.info(f"ML-ready variant with temporal lag -> {ml_output_path}")


def task_validate_schema(**context):
    """Run schema validation on the fused dataset."""
    import pandas as pd
    from scripts.utils.schema_loader import get_registry
    from scripts.validation.validate_schema import run_validation

    fused_path    = context["ti"].xcom_pull(key="fused_features_path")
    fused_df      = pd.read_parquet(fused_path)
    registry      = get_registry()
    resolution_km = context["params"].get("resolution_km", DEFAULT_RESOLUTION_KM)

    passed, results = run_validation(fused_df, registry, resolution_km=resolution_km)
    errors = results.get("errors", [])
    warnings = results.get("warnings", [])
    validation_results = {"passed": passed, "errors": errors, "warnings": warnings,
                          "issues": errors + warnings}

    if errors:
        logger.error(
            f"Validation ERRORS ({len(errors)}): " + "; ".join(errors[:5])
        )
    if warnings:
        logger.info(
            f"Validation warnings ({len(warnings)}): " + "; ".join(warnings[:5])
        )
    if not errors and not warnings:
        logger.info("Schema validation passed — all checks OK")

    context["ti"].xcom_push(key="validation_results", value=validation_results)


def task_detect_anomalies(**context):
    """Run seasonal-baseline anomaly detection (soft failure — does not block export)."""
    import pandas as pd
    from scripts.utils.schema_loader import get_registry
    from scripts.validation.detect_anomalies import detect_anomalies

    fused_path = context["ti"].xcom_pull(key="fused_features_path")
    fused_df   = pd.read_parquet(fused_path)
    registry   = get_registry()

    anomalies_found = detect_anomalies(
        fused_df=fused_df,
        registry=registry,
        execution_date=context["execution_date"],
    )

    if anomalies_found:
        logger.warning(
            f"Anomalies in {len(anomalies_found)} features: "
            + ", ".join(a["feature"] for a in anomalies_found)
        )
        notify_anomaly_alert(anomalies_found)
    else:
        logger.info("No anomalies detected")

    context["ti"].xcom_push(key="anomalies", value=anomalies_found)


# _send_anomaly_alert removed — use notify_anomaly_alert from dags.utils.slack_notify


def task_export_to_parquet(**context):
    """Export with region/year/month partitioning (Improvement 4c).

    Output:
      data/processed/22km/region=california/year=2026/month=02/features_2026-02-09.parquet
      data/processed/22km/region=texas/year=2026/month=02/features_2026-02-09.parquet
    """
    import pandas as pd

    # Use ML-ready variant (has lag columns) — falls back to plain fused if missing
    fused_path = (
        context["ti"].xcom_pull(key="fused_ml_features_path")
        or context["ti"].xcom_pull(key="fused_features_path")
    )
    fused_df      = pd.read_parquet(fused_path)
    execution_date = context["execution_date"]
    resolution_km = context["params"].get("resolution_km", DEFAULT_RESOLUTION_KM)

    ts_str = execution_date.strftime("%Y-%m-%dT%H%M")
    year   = execution_date.strftime("%Y")
    month  = execution_date.strftime("%m")

    exported_paths = []

    if "region" in fused_df.columns and fused_df["region"].notna().any():
        for region in fused_df["region"].dropna().unique():
            region_df = fused_df[fused_df["region"] == region].copy()
            region_df["date"] = ts_str

            output_dir = (
                PROCESSED_DIR / f"{resolution_km}km"
                / f"region={region}" / f"year={year}" / f"month={month}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"features_{ts_str}.parquet"
            region_df = _cast_parquet_compatible(region_df)
            region_df.to_parquet(output_path, index=False, version="1.0")
            exported_paths.append(str(output_path))
            logger.info(f"Exported {region}: {len(region_df)} rows → {output_path}")
    else:
        logger.warning("'region' column absent — falling back to legacy date= partition")
        fused_df["date"] = ts_str
        output_dir = PROCESSED_DIR / f"{resolution_km}km" / f"date={ts_str}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "features.parquet"
        fused_df = _cast_parquet_compatible(fused_df)
        fused_df.to_parquet(output_path, index=False, version="1.0")
        exported_paths.append(str(output_path))

    export_root = str(PROCESSED_DIR / f"{resolution_km}km")
    context["ti"].xcom_push(key="export_path",  value=export_root)
    context["ti"].xcom_push(key="export_paths", value=exported_paths)

    # --- OBJ-2: Export plain fused features (no lag columns) partitioned for fire spread ---
    fused_plain_path = context["ti"].xcom_pull(key="fused_features_path")
    fused_exported_paths = []
    if fused_plain_path:
        fused_plain_df = pd.read_parquet(fused_plain_path)
        if "region" in fused_plain_df.columns and fused_plain_df["region"].notna().any():
            for region in fused_plain_df["region"].dropna().unique():
                region_df = fused_plain_df[fused_plain_df["region"] == region].copy()
                region_df["date"] = ts_str
                output_dir = (
                    PROCESSED_DIR / "fused" / f"{resolution_km}km"
                    / f"region={region}" / f"year={year}" / f"month={month}"
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"fused_{ts_str}.parquet"
                region_df = _cast_parquet_compatible(region_df)
                region_df.to_parquet(output_path, index=False, version="1.0")
                fused_exported_paths.append(str(output_path))
                logger.info(f"[OBJ-2] Fused export {region}: {len(region_df)} rows → {output_path}")
    context["ti"].xcom_push(key="fused_export_paths", value=fused_exported_paths)

    # Upload all exported Parquet files to GCS (OBJ-1 ML-ready + OBJ-2 fused)
    bucket_name = os.environ.get("GCS_BUCKET_NAME")
    if bucket_name:
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            for local_path in exported_paths + fused_exported_paths:
                rel_path = Path(local_path).relative_to(PROJECT_ROOT)
                blob = bucket.blob(str(rel_path))
                blob.upload_from_filename(local_path)
                logger.info(f"GCS upload: gs://{bucket_name}/{rel_path}")
        except Exception as e:
            logger.warning(f"GCS upload failed (non-fatal): {e}")
    else:
        logger.info("GCS_BUCKET_NAME not set — skipping GCS upload")



# ---------------------------------------------------------------------------
# DAG Definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    on_failure_callback=notify_slack,
    description="Wildfire data pipeline with regional sharding (CA + TX parallel TaskGroups)",
    schedule_interval=SCHEDULE_INTERVAL,
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,   # DVC lock: never run two instances concurrently
    tags=["wildfire", "mlops", "data-pipeline"],
    params={
        "resolution_km": DEFAULT_RESOLUTION_KM,
        "weather_lookback_hours": 6,   # must match backfill DEFAULT_FREQ_HOURS=6
        # Watchdog trigger params (set by watchdog_sensor_dag on fire detection)
        "trigger_source": "cron",         # "cron" | "watchdog_active" | "watchdog_emergency"
        "fire_cells": [],                 # H3 cell IDs confirmed by watchdog
        "fire_frp_mw": 0.0,              # Max FRP at time of trigger (MW)
        "mode": "quiet",                  # watchdog mode that triggered this run
        "regions": [],                    # if empty, run all regions
        "detection_range_km": 25,         # focal grid outer boundary
        "h3_ring_max": 5,                 # focal grid ring count
        "triggered_by_watchdog": False,
    },
) as dag:

    # ------------------------------------------------------------------
    # Shared static branch (runs in parallel with region TaskGroups)
    # ------------------------------------------------------------------
    check_static_cache = ShortCircuitOperator(
        task_id="check_static_cache",
        python_callable=task_check_static_cache,
        provide_context=True,
        # Skip propagates only to load_static_layers, not beyond.
        # fuse_features handles partial upstream via trigger_rule='none_failed'.
        ignore_downstream_trigger_rules=False,
    )

    load_static_layers = PythonOperator(
        task_id="load_static_layers",
        python_callable=task_load_static_layers,
        provide_context=True,
    )

    check_static_cache >> load_static_layers

    # ------------------------------------------------------------------
    # Regional TaskGroups (Improvement 1b)
    # One TaskGroup per region: ingest_firms + ingest_weather run in
    # parallel within each group; process tasks follow their respective ingest.
    # ------------------------------------------------------------------
    region_task_groups = {}

    for region_key in REGIONS:
        with TaskGroup(group_id=f"region_{region_key}") as tg:

            ingest_f = PythonOperator(
                task_id="ingest_firms",
                python_callable=task_ingest_firms,
                op_kwargs={"region": region_key},
                provide_context=True,
            )

            ingest_w = PythonOperator(
                task_id="ingest_weather",
                python_callable=task_ingest_weather,
                op_kwargs={"region": region_key},
                provide_context=True,
                pool="open_meteo_pool",  # serialize across regions to avoid 429s
            )

            process_f = PythonOperator(
                task_id="process_firms",
                python_callable=task_process_firms,
                op_kwargs={"region": region_key},
                provide_context=True,
            )

            process_w = PythonOperator(
                task_id="process_weather",
                python_callable=task_process_weather,
                op_kwargs={"region": region_key},
                provide_context=True,
            )

            # GOES + HRRR: collect raw data for future use, do not block fusion
            ingest_goes = PythonOperator(
                task_id="ingest_goes",
                python_callable=task_ingest_goes,
                op_kwargs={"region": region_key},
                provide_context=True,
            )

            ingest_hrrr = PythonOperator(
                task_id="ingest_hrrr",
                python_callable=task_ingest_hrrr,
                op_kwargs={"region": region_key},
                provide_context=True,
            )

            # Core path: ingest → process (firms and weather in parallel)
            ingest_f >> process_f
            ingest_w >> process_w
            # GOES + HRRR run independently; failures are skipped, not blocking
            ingest_f >> ingest_goes
            ingest_w >> ingest_hrrr

        region_task_groups[region_key] = tg

    # ------------------------------------------------------------------
    # Field telemetry ingestion (runs in parallel with region TaskGroups)
    # ------------------------------------------------------------------
    ingest_field = PythonOperator(
        task_id="ingest_field_telemetry",
        python_callable=task_ingest_field_telemetry,
        provide_context=True,
        trigger_rule="all_done",  # Non-blocking — runs even if regions fail
    )

    # ------------------------------------------------------------------
    # Fusion — waits for ALL region TaskGroups + shared static + field telemetry
    # trigger_rule='none_failed' handles the static ShortCircuit skip gracefully
    # ------------------------------------------------------------------
    fuse = PythonOperator(
        task_id="fuse_features",
        python_callable=task_fuse_features,
        provide_context=True,
        trigger_rule="none_failed",
    )

    # Connect all branches into fusion
    load_static_layers >> fuse
    ingest_field >> fuse
    for tg in region_task_groups.values():
        tg >> fuse

    # ------------------------------------------------------------------
    # Validation → anomaly detection → export → DVC versioning
    # ------------------------------------------------------------------
    validate = PythonOperator(
        task_id="validate_schema",
        python_callable=task_validate_schema,
        provide_context=True,
    )

    detect_anomalies = PythonOperator(
        task_id="detect_anomalies",
        python_callable=task_detect_anomalies,
        provide_context=True,
        trigger_rule="all_done",  # Runs even if validation raised a warning
    )

    export = PythonOperator(
        task_id="export_to_parquet",
        python_callable=task_export_to_parquet,
        provide_context=True,
    )

    version = BashOperator(
        task_id="version_with_dvc",
        bash_command="""
            # Non-blocking: DVC/git are unavailable inside Docker containers.
            # Wrap in a subshell so any failure exits 0 — the DAG continues.
            (
                set -euo pipefail
                echo "=== DVC version step ==="
                mkdir -p dvc

                # Track ML-ready partitioned data (OBJ-1 input)
                dvc add "data/processed/{{ params.resolution_km }}km" -f
                cp "data/processed/{{ params.resolution_km }}km.dvc" \
                   "dvc/processed_{{ params.resolution_km }}km.dvc"
                echo "DVC pointer updated: dvc/processed_{{ params.resolution_km }}km.dvc"

                # Track plain fused features (OBJ-2 input)
                if [ -d "data/processed/fused/{{ params.resolution_km }}km" ]; then
                    dvc commit data/processed/fused -f
                    echo "DVC commit complete: data/processed/fused"
                else
                    echo "Fused {{ params.resolution_km }}km dir not yet populated — skipping"
                fi

                echo "=== DVC version step complete ==="
                echo "Run on host: git add dvc/ && git commit -m 'data: run {{ execution_date }}' && dvc push"
            ) || echo "DVC versioning skipped (non-fatal — git/DVC not available in this environment)"
        """,
        cwd="/opt/airflow",
        dag=dag,
    )

    # ------------------------------------------------------------------
    # Inference — score all grid cells using the production model.
    # Reads fused_ml_features_path XCom, strips pipeline-only columns,
    # runs full_pipeline(is_inference=True), scores, writes to GCS.
    # trigger_rule="all_done": runs even if version_with_dvc is skipped.
    # ------------------------------------------------------------------
    def task_run_inference(**context):
        import json
        import io
        import os
        import sys
        from datetime import datetime, timezone
        from pathlib import Path as _Path
        import pandas as _pd
        import yaml as _yaml

        # Add model-pipeline to sys.path
        # In Docker: MODEL_PIPELINE_ROOT=/opt/model-pipeline (set in docker-compose.yaml)
        # Locally: falls back to sibling directory of Data-Pipeline
        model_pipeline_root = _Path(
            os.environ.get("MODEL_PIPELINE_ROOT", str(PROJECT_ROOT.parent / "model-pipeline"))
        )
        if str(model_pipeline_root) not in sys.path:
            sys.path.insert(0, str(model_pipeline_root))

        from src.preprocessing.feature_engineering import full_pipeline

        _PIPELINE_ONLY_COLS = [
            "active_fire_count", "mean_frp", "median_frp",
            "max_confidence", "nearest_fire_distance_km",
            "fire_detected_binary",
            "canopy_base_height_m", "canopy_bulk_density", "evt_national_class",
        ]

        def assign_risk_tier(score: float) -> str:
            for tier, lower in [("CRITICAL", 0.65), ("HIGH", 0.365), ("MEDIUM", 0.15)]:
                if score >= lower:
                    return tier
            return "LOW"

        # Resolve input: XCom from same run, or latest exported file on disk
        fused_ml_path = context["ti"].xcom_pull(
            task_ids="fuse_features", key="fused_ml_features_path"
        )

        if not fused_ml_path:
            logger.info("run_inference: XCom empty — scanning export dir for latest file")
            resolution_km = context["params"].get("resolution_km", DEFAULT_RESOLUTION_KM)
            export_root = PROCESSED_DIR / f"{resolution_km}km"
            candidates = sorted(
                export_root.glob("region=*/year=*/month=*/features_*.parquet"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                logger.error("run_inference: no exported parquet files found under %s — aborting", export_root)
                return
            latest_mtime = candidates[0].stat().st_mtime
            latest_files = [p for p in candidates if p.stat().st_mtime >= latest_mtime - 60]
            fused_df = _pd.concat(
                [_pd.read_parquet(p) for p in latest_files], ignore_index=True
            )
            logger.info("run_inference: loaded %d rows from %d file(s): %s",
                        len(fused_df), len(latest_files), [p.name for p in latest_files])
        else:
            fused_df = _pd.read_parquet(fused_ml_path)
            logger.info("run_inference: read %d rows from XCom path %s", len(fused_df), fused_ml_path)

        # Save FIRMS aggregates per region BEFORE dropping pipeline-only cols
        firms_by_region = {}
        if "active_fire_count" in fused_df.columns and "region" in fused_df.columns:
            for _r in fused_df["region"].dropna().unique():
                _rdf = fused_df[fused_df["region"] == _r]
                _count = int(_rdf["active_fire_count"].sum())
                _hotspots = []
                if _count > 0 and "mean_frp" in _rdf.columns:
                    for _, _row in _rdf[_rdf["active_fire_count"] > 0].iterrows():
                        _hotspots.append({
                            "lat": float(_row.get("latitude", 0)),
                            "lon": float(_row.get("longitude", 0)),
                            "frp": float(_row.get("mean_frp", 0)),
                            "confidence": float(_row.get("max_confidence", 80)),
                        })
                firms_by_region[_r] = {"count": _count, "hotspots": _hotspots}

        drop_cols = [c for c in _PIPELINE_ONLY_COLS if c in fused_df.columns]
        if drop_cols:
            fused_df = fused_df.drop(columns=drop_cols)

        cfg_path = model_pipeline_root / "configs" / "model_config.yaml"
        with open(cfg_path, encoding="utf-8") as _f:
            cfg = _yaml.safe_load(_f)

        bucket = os.environ.get("GCS_BUCKET_NAME", cfg["data"]["gcs_bucket"])
        run_timestamp = datetime.now(timezone.utc)

        from src.tracking.vertex_registry import VertexRegistry
        vai = cfg["tracking"]["vertex_ai"]
        project_id = os.environ.get("GCP_PROJECT_ID", vai.get("project_id", ""))
        location = vai.get("location", "us-central1")

        all_scored = []
        all_critical = []
        inference_xcom = {}  # {region -> enriched inference dict} — written to GCS + XCom

        for region in fused_df["region"].dropna().unique():
            logger.info("run_inference: scoring region=%s", region)
            region_df = fused_df[fused_df["region"] == region].copy()

            try:
                registry = VertexRegistry(
                    project_id=project_id,
                    location=location,
                    display_name=f"wildfire-ignition-{region}",
                    gcs_bucket=bucket,
                )
                model, medians, threshold = registry.load_production()
                logger.info("[%s] Loaded Vertex AI production model, threshold=%.4f", region, threshold)
            except Exception as exc:
                logger.error("[%s] Model load failed: %s — skipping", region, exc)
                continue

            try:
                X, _ = full_pipeline(region_df, model_type="xgb", is_inference=True, fit_medians=medians)
            except Exception as exc:
                logger.error("[%s] Preprocessing failed: %s — skipping", region, exc)
                continue

            import xgboost as _xgb
            import lightgbm as _lgb
            try:
                if isinstance(model, _xgb.Booster):
                    y_prob = model.predict(_xgb.DMatrix(X))
                elif isinstance(model, _lgb.Booster):
                    y_prob = model.predict(X)
                elif hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X)[:, 1]
                else:
                    y_prob = model.predict(X)
            except Exception as exc:
                logger.error("[%s] Scoring failed: %s — skipping", region, exc)
                continue

            id_cols = ["grid_id", "region"]
            for opt in ("latitude", "longitude"):
                if opt in region_df.columns:
                    id_cols.append(opt)
            scored_df = region_df[id_cols].copy().reset_index(drop=True)
            scored_df["timestamp"]       = run_timestamp
            scored_df["fire_risk_score"] = y_prob
            scored_df["fire_risk_flag"]  = (y_prob >= threshold).astype(int)
            scored_df["risk_tier"]       = [assign_risk_tier(s) for s in y_prob]
            scored_df["model_version"]   = "production"
            scored_df["threshold_used"]  = threshold

            n_flagged = int(scored_df["fire_risk_flag"].sum())
            n_crit    = int((scored_df["risk_tier"] == "CRITICAL").sum())
            logger.info("[%s] flagged=%d  CRITICAL=%d  max_score=%.4f",
                        region, n_flagged, n_crit, float(scored_df["fire_risk_score"].max()))

            # Build cells_list once — reused by GCS write
            _cell_cols = ["grid_id", "fire_risk_score", "fire_risk_flag", "risk_tier"] + (
                ["latitude", "longitude"] if "latitude" in scored_df.columns else []
            )
            cells_list = scored_df[_cell_cols].to_dict(orient="records")

            # Weather telemetry from region_df (weather cols survive the drop)
            _telemetry = {}
            if "temperature_2m" in region_df.columns:
                _telemetry["temperature_max"] = round(float(region_df["temperature_2m"].max()), 2)
            if "wind_speed_10m" in region_df.columns:
                _telemetry["wind_speed_mph"] = round(float(region_df["wind_speed_10m"].mean() * 0.6214), 2)
            if "relative_humidity_2m" in region_df.columns:
                _telemetry["relative_humidity"] = round(float(region_df["relative_humidity_2m"].mean()), 2)
            if "soil_moisture_0_to_7cm" in region_df.columns:
                _telemetry["soil_moisture"] = round(float(region_df["soil_moisture_0_to_7cm"].mean()), 4)

            _firms = firms_by_region.get(region, {"count": 0, "hotspots": []})

            # Enriched JSON — single source of truth for OBJ-3 server
            region_payload = {
                "run_timestamp": run_timestamp.isoformat(),
                "model_version": "production",
                "threshold": threshold,
                "region": region,
                "cells": cells_list,
                "summary": {
                    "total_cells":      len(scored_df),
                    "flagged_cells":    n_flagged,
                    "max_risk_score":   float(scored_df["fire_risk_score"].max()),
                    "risk_tier_counts": scored_df["risk_tier"].value_counts().to_dict(),
                },
                "firms_hotspot_count": _firms["count"],
                "firms_hotspots":      _firms["hotspots"],
                "telemetry":           _telemetry,
            }
            inference_xcom[region] = region_payload

            try:
                from google.cloud import storage as _gcs
                client = _gcs.Client()
                bkt = client.bucket(bucket)
                ts_str = run_timestamp.strftime("%Y%m%dT%H%MZ")
                year   = run_timestamp.year
                month  = f"{run_timestamp.month:02d}"

                buf = io.BytesIO()
                scored_df.to_parquet(buf, index=False)
                bkt.blob(
                    f"inference/region={region}/year={year}/month={month}/inference_{ts_str}.parquet"
                ).upload_from_string(buf.getvalue(), content_type="application/octet-stream")

                bkt.blob(f"inference/latest/{region}_latest.json").upload_from_string(
                    json.dumps(region_payload, indent=2),
                    content_type="application/json",
                )
                logger.info("[%s] GCS write complete", region)
            except Exception as exc:
                logger.warning("[%s] GCS write failed (non-fatal): %s", region, exc)

            all_scored.append(scored_df)
            if n_crit > 0:
                all_critical.extend(
                    scored_df[scored_df["risk_tier"] == "CRITICAL"][
                        ["grid_id", "region", "fire_risk_score"]
                    ].to_dict(orient="records")
                )

        if not all_scored:
            logger.warning("run_inference: no regions scored successfully")
            return

        if all_critical:
            try:
                from src.notifications.alerter import SlackAlerter
                top = all_critical[0]
                SlackAlerter().alert_critical_fire_risk(
                    region=str(top.get("region", "unknown")),
                    grid_id=str(top.get("grid_id", "unknown")),
                    probability=float(top.get("fire_risk_score", 0.0)),
                )
            except Exception as exc:
                logger.warning("Slack alert failed (non-blocking): %s", exc)

        context["ti"].xcom_push(key="inference_results", value=inference_xcom)
        logger.info("run_inference complete — %d regions, %d CRITICAL cells",
                    len(all_scored), len(all_critical))

        # ── Trigger OBJ-3 server (non-blocking) ──────────────────────────────
        obj3_url = os.environ.get("OBJ3_DASHBOARD_URL", "http://obj3-dashboard:8000")
        scored_regions = list(inference_xcom.keys())
        try:
            import requests as _req
            _res_km = context["params"].get("resolution_km", DEFAULT_RESOLUTION_KM)
            resp = _req.post(
                f"{obj3_url}/api/generate-from-pipeline",
                json={"regions": scored_regions, "bucket": bucket, "resolution_km": _res_km},
                timeout=300,
            )
            if resp.status_code == 200:
                logger.info("OBJ-3 trigger OK: %s", resp.json())
            else:
                logger.warning("OBJ-3 trigger returned %d: %s", resp.status_code, resp.text[:500])
        except Exception as exc:
            logger.warning("OBJ-3 trigger failed (non-blocking): %s", exc)

    run_inference = PythonOperator(
        task_id="run_inference",
        python_callable=task_run_inference,
        provide_context=True,
        trigger_rule="all_done",
    )

    fuse >> validate >> detect_anomalies >> export >> version >> run_inference


# ---------------------------------------------------------------------------
# DAG import validation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"DAG '{DAG_ID}' parsed successfully.")
    print(f"Tasks: {[t.task_id for t in dag.tasks]}")
    print(f"Task count: {len(dag.tasks)}")
    dag.test()