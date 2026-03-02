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

# ---------------------------------------------------------------------------
# DAG-level configuration
# ---------------------------------------------------------------------------
DAG_ID = "wildfire_data_pipeline"
SCHEDULE_INTERVAL = "0 */6 * * *"  # Fallback cron; watchdog_sensor_dag overrides

# Resolution tiers (Improvement 1a + watchdog escalation):
#   64 km (H3 res 2) — coarse default scan, ~200 cells CA+TX
#   22 km (H3 res 5) — fire-confirmed detailed scan, ~800-1000 cells CA
DEFAULT_RESOLUTION_KM = 64  # Watchdog escalates to 22 on confirmed fire

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
    """Fetch weather for a single region's grid cells."""
    from scripts.ingestion.ingest_weather import fetch_weather_data
    from scripts.utils.grid_utils import generate_grid_for_bbox

    execution_date = context["execution_date"]

    # If resolution_km is missing and we default to 64 km on a watchdog-triggered
    # 22 km run, the weather grid_ids won't match FIRMS grid_ids in fusion →
    # every row joins to null weather features with no error raised.
    # The DAG params dict always sets "resolution_km" as a default, so this
    # assertion only fires if that default is accidentally removed.
    resolution_km = context["params"].get("resolution_km")
    if resolution_km is None:
        raise ValueError(
            "resolution_km is missing from DAG params in task_ingest_weather. "
            "This would silently produce a grid_id mismatch during fusion — "
            "weather features would be null for every row. "
            "Ensure the DAG params dict includes 'resolution_km'."
        )

    # (15-min interval) can request a narrower window (e.g. 2h) for fresher data,
    # while cron runs keep the standard 24h window.
    lookback_hours = context["params"].get("weather_lookback_hours", 24)

    bbox = REGIONS[region]["bbox"]

    # Generate region-specific grid centroids only — no full-grid needed here.
    # generate_grid_for_bbox is cheaper than generate_full_grid at parse time.
    grid = generate_grid_for_bbox(bbox, resolution_km)
    grid_centroids = grid[["grid_id", "latitude", "longitude"]]

    output_path = fetch_weather_data(
        grid_centroids=grid_centroids,
        execution_date=execution_date,
        lookback_hours=lookback_hours,
        output_dir=str(RAW_DIR / "weather"),
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


# ---------------------------------------------------------------------------
# Fusion and downstream tasks (shared — wait for all regions)
# ---------------------------------------------------------------------------

def task_fuse_features(**context):
    """Join all regions data into the unified feature table.

    When triggered by watchdog with confirmed fire cells, generates a focal
    grid (5-25 km detection zone) for dense coverage around the fire.
    Cron-triggered runs use the full regional grid.
    """
    from scripts.fusion.fuse_features import fuse_features
    import pandas as pd

    execution_date = context["execution_date"]
    resolution_km  = context["params"].get("resolution_km", DEFAULT_RESOLUTION_KM)
    fire_cells     = context["params"].get("fire_cells", [])
    h3_ring_max    = context["params"].get("h3_ring_max", 5)
    trigger_source = context["params"].get("trigger_source", "cron")

    firms_dfs, weather_dfs = [], []

    for region in REGIONS:
        firms_path   = context["ti"].xcom_pull(key=f"firms_features_path_{region}")
        weather_path = context["ti"].xcom_pull(key=f"weather_features_path_{region}")
        if firms_path:
            df = pd.read_parquet(firms_path)
            df["region"] = region
            firms_dfs.append(df)
        if weather_path:
            weather_dfs.append(pd.read_parquet(weather_path))

    firms_df   = pd.concat(firms_dfs,   ignore_index=True) if firms_dfs   else pd.DataFrame()
    weather_df = pd.concat(weather_dfs, ignore_index=True) if weather_dfs else pd.DataFrame()
    static_path = (
            context["ti"].xcom_pull(task_ids="check_static_cache", key="static_features_path")
            or context["ti"].xcom_pull(task_ids="load_static_layers", key="static_features_path")
    )
    static_df   = pd.read_parquet(static_path) if static_path else pd.DataFrame()

    # Generate focal grid when watchdog provided fire cells
    if fire_cells and trigger_source != "cron":
        try:
            from scripts.utils.grid_utils import generate_fire_focal_grid
            focal_grid = generate_fire_focal_grid(
                fire_cell_ids=fire_cells, ring_min=1, ring_max=h3_ring_max,
            )
            context["ti"].xcom_push(key="focal_grid_cell_count", value=len(focal_grid))
            logger.info(
                f"Focal grid: {len(focal_grid)} cells "
                f"(fire={sum(focal_grid['cell_type']=='fire')}, "
                f"zone={sum(focal_grid['cell_type']=='detection_zone')})"
            )
        except Exception as e:
            logger.warning(f"Focal grid generation failed: {e}")

    fused = fuse_features(
        firms_features=firms_df,
        weather_features=weather_df,
        static_features=static_df,
        execution_date=pd.Timestamp(str(execution_date)),
        resolution_km=resolution_km,
    )

    output_path = PROCESSED_DIR / "fused" / "fused_features_latest.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fused.to_parquet(output_path, index=False)
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

    # --- Priority resolution (Sprint 3c) ---
    # Apply ground truth overrides if any field telemetry data is available.
    try:
        from scripts.fusion.priority_resolver import resolve_priorities
        ml_fused = resolve_priorities(
            fused_df=ml_fused,
            ground_truth_df=pd.DataFrame(),  # No ground truth during initial test
            config_path=None,
        )
    except Exception as e:
        logger.warning(f"Priority resolution skipped: {e}")

    ml_output_path = PROCESSED_DIR / "fused" / "fused_features_ml_latest.parquet"
    ml_fused.to_parquet(ml_output_path, index=False)
    context["ti"].xcom_push(key="fused_ml_features_path", value=str(ml_output_path))

    region_counts = fused["region"].value_counts().to_dict() if "region" in fused.columns else {}
    logger.info(
        f"Fusion: {len(fused)} rows (regions: {region_counts}, "
        f"src: {trigger_source}, res: {resolution_km}km) -> {output_path}"
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
    validation_results = {"passed": passed, "issues": results.get("issues", [])}

    if validation_results["issues"]:
        logger.warning(
            f"Validation issues ({len(validation_results['issues'])}): "
            + "; ".join(validation_results["issues"][:5])
        )
    else:
        logger.info("Schema validation passed — all checks OK")

    context["ti"].xcom_push(key="validation_results", value=validation_results)

    if not validation_results["passed"]:
        logger.warning(
            f"Schema validation issues (non-fatal): {validation_results['issues'][:5]}"
        )


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
        _send_anomaly_alert(anomalies_found)
    else:
        logger.info("No anomalies detected")

    context["ti"].xcom_push(key="anomalies", value=anomalies_found)


def _send_anomaly_alert(anomalies: list[dict]):
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        return
    try:
        import requests
        msg = (
            ":warning: *Wildfire Pipeline Anomaly Alert*\n"
            + "\n".join(
                f"• `{a['feature']}`: {a['outlier_count']} outliers "
                f"(z>{a['z_threshold']}, {a['season']})"
                for a in anomalies
            )
        )
        requests.post(webhook_url, json={"text": msg}, timeout=10)
    except Exception as e:
        logger.warning(f"Slack alert failed: {e}")


def task_export_to_parquet(**context):
    """Export with region/year/month partitioning (Improvement 4c).

    Output:
      data/processed/22km/region=california/year=2026/month=02/features_2026-02-09.parquet
      data/processed/22km/region=texas/year=2026/month=02/features_2026-02-09.parquet
    """
    import pandas as pd

    fused_path    = context["ti"].xcom_pull(key="fused_features_path")
    fused_df      = pd.read_parquet(fused_path)
    execution_date = context["execution_date"]
    resolution_km = context["params"].get("resolution_km", DEFAULT_RESOLUTION_KM)

    date_str = execution_date.strftime("%Y-%m-%d")
    year     = execution_date.strftime("%Y")
    month    = execution_date.strftime("%m")

    exported_paths = []

    if "region" in fused_df.columns and fused_df["region"].notna().any():
        for region in fused_df["region"].dropna().unique():
            region_df = fused_df[fused_df["region"] == region].copy()
            region_df["date"] = date_str

            output_dir = (
                PROCESSED_DIR / f"{resolution_km}km"
                / f"region={region}" / f"year={year}" / f"month={month}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"features_{date_str}.parquet"
            region_df.to_parquet(output_path, index=False)
            exported_paths.append(str(output_path))
            logger.info(f"Exported {region}: {len(region_df)} rows → {output_path}")
    else:
        logger.warning("'region' column absent — falling back to legacy date= partition")
        fused_df["date"] = date_str
        output_dir = PROCESSED_DIR / f"{resolution_km}km" / f"date={date_str}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "features.parquet"
        fused_df.to_parquet(output_path, index=False)
        exported_paths.append(str(output_path))

    export_root = str(PROCESSED_DIR / f"{resolution_km}km")
    context["ti"].xcom_push(key="export_path",  value=export_root)
    context["ti"].xcom_push(key="export_paths", value=exported_paths)


def task_export_spatial(**context):
    """Track B: Export spatial grid arrays for CNN/GCN models.

    Produces:
      - spatial_grid_{date}.npz: 3D array (H × W × C)
      - adjacency_{date}.npz: sparse COO adjacency matrix
    """
    import pandas as pd
    from scripts.export.export_spatial import export_spatial_grid, export_adjacency_matrix

    fused_ml_path = context["ti"].xcom_pull(key="fused_ml_features_path")
    if not fused_ml_path:
        # Fallback to raw fused if ML-ready variant not available
        fused_ml_path = context["ti"].xcom_pull(key="fused_features_path")

    fused_df = pd.read_parquet(fused_ml_path)
    execution_date = context["execution_date"]
    resolution_km = context["params"].get("resolution_km", DEFAULT_RESOLUTION_KM)
    date_str = execution_date.strftime("%Y-%m-%d")

    output_dir = str(PROCESSED_DIR / "spatial" / f"{resolution_km}km")

    grid_path = export_spatial_grid(
        fused_df, output_dir, resolution_km, date_str
    )
    adj_path = export_adjacency_matrix(
        fused_df, output_dir, resolution_km, date_str
    )

    context["ti"].xcom_push(key="spatial_grid_path", value=str(grid_path))
    context["ti"].xcom_push(key="adjacency_path", value=str(adj_path))
    logger.info(f"Spatial export complete: grid={grid_path}, adj={adj_path}")


# ---------------------------------------------------------------------------
# DAG Definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Wildfire data pipeline with regional sharding (CA + TX parallel TaskGroups)",
    schedule_interval=SCHEDULE_INTERVAL,
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,   # DVC lock: never run two instances concurrently
    tags=["wildfire", "mlops", "data-pipeline"],
    params={
        "resolution_km": DEFAULT_RESOLUTION_KM,
        "weather_lookback_hours": 24,
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

            # Within-group dependencies:
            # ingest runs first, process follows; firms and weather run in parallel
            ingest_f >> process_f
            ingest_w >> process_w

        region_task_groups[region_key] = tg

    # ------------------------------------------------------------------
    # Fusion — waits for ALL region TaskGroups + shared static branch
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

    # Real DVC BashOperator — restored from base (lisun had a logger stub).
    # bash -c is explicit: works on WSL2, macOS Docker, Windows 10 Docker Desktop.
    # Improvement 4c: tracks resolution_km dir tree (covers all region sub-dirs).
    version = BashOperator(
        task_id="version_with_dvc",
        bash_command="""
            set -euo pipefail

            echo "=== DVC version step ==="

            # DVC needs a git repo context
            if [ ! -d .git ] || [ -z "$(ls -A .git)" ]; then
                git init
                git config user.email "airflow@wildfire.local"
                git config user.name "Airflow"
            fi

            if ! dvc remote list | grep -q .; then
                echo "ERROR: No DVC remote configured."
                exit 1
            fi

            echo "Tracking data/processed/fused ..."
            dvc add data/processed/fused -f

            echo "Tracking data/processed/{{ params.resolution_km }}km ..."
            dvc add data/processed/{{ params.resolution_km }}km -f

            echo "Pushing to GCS remote ..."
            dvc push data/processed/fused.dvc data/processed/{{ params.resolution_km }}km.dvc

            echo "=== DVC version step complete ==="
        """,
        cwd="/opt/airflow",
        dag=dag,
    )

    export_spatial = PythonOperator(
        task_id="export_spatial",
        python_callable=task_export_spatial,
        provide_context=True,
    )

    # Track A (tabular) and Track B (spatial) run in parallel after anomaly detection
    fuse >> validate >> detect_anomalies >> [export, export_spatial] >> version


# ---------------------------------------------------------------------------
# DAG import validation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"DAG '{DAG_ID}' parsed successfully.")
    print(f"Tasks: {[t.task_id for t in dag.tasks]}")
    print(f"Task count: {len(dag.tasks)}")
    dag.test()
