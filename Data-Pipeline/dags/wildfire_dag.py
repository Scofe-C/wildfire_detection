"""
Wildfire Data Pipeline DAG
==========================
Main Airflow DAG that orchestrates the end-to-end data pipeline:
  ingest → process → fuse → validate → detect anomalies → export → version

Owner: Person E
Schedule: Every 6 hours (00:00, 06:00, 12:00, 18:00 UTC)

This file must be importable without errors — if any import fails,
Airflow will silently skip this DAG. The CI pipeline validates this
by running: python dags/wildfire_dag.py
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

# ---------------------------------------------------------------------------
# DAG-level configuration
# ---------------------------------------------------------------------------
DAG_ID = "wildfire_data_pipeline"
SCHEDULE_INTERVAL = "0 */6 * * *"  # Every 6 hours
DEFAULT_RESOLUTION_KM = 64

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
STATIC_DIR = DATA_DIR / "static"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure the project scripts are importable
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
# Task callables
# ---------------------------------------------------------------------------
def task_ingest_firms(**context):
    """Airflow task: Fetch FIRMS active fire detections."""
    from scripts.ingestion.ingest_firms import fetch_firms_data

    execution_date = context["execution_date"]
    resolution_km = context["params"].get("resolution_km", DEFAULT_RESOLUTION_KM)

    output_path = fetch_firms_data(
        execution_date=execution_date,
        resolution_km=resolution_km,
        lookback_hours=24,
        output_dir=str(RAW_DIR / "firms"),
    )

    # Push output path to XCom for downstream tasks
    context["ti"].xcom_push(key="firms_raw_path", value=str(output_path))
    logger.info(f"FIRMS ingestion complete → {output_path}")


def task_ingest_weather(**context):
    """Airflow task: Fetch weather data from Open-Meteo with NWS fallback."""
    from scripts.ingestion.ingest_weather import fetch_weather_data
    from scripts.utils.grid_utils import generate_full_grid

    execution_date = context["execution_date"]
    resolution_km = context["params"].get("resolution_km", DEFAULT_RESOLUTION_KM)

    # Generate grid centroids for weather API queries
    grid = generate_full_grid(resolution_km)
    grid_centroids = grid[["grid_id", "latitude", "longitude"]]

    output_path = fetch_weather_data(
        grid_centroids=grid_centroids,
        execution_date=execution_date,
        lookback_hours=24,
        output_dir=str(RAW_DIR / "weather"),
    )

    context["ti"].xcom_push(key="weather_raw_path", value=str(output_path))
    logger.info(f"Weather ingestion complete → {output_path}")


def task_check_static_cache(**context):
    """Airflow task: Check if static layers are cached.

    Returns True if cache exists (skips download), False if download needed.
    Used with ShortCircuitOperator to conditionally skip load_static_layers.
    """
    resolution_km = context["params"].get("resolution_km", DEFAULT_RESOLUTION_KM)
    cache_path = STATIC_DIR / f"static_features_{resolution_km}km.parquet"

    if cache_path.exists():
        logger.info(f"Static layer cache found: {cache_path}")
        context["ti"].xcom_push(
            key="static_features_path", value=str(cache_path)
        )
        return False  # ShortCircuit: skip downstream load_static_layers
    else:
        logger.info(f"No static layer cache at {cache_path}. Download needed.")
        return True  # Continue to load_static_layers


def task_load_static_layers(**context):
    """Airflow task: Download and process LANDFIRE + SRTM static layers.

    This is the most expensive task but only runs once per resolution level.
    Person C implements the actual processing logic.
    """
    resolution_km = context["params"].get("resolution_km", DEFAULT_RESOLUTION_KM)

    # ---------------------------------------------------------------
    # PERSON C: Implement the static layer processing here.
    # This stub demonstrates the interface contract.
    # ---------------------------------------------------------------
    # from scripts.processing.process_static import load_and_process_static
    #
    # output_path = load_and_process_static(
    #     resolution_km=resolution_km,
    #     output_dir=str(STATIC_DIR),
    # )

    # Placeholder: create empty parquet with expected columns
    import pandas as pd
    from scripts.utils.grid_utils import generate_full_grid

    grid = generate_full_grid(resolution_km)
    static_df = grid[["grid_id"]].copy()
    static_df["fuel_model_fbfm40"] = None
    static_df["canopy_cover_pct"] = None
    static_df["vegetation_type"] = None
    static_df["ndvi"] = None
    static_df["elevation_m"] = None
    static_df["slope_degrees"] = None
    static_df["aspect_degrees"] = None
    static_df["dominant_fuel_fraction"] = None

    output_path = STATIC_DIR / f"static_features_{resolution_km}km.parquet"
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    static_df.to_parquet(output_path, index=False)

    context["ti"].xcom_push(key="static_features_path", value=str(output_path))
    logger.info(f"Static layers processed → {output_path}")


def task_process_firms(**context):
    """Airflow task: Aggregate FIRMS point data to grid-level features."""
    from scripts.processing.process_firms import process_firms_data

    raw_path = context["ti"].xcom_pull(key="firms_raw_path")
    resolution_km = context["params"].get("resolution_km", DEFAULT_RESOLUTION_KM)

    firms_features = process_firms_data(
        raw_csv_path=raw_path,
        resolution_km=resolution_km,
    )

    # Save intermediate result
    output_path = PROCESSED_DIR / "firms" / "firms_features_latest.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    firms_features.to_parquet(output_path, index=False)

    context["ti"].xcom_push(key="firms_features_path", value=str(output_path))


def task_process_weather(**context):
    """Airflow task: Process raw weather data into grid-aligned features.

    Person B implements the actual processing logic including derived
    feature computation.
    """
    raw_path = context["ti"].xcom_pull(key="weather_raw_path")
    resolution_km = context["params"].get("resolution_km", DEFAULT_RESOLUTION_KM)

    # ---------------------------------------------------------------
    # PERSON B: Implement weather processing here.
    # This stub loads the raw data and passes it through.
    # Replace with: from scripts.processing.process_weather import process_weather_data
    # ---------------------------------------------------------------
    import pandas as pd

    weather_df = pd.read_csv(raw_path)

    output_path = PROCESSED_DIR / "weather" / "weather_features_latest.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weather_df.to_parquet(output_path, index=False)

    context["ti"].xcom_push(key="weather_features_path", value=str(output_path))


def task_fuse_features(**context):
    """Airflow task: Join all data sources into the unified feature table."""
    from scripts.fusion.fuse_features import fuse_features
    import pandas as pd

    execution_date = context["execution_date"]
    resolution_km = context["params"].get("resolution_km", DEFAULT_RESOLUTION_KM)

    # Load intermediate results from upstream tasks
    firms_path = context["ti"].xcom_pull(key="firms_features_path")
    weather_path = context["ti"].xcom_pull(key="weather_features_path")
    static_path = context["ti"].xcom_pull(key="static_features_path")

    firms_df = pd.read_parquet(firms_path) if firms_path else pd.DataFrame()
    weather_df = pd.read_parquet(weather_path) if weather_path else pd.DataFrame()
    static_df = pd.read_parquet(static_path) if static_path else pd.DataFrame()

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
    logger.info(f"Feature fusion complete: {len(fused)} rows → {output_path}")


def task_validate_schema(**context):
    """Airflow task: Run Great Expectations validation on fused data.

    Person D implements the full validation suite.
    """
    import pandas as pd
    from scripts.utils.schema_loader import get_registry

    fused_path = context["ti"].xcom_pull(key="fused_features_path")
    fused_df = pd.read_parquet(fused_path)
    registry = get_registry()

    # ---------------------------------------------------------------
    # PERSON D: Replace this stub with Great Expectations validation.
    # from scripts.validation.validate_schema import run_validation
    # result = run_validation(fused_df)
    # ---------------------------------------------------------------

    validation_results = {"passed": True, "issues": []}

    # Basic programmatic validation as a starting point
    # Check non-nullable columns
    for col in registry.get_non_nullable_columns():
        if col in fused_df.columns and fused_df[col].isnull().any():
            null_rate = fused_df[col].isnull().mean()
            validation_results["issues"].append(
                f"Non-nullable column '{col}' has {null_rate:.2%} nulls"
            )
            validation_results["passed"] = False

    # Check value ranges
    for col, rules in registry.get_validation_rules().items():
        if col not in fused_df.columns:
            continue
        if "min" in rules:
            violations = (fused_df[col].dropna() < rules["min"]).sum()
            if violations > 0:
                validation_results["issues"].append(
                    f"Column '{col}': {violations} values below min={rules['min']}"
                )
        if "max" in rules:
            violations = (fused_df[col].dropna() > rules["max"]).sum()
            if violations > 0:
                validation_results["issues"].append(
                    f"Column '{col}': {violations} values above max={rules['max']}"
                )

    # Check null rates
    max_null = registry.max_null_rate
    for col in fused_df.columns:
        null_rate = fused_df[col].isnull().mean()
        if null_rate > max_null:
            validation_results["issues"].append(
                f"Column '{col}': null rate {null_rate:.2%} exceeds threshold {max_null:.0%}"
            )

    if validation_results["issues"]:
        logger.warning(
            f"Validation issues ({len(validation_results['issues'])}): "
            + "; ".join(validation_results["issues"][:5])
        )
    else:
        logger.info("Schema validation passed — all checks OK")

    context["ti"].xcom_push(key="validation_results", value=validation_results)

    if not validation_results["passed"]:
        raise ValueError(
            f"Schema validation failed with {len(validation_results['issues'])} issues. "
            f"First issue: {validation_results['issues'][0]}"
        )


def task_detect_anomalies(**context):
    """Airflow task: Run seasonal-baseline anomaly detection.

    Person D implements the full anomaly detection logic.
    """
    import pandas as pd
    from scripts.utils.schema_loader import get_registry

    fused_path = context["ti"].xcom_pull(key="fused_features_path")
    fused_df = pd.read_parquet(fused_path)
    registry = get_registry()

    execution_date = context["execution_date"]
    current_month = execution_date.month

    # ---------------------------------------------------------------
    # PERSON D: Replace with full seasonal baseline anomaly detection.
    # from scripts.validation.detect_anomalies import detect_anomalies
    # anomalies = detect_anomalies(fused_df, execution_date)
    # ---------------------------------------------------------------

    anomaly_config = registry.anomaly_config
    z_threshold = registry.get_z_threshold(current_month)
    monitored = anomaly_config.get("monitored_features", [])

    anomalies_found = []
    for col in monitored:
        if col not in fused_df.columns:
            continue
        values = fused_df[col].dropna()
        if len(values) < 10:
            continue

        mean_val = values.mean()
        std_val = values.std()
        if std_val == 0:
            continue

        # Check for any values exceeding z-score threshold
        z_scores = ((values - mean_val) / std_val).abs()
        outlier_count = (z_scores > z_threshold).sum()

        if outlier_count > 0:
            anomalies_found.append({
                "feature": col,
                "outlier_count": int(outlier_count),
                "z_threshold": z_threshold,
                "season": "fire_season" if current_month in registry.fire_season_months else "off_season",
            })

    if anomalies_found:
        logger.warning(
            f"Anomalies detected in {len(anomalies_found)} features: "
            + ", ".join(a["feature"] for a in anomalies_found)
        )
        # Send alert (soft failure — does not block pipeline)
        _send_anomaly_alert(anomalies_found)
    else:
        logger.info("No anomalies detected")

    context["ti"].xcom_push(key="anomalies", value=anomalies_found)


def _send_anomaly_alert(anomalies: list[dict]):
    """Send anomaly alert via Slack webhook (if configured)."""
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logger.info("SLACK_WEBHOOK_URL not set — skipping Slack alert")
        return

    try:
        import requests

        message = (
            ":warning: *Wildfire Pipeline Anomaly Alert*\n"
            + "\n".join(
                f"• `{a['feature']}`: {a['outlier_count']} outliers "
                f"(z>{a['z_threshold']}, {a['season']})"
                for a in anomalies
            )
        )
        requests.post(webhook_url, json={"text": message}, timeout=10)
        logger.info("Anomaly alert sent to Slack")
    except Exception as e:
        logger.warning(f"Failed to send Slack alert: {e}")


def task_export_to_parquet(**context):
    """Airflow task: Export validated data to partitioned Parquet on GCS."""
    import pandas as pd

    fused_path = context["ti"].xcom_pull(key="fused_features_path")
    fused_df = pd.read_parquet(fused_path)
    execution_date = context["execution_date"]
    resolution_km = context["params"].get("resolution_km", DEFAULT_RESOLUTION_KM)

    # Add date partition column
    date_str = execution_date.strftime("%Y-%m-%d")
    fused_df["date"] = date_str

    # Write locally (DVC will push to GCS)
    output_dir = PROCESSED_DIR / f"{resolution_km}km" / f"date={date_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "features.parquet"
    fused_df.to_parquet(output_path, index=False)

    context["ti"].xcom_push(key="export_path", value=str(output_path))
    logger.info(f"Exported to {output_path}")


# ---------------------------------------------------------------------------
# DAG Definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="End-to-end wildfire data pipeline: ingest → process → fuse → validate → export",
    schedule_interval=SCHEDULE_INTERVAL,
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,  # Prevent concurrent runs (DVC Git lock conflicts)
    tags=["wildfire", "mlops", "data-pipeline"],
    params={"resolution_km": DEFAULT_RESOLUTION_KM},
) as dag:

    # --- Ingestion tasks (parallel) ---
    ingest_firms = PythonOperator(
        task_id="ingest_firms",
        python_callable=task_ingest_firms,
        provide_context=True,
    )

    ingest_weather = PythonOperator(
        task_id="ingest_weather",
        python_callable=task_ingest_weather,
        provide_context=True,
    )

    # --- Static layer tasks (conditional) ---
    check_static_cache = ShortCircuitOperator(
        task_id="check_static_cache",
        python_callable=task_check_static_cache,
        provide_context=True,
        ignore_downstream_trigger_rules=False,
    )

    load_static_layers = PythonOperator(
        task_id="load_static_layers",
        python_callable=task_load_static_layers,
        provide_context=True,
    )

    # --- Processing tasks (parallel, after respective ingestion) ---
    process_firms = PythonOperator(
        task_id="process_firms",
        python_callable=task_process_firms,
        provide_context=True,
    )

    process_weather = PythonOperator(
        task_id="process_weather",
        python_callable=task_process_weather,
        provide_context=True,
    )

    # --- Fusion ---
    fuse = PythonOperator(
        task_id="fuse_features",
        python_callable=task_fuse_features,
        provide_context=True,
        trigger_rule="none_failed",  # Run even if static cache check short-circuits
    )

    # --- Validation ---
    validate = PythonOperator(
        task_id="validate_schema",
        python_callable=task_validate_schema,
        provide_context=True,
    )

    # --- Anomaly detection (soft failure — does not block export) ---
    detect_anomalies = PythonOperator(
        task_id="detect_anomalies",
        python_callable=task_detect_anomalies,
        provide_context=True,
        trigger_rule="all_done",  # Run even if validation has warnings
    )

    # --- Export ---
    export = PythonOperator(
        task_id="export_to_parquet",
        python_callable=task_export_to_parquet,
        provide_context=True,
    )

    # --- DVC versioning (must be last) ---
    # Strategy (Option B): dvc add + dvc push only.
    # Data is pushed to GCS immediately so nothing is lost.
    # The resulting .dvc metadata files are updated on the host via
    # the volume mount — the developer commits them as part of the
    # normal PR flow. We intentionally do NOT run git commands from
    # inside the container to avoid .git mount complexity and
    # automated-commit footguns.
    #
    # Idempotency: dvc add re-hashes (safe to retry), dvc push skips
    # already-pushed files. The DAG's max_active_runs=1 prevents
    # concurrent DVC operations.
    version = BashOperator(
        task_id="version_with_dvc",
        bash_command="""
            set -euo pipefail

            echo "=== DVC version step ==="

            # Preflight: verify a DVC remote is configured
            if ! dvc remote list | grep -q .; then
                echo "ERROR: No DVC remote configured."
                echo "Run on host: dvc remote add -d gcs_remote gs://<bucket>/dvc-store"
                exit 1
            fi

            # Track the fused feature directory (intermediate)
            echo "Tracking data/processed/fused ..."
            dvc add data/processed/fused

            # Track the exported partitioned output (delivery artifact)
            echo "Tracking data/processed/{{ params.resolution_km }}km ..."
            dvc add data/processed/{{ params.resolution_km }}km

            # Push to GCS remote
            echo "Pushing to remote ..."
            dvc push

            echo "=== DVC version step complete ==="
        """,
        dag=dag,
    )

    # --- Task dependencies ---
    # Three parallel ingestion branches
    ingest_firms >> process_firms
    ingest_weather >> process_weather
    check_static_cache >> load_static_layers

    # Fusion waits for all three branches
    [process_firms, process_weather, load_static_layers] >> fuse

    # Sequential validation → anomaly detection → export → version
    fuse >> validate >> detect_anomalies >> export >> version


# ---------------------------------------------------------------------------
# DAG import validation (run this file directly to check for import errors)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"DAG '{DAG_ID}' parsed successfully.")
    print(f"Tasks: {[t.task_id for t in dag.tasks]}")
    print(f"Task count: {len(dag.tasks)}")
    dag.test()