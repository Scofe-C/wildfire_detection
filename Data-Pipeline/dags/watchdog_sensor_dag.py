"""
Watchdog Sensor DAG
====================
Runs continuously (schedule: every 2 minutes) using a PythonSensor in
reschedule mode. Polls GCS for trigger files written by the Cloud Function.

On trigger file found:
  1. Read trigger params (resolution_km, regions, fire_cells, mode)
  2. Trigger wildfire_dag with those params via Airflow REST API (local call)
  3. Delete the trigger file from GCS (prevents double-processing)
  4. Log the trigger event

This DAG never calls external APIs itself — it only reads GCS.
The Cloud Function handles all fire detection logic.

Option B design rationale:
  - No public network exposure required (no ngrok / static IP)
  - Sensor uses mode='reschedule' → releases worker slot between polls
  - max_active_runs=1 prevents multiple sensor instances
  - reschedule mode + 2-min schedule = worst-case 3-min latency after fire confirmed

GCS trigger file schema (written by Cloud Function):
    {
        "trigger_id": "uuid4",
        "triggered_at": "2026-02-17T18:00:00Z",
        "trigger_source": "watchdog_emergency",
        "resolution_km": 22,
        "regions": ["california"],
        "fire_cells": ["8e283082ddbffff"],
        "fire_frp_mw": 250.0,
        "mode": "emergency",
        "detection_range_km": 25,
        "h3_ring_max": 5
    }
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.dates import days_ago

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

DAG_ID_SENSOR   = "watchdog_sensor"
DAG_ID_PIPELINE = "wildfire_data_pipeline"

DEFAULT_RESOLUTION_KM = 22


# ---------------------------------------------------------------------------
# Sensor callable — runs every 2 minutes in reschedule mode
# ---------------------------------------------------------------------------

def check_for_fire_triggers(**context) -> bool:
    """Check GCS for pending trigger files from the Cloud Function.

    Returns True (sensor satisfied) if ≥1 trigger file found and processed.
    Returns False (reschedule) if no triggers pending.

    Uses mode='reschedule' — worker slot is released between calls.
    """
    from scripts.utils.gcs_state import list_pending_triggers, delete_trigger

    triggers = list_pending_triggers()

    if not triggers:
        logger.info("Watchdog sensor: no pending triggers in GCS")
        return False

    logger.info(f"Watchdog sensor: {len(triggers)} pending trigger(s) found")

    # Process the highest-priority trigger (emergency > active > quiet)
    priority_order = {"emergency": 0, "active": 1, "watchdog_emergency": 0,
                      "watchdog_active": 1, "watchdog_quiet": 2}
    triggers_sorted = sorted(
        triggers,
        key=lambda t: priority_order.get(t["data"].get("mode", "quiet"), 99),
    )
    trigger = triggers_sorted[0]
    trigger_data = trigger["data"]
    gcs_path = trigger["gcs_path"]

    logger.info(
        f"Processing trigger {trigger_data.get('trigger_id', '?')}: "
        f"mode={trigger_data.get('mode')}, "
        f"resolution_km={trigger_data.get('resolution_km')}, "
        f"regions={trigger_data.get('regions')}, "
        f"fire_cells={len(trigger_data.get('fire_cells', []))}"
    )

    # Push trigger data to XCom for the process_trigger task
    context["ti"].xcom_push(key="trigger_data", value=trigger_data)
    context["ti"].xcom_push(key="trigger_gcs_path", value=gcs_path)

    return True


def process_fire_trigger(**context):
    """Trigger the main wildfire pipeline with parameters from the GCS trigger file.

    Uses Airflow's internal TriggerDagRunOperator logic via the DagBag API
    — no HTTP call needed, no network exposure.
    """
    from airflow.api.common.trigger_dag import trigger_dag
    from scripts.utils.gcs_state import delete_trigger

    trigger_data = context["ti"].xcom_pull(key="trigger_data")
    gcs_path     = context["ti"].xcom_pull(key="trigger_gcs_path")

    if not trigger_data:
        logger.warning("No trigger_data in XCom — skipping")
        return

    trigger_id   = trigger_data.get("trigger_id", "unknown")
    resolution   = trigger_data.get("resolution_km", DEFAULT_RESOLUTION_KM)
    fire_cells   = trigger_data.get("fire_cells", [])
    mode         = trigger_data.get("mode", "active")
    regions      = trigger_data.get("regions", ["california", "texas"])
    frp          = trigger_data.get("fire_frp_mw", 0.0)
    h3_ring_max  = trigger_data.get("h3_ring_max", 5)
    detect_range = trigger_data.get("detection_range_km", 25)

    run_id = f"watchdog__{trigger_id}__{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"

    conf = {
        "resolution_km": resolution,
        "trigger_source": trigger_data.get("trigger_source", "watchdog"),
        "fire_cells": fire_cells,
        "mode": mode,
        "regions": regions,
        "fire_frp_mw": frp,
        "h3_ring_max": h3_ring_max,
        "detection_range_km": detect_range,
        "triggered_by_watchdog": True,
        "original_trigger_id": trigger_id,
    }

    try:
        trigger_dag(
            dag_id=DAG_ID_PIPELINE,
            run_id=run_id,
            conf=conf,
            replace_microseconds=False,
        )
        logger.info(
            f"✓ Triggered {DAG_ID_PIPELINE} with run_id={run_id} "
            f"(mode={mode}, resolution={resolution}km, "
            f"{len(fire_cells)} fire cells, FRP={frp:.0f} MW)"
        )
    except Exception as e:
        logger.error(f"Failed to trigger {DAG_ID_PIPELINE}: {e}")
        raise   # Re-raise so the task fails and retries

    # Delete the processed trigger file from GCS
    if gcs_path:
        deleted = delete_trigger(gcs_path)
        if deleted:
            logger.info(f"Trigger file deleted: gs://.../{gcs_path}")
        else:
            logger.warning(f"Could not delete trigger file: {gcs_path}")

    # If there are more pending triggers (e.g. both CA and TX fired), log them
    # They'll be processed on the next sensor cycle (2 min)
    from scripts.utils.gcs_state import list_pending_triggers
    remaining = list_pending_triggers()
    if remaining:
        logger.info(
            f"{len(remaining)} additional trigger(s) pending — "
            f"will be processed on next sensor cycle"
        )


def compute_region_manifest(**context):
    """Compute data sufficiency manifest for model training.

    Runs after a successful pipeline trigger. Reads the latest exported
    parquet files and writes per-region sufficiency stats to GCS.

    Manifest schema:
        {
            "california": {
                "row_count": 45000,
                "fire_event_days": 62,
                "unique_fire_cells": 48,
                "quality_flag_ok_pct": 0.91,
                "sufficient_for_training": true,
                "computed_at": "2026-02-17T18:00:00Z"
            },
            "texas": { ... }
        }
    """
    import pandas as pd
    from pathlib import Path
    from google.cloud import storage

    bucket_name = os.environ.get("GCS_BUCKET_NAME")
    resolution_km = context["params"].get("resolution_km", DEFAULT_RESOLUTION_KM)

    processed_dir = PROJECT_ROOT / "data" / "processed" / f"{resolution_km}km"
    manifest = {}

    thresholds_map = {
        "california": {
            "min_fire_event_days": 50,
            "min_temporal_coverage_days": 90,
            "min_fire_cells": 30,
            "min_quality_flag_pct": 0.80,
        },
        "texas": {
            "min_fire_event_days": 50,
            "min_temporal_coverage_days": 90,
            "min_fire_cells": 30,
            "min_quality_flag_pct": 0.80,
        },
    }

    for region_dir in processed_dir.glob("region=*"):
        region = region_dir.name.replace("region=", "")
        parquet_files = list(region_dir.rglob("*.parquet"))

        if not parquet_files:
            logger.info(f"No parquet files for {region} — skipping manifest")
            continue

        try:
            dfs = [pd.read_parquet(f) for f in parquet_files[:50]]  # cap at 50 files
            df = pd.concat(dfs, ignore_index=True)

            fire_event_days = 0
            unique_fire_cells = 0
            quality_ok_pct = 0.0

            if "date" in df.columns and "fire_detected_binary" in df.columns:
                fire_days_df = df[df["fire_detected_binary"] == 1]
                fire_event_days = fire_days_df["date"].nunique() if "date" in fire_days_df.columns else 0

            if "grid_id" in df.columns and "fire_detected_binary" in df.columns:
                unique_fire_cells = df[df["fire_detected_binary"] == 1]["grid_id"].nunique()

            if "data_quality_flag" in df.columns:
                quality_ok_pct = (df["data_quality_flag"] <= 2).mean()

            thresholds = thresholds_map.get(region, thresholds_map["california"])
            sufficient = (
                fire_event_days >= thresholds["min_fire_event_days"]
                and unique_fire_cells >= thresholds["min_fire_cells"]
                and quality_ok_pct >= thresholds["min_quality_flag_pct"]
            )

            manifest[region] = {
                "row_count": len(df),
                "fire_event_days": fire_event_days,
                "unique_fire_cells": unique_fire_cells,
                "quality_flag_ok_pct": round(quality_ok_pct, 3),
                "sufficient_for_training": sufficient,
                "computed_at": datetime.utcnow().isoformat(),
                "thresholds_used": thresholds,
            }

            logger.info(
                f"Manifest [{region}]: {fire_event_days} fire days, "
                f"{unique_fire_cells} fire cells, "
                f"quality={quality_ok_pct:.1%}, "
                f"sufficient={sufficient}"
            )
        except Exception as e:
            logger.warning(f"Manifest computation failed for {region}: {e}")
            continue

    if not manifest:
        logger.info("No manifest data computed — no parquet files found")
        return

    # Write to GCS
    if bucket_name:
        try:
            client = storage.Client()
            gcs_path = f"watchdog/manifests/manifest_{resolution_km}km.json"
            client.bucket(bucket_name).blob(gcs_path).upload_from_string(
                json.dumps(manifest, indent=2, default=str),
                content_type="application/json",
            )
            logger.info(f"Manifest written to gs://{bucket_name}/{gcs_path}")
        except Exception as e:
            logger.warning(f"Could not write manifest to GCS: {e}")
    else:
        logger.info(f"Manifest (no GCS): {json.dumps(manifest, indent=2)}")

    context["ti"].xcom_push(key="region_manifest", value=manifest)


# ---------------------------------------------------------------------------
# DAG Definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id=DAG_ID_SENSOR,
    default_args={
        "owner": "wildfire-team",
        "depends_on_past": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=1),
        "execution_timeout": timedelta(minutes=5),
    },
    description=(
        "Polls GCS for fire trigger files from Cloud Function watchdog. "
        "Triggers wildfire_dag on confirmed detection."
    ),
    schedule_interval="*/2 * * * *",   # Poll every 2 minutes
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,   # Only one sensor active at a time
    tags=["wildfire", "watchdog", "gcp"],
    params={"resolution_km": DEFAULT_RESOLUTION_KM},
) as dag:

    # PythonSensor with mode='reschedule':
    # - Returns False → releases worker slot, reschedules in poke_interval
    # - Returns True  → sensor satisfied, downstream tasks run
    # - timeout=120s  → fail and retry if no trigger found within 2 min
    #   (the schedule_interval will re-run it anyway)
    check_trigger = PythonSensor(
        task_id="check_for_fire_triggers",
        python_callable=check_for_fire_triggers,
        mode="reschedule",          # Don't hold a worker slot between polls
        poke_interval=60,           # Re-check GCS every 60 seconds
        timeout=120,                # Soft-fail after 2 min if no trigger
        soft_fail=True,             # Don't mark as failed — just skip
        provide_context=True,
    )

    process_trigger_task = PythonOperator(
        task_id="process_fire_trigger",
        python_callable=process_fire_trigger,
        provide_context=True,
    )

    manifest_task = PythonOperator(
        task_id="compute_region_manifest",
        python_callable=compute_region_manifest,
        provide_context=True,
        trigger_rule="all_done",  # Run even if process_trigger had issues
    )

    check_trigger >> process_trigger_task >> manifest_task


if __name__ == "__main__":
    print(f"Watchdog sensor DAG '{DAG_ID_SENSOR}' parsed OK")
    print(f"Tasks: {[t.task_id for t in dag.tasks]}")
