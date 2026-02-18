"""
GCS State Manager
=================
Read/write watchdog state JSON to GCS with conditional writes to prevent
race conditions when multiple Cloud Function invocations run concurrently.

All state is stored under gs://{GCS_BUCKET_NAME}/watchdog/

State schema:
    {
        "mode": "quiet" | "active" | "emergency",
        "last_updated": "2026-02-17T18:00:00Z",
        "last_fire_detected": null | "2026-02-17T18:00:00Z",
        "consecutive_fire_scans": 0,
        "consecutive_no_fire_scans": 0,
        "active_fire_cells": [],         # list of H3 cell IDs currently on fire
        "false_alarm_count_today": 0,
        "emergency_activated_at": null | "2026-02-17T18:00:00Z",
        "revert_at": null | "2026-02-17T18:30:00Z",  # FA revert deadline
        "prior_mode": null | "quiet" | "active"       # mode before FA suppression
    }
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Default state for a fresh deployment
DEFAULT_STATE = {
    "mode": "quiet",
    "last_updated": None,
    "last_fire_detected": None,
    "consecutive_fire_scans": 0,
    "consecutive_no_fire_scans": 0,
    "active_fire_cells": [],
    "false_alarm_count_today": 0,
    "emergency_activated_at": None,
    "revert_at": None,
    "prior_mode": None,
}


def _get_bucket_name() -> str:
    bucket = os.environ.get("GCS_BUCKET_NAME")
    if not bucket:
        raise RuntimeError(
            "GCS_BUCKET_NAME environment variable not set. "
            "Set it in .env or Cloud Function environment."
        )
    return bucket


def read_state(config_path: Optional[str] = None) -> dict:
    """Read current watchdog state from GCS.

    Returns DEFAULT_STATE if no state file exists yet (first run).
    Never raises on missing file — safe to call on cold start.
    """
    from google.cloud import storage
    from scripts.utils.schema_loader import get_registry

    registry = get_registry(config_path)
    gcs_path = registry.config.get("watchdog", {}).get("gcs_paths", {}).get(
        "state", "watchdog/state/current.json"
    )
    bucket_name = _get_bucket_name()

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)

        if not blob.exists():
            logger.info("No watchdog state file found — using defaults (first run)")
            return dict(DEFAULT_STATE)

        data = json.loads(blob.download_as_text())
        # Merge with defaults so new fields added to DEFAULT_STATE don't break old state
        merged = dict(DEFAULT_STATE)
        merged.update(data)
        return merged

    except Exception as e:
        logger.error(f"Failed to read watchdog state from GCS: {e}")
        logger.warning("Falling back to DEFAULT_STATE")
        return dict(DEFAULT_STATE)


def write_state(state: dict, config_path: Optional[str] = None) -> bool:
    """Write watchdog state to GCS.

    Uses if_generation_match for conditional write on updates to prevent
    race conditions when multiple Cloud Function instances run concurrently.
    Returns True on success, False if write was rejected (another instance won).
    """
    from google.cloud import storage
    from google.api_core.exceptions import PreconditionFailed
    from scripts.utils.schema_loader import get_registry

    registry = get_registry(config_path)
    gcs_path = registry.config.get("watchdog", {}).get("gcs_paths", {}).get(
        "state", "watchdog/state/current.json"
    )
    bucket_name = _get_bucket_name()

    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(state, indent=2, default=str)

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)

        # Get current generation for conditional write
        blob.reload() if blob.exists() else None
        generation = blob.generation if blob.exists() else 0

        blob.upload_from_string(
            payload,
            content_type="application/json",
            if_generation_match=generation,
        )
        logger.info(f"Watchdog state written to gs://{bucket_name}/{gcs_path}")
        return True

    except PreconditionFailed:
        logger.warning(
            "State write rejected (generation mismatch) — another invocation "
            "updated state concurrently. This instance's update is skipped."
        )
        return False
    except Exception as e:
        logger.error(f"Failed to write watchdog state: {e}")
        return False


def write_trigger(
    trigger_data: dict,
    config_path: Optional[str] = None,
) -> Optional[str]:
    """Write a pipeline trigger file to GCS.

    The local Airflow watchdog_sensor_dag polls for these files.
    Returns the GCS path of the written trigger, or None on failure.

    Trigger file schema:
        {
            "trigger_id": "uuid4",
            "triggered_at": "2026-02-17T18:00:00Z",
            "trigger_source": "goes_nrt_confirmed" | "emergency" | "manual",
            "resolution_km": 22,
            "regions": ["california"],
            "fire_cells": ["8e283082ddbffff", ...],
            "fire_frp_mw": 75.5,
            "mode": "emergency"
        }
    """
    import uuid
    from google.cloud import storage
    from scripts.utils.schema_loader import get_registry

    registry = get_registry(config_path)
    triggers_prefix = registry.config.get("watchdog", {}).get("gcs_paths", {}).get(
        "triggers", "watchdog/triggers/"
    )
    bucket_name = _get_bucket_name()

    trigger_id = str(uuid.uuid4())
    trigger_data["trigger_id"] = trigger_id
    trigger_data["triggered_at"] = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(trigger_data, indent=2, default=str)

    gcs_path = f"{triggers_prefix.rstrip('/')}/{trigger_id}.json"

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(payload, content_type="application/json")
        logger.info(f"Pipeline trigger written: gs://{bucket_name}/{gcs_path}")
        return gcs_path
    except Exception as e:
        logger.error(f"Failed to write trigger file: {e}")
        return None


def write_false_alarm_record(
    detection_data: dict,
    gate_failed: str,
    config_path: Optional[str] = None,
) -> None:
    """Write a false alarm audit record to GCS for later review."""
    import uuid
    from google.cloud import storage
    from scripts.utils.schema_loader import get_registry

    registry = get_registry(config_path)
    fa_prefix = registry.config.get("watchdog", {}).get("gcs_paths", {}).get(
        "false_alarms", "watchdog/false_alarms/"
    )
    bucket_name = _get_bucket_name()

    record = {
        "record_id": str(uuid.uuid4()),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "gate_failed": gate_failed,
        "detection_data": detection_data,
    }
    gcs_path = f"{fa_prefix.rstrip('/')}/{record['record_id']}.json"

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        bucket.blob(gcs_path).upload_from_string(
            json.dumps(record, indent=2, default=str),
            content_type="application/json",
        )
        logger.info(f"False alarm record written: gs://{bucket_name}/{gcs_path}")
    except Exception as e:
        logger.warning(f"Failed to write false alarm record: {e}")


def write_emergency_log(
    event: str,
    details: dict,
    config_path: Optional[str] = None,
) -> None:
    """Append an emergency lifecycle event to GCS audit log."""
    import uuid
    from google.cloud import storage
    from scripts.utils.schema_loader import get_registry

    registry = get_registry(config_path)
    emg_prefix = registry.config.get("watchdog", {}).get("gcs_paths", {}).get(
        "emergency_log", "watchdog/emergency/"
    )
    bucket_name = _get_bucket_name()

    record = {
        "event_id": str(uuid.uuid4()),
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": details,
    }
    gcs_path = f"{emg_prefix.rstrip('/')}/{record['event_id']}.json"

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        bucket.blob(gcs_path).upload_from_string(
            json.dumps(record, indent=2, default=str),
            content_type="application/json",
        )
    except Exception as e:
        logger.warning(f"Failed to write emergency log: {e}")


def read_industrial_sources(config_path: Optional[str] = None) -> list[dict]:
    """Read industrial heat source exclusion list from GCS.

    Returns a list of dicts:
        [{"name": "Tesoro Refinery", "lat": 37.9, "lon": -122.0, "radius_km": 2.0}, ...]

    Returns empty list if file not found (no exclusions applied — safe default).
    The file is updateable at gs://{bucket}/watchdog/config/industrial_sources.json
    without any code redeployment.
    """
    from google.cloud import storage
    from scripts.utils.schema_loader import get_registry

    registry = get_registry(config_path)
    gcs_path = registry.config.get("watchdog", {}).get("gcs_paths", {}).get(
        "industrial_sources", "watchdog/config/industrial_sources.json"
    )
    bucket_name = _get_bucket_name()

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        if not blob.exists():
            logger.info(
                f"No industrial sources file at gs://{bucket_name}/{gcs_path} — "
                "gate G4 will not exclude any locations"
            )
            return []
        data = json.loads(blob.download_as_text())
        sources = data if isinstance(data, list) else data.get("sources", [])
        logger.info(f"Loaded {len(sources)} industrial exclusion sources")
        return sources
    except Exception as e:
        logger.warning(f"Failed to read industrial sources: {e} — G4 skipped")
        return []


def delete_trigger(gcs_path: str) -> bool:
    """Delete a trigger file after the local Airflow sensor has processed it."""
    from google.cloud import storage

    bucket_name = _get_bucket_name()
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        bucket.blob(gcs_path).delete()
        logger.info(f"Trigger file deleted: gs://{bucket_name}/{gcs_path}")
        return True
    except Exception as e:
        logger.warning(f"Failed to delete trigger {gcs_path}: {e}")
        return False


def list_pending_triggers(config_path: Optional[str] = None) -> list[dict]:
    """List all unprocessed trigger files in GCS.

    Called by the local Airflow watchdog_sensor_dag to find pending triggers.
    Returns list of dicts: [{"gcs_path": "...", "data": {...}}, ...]
    """
    from google.cloud import storage
    from scripts.utils.schema_loader import get_registry

    registry = get_registry(config_path)
    triggers_prefix = registry.config.get("watchdog", {}).get("gcs_paths", {}).get(
        "triggers", "watchdog/triggers/"
    )
    bucket_name = _get_bucket_name()

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=triggers_prefix))
        triggers = []
        for blob in blobs:
            if not blob.name.endswith(".json"):
                continue
            try:
                data = json.loads(blob.download_as_text())
                triggers.append({"gcs_path": blob.name, "data": data})
            except Exception as e:
                logger.warning(f"Skipping malformed trigger {blob.name}: {e}")
        return triggers
    except Exception as e:
        logger.error(f"Failed to list pending triggers: {e}")
        return []
