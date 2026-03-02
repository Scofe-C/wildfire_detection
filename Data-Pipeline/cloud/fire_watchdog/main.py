"""
Cloud Function: fire_watchdog
==============================
Entry point for the GCP Cloud Function that runs on a Cloud Scheduler trigger.

Responsibilities:
  1. Read current watchdog state from GCS
  2. Determine active mode (quiet / active / emergency) from state + season
  3. Poll FIRMS GOES_NRT for both CA and TX bounding boxes
  4. Run four-gate false alarm check on any detections
  5. On confirmed fire: write GCS trigger file for local Airflow sensor
  6. On false alarm: write FA record, schedule state revert
  7. Update watchdog state in GCS (conditional write — race safe)

Option B architecture:
  Cloud Function does NOT call Airflow REST API.
  Instead it writes a trigger JSON to gs://{bucket}/watchdog/triggers/
  Local Airflow watchdog_sensor_dag polls that prefix every 60 seconds.

Environment variables (set in Cloud Function config or .env.yaml):
  FIRMS_MAP_KEY          — NASA FIRMS API key
  GCS_BUCKET_NAME        — GCS bucket for state, triggers, manifests
  SLACK_WEBHOOK_URL      — Optional Slack webhook for emergency alerts
  GOOGLE_CLOUD_PROJECT   — GCP project ID (auto-set by Cloud Functions runtime)

Deploy:
  See cloud/deploy.sh for one-command deployment.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

# Cloud Function runtime does not have the pipeline's sys.path.
# We use inline imports and a minimal dependency set.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CA and TX bounding boxes — duplicated here to avoid importing schema_loader
# in the Cloud Function (keeps cold start fast, no heavy deps)
REGION_BBOXES = {
    "california": [-124.48, 32.53, -114.13, 42.01],
    "texas":      [-106.65, 25.84,  -93.51, 36.50],
}

# Fire season months — duplicated from schema_config.yaml
FIRE_SEASON_MONTHS = [6, 7, 8, 9, 10, 11]


def fire_watchdog(request=None, context=None):
    """Cloud Function entry point.

    Called by Cloud Scheduler (HTTP trigger) or Pub/Sub (event trigger).
    Both invocation styles are supported — request body is optional.

    Args:
        request: flask.Request (HTTP trigger) or None (Pub/Sub).
        context: Cloud Functions context object (Pub/Sub trigger).

    Returns:
        JSON response dict with watchdog result.
    """
    run_id = _new_run_id()
    logger.info(f"=== fire_watchdog invocation {run_id} ===")

    try:
        # ------------------------------------------------------------------
        # 1. Load dependencies (deferred to avoid slow cold starts on quiet polls)
        # ------------------------------------------------------------------
        from google.cloud import storage
        client = storage.Client()
        bucket_name = _require_env("GCS_BUCKET_NAME")

        # Load watchdog config from GCS (avoid importing schema_loader)
        watchdog_config = _load_watchdog_config(client, bucket_name)

        # ------------------------------------------------------------------
        # 2. Read current state
        # ------------------------------------------------------------------
        state = _read_state(client, bucket_name, watchdog_config)
        current_mode = state.get("mode", "quiet")

        # Determine if we should be in active mode based on season
        now = datetime.now(timezone.utc)
        in_fire_season = now.month in FIRE_SEASON_MONTHS
        if current_mode == "quiet" and in_fire_season:
            current_mode = "active"
            state["mode"] = "active"
            logger.info("Transitioning to 'active' mode (fire season)")

        logger.info(f"Current mode: {current_mode}, fire season: {in_fire_season}")

        # ------------------------------------------------------------------
        # 3. Poll GOES_NRT for each region (CA and TX run sequentially here;
        #    in production consider two separate Cloud Function invocations)
        # ------------------------------------------------------------------
        poll_config = watchdog_config.get("goes_nrt", {})
        lookback_min = poll_config.get("client_filter_minutes", 60)
        min_frp = poll_config.get("min_frp_mw", 10.0)
        api_key = os.environ.get("FIRMS_MAP_KEY")

        all_results = {}

        for region, bbox in REGION_BBOXES.items():
            logger.info(f"Polling GOES NRT for {region}: {bbox}")
            detections = _fetch_goes_nrt(
                api_key=api_key,
                bbox=bbox,
                lookback_minutes=lookback_min,
                min_frp_mw=min_frp,
            )
            all_results[region] = detections
            logger.info(f"{region}: {len(detections)} raw GOES detections")

        # ------------------------------------------------------------------
        # 4. Run four-gate false alarm check per region
        # ------------------------------------------------------------------
        industrial_sources = _load_industrial_sources(client, bucket_name, watchdog_config)

        confirmed_regions = {}
        fa_regions = {}

        for region, detections in all_results.items():
            if not detections:
                continue

            gate_result = _run_gates(
                detections=detections,
                region=region,
                state=state,
                watchdog_config=watchdog_config,
                industrial_sources=industrial_sources,
                client=client,
                bucket_name=bucket_name,
            )

            if gate_result["confirmed"]:
                confirmed_regions[region] = gate_result
                logger.info(
                    f"✓ Fire CONFIRMED in {region}: "
                    f"{len(gate_result['fire_cells'])} cells, "
                    f"FRP={gate_result['max_frp']:.0f} MW"
                )
            else:
                fa_regions[region] = gate_result
                logger.info(
                    f"✗ False alarm in {region} at gate {gate_result['gate_failed']}"
                )
                # Write FA record
                _write_false_alarm(
                    client, bucket_name, watchdog_config,
                    region, gate_result
                )

        # ------------------------------------------------------------------
        # 5. Update state and write triggers for confirmed detections
        # ------------------------------------------------------------------
        trigger_paths = []

        for region, gate_result in confirmed_regions.items():
            # Check / update emergency status
            from scripts.detection.emergency import evaluate_emergency, get_pipeline_params_for_mode

            # Inline gcs_state proxy
            state = evaluate_emergency(
                state=state,
                confirmed_cells=gate_result["fire_cells"],
                max_frp=gate_result["max_frp"],
                watchdog_config=watchdog_config,
                gcs_state=None,  # emergency log written separately below
            )

            # Write emergency log if activated
            if state.get("mode") == "emergency" and not state.get("emergency_activated_at_logged"):
                _write_emergency_log(client, bucket_name, watchdog_config, "activated", {
                    "region": region,
                    "fire_cells": gate_result["fire_cells"],
                    "frp_mw": gate_result["max_frp"],
                    "run_id": run_id,
                })
                state["emergency_activated_at_logged"] = True

            # Build trigger params
            pipeline_params = get_pipeline_params_for_mode(
                mode=state["mode"],
                fire_cells=gate_result["fire_cells"],
                watchdog_config=watchdog_config,
                region=region,
            )
            pipeline_params["run_id"] = run_id

            # Write trigger file for local Airflow sensor
            trigger_path = _write_trigger(
                client, bucket_name, watchdog_config, pipeline_params
            )
            if trigger_path:
                trigger_paths.append(trigger_path)
                logger.info(f"Trigger written: gs://{bucket_name}/{trigger_path}")

        # Handle false alarm reverts
        if fa_regions and not confirmed_regions:
            revert_cfg = watchdog_config.get("false_alarm", {})
            revert_minutes = revert_cfg.get("revert_after_minutes", 30)
            from datetime import timedelta
            revert_at = (datetime.now(timezone.utc) + timedelta(minutes=revert_minutes)).isoformat()
            state["revert_at"] = revert_at
            state["prior_mode"] = state.get("mode", current_mode)
            logger.info(
                f"False alarm(s) detected — will revert to {state['prior_mode']} "
                f"mode at {revert_at}"
            )

        # Check if a pending revert has elapsed
        revert_at = state.get("revert_at")
        if revert_at:
            try:
                revert_dt = datetime.fromisoformat(revert_at.replace("Z", "+00:00"))
                if datetime.now(timezone.utc) >= revert_dt:
                    prior = state.get("prior_mode", "quiet")
                    logger.info(f"Revert timer elapsed — reverting to mode '{prior}'")
                    state["mode"] = prior
                    state["revert_at"] = None
                    state["prior_mode"] = None
            except Exception as e:
                logger.warning(f"Could not parse revert_at: {e}")

        # Update consecutive scan counters
        if confirmed_regions:
            state["consecutive_fire_scans"] = state.get("consecutive_fire_scans", 0) + 1
            state["consecutive_no_fire_scans"] = 0
            state["last_fire_detected"] = datetime.now(timezone.utc).isoformat()
        else:
            state["consecutive_no_fire_scans"] = state.get("consecutive_no_fire_scans", 0) + 1
            state["consecutive_fire_scans"] = 0

        # ------------------------------------------------------------------
        # 6. Write updated state (conditional write — race safe)
        # ------------------------------------------------------------------
        _write_state(client, bucket_name, watchdog_config, state)

        result = {
            "run_id": run_id,
            "mode": state.get("mode"),
            "confirmed_regions": list(confirmed_regions.keys()),
            "false_alarm_regions": list(fa_regions.keys()),
            "trigger_files_written": len(trigger_paths),
            "trigger_paths": trigger_paths,
        }
        logger.info(f"=== watchdog complete: {result} ===")
        return result

    except Exception as e:
        logger.error(f"fire_watchdog FAILED: {e}", exc_info=True)
        return {"error": str(e), "run_id": run_id}


# ---------------------------------------------------------------------------
# Helper functions (inline — no imports from pipeline scripts to keep
# Cloud Function dependencies minimal and cold start fast)
# ---------------------------------------------------------------------------

def _new_run_id() -> str:
    import uuid
    return str(uuid.uuid4())[:8]


def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"Required env var {key} not set")
    return val


def _load_watchdog_config(client, bucket_name: str) -> dict:
    """Load watchdog section from schema_config.yaml stored in GCS.

    Falls back to hardcoded defaults if config not in GCS.
    The config is uploaded during `deploy.sh`.
    """
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob("watchdog/config/schema_config.yaml")
        if blob.exists():
            import yaml
            content = blob.download_as_text()
            full_config = yaml.safe_load(content)
            return full_config.get("watchdog", {})
    except Exception as e:
        logger.warning(f"Could not load watchdog config from GCS: {e}")

    # Hardcoded fallback defaults
    return {
        "modes": {
            "quiet":     {"poll_interval_minutes": 30, "resolution_km": 64},
            "active":    {"poll_interval_minutes": 15, "resolution_km": 64},
            "emergency": {"poll_interval_minutes": 5,  "resolution_km": 22},
        },
        "fire_season_months": [6, 7, 8, 9, 10, 11],
        "goes_nrt": {"client_filter_minutes": 60, "min_frp_mw": 10.0},
        "false_alarm": {
            "min_neighbor_detections": 2,
            "min_consecutive_scans": 2,
            "viirs_bypass_frp_mw": 50.0,
            "revert_after_minutes": 30,
        },
        "emergency": {
            "min_frp_mw": 200.0,
            "min_expanding_scans": 2,
            "deactivate_no_expansion_scans": 3,
            "deactivate_frp_mw": 50.0,
            "deactivate_low_frp_scans": 2,
        },
        "detection": {"h3_ring_min": 1, "h3_ring_max": 5, "max_range_km": 25},
        "gcs_paths": {
            "state": "watchdog/state/current.json",
            "triggers": "watchdog/triggers/",
            "false_alarms": "watchdog/false_alarms/",
            "emergency_log": "watchdog/emergency/",
            "industrial_sources": "watchdog/config/industrial_sources.json",
        },
    }


def _read_state(client, bucket_name: str, watchdog_config: dict) -> dict:
    default = {
        "mode": "quiet", "last_updated": None, "last_fire_detected": None,
        "consecutive_fire_scans": 0, "consecutive_no_fire_scans": 0,
        "active_fire_cells": [], "false_alarm_count_today": 0,
        "emergency_activated_at": None, "revert_at": None, "prior_mode": None,
        "consecutive_expanding_scans": 0, "consecutive_no_expansion_scans": 0,
        "consecutive_low_frp_scans": 0,
    }
    gcs_path = watchdog_config.get("gcs_paths", {}).get("state", "watchdog/state/current.json")
    try:
        blob = client.bucket(bucket_name).blob(gcs_path)
        if not blob.exists():
            return default
        data = json.loads(blob.download_as_text())
        merged = dict(default)
        merged.update(data)
        return merged
    except Exception as e:
        logger.warning(f"State read failed: {e} — using defaults")
        return default


def _write_state(client, bucket_name: str, watchdog_config: dict, state: dict) -> bool:
    from google.api_core.exceptions import PreconditionFailed
    gcs_path = watchdog_config.get("gcs_paths", {}).get("state", "watchdog/state/current.json")
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(state, indent=2, default=str)
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.reload() if blob.exists() else None
        generation = blob.generation if blob.exists() else 0
        blob.upload_from_string(
            payload, content_type="application/json",
            if_generation_match=generation,
        )
        return True
    except PreconditionFailed:
        logger.warning("State write rejected (concurrent update) — skipping")
        return False
    except Exception as e:
        logger.error(f"State write failed: {e}")
        return False


def _fetch_goes_nrt(api_key, bbox, lookback_minutes, min_frp_mw) -> list[dict]:
    """Lightweight FIRMS GOES_NRT fetch (inline to keep CF deps minimal)."""
    import io, time
    import pandas as pd
    import requests

    if not api_key:
        logger.warning("No FIRMS_MAP_KEY — GOES NRT skipped")
        return []

    west, south, east, north = bbox
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv"
        f"/{api_key}/GOES_NRT/{west},{south},{east},{north}/1"
    )
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)

    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code == 200:
                content = resp.text.strip()
                if not content or content.startswith("No data") or len(content.splitlines()) <= 1:
                    return []
                df = pd.read_csv(io.StringIO(content))
                if df.empty:
                    return []
                # Parse datetime
                if "acq_date" in df.columns and "acq_time" in df.columns:
                    df["acq_time_str"] = df["acq_time"].astype(str).str.zfill(4)
                    df["acq_datetime"] = pd.to_datetime(
                        df["acq_date"].astype(str) + " " + df["acq_time_str"],
                        format="%Y-%m-%d %H%M", utc=True, errors="coerce"
                    )
                else:
                    df["acq_datetime"] = pd.Timestamp.now(tz="UTC")
                df = df[df["acq_datetime"] >= cutoff]
                if "frp" in df.columns:
                    df = df[df["frp"].fillna(0) >= min_frp_mw]
                lat_col = "latitude" if "latitude" in df.columns else "lat"
                lon_col = "longitude" if "longitude" in df.columns else "lon"
                detections = []
                for _, row in df.iterrows():
                    try:
                        detections.append({
                            "lat": float(row[lat_col]),
                            "lon": float(row[lon_col]),
                            "frp": float(row["frp"]) if "frp" in df.columns and pd.notna(row.get("frp")) else None,
                            "acq_datetime": row["acq_datetime"],
                            "source": "GOES_NRT",
                            "confidence": str(row.get("confidence", "n")),
                        })
                    except Exception:
                        continue
                return detections
            if resp.status_code == 429:
                time.sleep(30 * (attempt + 1))
        except Exception as e:
            logger.warning(f"GOES NRT attempt {attempt+1}: {e}")
            time.sleep(5 * (attempt + 1))
    return []


def _run_gates(detections, region, state, watchdog_config, industrial_sources,
               client, bucket_name) -> dict:
    """Run the four-gate false alarm check."""
    from scripts.detection.fire_detector import FireDetector
    import scripts.utils.gcs_state as gcs_state_module

    detector = FireDetector(
        watchdog_config=watchdog_config,
        state=state,
        gcs_state=gcs_state_module,
    )

    prev_cells_raw = state.get("active_fire_cells", [])
    prev_detections = [{"lat": 0, "lon": 0}] if prev_cells_raw else None

    return detector.evaluate(
        detections=detections,
        region=region,
        previous_scan_detections=prev_detections,
        industrial_sources=industrial_sources,
    )


def _load_industrial_sources(client, bucket_name: str, watchdog_config: dict) -> list:
    gcs_path = watchdog_config.get("gcs_paths", {}).get(
        "industrial_sources", "watchdog/config/industrial_sources.json"
    )
    try:
        blob = client.bucket(bucket_name).blob(gcs_path)
        if not blob.exists():
            return []
        data = json.loads(blob.download_as_text())
        return data if isinstance(data, list) else data.get("sources", [])
    except Exception as e:
        logger.warning(f"Could not load industrial sources: {e}")
        return []


def _write_trigger(client, bucket_name: str, watchdog_config: dict, params: dict) -> str:
    import uuid
    trigger_id = str(uuid.uuid4())
    params["trigger_id"] = trigger_id
    params["triggered_at"] = datetime.now(timezone.utc).isoformat()
    prefix = watchdog_config.get("gcs_paths", {}).get("triggers", "watchdog/triggers/")
    gcs_path = f"{prefix.rstrip('/')}/{trigger_id}.json"
    try:
        client.bucket(bucket_name).blob(gcs_path).upload_from_string(
            json.dumps(params, indent=2, default=str),
            content_type="application/json",
        )
        return gcs_path
    except Exception as e:
        logger.error(f"Failed to write trigger: {e}")
        return ""


def _write_false_alarm(client, bucket_name, watchdog_config, region, gate_result):
    import uuid
    prefix = watchdog_config.get("gcs_paths", {}).get("false_alarms", "watchdog/false_alarms/")
    record = {
        "id": str(uuid.uuid4()),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "region": region,
        "gate_failed": gate_result.get("gate_failed"),
        "summary": gate_result.get("detection_summary", {}),
    }
    gcs_path = f"{prefix.rstrip('/')}/{record['id']}.json"
    try:
        client.bucket(bucket_name).blob(gcs_path).upload_from_string(
            json.dumps(record, indent=2), content_type="application/json"
        )
    except Exception as e:
        logger.warning(f"FA record write failed: {e}")


def _write_emergency_log(client, bucket_name, watchdog_config, event, details):
    import uuid
    prefix = watchdog_config.get("gcs_paths", {}).get("emergency_log", "watchdog/emergency/")
    record = {
        "id": str(uuid.uuid4()),
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": details,
    }
    gcs_path = f"{prefix.rstrip('/')}/{record['id']}.json"
    try:
        client.bucket(bucket_name).blob(gcs_path).upload_from_string(
            json.dumps(record, indent=2), content_type="application/json"
        )
    except Exception as e:
        logger.warning(f"Emergency log write failed: {e}")
