"""Field telemetry ingestion — drone, firefighter, and ICS-209 observations.

Accepts ground-truth observations from field sources and converts them
into DataFrames compatible with the fusion pipeline.

Sources:
    - drone: UAV thermal/visual observations with GPS + FRP
    - firefighter: ground crew reports with lat/lon + confidence
    - ics209: structured ICS-209 incident status summary data

Field telemetry has priority=1 (ground_truth) in the data hierarchy,
overriding satellite (priority=2) and model inference (priority=3)
when spatially and temporally overlapping.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_SOURCE_TYPES: list[str] = ["drone", "firefighter", "ics209"]

_REQUIRED_FIELDS = {"source_type", "priority", "latitude", "longitude", "timestamp", "confidence"}

_OPTIONAL_DEFAULTS: dict[str, Any] = {
    "frp": None,
    "report_text": None,
    "spatial_trust_radius_km": 5.0,
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_field_telemetry(payload: Any) -> tuple[bool, list[str]]:
    """Validate a single field telemetry payload.

    Parameters
    ----------
    payload:
        Dictionary with field telemetry data.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list_of_issues). Empty issues list if valid.
    """
    issues: list[str] = []

    if not isinstance(payload, dict):
        return False, ["Payload must be a dictionary"]

    # Check required fields
    for field in _REQUIRED_FIELDS:
        if field not in payload:
            issues.append(f"Missing required field: {field}")

    if issues:
        return False, issues

    # Validate source_type
    if payload["source_type"] not in VALID_SOURCE_TYPES:
        issues.append(
            f"Invalid source_type: {payload['source_type']!r}. "
            f"Must be one of {VALID_SOURCE_TYPES}"
        )

    # Validate ranges
    lat = payload.get("latitude")
    if isinstance(lat, (int, float)) and not (-90 <= lat <= 90):
        issues.append(f"latitude out of range: {lat} (must be -90 to 90)")

    lon = payload.get("longitude")
    if isinstance(lon, (int, float)) and not (-180 <= lon <= 180):
        issues.append(f"longitude out of range: {lon} (must be -180 to 180)")

    conf = payload.get("confidence")
    if isinstance(conf, (int, float)) and not (0 <= conf <= 100):
        issues.append(f"confidence out of range: {conf} (must be 0 to 100)")

    return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# DataFrame conversion
# ---------------------------------------------------------------------------

def field_telemetry_to_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert a single validated field telemetry payload to a 1-row DataFrame.

    Parameters
    ----------
    payload:
        Validated field telemetry dict.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with pipeline-compatible columns.

    Raises
    ------
    ValueError
        If the payload fails validation.
    """
    is_valid, issues = validate_field_telemetry(payload)
    if not is_valid:
        raise ValueError(f"Invalid field telemetry: {'; '.join(issues)}")

    # Apply optional defaults
    for key, default in _OPTIONAL_DEFAULTS.items():
        payload.setdefault(key, default)

    row = {
        "latitude": float(payload["latitude"]),
        "longitude": float(payload["longitude"]),
        "timestamp": payload["timestamp"],
        "source_type": payload["source_type"],
        "confidence": int(payload["confidence"]),
        "frp": payload.get("frp"),
        "report_text": payload.get("report_text"),
        "spatial_trust_radius_km": float(payload.get("spatial_trust_radius_km", 5.0)),
        # Pipeline-compatible columns
        "fire_detected_binary": 1,          # Ground truth = fire confirmed
        "data_source_priority": int(payload.get("priority", 1)),  # 1 = ground truth
        "data_quality_flag": 0,             # 0 = highest quality (direct observation)
    }

    return pd.DataFrame([row])


def batch_field_telemetry_to_dataframe(
    payloads: list[dict[str, Any]],
) -> pd.DataFrame:
    """Convert a batch of field telemetry payloads to a DataFrame.

    Invalid payloads are skipped with a warning (not raised).

    Parameters
    ----------
    payloads:
        List of field telemetry dicts.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame. May be empty if all payloads are invalid.
    """
    frames: list[pd.DataFrame] = []
    for i, payload in enumerate(payloads):
        try:
            df = field_telemetry_to_dataframe(payload)
            frames.append(df)
        except (ValueError, TypeError) as exc:
            logger.warning("Skipping invalid field telemetry [%d]: %s", i, exc)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# File-based ingestion
# ---------------------------------------------------------------------------

def load_pending_field_telemetry(
    input_dir: str | Path,
) -> list[dict[str, Any]]:
    """Load pending field telemetry JSON files from a watched directory.

    Reads all ``*.json`` files from ``input_dir``, validates each payload,
    and moves processed files to ``input_dir/processed/``.

    Parameters
    ----------
    input_dir:
        Directory to scan for JSON telemetry files.

    Returns
    -------
    list[dict]
        List of valid payloads. Invalid files are logged and moved
        to ``input_dir/rejected/``.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        return []

    processed_dir = input_dir / "processed"
    rejected_dir = input_dir / "rejected"
    processed_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    valid_payloads: list[dict[str, Any]] = []

    for json_file in sorted(input_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))

            # Handle both single payload and list of payloads
            items = data if isinstance(data, list) else [data]

            file_valid = 0
            for item in items:
                is_valid, issues = validate_field_telemetry(item)
                if is_valid:
                    valid_payloads.append(item)
                    file_valid += 1
                else:
                    logger.warning(
                        "Invalid payload in %s: %s", json_file.name, issues,
                    )

            # Move to processed (even if some items were invalid)
            dest = processed_dir / json_file.name
            shutil.move(str(json_file), str(dest))
            logger.info(
                "Processed %s: %d/%d valid observations",
                json_file.name, file_valid, len(items),
            )

        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to read %s: %s", json_file.name, exc)
            dest = rejected_dir / json_file.name
            shutil.move(str(json_file), str(dest))

    if valid_payloads:
        logger.info(
            "Loaded %d field telemetry observations from %s",
            len(valid_payloads), input_dir,
        )

    return valid_payloads
