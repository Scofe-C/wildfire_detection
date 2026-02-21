"""
FIRMS Data Ingestion
====================
Fetches active fire detection data from NASA FIRMS API for VIIRS and MODIS.
Outputs raw CSV to staging.

Owner: Person A
"""

import io
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import requests

from scripts.utils.rate_limiter import RateLimiter, create_firms_limiter
from scripts.utils.schema_loader import get_registry

logger = logging.getLogger(__name__)

EXPECTED_FIRMS_COLUMNS = [
    "latitude",
    "longitude",
    "brightness",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "version",
    "bright_t31",
    "frp",
    "daynight",
]


def _coerce_datetime(dt: Union[datetime, object]) -> datetime:
    """
    Airflow sometimes passes a lazy/proxy pendulum object.
    Convert to a real python datetime, timezone-aware (UTC).
    """
    try:
        ts = pd.Timestamp(dt)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.to_pydatetime()
    except Exception:
        # Last resort: assume it's already a datetime-like
        if isinstance(dt, datetime):
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt
        return datetime.now(timezone.utc)


def fetch_firms_data(
    execution_date: datetime,
    resolution_km: int = 22,
    lookback_hours: int = 24,
    output_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    region: Optional[str] = None,
) -> Path:
    """Fetch FIRMS active fire detections.

    Args:
        region: If provided (e.g. 'california'), fetch only that region's bbox.
                If None, fetches all configured regions (legacy / non-sharded mode).

    Returns:
        Path to raw FIRMS CSV for the requested region(s).
    """
    execution_date = _coerce_datetime(execution_date)

    registry = get_registry(config_path)
    firms_config = registry.get_source_config("firms")
    all_bboxes = registry.geographic_bboxes

    # --- Improvement 1b: scope to a single region when sharding ---
    if region is not None:
        if region not in all_bboxes:
            raise ValueError(
                f"Region '{region}' not found in schema config. "
                f"Available: {list(all_bboxes.keys())}"
            )
        bboxes = {region: all_bboxes[region]}
        logger.info(f"FIRMS ingestion scoped to region='{region}'")
    else:
        bboxes = all_bboxes

    api_key = os.environ.get(firms_config["api_key_env_var"])

    # If no API key, write an empty CSV so the pipeline can proceed.
    if not api_key:
        if output_dir is None:
            output_dir = str(Path("data") / "raw" / "firms")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        date_str = execution_date.strftime("%Y%m%d_%H%M%S")
        out = out_dir / f"firms_empty_{date_str}.csv"

        pd.DataFrame(
            columns=["latitude", "longitude", "acq_date", "acq_time", "frp", "confidence"]
        ).to_csv(out, index=False)

        logger.warning(
            f"No FIRMS API key ({firms_config['api_key_env_var']}) — wrote empty FIRMS CSV: {out}"
        )
        return out

    base_url = firms_config["base_url"]
    sensors = firms_config["sensors"]
    limiter = create_firms_limiter(config_path)

    day_range = max(1, (lookback_hours + 23) // 24)

    all_detections = []

    for region_name, bbox in bboxes.items():
        west, south, east, north = bbox
        bbox_str = f"{west},{south},{east},{north}"

        for sensor in sensors:
            logger.info(
                f"Fetching FIRMS {sensor} data for {region_name} (day_range={day_range})"
            )

            detections = _fetch_single_request(
                base_url=base_url,
                api_key=api_key,
                sensor=sensor,
                bbox_str=bbox_str,
                day_range=day_range,
                limiter=limiter,
                max_retries=firms_config.get("max_retries", 3),
                timeout=firms_config.get("timeout_seconds", 30),
            )

            if detections is not None and len(detections) > 0:
    # tag each detection row with its source region and sensor
                detections["region"] = region_name
                detections["sensor_source"] = sensor
                all_detections.append(detections)
                logger.info(f"  → {len(detections)} detections from {sensor} in {region_name}")
            else:
                logger.info(f"  → 0 detections from {sensor} in {region_name} (normal)")

    if all_detections:
        combined = pd.concat(all_detections, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=EXPECTED_FIRMS_COLUMNS + ["region", "sensor_source"])
        logger.info("No fire detections across any region/sensor (normal in off-season)")

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "firms"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = execution_date.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"firms_raw_{date_str}.csv"
    combined.to_csv(output_path, index=False)

    logger.info(f"FIRMS ingestion complete: {len(combined)} total detections → {output_path}")
    return output_path


def _fetch_single_request(
    base_url: str,
    api_key: str,
    sensor: str,
    bbox_str: str,
    day_range: int,
    limiter: RateLimiter,
    max_retries: int = 3,
    timeout: int = 30,
) -> Optional[pd.DataFrame]:
    url = f"{base_url}/csv/{api_key}/{sensor}/{bbox_str}/{day_range}"

    for attempt in range(max_retries):
        try:
            with limiter.acquire():
                response = requests.get(url, timeout=timeout)

            if response.status_code == 200:
                content = response.text.strip()
                if not content or content.startswith("No data"):
                    return None

                df = pd.read_csv(io.StringIO(content))
                if len(df) == 0:
                    return None
                return df

            if response.status_code == 429:
                delay = limiter.get_backoff_delay()
                logger.warning(
                    f"FIRMS rate limited (429). Backing off {delay:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                limiter.record_failure()
                time.sleep(delay)
                continue

            # Bug #3 fix: 4xx errors (except 429) are non-retryable —
            # fail immediately to avoid wasting time on auth/not-found.
            if 400 <= response.status_code < 500:
                logger.error(
                    f"FIRMS API non-retryable error: HTTP {response.status_code} "
                    f"for {sensor}. Response: {response.text[:200]}"
                )
                return None

            # 5xx — transient server error, retry with backoff
            logger.error(
                f"FIRMS API error: HTTP {response.status_code} for {sensor}. "
                f"Response: {response.text[:200]}"
            )
            limiter.record_failure()
            time.sleep(limiter.get_backoff_delay())

        except requests.exceptions.Timeout:
            logger.warning(
                f"FIRMS request timed out after {timeout}s "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            limiter.record_failure()
            time.sleep(limiter.get_backoff_delay())

        except requests.exceptions.ConnectionError as e:
            logger.error(f"FIRMS connection error: {e}")
            limiter.record_failure()
            time.sleep(limiter.get_backoff_delay())

    logger.error(
        f"FIRMS fetch failed after {max_retries} retries for sensor={sensor}, bbox={bbox_str}"
    )
    return None


def validate_firms_raw(df: pd.DataFrame) -> tuple[bool, list[str]]:
    issues: list[str] = []

    if df.empty:
        return True, []

    required_cols = ["latitude", "longitude", "frp", "confidence", "acq_date"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")

    if "latitude" in df.columns:
        out_of_range = df[(df["latitude"] < 20.0) | (df["latitude"] > 50.0)]
        if len(out_of_range) > 0:
            issues.append(f"{len(out_of_range)} detections with latitude outside [20, 50]")

    if "longitude" in df.columns:
        out_of_range = df[(df["longitude"] < -130.0) | (df["longitude"] > -85.0)]
        if len(out_of_range) > 0:
            issues.append(f"{len(out_of_range)} detections with longitude outside [-130, -85]")

    if "frp" in df.columns:
        negative_frp = df[df["frp"] < 0]
        if len(negative_frp) > 0:
            issues.append(f"{len(negative_frp)} detections with negative FRP (sensor artifact)")

    return (len(issues) == 0), issues
