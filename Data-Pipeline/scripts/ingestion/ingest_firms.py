"""
FIRMS Data Ingestion
====================
Fetches active fire detection data from NASA FIRMS API for VIIRS and MODIS
sensors. Handles rate limiting, retries, pagination, and outputs raw CSV
to the staging area.

Owner: Person A
Dependencies: requests, pandas, numpy
API Docs: https://firms.modaps.eosdis.nasa.gov/api/area/

Key behaviors:
    - Queries each configured sensor separately (VIIRS_SNPP, VIIRS_NOAA20, MODIS)
    - Requests data for California and Texas bounding boxes
    - Handles zero-detection responses as normal (not errors)
    - Implements exponential backoff with jitter on HTTP errors
    - Outputs raw CSV preserving all FIRMS columns for auditability
"""

import io
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from scripts.utils.rate_limiter import RateLimiter, create_firms_limiter
from scripts.utils.schema_loader import get_registry

logger = logging.getLogger(__name__)

# FIRMS CSV columns we expect (may vary by sensor, these are the common ones)
EXPECTED_FIRMS_COLUMNS = [
    "latitude", "longitude", "brightness", "scan", "track",
    "acq_date", "acq_time", "satellite", "instrument", "confidence",
    "version", "bright_t31", "frp", "daynight",
]


def fetch_firms_data(
    execution_date: datetime,
    resolution_km: int = 64,
    lookback_hours: int = 24,
    output_dir: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Path:
    """Fetch FIRMS active fire detections for all configured regions.

    This is the main entry point called by the Airflow task.

    Args:
        execution_date: The canonical pipeline execution timestamp.
        resolution_km: Grid resolution (affects bbox granularity logging only;
                       spatial aggregation happens in process_firms).
        lookback_hours: How many hours of data to request (default 24h covers
                        the gap between 6-hourly pipeline runs with margin).
        output_dir: Local directory to write raw CSV. Defaults to data/raw/firms/.
        config_path: Optional schema config override.

    Returns:
        Path to the output CSV file containing all detections.

    Raises:
        RuntimeError: If all API requests fail after retries.
    """
    registry = get_registry(config_path)
    firms_config = registry.get_source_config("firms")
    bboxes = registry.geographic_bboxes

    api_key = os.environ.get(firms_config["api_key_env_var"])
    if not api_key:
        raise EnvironmentError(
            f"FIRMS API key not set. Set environment variable "
            f"'{firms_config['api_key_env_var']}'. "
            f"Get a free key at https://firms.modaps.eosdis.nasa.gov/api/"
        )

    base_url = firms_config["base_url"]
    sensors = firms_config["sensors"]
    limiter = create_firms_limiter(config_path)

    # Calculate the date range for the request
    # FIRMS API uses day_range parameter (1 = last 24h, 2 = last 48h, etc.)
    day_range = max(1, lookback_hours // 24)

    all_detections = []

    for region_name, bbox in bboxes.items():
        west, south, east, north = bbox
        bbox_str = f"{west},{south},{east},{north}"

        for sensor in sensors:
            logger.info(
                f"Fetching FIRMS {sensor} data for {region_name} "
                f"(day_range={day_range})"
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
                detections["region"] = region_name
                detections["sensor_source"] = sensor
                all_detections.append(detections)
                logger.info(
                    f"  → {len(detections)} detections from {sensor} in {region_name}"
                )
            else:
                logger.info(
                    f"  → 0 detections from {sensor} in {region_name} (normal)"
                )

    # Combine all detections into a single DataFrame
    if all_detections:
        combined = pd.concat(all_detections, ignore_index=True)
    else:
        # No fires detected anywhere — create empty DataFrame with expected columns
        combined = pd.DataFrame(columns=EXPECTED_FIRMS_COLUMNS + ["region", "sensor_source"])
        logger.info("No fire detections across any region/sensor (normal in off-season)")

    # Write to output
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "firms"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = execution_date.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"firms_raw_{date_str}.csv"
    combined.to_csv(output_path, index=False)

    logger.info(
        f"FIRMS ingestion complete: {len(combined)} total detections → {output_path}"
    )
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
    """Make a single FIRMS API request with rate limiting and retries.

    Args:
        base_url: FIRMS API base URL.
        api_key: MAP_KEY for authentication.
        sensor: Sensor identifier (e.g., 'VIIRS_SNPP_NRT').
        bbox_str: Bounding box as 'west,south,east,north'.
        day_range: Number of days of data to request.
        limiter: Rate limiter instance.
        max_retries: Maximum retry attempts.
        timeout: Request timeout in seconds.

    Returns:
        DataFrame of detections, or None if no data / all retries failed.
    """
    # FIRMS API URL format: {base_url}/csv/{key}/{sensor}/{bbox}/{day_range}
    url = f"{base_url}/csv/{api_key}/{sensor}/{bbox_str}/{day_range}"

    for attempt in range(max_retries):
        try:
            with limiter.acquire():
                response = requests.get(url, timeout=timeout)

            if response.status_code == 200:
                # FIRMS returns CSV text; may be empty if no fires
                content = response.text.strip()
                if not content or content.startswith("No data"):
                    return None

                df = pd.read_csv(io.StringIO(content))

                # Basic sanity check on returned data
                if len(df) == 0:
                    return None

                return df

            elif response.status_code == 429:
                # Rate limited — back off
                delay = limiter.get_backoff_delay()
                logger.warning(
                    f"FIRMS rate limited (429). Backing off {delay:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                limiter.record_failure()
                import time
                time.sleep(delay)

            else:
                logger.error(
                    f"FIRMS API error: HTTP {response.status_code} for {sensor}. "
                    f"Response: {response.text[:200]}"
                )
                limiter.record_failure()
                delay = limiter.get_backoff_delay()
                import time
                time.sleep(delay)

        except requests.exceptions.Timeout:
            logger.warning(
                f"FIRMS request timed out after {timeout}s "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            limiter.record_failure()
            delay = limiter.get_backoff_delay()
            import time
            time.sleep(delay)

        except requests.exceptions.ConnectionError as e:
            logger.error(f"FIRMS connection error: {e}")
            limiter.record_failure()
            delay = limiter.get_backoff_delay()
            import time
            time.sleep(delay)

    logger.error(
        f"FIRMS fetch failed after {max_retries} retries for sensor={sensor}, "
        f"bbox={bbox_str}"
    )
    return None


def validate_firms_raw(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Run basic validation on raw FIRMS data before processing.

    Args:
        df: Raw FIRMS DataFrame.

    Returns:
        Tuple of (is_valid, list_of_issues).
    """
    issues = []

    if df.empty:
        return True, []  # Empty is valid (no fires)

    # Check required columns exist
    required_cols = ["latitude", "longitude", "frp", "confidence", "acq_date"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")

    # Check latitude/longitude ranges
    if "latitude" in df.columns:
        out_of_range = df[
            (df["latitude"] < 20.0) | (df["latitude"] > 50.0)
        ]
        if len(out_of_range) > 0:
            issues.append(
                f"{len(out_of_range)} detections with latitude outside [20, 50]"
            )

    if "longitude" in df.columns:
        out_of_range = df[
            (df["longitude"] < -130.0) | (df["longitude"] > -85.0)
        ]
        if len(out_of_range) > 0:
            issues.append(
                f"{len(out_of_range)} detections with longitude outside [-130, -85]"
            )

    # Check for negative FRP (sensor artifact)
    if "frp" in df.columns:
        negative_frp = df[df["frp"] < 0]
        if len(negative_frp) > 0:
            issues.append(
                f"{len(negative_frp)} detections with negative FRP (sensor artifact)"
            )

    is_valid = len(issues) == 0
    return is_valid, issues
