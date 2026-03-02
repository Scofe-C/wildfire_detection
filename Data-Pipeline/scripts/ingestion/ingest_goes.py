"""
GOES NRT Fire Detection Ingestion
==================================
Lightweight FIRMS GOES_NRT poll used by the Cloud Function watchdog.
Returns raw detections for the last N minutes within a bounding box.

Separate from ingest_firms.py intentionally:
  - Minimal dependencies (no grid_utils, no schema_loader heavy path)
  - Returns structured detection objects, not raw CSV files
  - Used by fire_detector.py and the Cloud Function, not by the Airflow DAG
  - Can run inside Cloud Function (64 MB memory budget)

GOES NRT via FIRMS:
  - Updates every 10 minutes (CONUS ABI scan cadence)
  - Latency: ~20–30 minutes from observation to API availability
  - Spatial resolution: ~2 km pixels
  - Rate cost: ~1–5 FIRMS transactions per bounding box query
"""

import io
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# FIRMS API endpoint — same base URL as ingest_firms.py
FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area"
GOES_NRT_SOURCE = "GOES_NRT"

# Minimum FRP (MW) to treat a GOES pixel as a fire candidate
# Below this threshold, pixels are likely warm surfaces (roads, parking lots)
DEFAULT_MIN_FRP_MW = 10.0

# Default client-side lookback window
DEFAULT_LOOKBACK_MINUTES = 60


def fetch_goes_nrt_detections(
    bbox: list[float],
    lookback_minutes: int = DEFAULT_LOOKBACK_MINUTES,
    min_frp_mw: float = DEFAULT_MIN_FRP_MW,
    api_key: Optional[str] = None,
    max_retries: int = 3,
    timeout_seconds: int = 20,
    source: Optional[str] = None,
) -> list[dict]:
    """Poll FIRMS GOES_NRT for fire detections in a bounding box.

    Always requests day_range=1 (API minimum), then filters client-side
    to the last `lookback_minutes`. Empty list = no fires detected.

    Args:
        bbox: [west, south, east, north] in WGS84 degrees.
        lookback_minutes: How far back to look for detections (client-side filter).
        min_frp_mw: Minimum Fire Radiative Power to consider a candidate.
        api_key: FIRMS MAP_KEY. Falls back to FIRMS_MAP_KEY env var if None.
        max_retries: Number of HTTP retries on failure.
        timeout_seconds: Per-request timeout.
        source: FIRMS source identifier (e.g. "GOES_NRT", "VIIRS_SNPP_NRT").
            Defaults to module-level GOES_NRT_SOURCE if not provided.
            Prefer passing this parameter instead of mutating the global.

    Returns:
        List of detection dicts:
        [{"lat": 37.5, "lon": -120.2, "frp": 45.3, "acq_datetime": datetime, ...}]
        Empty list if no fires or API unavailable.
    """
    effective_source = source or GOES_NRT_SOURCE
    api_key = api_key or os.environ.get("FIRMS_MAP_KEY")
    if not api_key:
        logger.warning("No FIRMS_MAP_KEY — GOES NRT quick-check skipped")
        return []

    west, south, east, north = bbox
    bbox_str = f"{west},{south},{east},{north}"
    url = f"{FIRMS_BASE_URL}/csv/{api_key}/{effective_source}/{bbox_str}/1"

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout_seconds)

            if resp.status_code == 200:
                content = resp.text.strip()
                if not content or content.startswith("No data") or len(content.splitlines()) <= 1:
                    logger.info("GOES NRT: no detections in bounding box")
                    return []

                df = pd.read_csv(io.StringIO(content))
                if df.empty:
                    return []

                # Parse acquisition datetime from separate date + time columns
                df = _parse_acq_datetime(df)

                # Client-side filter to lookback window
                df = df[df["acq_datetime"] >= cutoff].copy()

                if df.empty:
                    logger.info(f"GOES NRT: {len(df)} detections in last {lookback_minutes} min")
                    return []

                # Apply minimum FRP threshold
                if "frp" in df.columns:
                    df = df[df["frp"].fillna(0) >= min_frp_mw]

                if df.empty:
                    logger.info(
                        f"GOES NRT: all detections below FRP threshold "
                        f"({min_frp_mw} MW)"
                    )
                    return []

                detections = _df_to_detections(df)
                logger.info(
                    f"GOES NRT: {len(detections)} candidate detections "
                    f"(FRP >= {min_frp_mw} MW, last {lookback_minutes} min)"
                )
                return detections

            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                logger.warning(f"FIRMS rate limited (429). Waiting {wait}s")
                time.sleep(wait)
                continue

            logger.error(f"FIRMS HTTP {resp.status_code}: {resp.text[:200]}")
            time.sleep(5 * (attempt + 1))

        except requests.exceptions.Timeout:
            logger.warning(f"FIRMS timeout (attempt {attempt + 1}/{max_retries})")
            time.sleep(5 * (attempt + 1))
        except requests.exceptions.ConnectionError as e:
            logger.error(f"FIRMS connection error: {e}")
            time.sleep(10 * (attempt + 1))
        except Exception as e:
            logger.error(f"Unexpected GOES NRT fetch error: {e}")
            return []

    logger.error(f"GOES NRT: all {max_retries} attempts failed")
    return []


def fetch_goes_s3_detections(
    bbox: list[float],
    lookback_minutes: int = 30,
    use_probable: bool = False,
) -> list[dict]:
    """Read GOES-R ABI FDC product directly from AWS S3 (no auth required).

    Lower latency than FIRMS GOES_NRT (~15 min vs ~25 min) but requires
    netCDF4/xarray. Falls back gracefully if those aren't installed.

    GOES-18 covers western CONUS (California).
    GOES-19 covers eastern CONUS (Texas) — replaced GOES-16 in April 2025.

    Args:
        bbox: [west, south, east, north].
        lookback_minutes: How far back to scan for FDC files.
        use_probable: Include probable fire codes (more sensitive, more FA risk).

    Returns:
        List of detection dicts (same schema as fetch_goes_nrt_detections).
    """
    try:
        import xarray as xr
        import numpy as np
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError:
        logger.info(
            "xarray/boto3 not available — GOES S3 direct access skipped. "
            "Falling back to FIRMS GOES_NRT."
        )
        return []

    west, south, east, north = bbox

    # Determine which GOES satellite covers this bbox
    # GOES-18 (West) best for CA (western longitudes), GOES-19 (East) for TX
    lon_center = (west + east) / 2
    bucket_name = "noaa-goes18" if lon_center < -100 else "noaa-goes19"

    FIRE_CODES = [10, 11, 30, 31]
    PROBABLE_CODES = [13, 14, 15, 33, 34, 35]
    target_codes = FIRE_CODES + (PROBABLE_CODES if use_probable else [])

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)

    try:
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        detections = []

        # List files for the last hour
        now = datetime.now(timezone.utc)
        for hour_offset in range(2):  # current hour + previous hour
            check_time = now - timedelta(hours=hour_offset)
            julian_day = check_time.timetuple().tm_yday
            prefix = f"ABI-L2-FDCC/{check_time.year}/{julian_day:03d}/{check_time.hour:02d}/"

            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            files = [obj["Key"] for obj in response.get("Contents", [])]

            for key in sorted(files)[-6:]:  # last 6 files = ~30 min at 5-min cadence
                file_time = _parse_goes_filename_time(key)
                if file_time and file_time < cutoff:
                    continue

                try:
                    url = f"https://{bucket_name}.s3.amazonaws.com/{key}"
                    ds = xr.open_dataset(url, engine="netcdf4")
                    fire_mask = ds["Mask"].values
                    lats = ds["goes_imager_projection"].attrs.get("latitude_of_projection_origin", 0)

                    # Get lat/lon arrays for pixel coordinates
                    # GOES uses a fixed grid projection; simplified bbox intersection
                    # For production, use goes2go or pyproj for accurate reprojection
                    fire_pixels = np.argwhere(np.isin(fire_mask, target_codes))

                    if len(fire_pixels) == 0:
                        ds.close()
                        continue

                    # Approximate pixel → lat/lon using dataset coordinate arrays
                    if "x" in ds.coords and "y" in ds.coords:
                        x_coords = ds["x"].values
                        y_coords = ds["y"].values
                        for py, px in fire_pixels[:100]:  # cap at 100 pixels per file
                            # GOES projection simplified approximation
                            # Full implementation: use pyproj with GOES projection params
                            approx_lon = float(x_coords[px]) * 180 / 3.14159  # placeholder
                            approx_lat = float(y_coords[py]) * 180 / 3.14159  # placeholder
                            if west <= approx_lon <= east and south <= approx_lat <= north:
                                detections.append({
                                    "lat": approx_lat,
                                    "lon": approx_lon,
                                    "frp": None,   # FDC mask doesn't include FRP directly
                                    "acq_datetime": file_time or now,
                                    "source": "GOES_S3",
                                    "confidence": "high" if fire_mask[py, px] in FIRE_CODES else "medium",
                                    "satellite": bucket_name,
                                })
                    ds.close()

                except Exception as file_err:
                    logger.debug(f"Skipping GOES S3 file {key}: {file_err}")
                    continue

        logger.info(f"GOES S3: {len(detections)} fire pixels in bbox")
        return detections

    except Exception as e:
        logger.warning(f"GOES S3 access failed: {e} — falling back to FIRMS GOES_NRT")
        return []


def _parse_acq_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Parse FIRMS acq_date + acq_time columns into a single UTC datetime."""
    df = df.copy()
    try:
        if "acq_date" in df.columns and "acq_time" in df.columns:
            df["acq_time_str"] = df["acq_time"].astype(str).str.zfill(4)
            df["acq_datetime"] = pd.to_datetime(
                df["acq_date"].astype(str) + " " + df["acq_time_str"],
                format="%Y-%m-%d %H%M",
                utc=True,
                errors="coerce",
            )
        else:
            df["acq_datetime"] = pd.Timestamp.now(tz="UTC")
    except Exception as e:
        logger.warning(f"Could not parse acq_datetime: {e}")
        df["acq_datetime"] = pd.Timestamp.now(tz="UTC")
    return df


def _df_to_detections(df: pd.DataFrame) -> list[dict]:
    """Convert FIRMS CSV DataFrame to structured detection list."""
    detections = []
    lat_col = "latitude" if "latitude" in df.columns else "lat"
    lon_col = "longitude" if "longitude" in df.columns else "lon"

    for _, row in df.iterrows():
        try:
            detection = {
                "lat": float(row.get(lat_col, 0)),
                "lon": float(row.get(lon_col, 0)),
                "frp": float(row.get("frp", 0)) if pd.notna(row.get("frp")) else None,
                "acq_datetime": row.get("acq_datetime", datetime.now(timezone.utc)),
                "source": GOES_NRT_SOURCE,
                "confidence": str(row.get("confidence", "n")),
                "satellite": str(row.get("satellite", "")),
                "bright_ti4": float(row.get("bright_ti4", 0)) if pd.notna(row.get("bright_ti4")) else None,
            }
            detections.append(detection)
        except Exception:
            continue
    return detections


def _parse_goes_filename_time(key: str) -> Optional[datetime]:
    """Extract scan start time from GOES-R filename.

    Filename format: OR_ABI-L2-FDCC-M6_G18_s{YYYYDDDHHMMSS}_{...}.nc
    """
    try:
        # Find the 's' prefix for scan start time
        parts = key.split("_")
        for part in parts:
            if part.startswith("s") and len(part) >= 14:
                ts_str = part[1:14]  # YYYYDDDHHMMSS
                year = int(ts_str[0:4])
                doy  = int(ts_str[4:7])
                hour = int(ts_str[7:9])
                minute = int(ts_str[9:11])
                dt = datetime(year, 1, 1, hour, minute, tzinfo=timezone.utc) + timedelta(days=doy - 1)
                return dt
    except Exception:
        pass
    return None
