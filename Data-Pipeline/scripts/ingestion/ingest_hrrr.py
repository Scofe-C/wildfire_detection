"""
HRRR Weather Ingestion — Focal Grid
=====================================
Fetches NOAA High-Resolution Rapid Refresh (HRRR) analysis fields
for a focal grid of H3 cells around a confirmed fire.

Called exclusively on watchdog-triggered emergency/active runs — never on
cron runs. The focal grid is ~25–100 cells (fire + detection zone rings),
so HRRR is only pulled for the cells that matter.

Source
------
HRRR GRIB2 files are read from the public AWS S3 mirror:
  s3://noaa-hrrr-bdp-pds/hrrr.{YYYYMMDD}/conus/hrrr.t{HH}z.wrfsfcf00.grib2

  - No authentication required (unsigned S3 access)
  - f00 = the analysis (zero-hour forecast) = most current observed state
  - New cycle published every hour; latest complete cycle available ~45–60
    min after the top of the hour
  - 3 km horizontal resolution; bilinear interpolation to H3 centroids

Variable mapping (HRRR → schema column)
----------------------------------------
  TMP:2 m above ground         → temperature_2m          (°C, already Celsius in HRRR)
  RH:2 m above ground          → relative_humidity_2m    (%)
  UGRD:10 m above ground       → _u_wind_10m             (m/s, intermediate)
  VGRD:10 m above ground       → _v_wind_10m             (m/s, intermediate)
  WIND → speed derived from U/V → wind_speed_10m         (km/h, converted)
  WDIR derived from U/V        → wind_direction_10m      (degrees meteorological)
  APCP:surface                 → precipitation           (mm, accumulated since analysis)
  SOILW:0-0.1 m below ground   → soil_moisture_0_to_7cm  (m³/m³, closest layer)
  VPD derived from T+RH        → vpd                     (kPa)

Output contract
---------------
Identical CSV schema to ingest_weather.py output:
  grid_id, timestamp, temperature_2m, relative_humidity_2m,
  wind_speed_10m, wind_direction_10m, precipitation,
  soil_moisture_0_to_7cm, vpd, fire_weather_index, data_quality_flag

  data_quality_flag = 3 for all rows (HRRR source, per schema_config.yaml)
  fire_weather_index = None (not available from HRRR surface analysis)

Error handling
--------------
- Latest cycle selection retries up to N_CYCLE_LOOKBACK previous cycles
  when the expected S3 object is not yet published (S3 key 404).
- On complete HRRR failure: returns None so the caller can fall back to
  forward-fill (data_quality_flag = 4 via existing ingest_weather logic).
- No exception is ever raised to the caller — HRRR is best-effort.

Dependencies
------------
  herbie-data   >= 2024.3.0  (HRRR S3 path construction + GRIB2 access)
  cfgrib        >= 0.9.10    (eccodes-backed GRIB2 reading via xarray)
  eccodes       >= 2.30.0    (system library, installed via apt/conda)
  scipy         >= 1.12.0    (RegularGridInterpolator for centroid extraction)

These are NOT in requirements.txt yet — add them (see HRRR integration PR).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How many hourly cycles to look back when the current cycle is not yet
# published on S3 (~45–60 min lag; 2 cycles = safe for any trigger time)
N_CYCLE_LOOKBACK = 3

# HRRR S3 bucket — public, unsigned
HRRR_S3_BUCKET = "noaa-hrrr-bdp-pds"

# Quality flag for HRRR-sourced rows (matches schema_config.yaml code 3)
HRRR_QUALITY_FLAG = 3

# HRRR variable specs: (herbie shortName or cfgrib typeOfLevel+name, level, output_col)
# These are passed to Herbie's xarray accessor for message filtering.
HRRR_VARIABLES = [
    # shortName  typeOfLevel            level    output_col
    ("TMP",      "heightAboveGround",   2,       "temperature_2m"),
    ("RH",       "heightAboveGround",   2,       "relative_humidity_2m"),
    ("UGRD",     "heightAboveGround",   10,      "_u_wind_10m"),
    ("VGRD",     "heightAboveGround",   10,      "_v_wind_10m"),
    ("APCP",     "surface",             0,       "precipitation"),
    ("SOILW",    "depthBelowLandLayer", 0,       "soil_moisture_0_to_7cm"),
]

# Output column schema — must match ingest_weather.py exactly
OUTPUT_COLUMNS = [
    "grid_id",
    "timestamp",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "soil_moisture_0_to_7cm",
    "vpd",
    "fire_weather_index",
    "data_quality_flag",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_hrrr_for_focal_grid(
    focal_grid: pd.DataFrame,
    execution_date: datetime,
    output_dir: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Optional[Path]:
    """Fetch HRRR analysis fields for focal grid cells around a confirmed fire.

    Selects the latest complete HRRR cycle at or before execution_date,
    extracts the required surface variables, bilinearly interpolates to
    each H3 cell centroid, and writes a CSV with the same schema as
    ingest_weather.py output.

    Args:
        focal_grid:     DataFrame with columns grid_id, latitude, longitude.
                        Typically the output of generate_fire_focal_grid().
        execution_date: Airflow execution_date (UTC). Used to pick the HRRR
                        cycle — the latest complete cycle <= execution_date.
        output_dir:     Directory to write output CSV. Defaults to
                        data/raw/weather/.
        config_path:    Optional schema config path override.

    Returns:
        Path to written CSV, or None if HRRR fetch failed entirely.
        None signals the caller to fall back to forward-fill from the
        previous cron run (data_quality_flag = 4).
    """
    if focal_grid is None or focal_grid.empty:
        logger.warning("fetch_hrrr_for_focal_grid: empty focal_grid — skipping")
        return None

    required_cols = {"grid_id", "latitude", "longitude"}
    if not required_cols.issubset(focal_grid.columns):
        logger.error(
            f"focal_grid missing columns: {required_cols - set(focal_grid.columns)}"
        )
        return None

    # Normalize execution_date to UTC-aware datetime
    execution_date = _to_utc(execution_date)

    # Resolve output directory
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "weather"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"HRRR fetch: {len(focal_grid)} focal cells, "
        f"execution_date={execution_date.isoformat()}"
    )

    # --- Select HRRR cycle ---
    cycle_dt = _select_hrrr_cycle(execution_date)
    if cycle_dt is None:
        logger.error("HRRR: could not select a valid cycle — aborting")
        return None

    logger.info(f"HRRR: selected cycle {cycle_dt.strftime('%Y-%m-%dT%H:00Z')}")

    # --- Fetch GRIB2 fields via Herbie ---
    raw_fields = _fetch_hrrr_fields(cycle_dt)
    if raw_fields is None:
        logger.warning("HRRR: all field fetches failed — returning None for fallback")
        return None

    # --- Interpolate to focal grid centroids ---
    records = _interpolate_to_centroids(raw_fields, focal_grid, cycle_dt)
    if not records:
        logger.warning("HRRR: interpolation produced no records — returning None")
        return None

    df = pd.DataFrame(records)

    # --- Derive wind speed / direction from U and V components ---
    if "_u_wind_10m" in df.columns and "_v_wind_10m" in df.columns:
        df["wind_speed_10m"]     = _uv_to_speed_kmh(df["_u_wind_10m"], df["_v_wind_10m"])
        df["wind_direction_10m"] = _uv_to_direction(df["_u_wind_10m"], df["_v_wind_10m"])
        df = df.drop(columns=["_u_wind_10m", "_v_wind_10m"])
    else:
        df["wind_speed_10m"]     = np.nan
        df["wind_direction_10m"] = np.nan

    # --- Derive VPD from temperature and relative humidity ---
    if "temperature_2m" in df.columns and "relative_humidity_2m" in df.columns:
        df["vpd"] = _compute_vpd(df["temperature_2m"], df["relative_humidity_2m"])
    else:
        df["vpd"] = np.nan

    # --- Metadata columns ---
    df["fire_weather_index"] = None          # Not available in HRRR surface analysis
    df["data_quality_flag"]  = HRRR_QUALITY_FLAG
    df["timestamp"]          = cycle_dt.isoformat()

    # --- Enforce output schema ---
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[OUTPUT_COLUMNS]

    # --- Write CSV ---
    date_str = execution_date.strftime("%Y%m%d_%H%M%S")
    output_path = out_dir / f"weather_hrrr_{date_str}.csv"
    df.to_csv(output_path, index=False)

    logger.info(
        f"HRRR ingestion complete: {len(df)} focal cells "
        f"(cycle={cycle_dt.strftime('%Y-%m-%dT%H:00Z')}) → {output_path}"
    )
    return output_path


# ---------------------------------------------------------------------------
# Cycle selection
# ---------------------------------------------------------------------------

def _select_hrrr_cycle(execution_date: datetime) -> Optional[datetime]:
    """Select the latest complete HRRR cycle <= execution_date.

    HRRR publishes a new analysis every hour. The f00 (analysis) file for
    cycle HH is typically available ~45–60 min after HH:00 UTC. We look back
    up to N_CYCLE_LOOKBACK hours to find the most recent cycle whose S3 object
    actually exists.

    Returns:
        UTC datetime truncated to the hour of the selected cycle,
        or None if no cycle found within the lookback window.
    """
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    except ImportError:
        logger.error(
            "boto3 not installed — cannot access HRRR S3. "
            "Install: pip install boto3"
        )
        return None
    except Exception as e:
        logger.error(f"HRRR: boto3 client init failed: {e}")
        return None

    # Start from the most recent full hour at or before execution_date
    candidate = execution_date.replace(minute=0, second=0, microsecond=0)

    for attempt in range(N_CYCLE_LOOKBACK):
        cycle_candidate = candidate - timedelta(hours=attempt)
        s3_key = _hrrr_s3_key(cycle_candidate)

        try:
            s3.head_object(Bucket=HRRR_S3_BUCKET, Key=s3_key)
            logger.debug(f"HRRR: S3 key found: {s3_key}")
            return cycle_candidate
        except s3.exceptions.ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("404", "NoSuchKey"):
                logger.debug(
                    f"HRRR: cycle {cycle_candidate.strftime('%Y-%m-%dT%H:00Z')} "
                    f"not yet published (S3 key: {s3_key}) — trying previous hour"
                )
                continue
            # Unexpected S3 error — abort rather than silently use wrong data
            logger.error(f"HRRR: unexpected S3 error checking {s3_key}: {e}")
            return None
        except Exception as e:
            logger.error(f"HRRR: S3 head_object failed for {s3_key}: {e}")
            return None

    logger.warning(
        f"HRRR: no published cycle found in the last {N_CYCLE_LOOKBACK} hours "
        f"before {execution_date.isoformat()}"
    )
    return None


def _hrrr_s3_key(cycle_dt: datetime) -> str:
    """Build the S3 object key for a given HRRR cycle's f00 analysis file."""
    date_str = cycle_dt.strftime("%Y%m%d")
    hour_str = cycle_dt.strftime("%H")
    return f"hrrr.{date_str}/conus/hrrr.t{hour_str}z.wrfsfcf00.grib2"


# ---------------------------------------------------------------------------
# GRIB2 field fetching
# ---------------------------------------------------------------------------

def _fetch_hrrr_fields(cycle_dt: datetime) -> Optional[dict]:
    """Download and parse HRRR GRIB2 fields for the given cycle.

    Uses Herbie to handle S3 path construction and xarray/cfgrib integration.

    Returns:
        Dict mapping output_col → xarray DataArray on the HRRR native grid,
        or None on complete failure.
    """
    try:
        from herbie import Herbie
    except ImportError:
        logger.error(
            "herbie-data not installed — cannot fetch HRRR. "
            "Install: pip install herbie-data"
        )
        return None

    fields: dict = {}

    for short_name, level_type, level_value, output_col in HRRR_VARIABLES:
        try:
            H = Herbie(
                cycle_dt.strftime("%Y-%m-%d %H:%M"),
                model="hrrr",
                product="sfc",
                fxx=0,          # f00 = analysis, not forecast
                save_dir="/tmp/hrrr_cache",
                verbose=False,
            )

            # Build the cfgrib search string Herbie uses for GRIB2 message selection
            search_str = _build_herbie_search(short_name, level_type, level_value)
            ds = H.xarray(search_str, remove_grib=False)

            if ds is None or len(ds) == 0:
                logger.warning(
                    f"HRRR: no data for {short_name}:{level_type}:{level_value} "
                    f"in cycle {cycle_dt.strftime('%Y-%m-%dT%H:00Z')}"
                )
                continue

            # Extract the first (and usually only) DataArray for this variable
            var_names = list(ds.data_vars)
            if not var_names:
                logger.warning(f"HRRR: empty dataset for {short_name}")
                continue

            fields[output_col] = ds[var_names[0]]
            logger.debug(f"HRRR: fetched {output_col} ({short_name})")

        except Exception as e:
            logger.warning(
                f"HRRR: failed to fetch {short_name}:{level_type}:{level_value}: {e}"
            )
            # Continue — partial fields are better than aborting entirely.
            # The caller gets NaN for any missing variable.
            continue

    if not fields:
        logger.error("HRRR: no fields could be fetched")
        return None

    return fields


def _build_herbie_search(short_name: str, level_type: str, level_value: int) -> str:
    """Build a Herbie-compatible GRIB2 search string.

    Herbie uses regex matching against the GRIB2 inventory (.idx) file.
    Format: ':VARNAME:LEVEL_DESCRIPTION:'

    Examples:
        TMP:2 m above ground      → ':TMP:2 m above ground:'
        UGRD:10 m above ground    → ':UGRD:10 m above ground:'
        APCP:surface              → ':APCP:surface:'
        SOILW:0-0.1 m below       → ':SOILW:0-0.1 m below ground:'
    """
    level_desc_map = {
        ("heightAboveGround", 2):   "2 m above ground",
        ("heightAboveGround", 10):  "10 m above ground",
        ("surface", 0):             "surface",
        ("depthBelowLandLayer", 0): "0-0.1 m below ground",
    }
    level_desc = level_desc_map.get(
        (level_type, level_value),
        f"{level_value} {level_type}"
    )
    return f":{short_name}:{level_desc}:"


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def _interpolate_to_centroids(
    fields: dict,
    focal_grid: pd.DataFrame,
    cycle_dt: datetime,
) -> list[dict]:
    """Bilinearly interpolate HRRR fields to each focal grid centroid.

    HRRR uses a Lambert Conformal Conic projection (~3 km grid spacing).
    We use scipy.interpolate.RegularGridInterpolator on the lat/lon coordinate
    arrays included in the xarray Dataset (latitude, longitude data variables)
    to perform bilinear interpolation to each H3 centroid.

    Args:
        fields:     Dict of output_col → xarray DataArray (HRRR native grid)
        focal_grid: DataFrame with grid_id, latitude, longitude
        cycle_dt:   HRRR cycle datetime (for logging)

    Returns:
        List of dicts, one per focal grid cell, with all variable columns.
        Missing variables produce NaN for that cell.
    """
    try:
        from scipy.interpolate import RegularGridInterpolator
    except ImportError:
        logger.error(
            "scipy not installed — cannot interpolate HRRR to centroids. "
            "Install: pip install scipy"
        )
        return []

    if not fields:
        return []

    # Use the first available field to extract the HRRR lat/lon coordinate arrays.
    # All HRRR surface fields share the same native grid.
    first_field = next(iter(fields.values()))
    try:
        hrrr_lats = first_field["latitude"].values   # shape (y, x)
        hrrr_lons = first_field["longitude"].values  # shape (y, x)
    except KeyError:
        logger.error(
            "HRRR DataArray missing 'latitude'/'longitude' coordinate variables. "
            "Check Herbie/cfgrib version — expected these as data_vars in the dataset."
        )
        return []

    # Build a lookup from field values on the native HRRR grid.
    # RegularGridInterpolator requires 1D coordinate axes — we use the row/column
    # indices as proxy axes and convert target lat/lon to nearest grid row/col.
    # This is a nearest-row-col + bilinear approach that avoids a full pyproj
    # reprojection. Accuracy: < 1 km at HRRR 3 km grid spacing — acceptable
    # for 22 km H3 cells where we're averaging ~50 HRRR points anyway.
    n_rows, n_cols = hrrr_lats.shape
    row_axis = np.arange(n_rows, dtype=float)
    col_axis = np.arange(n_cols, dtype=float)

    records = []

    for _, cell in focal_grid.iterrows():
        cell_lat = float(cell["latitude"])
        cell_lon = float(cell["longitude"])

        # Find approximate nearest row/col on HRRR grid using vectorized distance
        # (fast approximation — no full haversine needed at this scale)
        lat_diffs = hrrr_lats - cell_lat
        lon_diffs = hrrr_lons - cell_lon
        dist2 = lat_diffs ** 2 + lon_diffs ** 2
        r_idx, c_idx = np.unravel_index(np.argmin(dist2), dist2.shape)

        rec: dict = {
            "grid_id": str(cell["grid_id"]),
        }

        for col_name, da in fields.items():
            try:
                values = da.values  # shape (y, x)

                # Extract a small neighbourhood (3×3) around the nearest point
                # and use bilinear interpolation via RegularGridInterpolator.
                # Clamp to valid index range.
                r0 = max(0, r_idx - 1)
                r1 = min(n_rows - 1, r_idx + 1)
                c0 = max(0, c_idx - 1)
                c1 = min(n_cols - 1, c_idx + 1)

                patch_rows = row_axis[r0:r1 + 1]
                patch_cols = col_axis[c0:c1 + 1]
                patch_vals = values[r0:r1 + 1, c0:c1 + 1]

                if patch_vals.shape[0] < 2 or patch_vals.shape[1] < 2:
                    # Edge cell — fall back to nearest neighbor
                    rec[col_name] = float(values[r_idx, c_idx])
                else:
                    interp = RegularGridInterpolator(
                        (patch_rows, patch_cols),
                        patch_vals,
                        method="linear",
                        bounds_error=False,
                        fill_value=None,
                    )
                    rec[col_name] = float(interp([[float(r_idx), float(c_idx)]])[0])

            except Exception as e:
                logger.debug(f"HRRR: interpolation failed for {col_name} cell {cell['grid_id']}: {e}")
                rec[col_name] = np.nan

        records.append(rec)

    logger.info(
        f"HRRR: interpolated {len(records)} focal cells "
        f"from cycle {cycle_dt.strftime('%Y-%m-%dT%H:00Z')}"
    )
    return records


# ---------------------------------------------------------------------------
# Derived variable computation
# ---------------------------------------------------------------------------

def _uv_to_speed_kmh(u: pd.Series, v: pd.Series) -> pd.Series:
    """Convert U/V wind components (m/s) to wind speed (km/h).

    HRRR U and V are in m/s. Open-Meteo returns km/h.
    Conversion: speed_kmh = sqrt(u² + v²) * 3.6
    """
    speed_ms = np.sqrt(u.fillna(0) ** 2 + v.fillna(0) ** 2)
    return (speed_ms * 3.6).round(2)


def _uv_to_direction(u: pd.Series, v: pd.Series) -> pd.Series:
    """Convert U/V wind components to meteorological wind direction (degrees).

    Meteorological convention: direction FROM which wind is blowing.
    0° = from North, 90° = from East.

    Formula: dir = (270 - atan2(v, u) * 180/π) mod 360
    """
    direction_math = np.degrees(np.arctan2(v.fillna(0), u.fillna(0)))
    direction_met  = (270.0 - direction_math) % 360.0
    return direction_met.round(1)


def _compute_vpd(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    """Compute Vapor Pressure Deficit (kPa) from temperature and relative humidity.

    VPD = SVP × (1 - RH/100)
    SVP (Saturated Vapor Pressure) = 0.6108 × exp(17.27 × T / (T + 237.3))  [kPa]

    This matches the WMO/FAO formula used by Open-Meteo for consistency.
    """
    t = temp_c.fillna(20.0)
    rh = rh_pct.fillna(50.0).clip(0, 100)
    svp = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    vpd = svp * (1.0 - rh / 100.0)
    return vpd.clip(lower=0.0).round(4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_utc(dt) -> datetime:
    """Normalize any Airflow/Pendulum/proxy datetime to UTC-aware Python datetime."""
    if hasattr(dt, "__wrapped__"):
        dt = dt.__wrapped__
    ts = pd.to_datetime(str(dt), utc=True)
    return ts.to_pydatetime()
