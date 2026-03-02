"""
Weather Data Processing
=======================
Transforms raw weather rows (hourly) into grid-level features for fusion.

Input:  Raw CSV from ingest_weather (data/raw/weather/*.csv)
Output: DataFrame with grid_id + weather features (one row per grid cell)

Derived features computed here (assignment Section 3.4):
  - days_since_last_precipitation : days since last hour with precip > 1 mm
  - cumulative_wind_run_24h       : total km of wind travel over the raw window
  - drought_index_proxy           : composite 0-1 score from soil moisture,
                                    temperature, and precipitation gap
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Precipitation threshold for "meaningful rain" (mm per hour)
PRECIP_THRESHOLD_MM = 1.0

# Drought proxy weights (must sum to 1.0)
_DROUGHT_W_SOIL    = 0.40   # low soil moisture → higher drought
_DROUGHT_W_TEMP    = 0.25   # high temperature  → higher drought
_DROUGHT_W_PRECIP  = 0.35   # long dry spell    → higher drought

# Columns expected from ingestion
WEATHER_COLS = [
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


def _circular_mean_degrees(series: pd.Series) -> float:
    """Compute circular mean of angles in degrees (0–360).

    Standard arithmetic mean gives wrong results for circular variables:
    e.g. mean(350°, 10°) = 180° instead of the correct 0° (north).
    Uses atan2 on unit-circle sine/cosine components.
    """
    rads = np.deg2rad(series.dropna())
    if len(rads) == 0:
        return np.nan
    return float(np.rad2deg(np.arctan2(np.sin(rads).mean(), np.cos(rads).mean())) % 360)

def process_weather_data(
    raw_csv_path: str,
    resolution_km: int = 64,
    config_path: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate hourly weather data into one row per grid cell.

    Aggregation strategy:
      - mean  : temperature, humidity, wind speed/direction, soil moisture, vpd
      - sum   : precipitation (total over window)
      - max   : fire_weather_index
      - min   : data_quality_flag (prefer 0 over 2)

    Derived features (computed before aggregation):
      - days_since_last_precipitation
      - cumulative_wind_run_24h
      - drought_index_proxy

    Args:
        raw_csv_path: Path to raw weather CSV from ingest_weather.
        resolution_km: Grid resolution (unused here, passed for interface compat).
        config_path: Optional schema config path override.

    Returns:
        DataFrame with one row per grid_id and all weather + derived features.
        Returns empty DataFrame with grid_id column on any input failure.
    """
    p = Path(raw_csv_path)
    if not p.exists():
        logger.warning(f"Weather raw file not found: {raw_csv_path}. Returning empty.")
        return pd.DataFrame({"grid_id": []})

    df = pd.read_csv(p)

    if df.empty:
        logger.info("Empty weather raw CSV. Returning empty features.")
        return pd.DataFrame({"grid_id": []})

    if "grid_id" not in df.columns:
        logger.warning("Weather raw missing grid_id. Returning empty features.")
        return pd.DataFrame({"grid_id": []})

    df = df.copy()
    df["grid_id"] = df["grid_id"].astype(str)

    # Parse timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = pd.NaT

    # Ensure all expected columns exist
    for c in WEATHER_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # Coerce numerics
    numeric_cols = [
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
        "wind_direction_10m", "precipitation", "soil_moisture_0_to_7cm",
        "vpd", "fire_weather_index", "data_quality_flag",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ------------------------------------------------------------------
    # Derived feature: days_since_last_precipitation
    # ------------------------------------------------------------------
    df["days_since_last_precipitation"] = _compute_days_since_precip(df)

    # ------------------------------------------------------------------
    # Derived feature: cumulative_wind_run_24h
    # ------------------------------------------------------------------
    df["cumulative_wind_run_24h"] = _compute_wind_run(df)

    # ------------------------------------------------------------------
    # Derived feature: drought_index_proxy
    # ------------------------------------------------------------------
    df["drought_index_proxy"] = _compute_drought_proxy(df)

    # ------------------------------------------------------------------
    # Aggregate to one row per grid cell
    # ------------------------------------------------------------------
    agg_spec = {
        "temperature_2m":             "mean",
        "relative_humidity_2m":       "mean",
        "wind_speed_10m":             "mean",
        "wind_direction_10m":         _circular_mean_degrees,  # circular variable — arithmetic mean is wrong
        "precipitation":              "sum",
        "soil_moisture_0_to_7cm":     "mean",
        "vpd":                        "mean",
        "fire_weather_index":         "max",
        "data_quality_flag":          "min",
        # Derived features: take the value computed from the full window
        # (they are already per-cell scalars, so mean == any aggregation)
        "days_since_last_precipitation": "mean",
        "cumulative_wind_run_24h":       "mean",
        "drought_index_proxy":           "mean",
    }

    out = df.groupby("grid_id", as_index=False).agg(agg_spec)

    # Cast derived features to correct types
    out["days_since_last_precipitation"] = (
        out["days_since_last_precipitation"]
        .round(0)
        .clip(lower=0, upper=365)
        .astype("Int16")
    )
    out["cumulative_wind_run_24h"] = out["cumulative_wind_run_24h"].clip(lower=0)
    out["drought_index_proxy"] = out["drought_index_proxy"].clip(lower=0.0, upper=1.0)

    logger.info(
        f"Weather processing complete: {len(out)} grid cells, "
        f"{out['days_since_last_precipitation'].notna().sum()} with drought history"
    )
    return out


# ---------------------------------------------------------------------------
# Derived feature implementations
# ---------------------------------------------------------------------------

def _compute_days_since_precip(df: pd.DataFrame) -> pd.Series:
    """Days since any hour exceeded PRECIP_THRESHOLD_MM per grid cell.

    If the raw window contains a wet hour, result is 0.0 (rained recently).
    If no wet hour is found in the window, result is the number of days
    since the start of the window (approximated from timestamp range).

    For cells with no timestamp data, returns NaN.

    Args:
        df: Raw hourly weather DataFrame with grid_id, timestamp, precipitation.

    Returns:
        Series aligned to df.index with days_since values.
    """
    result = pd.Series(index=df.index, dtype=float)
    result[:] = np.nan

    if "precipitation" not in df.columns or df["precipitation"].isna().all():
        return result

    now_approx = df["timestamp"].max() if df["timestamp"].notna().any() else None

    for grid_id, group in df.groupby("grid_id"):
        precip = group["precipitation"].fillna(0)
        has_precip = precip >= PRECIP_THRESHOLD_MM

        if has_precip.any():
            # There was rain in this window → 0 days since precipitation
            result.loc[group.index] = 0.0
        else:
            # No rain detected — estimate how long ago it rained from window span
            if "timestamp" in group.columns and group["timestamp"].notna().any():
                ts = group["timestamp"].dropna()
                window_hours = (ts.max() - ts.min()).total_seconds() / 3600
                # The entire window is dry → at minimum window_hours/24 days
                # since last rain.  If the window is zero-width (single row),
                # default to 1.0 — we cannot determine actual drought length
                # from a single point, so 0.0 would falsely imply recent rain.
                days_dry = max(window_hours / 24.0, 1.0)
            else:
                days_dry = 1.0  # Default: assume 1 day if no timestamp
            result.loc[group.index] = days_dry

    return result


def _compute_wind_run(df: pd.DataFrame) -> pd.Series:
    """Cumulative wind run (km) over the available raw window per grid cell.

    Wind run = sum(wind_speed_km_h × hours_per_row).

    The raw data is hourly, so each row represents 1 hour. Converts km/h to km.

    Args:
        df: Raw hourly weather DataFrame with grid_id and wind_speed_10m (km/h).

    Returns:
        Series aligned to df.index with cumulative wind run values.
    """
    result = pd.Series(index=df.index, dtype=float)
    result[:] = np.nan

    if "wind_speed_10m" not in df.columns:
        return result

    # Each hourly row = 1 hour. wind_speed (km/h) × 1 h = km of wind run.
    # Sum across hours for each grid cell.
    wind_per_hour = df["wind_speed_10m"].fillna(0).clip(lower=0)

    for grid_id, group in df.groupby("grid_id"):
        cumulative = wind_per_hour.loc[group.index].sum()
        result.loc[group.index] = cumulative

    return result


def _compute_drought_proxy(df: pd.DataFrame) -> pd.Series:
    """Composite drought index proxy (0.0 = no drought, 1.0 = severe drought).

    Combines three normalized sub-scores:
      1. Soil moisture deficit (low soil moisture → high drought)
      2. Temperature stress (high temperature → high drought)
      3. Precipitation gap (days since rain → high drought)

    Each sub-score is normalized to [0, 1] using physiologically meaningful
    bounds for California and Texas fire conditions:
      - Soil moisture: 0.0 (bone dry) to 0.5 m³/m³ (saturated)
      - Temperature:   0°C (cold) to 45°C (extreme heat)
      - Days dry:      0 days (just rained) to 90 days (long drought)

    Args:
        df: Raw hourly weather DataFrame per grid cell.

    Returns:
        Series aligned to df.index with drought_proxy values in [0, 1].
    """
    result = pd.Series(index=df.index, dtype=float)
    result[:] = np.nan

    SOIL_MIN, SOIL_MAX = 0.0, 0.5
    TEMP_MIN, TEMP_MAX = 0.0, 45.0
    DAYS_MIN, DAYS_MAX = 0.0, 90.0

    # Compute per-grid aggregates first
    for grid_id, group in df.groupby("grid_id"):
        idx = group.index

        # Sub-score 1: soil moisture deficit (low = more drought)
        soil = df.loc[idx, "soil_moisture_0_to_7cm"].mean() if "soil_moisture_0_to_7cm" in df.columns else np.nan
        if pd.notna(soil):
            soil_score = 1.0 - np.clip((soil - SOIL_MIN) / (SOIL_MAX - SOIL_MIN), 0, 1)
        else:
            soil_score = 0.5  # neutral if missing

        # Sub-score 2: temperature stress (high = more drought)
        temp = df.loc[idx, "temperature_2m"].mean() if "temperature_2m" in df.columns else np.nan
        if pd.notna(temp):
            temp_score = np.clip((temp - TEMP_MIN) / (TEMP_MAX - TEMP_MIN), 0, 1)
        else:
            temp_score = 0.5

        # Sub-score 3: precipitation gap
        days_dry = df.loc[idx, "days_since_last_precipitation"].mean() if "days_since_last_precipitation" in df.columns else np.nan
        if pd.notna(days_dry):
            precip_score = np.clip((days_dry - DAYS_MIN) / (DAYS_MAX - DAYS_MIN), 0, 1)
        else:
            precip_score = 0.5

        drought = (
            _DROUGHT_W_SOIL   * soil_score
            + _DROUGHT_W_TEMP   * temp_score
            + _DROUGHT_W_PRECIP * precip_score
        )
        result.loc[idx] = float(np.clip(drought, 0.0, 1.0))

    return result
