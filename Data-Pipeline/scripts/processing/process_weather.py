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
import math
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
    # Extract month for FWI day-length factors before aggregation
    fwi_month = 7  # July default if no timestamps present
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        median_month = df["timestamp"].dt.month.median()
        if pd.notna(median_month):
            fwi_month = int(round(median_month))

    agg_spec = {
        "temperature_2m":             "mean",
        "relative_humidity_2m":       "mean",
        "wind_speed_10m":             "mean",
        "wind_direction_10m":         _circular_mean_degrees,  # circular variable — arithmetic mean is wrong
        "precipitation":              "sum",
        "soil_moisture_0_to_7cm":     "mean",
        "vpd":                        "mean",
        "data_quality_flag":          "min",
        # Derived features: take the value computed from the full window
        # (they are already per-cell scalars, so mean == any aggregation)
        "days_since_last_precipitation": "mean",
        "cumulative_wind_run_24h":       "mean",
        "drought_index_proxy":           "mean",
    }

    out = df.groupby("grid_id", as_index=False).agg(agg_spec)

    # Compute Canadian FWI from aggregated weather variables
    out["fire_weather_index"] = _compute_fwi(out, fwi_month)
    logger.info("Computed FWI for %d/%d cells (month=%d)",
                out["fire_weather_index"].notna().sum(), len(out), fwi_month)

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
# Canadian Fire Weather Index (FWI) — Van Wagner (1987)
# ---------------------------------------------------------------------------
# Reference: Van Wagner, C.E. (1987). Development and structure of the
# Canadian Forest Fire Weather Index System. Forestry Technical Report 35.
# Canadian Forestry Service, Ottawa.
#
# Standard startup values used when no previous-day codes are available.
# This gives a "current-conditions FWI" — accurate for relative fire danger
# comparison across cells; less accurate as an absolute cumulative drought index.
# ---------------------------------------------------------------------------

_FWI_FFMC0 = 85.0   # startup FFMC
_FWI_DMC0  =  6.0   # startup DMC
_FWI_DC0   = 15.0   # startup DC

# DMC/DC day-length adjustments by month (Jan=index 0)
_DMC_LE = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
_DC_LF  = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]


def _ffmc(T: float, H: float, W: float, ro: float) -> float:
    """Fine Fuel Moisture Code (FFMC)."""
    mo = 147.2 * (101.0 - _FWI_FFMC0) / (59.5 + _FWI_FFMC0)
    if ro > 0.5:
        rf = ro - 0.5
        if mo <= 150.0:
            mo = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
        else:
            mo = (mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
                  + 0.0015 * (mo - 150.0) ** 2 * math.sqrt(rf))
        mo = min(mo, 250.0)
    Ed = (0.942 * H ** 0.679 + 11.0 * math.exp((H - 100.0) / 10.0)
          + 0.18 * (21.1 - T) * (1.0 - math.exp(-0.115 * H)))
    Ew = (0.618 * H ** 0.753 + 10.0 * math.exp((H - 100.0) / 10.0)
          + 0.18 * (21.1 - T) * (1.0 - math.exp(-0.115 * H)))
    if mo > Ed:
        ko = (0.424 * (1.0 - (H / 100.0) ** 1.7)
              + 0.0694 * math.sqrt(W) * (1.0 - (H / 100.0) ** 8))
        m = Ed + (mo - Ed) * 10.0 ** -(ko * 0.581 * math.exp(0.0365 * T))
    elif mo < Ew:
        kl = (0.424 * (1.0 - ((100.0 - H) / 100.0) ** 1.7)
              + 0.0694 * math.sqrt(W) * (1.0 - ((100.0 - H) / 100.0) ** 8))
        m = Ew - (Ew - mo) * 10.0 ** -(kl * 0.581 * math.exp(0.0365 * T))
    else:
        m = mo
    return max(0.0, min(101.0, 59.5 * (250.0 - m) / (147.2 + m)))


def _dmc(T: float, H: float, ro: float, month: int) -> float:
    """Duff Moisture Code (DMC)."""
    P0 = _FWI_DMC0
    if ro > 1.5:
        re = 0.92 * ro - 1.27
        Mo = 20.0 + math.exp(5.6348 - P0 / 43.43)
        b = (100.0 / (0.5 + 0.3 * P0) if P0 <= 33.0
             else 14.0 - 1.3 * math.log(P0) if P0 <= 65.0
             else 6.2 * math.log(P0) - 17.2)
        Mr = Mo + 1000.0 * re / (48.77 + b * re)
        P0 = max(0.0, 244.72 - 43.43 * math.log(Mr - 20.0))
    K = 1.894 * (T + 1.1) * (100.0 - H) * _DMC_LE[month - 1] * 1e-6
    return max(0.0, P0 + 100.0 * K)


def _dc(T: float, ro: float, month: int) -> float:
    """Drought Code (DC)."""
    D0 = _FWI_DC0
    if ro > 2.8:
        Qo = 800.0 * math.exp(-D0 / 400.0)
        Qr = Qo + 3.937 * (0.83 * ro - 1.27)
        D0 = max(0.0, 400.0 * math.log(800.0 / Qr))
    V = max(0.0, 0.36 * (T + 2.8) + _DC_LF[month - 1])
    return D0 + 0.5 * V


def _isi(W: float, ffmc: float) -> float:
    """Initial Spread Index (ISI)."""
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    return (19.115 * math.exp(-0.1386 * m) * (1.0 + m ** 5.31 / 49_300_000.0)
            * math.exp(0.05039 * W))


def _bui(dmc: float, dc: float) -> float:
    """Buildup Index (BUI)."""
    if dmc == 0.0:
        return 0.0
    if dmc <= 0.4 * dc:
        return max(0.0, 0.8 * dmc * dc / (dmc + 0.4 * dc))
    return max(0.0, dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7))


def _fwi(isi: float, bui: float) -> float:
    """Fire Weather Index (FWI)."""
    fd = (0.626 * bui ** 0.809 + 2.0 if bui <= 80.0
          else 1000.0 / (25.0 + 108.64 * math.exp(-0.023 * bui)))
    B = 0.1 * isi * fd
    return max(0.0, math.exp(2.72 * (0.434 * math.log(B)) ** 0.647) if B > 1.0 else B)


def _compute_fwi(df_agg: pd.DataFrame, month: int) -> pd.Series:
    """Compute Canadian FWI for each row of the aggregated weather DataFrame.

    Uses standard startup codes (FFMC=85, DMC=6, DC=15) since no previous-day
    state is available in single-window mode.  Inputs are clipped to physically
    valid ranges before computation.

    Args:
        df_agg: One row per grid cell with temperature_2m, relative_humidity_2m,
                wind_speed_10m, precipitation columns.
        month:  Calendar month (1–12) used for day-length adjustments.

    Returns:
        float32 Series of FWI values in [0, 150], NaN where inputs are missing.
    """
    result = pd.Series(np.nan, index=df_agg.index, dtype="float32")
    required = {"temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"}
    if not required.issubset(df_agg.columns):
        return result

    for idx, row in df_agg.iterrows():
        T  = row["temperature_2m"]
        H  = row["relative_humidity_2m"]
        W  = row["wind_speed_10m"]
        ro = row["precipitation"]
        if any(pd.isna(v) for v in (T, H, W, ro)):
            continue
        T  = float(np.clip(T,  -50.0, 60.0))
        H  = float(np.clip(H,    1.0, 99.0))  # avoid 0/100 edge cases in log terms
        W  = float(max(0.0, W))
        ro = float(max(0.0, ro))
        try:
            ffmc_val = _ffmc(T, H, W, ro)
            fwi_val  = _fwi(_isi(W, ffmc_val), _bui(_dmc(T, H, ro, month), _dc(T, ro, month)))
            result.at[idx] = float(np.clip(fwi_val, 0.0, 150.0))
        except (ValueError, ZeroDivisionError, OverflowError):
            pass  # leave NaN for this cell if computation fails
    return result


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
