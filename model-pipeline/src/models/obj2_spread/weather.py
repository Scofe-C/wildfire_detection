"""
Weather formatting utilities for Cell2Fire.

Converts pipeline weather DataFrames into Cell2Fire's expected CSV format.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .exceptions import Cell2FireError

logger = logging.getLogger(__name__)

_COL_MAP: dict[str, str] = {
    "wind_speed_10m":       "ws",
    "wind_speed":           "ws",
    "ws":                   "ws",
    "wind_direction_10m":   "wd",
    "wind_direction":       "wd",
    "wd":                   "wd",
    "temperature_2m":       "tmp",
    "temperature":          "tmp",
    "tmp":                  "tmp",
    "relative_humidity_2m": "rh",
    "relative_humidity":    "rh",
    "rh":                   "rh",
}

_REQUIRED_COLS = ("ws", "wd", "tmp", "rh")
_TIMESTAMP_COLS = ("timestamp", "datetime", "time", "valid_time")


def format_weather_csv(
    weather_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Convert pipeline weather DataFrame to Cell2Fire CSV format.

    Cell2Fire expects columns:
        datetime, ws (m/s), wd (degrees), tmp (°C), rh (%)

    Parameters
    ----------
    weather_df : pd.DataFrame
        Weather data from the pipeline.
    output_path : str | Path
        Where to write the formatted CSV.

    Returns
    -------
    Path to the written CSV file.

    Raises
    ------
    Cell2FireError
        If required columns are missing after mapping.
    """
    output_path = Path(output_path)
    out_df = pd.DataFrame()

    # Find timestamp column
    for ts_col in _TIMESTAMP_COLS:
        if ts_col in weather_df.columns:
            out_df["datetime"] = pd.to_datetime(weather_df[ts_col])
            break
    else:
        raise Cell2FireError(
            f"No timestamp column found. Available: {list(weather_df.columns)}"
        )

    # Map weather columns
    for src_col, tgt_col in _COL_MAP.items():
        if src_col in weather_df.columns and tgt_col not in out_df.columns:
            out_df[tgt_col] = weather_df[src_col].values

    missing = [c for c in _REQUIRED_COLS if c not in out_df.columns]
    if missing:
        raise Cell2FireError(
            f"Missing weather columns after mapping: {missing}. "
            f"Available in input: {list(weather_df.columns)}"
        )

    out_df = out_df.sort_values("datetime").reset_index(drop=True)
    out_df.to_csv(output_path, index=False)
    logger.info("Weather CSV written: %d rows → %s", len(out_df), output_path)
    return output_path


def validate_weather_df(weather_df: pd.DataFrame) -> list[str]:
    """Check a weather DataFrame for issues before formatting.

    Parameters
    ----------
    weather_df : pd.DataFrame
        Input weather data.

    Returns
    -------
    List of warning strings. Empty if all OK.
    """
    warnings: list[str] = []

    if not any(c in weather_df.columns for c in _TIMESTAMP_COLS):
        warnings.append(f"No timestamp column. Expected one of: {_TIMESTAMP_COLS}")

    mapped = {_COL_MAP.get(c, c) for c in weather_df.columns}
    missing = [c for c in _REQUIRED_COLS if c not in mapped]
    if missing:
        warnings.append(f"Missing required weather columns: {missing}")

    for col in weather_df.columns:
        null_count = weather_df[col].isnull().sum()
        if null_count > 0:
            warnings.append(f"Column '{col}' has {null_count} null values")

    ws_col = next(
        (c for c in weather_df.columns if c in ("wind_speed_10m", "wind_speed", "ws")),
        None,
    )
    if ws_col:
        max_ws = weather_df[ws_col].max()
        if max_ws > 50:
            warnings.append(
                f"Wind speed max={max_ws:.1f} seems high — check units (expected m/s)"
            )

    return warnings
