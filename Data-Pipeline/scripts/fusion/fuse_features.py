"""
Feature Fusion
==============
Joins processed data from all sources (FIRMS, weather, static layers)
into the unified feature table defined by schema_config.yaml.

Owner: Person D
Dependencies: pandas, numpy

Key behaviors:
    - Left-joins from the complete grid master table to ensure every cell
      appears in every timestep (no cells dropped due to missing data)
    - Applies fill strategies from the feature registry for missing values
    - Tags data quality for each row
    - Produces the final DataFrame ready for validation and export
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from scripts.utils.grid_utils import generate_full_grid
from scripts.utils.schema_loader import get_registry

logger = logging.getLogger(__name__)


def fuse_features(
    firms_features: pd.DataFrame,
    weather_features: pd.DataFrame,
    static_features: pd.DataFrame,
    execution_date: pd.Timestamp,
    resolution_km: int = 64,
    config_path: Optional[str] = None,
) -> pd.DataFrame:
    """Fuse all data sources into the unified feature table.

    This is the main entry point called by the Airflow task.

    Args:
        firms_features: Processed FIRMS grid-level features from process_firms.
        weather_features: Processed weather data from process_weather.
        static_features: Static layers (LANDFIRE + SRTM) from load_static_layers.
        execution_date: Canonical pipeline execution timestamp.
        resolution_km: Grid resolution.
        config_path: Optional schema config override.

    Returns:
        Fused DataFrame conforming to the schema_config.yaml feature registry.
    """
    registry = get_registry(config_path)

    logger.info(
        f"Fusing features: {len(firms_features)} fire rows, "
        f"{len(weather_features)} weather rows, "
        f"{len(static_features)} static rows"
    )

    # --- Step 1: Generate the master grid (all cells, guaranteed complete) ---
    master_grid = generate_full_grid(resolution_km, config_path)
    master_grid = master_grid[["grid_id", "latitude", "longitude"]].copy()
    master_grid["timestamp"] = execution_date
    master_grid["resolution_km"] = resolution_km

    logger.info(f"Master grid: {len(master_grid)} cells")

    # --- Step 2: Left-join fire features ---
    fused = master_grid.merge(
        firms_features,
        on="grid_id",
        how="left",
        suffixes=("", "_fire"),
    )

    # Fill missing fire features with defaults (no fire detected)
    fire_defaults = {
        "active_fire_count": 0,
        "mean_frp": 0.0,
        "median_frp": 0.0,
        "max_confidence": 0,
        "nearest_fire_distance_km": -1.0,
        "fire_detected_binary": 0,
    }
    for col, default in fire_defaults.items():
        if col in fused.columns:
            fused[col] = fused[col].fillna(default)
        else:
            fused[col] = default

    # --- Step 3: Left-join weather features ---
    # Weather data may have multiple timestamps per grid cell (hourly);
    # aggregate to the 6-hour window matching execution_date
    if not weather_features.empty and "timestamp" in weather_features.columns:
        weather_agg = _aggregate_weather_to_window(
            weather_features, execution_date, registry.temporal_aggregation_hours
        )
    else:
        weather_agg = pd.DataFrame(columns=["grid_id"])

    fused = fused.merge(
        weather_agg,
        on="grid_id",
        how="left",
        suffixes=("", "_weather"),
    )

    # --- Step 4: Left-join static features ---
    if not static_features.empty:
        fused = fused.merge(
            static_features,
            on="grid_id",
            how="left",
            suffixes=("", "_static"),
        )

    # --- Step 5: Apply fill strategies for missing values ---
    fused = _apply_fill_strategies(fused, registry)

    # --- Step 6: Compute data quality flag ---
    fused["data_quality_flag"] = _compute_quality_flags(fused, weather_agg)

    # --- Step 7: Select and order columns per the feature registry ---
    expected_columns = registry.get_feature_names()
    for col in expected_columns:
        if col not in fused.columns:
            fused[col] = None
            logger.warning(f"Column '{col}' missing after fusion — filled with None")

    # Keep only registered columns, in registry order
    available = [c for c in expected_columns if c in fused.columns]
    fused = fused[available]

    # --- Step 8: Remove duplicate suffix columns from merges ---
    fused = fused.loc[:, ~fused.columns.duplicated()]

    logger.info(
        f"Fusion complete: {len(fused)} rows × {len(fused.columns)} columns. "
        f"Null rates: {fused.isnull().mean().to_dict()}"
    )
    return fused


def _aggregate_weather_to_window(
    weather_df: pd.DataFrame,
    execution_date: pd.Timestamp,
    window_hours: int,
) -> pd.DataFrame:
    """Aggregate hourly weather data to a single row per grid cell.

    Takes the mean of continuous variables within the time window.
    """
    weather_df = weather_df.copy()
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])

    window_start = execution_date - pd.Timedelta(hours=window_hours)
    window_end = execution_date

    # Filter to the time window
    mask = (weather_df["timestamp"] >= window_start) & (
        weather_df["timestamp"] <= window_end
    )
    windowed = weather_df[mask]

    if windowed.empty:
        logger.warning(
            f"No weather data in window [{window_start}, {window_end}]. "
            f"Using all available data as fallback."
        )
        windowed = weather_df

    # Aggregate numeric columns by grid cell
    numeric_cols = windowed.select_dtypes(include=[np.number]).columns.tolist()
    if "grid_id" in numeric_cols:
        numeric_cols.remove("grid_id")

    agg_dict = {col: "mean" for col in numeric_cols}

    # Precipitation should be summed, not averaged
    if "precipitation" in agg_dict:
        agg_dict["precipitation"] = "sum"

    if not agg_dict:
        return pd.DataFrame(columns=["grid_id"])

    aggregated = windowed.groupby("grid_id").agg(agg_dict).reset_index()
    return aggregated


def _apply_fill_strategies(
    df: pd.DataFrame, registry
) -> pd.DataFrame:
    """Apply configured fill strategies for nullable columns.

    Strategies from schema_config.yaml:
    - 'forward_fill': Use last known value (requires historical context)
    - 'zero': Fill with 0
    - numeric value: Fill with that specific value
    """
    fill_strategies = registry.get_fill_strategies()

    for col, strategy in fill_strategies.items():
        if col not in df.columns:
            continue

        null_count = df[col].isnull().sum()
        if null_count == 0:
            continue

        if strategy == "forward_fill":
            # In a single-timestep context, forward fill doesn't help.
            # Mark these as needing historical data (handled by the Airflow
            # task that loads previous pipeline output).
            logger.info(
                f"Column '{col}': {null_count} nulls marked for forward-fill "
                f"(requires historical context)"
            )
        elif strategy == "zero":
            df[col] = df[col].fillna(0)
            logger.info(f"Column '{col}': filled {null_count} nulls with 0")
        elif isinstance(strategy, (int, float)):
            df[col] = df[col].fillna(strategy)
            logger.info(f"Column '{col}': filled {null_count} nulls with {strategy}")

    return df


def _compute_quality_flags(
    fused: pd.DataFrame, weather_agg: pd.DataFrame
) -> pd.Series:
    """Compute per-row data quality flags.

    Flag values:
        0 = All sources present, fresh data
        1 = Weather data was forward-filled from previous run
        2 = Weather data came from NWS fallback
        3 = Partial data (some features missing)
    """
    flags = pd.Series(0, index=fused.index, dtype="int8")

    # Check if weather data quality flags were propagated
    if "data_quality_flag_weather" in fused.columns:
        flags = fused["data_quality_flag_weather"].fillna(0).astype("int8")

    # Check for rows with significant missing data
    registry_cols = [
        c for c in fused.columns
        if c not in ["grid_id", "latitude", "longitude", "timestamp",
                     "resolution_km", "data_quality_flag"]
    ]
    null_fraction = fused[registry_cols].isnull().mean(axis=1)
    flags = flags.where(null_fraction < 0.3, 3)

    return flags
