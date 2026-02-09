"""
FIRMS Data Processing
=====================
Transforms raw FIRMS point detections into grid-level aggregate features.
Handles spatial join to H3 grid, FRP outlier clipping, and computation
of all fire_context schema features.

Owner: Person A
Dependencies: pandas, numpy, h3

Input: Raw FIRMS CSV from ingest_firms
Output: DataFrame with grid-level fire features ready for fusion
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from scripts.utils.grid_utils import (
    generate_full_grid,
    points_to_grid_ids,
    get_cell_neighbors,
)
from scripts.utils.schema_loader import get_registry

logger = logging.getLogger(__name__)

# FRP clipping percentile to handle sensor saturation outliers
FRP_CLIP_PERCENTILE = 99.5


def process_firms_data(
    raw_csv_path: str,
    resolution_km: int = 64,
    config_path: Optional[str] = None,
) -> pd.DataFrame:
    """Process raw FIRMS detections into grid-level fire features.

    This is the main entry point called by the Airflow task.

    Args:
        raw_csv_path: Path to the raw FIRMS CSV from ingest_firms.
        resolution_km: Target grid resolution.
        config_path: Optional schema config override.

    Returns:
        DataFrame with columns: grid_id, active_fire_count, mean_frp,
        median_frp, max_confidence, nearest_fire_distance_km,
        fire_detected_binary.
    """
    logger.info(f"Processing FIRMS data from {raw_csv_path}")

    raw_df = pd.read_csv(raw_csv_path)

    if raw_df.empty:
        logger.info("No fire detections to process. Returning empty features.")
        return _empty_fire_features()

    # --- Step 1: Clean and validate ---
    raw_df = _clean_raw_firms(raw_df)

    # --- Step 2: Assign detections to H3 grid cells ---
    raw_df["grid_id"] = points_to_grid_ids(
        raw_df["latitude"].values,
        raw_df["longitude"].values,
        resolution_km=resolution_km,
        config_path=config_path,
    )

    # --- Step 3: Clip FRP outliers ---
    raw_df = _clip_frp_outliers(raw_df)

    # --- Step 4: Aggregate to grid level ---
    grid_features = _aggregate_to_grid(raw_df)

    # --- Step 5: Compute nearest fire distance ---
    grid_features = _compute_nearest_fire_distance(
        grid_features, resolution_km, config_path
    )

    logger.info(
        f"FIRMS processing complete: {len(grid_features)} cells with fire data "
        f"({grid_features['fire_detected_binary'].sum()} with active fires)"
    )
    return grid_features


def _clean_raw_firms(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw FIRMS data: drop invalid rows, normalize confidence.

    MODIS uses 'l'/'n'/'h' for confidence, VIIRS uses 0-100 integer.
    We normalize MODIS to numeric: l=30, n=60, h=90.
    """
    original_len = len(df)

    # Drop rows with missing coordinates
    df = df.dropna(subset=["latitude", "longitude"])

    # Normalize MODIS confidence to numeric
    if "confidence" in df.columns:
        confidence_map = {"l": 30, "n": 60, "h": 90, "low": 30, "nominal": 60, "high": 90}
        df["confidence"] = df["confidence"].apply(
            lambda x: confidence_map.get(str(x).lower().strip(), x)
        )
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    # Ensure FRP is numeric
    if "frp" in df.columns:
        df["frp"] = pd.to_numeric(df["frp"], errors="coerce")

    dropped = original_len - len(df)
    if dropped > 0:
        logger.info(f"Cleaned FIRMS data: dropped {dropped} invalid rows")

    return df


def _clip_frp_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Clip FRP values at the 99.5th percentile to handle sensor saturation."""
    if "frp" not in df.columns or df["frp"].isna().all():
        return df

    valid_frp = df["frp"].dropna()
    if len(valid_frp) == 0:
        return df

    clip_threshold = valid_frp.quantile(FRP_CLIP_PERCENTILE / 100.0)
    clipped_count = (df["frp"] > clip_threshold).sum()

    if clipped_count > 0:
        logger.info(
            f"Clipping {clipped_count} FRP values above {clip_threshold:.1f} MW "
            f"(99.5th percentile)"
        )
        df["frp"] = df["frp"].clip(upper=clip_threshold)

    # Also clip negative FRP (sensor artifacts)
    negative_count = (df["frp"] < 0).sum()
    if negative_count > 0:
        logger.info(f"Clipping {negative_count} negative FRP values to 0")
        df["frp"] = df["frp"].clip(lower=0)

    return df


def _aggregate_to_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate point detections to grid-level features."""
    agg_dict = {
        "latitude": "count",  # Will become active_fire_count
    }

    if "frp" in df.columns:
        agg_dict["frp"] = ["mean", "median"]
    if "confidence" in df.columns:
        agg_dict["confidence"] = "max"

    grid_agg = df.groupby("grid_id").agg(agg_dict)

    # Flatten multi-level column index
    grid_agg.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in grid_agg.columns
    ]

    # Rename to match schema
    rename_map = {
        "latitude_count": "active_fire_count",
        "frp_mean": "mean_frp",
        "frp_median": "median_frp",
        "confidence_max": "max_confidence",
    }
    grid_agg = grid_agg.rename(columns=rename_map)
    grid_agg = grid_agg.reset_index()

    # Add binary label
    grid_agg["fire_detected_binary"] = 1

    # Ensure all expected columns exist
    for col in ["active_fire_count", "mean_frp", "median_frp", "max_confidence"]:
        if col not in grid_agg.columns:
            grid_agg[col] = 0 if "count" in col else 0.0

    return grid_agg


def _compute_nearest_fire_distance(
    grid_features: pd.DataFrame,
    resolution_km: int,
    config_path: Optional[str] = None,
) -> pd.DataFrame:
    """Compute distance from each fire cell to the nearest other fire cell.

    For cells without fire, this will be set during fusion (when we have
    the full grid). Here we compute it only for cells WITH fire.
    """
    if len(grid_features) <= 1:
        grid_features["nearest_fire_distance_km"] = -1.0
        return grid_features

    try:
        import h3

        fire_cells = grid_features["grid_id"].tolist()
        distances = []

        for cell_id in fire_cells:
            min_dist = float("inf")
            for other_id in fire_cells:
                if other_id != cell_id:
                    # h3.h3_distance returns grid distance (number of cells)
                    try:
                        grid_dist = h3.h3_distance(cell_id, other_id)
                        # Approximate km distance
                        km_dist = grid_dist * resolution_km * 0.75  # Hex spacing factor
                        min_dist = min(min_dist, km_dist)
                    except h3.H3ValueError:
                        continue

            distances.append(min_dist if min_dist != float("inf") else -1.0)

        grid_features["nearest_fire_distance_km"] = distances

    except Exception as e:
        logger.warning(f"Could not compute nearest fire distance: {e}")
        grid_features["nearest_fire_distance_km"] = -1.0

    return grid_features


def _empty_fire_features() -> pd.DataFrame:
    """Return an empty DataFrame with the correct fire feature columns."""
    return pd.DataFrame(columns=[
        "grid_id", "active_fire_count", "mean_frp", "median_frp",
        "max_confidence", "nearest_fire_distance_km", "fire_detected_binary",
    ])
