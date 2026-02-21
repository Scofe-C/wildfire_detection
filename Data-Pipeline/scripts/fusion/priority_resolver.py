"""
Priority Hierarchy Engine
=========================
Resolves data source priority conflicts when ground truth data (drone,
firefighter, ICS-209) is available alongside satellite-derived fire
features.

Priority levels:
  1 = Ground truth (field telemetry)
  2 = Satellite (FIRMS VIIRS/MODIS)
  3 = Model inference (future ML predictions)

Behavior:
  - When Priority 1 data exists for a cell, it overrides satellite fire
    features for that cell AND neighbors within spatial_trust_radius_km.
  - Temporal decay: override expires after temporal_decay_hours.
  - Graceful no-op: when no ground truth is present, all rows stay at
    Priority 2 (satellite) and pass through unchanged.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default config values (used when schema_config.yaml is unavailable)
DEFAULT_PRIORITY_CONFIG = {
    "levels": {
        "ground_truth": 1,
        "satellite": 2,
        "model_inference": 3,
    },
    "spatial_trust_radius_km": 5.0,
    "temporal_decay_hours": 6,
}

# Fire feature columns that can be overridden by ground truth
OVERRIDABLE_FIRE_COLS = [
    "active_fire_count",
    "mean_frp",
    "median_frp",
    "max_confidence",
    "fire_detected_binary",
]


def _load_priority_config(config_path: Optional[str] = None) -> dict:
    """Load priority hierarchy config from schema_config.yaml or use defaults."""
    if config_path is None:
        # Try default location
        default_path = Path(__file__).resolve().parents[2] / "configs" / "schema_config.yaml"
        if default_path.exists():
            config_path = str(default_path)

    if config_path and Path(config_path).exists():
        try:
            import yaml
            with open(config_path, "r") as f:
                full_config = yaml.safe_load(f)
            if "priority_hierarchy" in full_config:
                return full_config["priority_hierarchy"]
        except Exception as e:
            logger.warning(f"Failed to load priority config: {e}")

    return DEFAULT_PRIORITY_CONFIG


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate haversine distance between two points in km."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _find_neighbors(
    fused_df: pd.DataFrame,
    lat: float,
    lon: float,
    radius_km: float,
) -> pd.Index:
    """Find grid cells within radius_km of a given lat/lon point."""
    if "latitude" not in fused_df.columns or "longitude" not in fused_df.columns:
        return pd.Index([])

    distances = fused_df.apply(
        lambda row: _haversine_km(lat, lon, row["latitude"], row["longitude"]),
        axis=1,
    )
    return fused_df.index[distances <= radius_km]


def resolve_priorities(
    fused_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    config_path: Optional[str] = None,
) -> pd.DataFrame:
    """Apply priority hierarchy to fused features.

    Args:
        fused_df: Main fused DataFrame (satellite-based features).
        ground_truth_df: Field telemetry DataFrame (drone/firefighter/ICS-209).
            Pass an empty DataFrame for no-op behavior.
        config_path: Optional path to schema_config.yaml.

    Returns:
        DataFrame with data_source_priority column and overridden fire features
        where ground truth data is available within spatial+temporal bounds.
    """
    result = fused_df.copy()
    config = _load_priority_config(config_path)

    satellite_priority = config.get("levels", {}).get("satellite", 2)
    ground_truth_priority = config.get("levels", {}).get("ground_truth", 1)
    spatial_radius = config.get("spatial_trust_radius_km", 5.0)
    temporal_decay_hours = config.get("temporal_decay_hours", 6)

    # Initialize priority column — default to satellite
    if "data_source_priority" not in result.columns:
        result["data_source_priority"] = satellite_priority

    # --- No-op path: no ground truth data ---
    if ground_truth_df is None or ground_truth_df.empty:
        logger.info(
            "Priority resolution: no ground truth data — "
            "all rows remain at Priority 2 (satellite)."
        )
        return result

    logger.info(
        f"Priority resolution: processing {len(ground_truth_df)} ground truth "
        f"observations (radius={spatial_radius}km, decay={temporal_decay_hours}h)"
    )

    # Parse timestamps if needed
    if "timestamp" in result.columns:
        result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce", utc=True)
    if "timestamp" in ground_truth_df.columns:
        ground_truth_df = ground_truth_df.copy()
        ground_truth_df["timestamp"] = pd.to_datetime(
            ground_truth_df["timestamp"], errors="coerce", utc=True
        )

    overrides_applied = 0

    for _, gt_row in ground_truth_df.iterrows():
        gt_lat = gt_row.get("latitude")
        gt_lon = gt_row.get("longitude")
        gt_ts = gt_row.get("timestamp")

        if gt_lat is None or gt_lon is None:
            continue

        # Find spatially nearby cells
        neighbors = _find_neighbors(result, gt_lat, gt_lon, spatial_radius)
        if len(neighbors) == 0:
            continue

        # Apply temporal decay filter
        if gt_ts is not None and "timestamp" in result.columns:
            time_diff = (result.loc[neighbors, "timestamp"] - gt_ts).abs()
            decay_limit = pd.Timedelta(hours=temporal_decay_hours)
            neighbors = neighbors[time_diff <= decay_limit]

        if len(neighbors) == 0:
            continue

        # Override fire features for nearby cells
        for col in OVERRIDABLE_FIRE_COLS:
            if col in gt_row.index and col in result.columns:
                gt_val = gt_row[col]
                if gt_val is not None and not (isinstance(gt_val, float) and np.isnan(gt_val)):
                    result.loc[neighbors, col] = gt_val

        # Upgrade priority
        result.loc[neighbors, "data_source_priority"] = ground_truth_priority
        overrides_applied += len(neighbors)

    logger.info(f"Priority resolution complete: {overrides_applied} cells overridden.")
    return result
