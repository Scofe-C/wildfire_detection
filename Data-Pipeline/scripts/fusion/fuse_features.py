"""
Feature Fusion
==============
Joins processed data from all sources (FIRMS, weather, static layers)
into the unified feature table defined by schema_config.yaml.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from scripts.utils.grid_utils import generate_full_grid
from scripts.utils.schema_loader import get_registry

logger = logging.getLogger(__name__)


def _ensure_grid_id_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Return a DataFrame that at least has a 'grid_id' column."""
    if df is None:
        return pd.DataFrame(columns=["grid_id"])
    if isinstance(df, pd.DataFrame) and (len(df) == 0):
        # empty df might have no columns at all
        if "grid_id" not in df.columns:
            return pd.DataFrame(columns=["grid_id"])
        return df
    return df


def _safe_merge(left: pd.DataFrame, right: Optional[pd.DataFrame], *, how: str = "left") -> pd.DataFrame:
    """
    Merge on grid_id safely.
    - If right is None / empty / missing grid_id -> return left unchanged.
    - Ensures grid_id dtype aligns as string.
    - Avoids duplicate columns from right (except grid_id).
    """
    right = _ensure_grid_id_df(right)

    if right is None or len(right) == 0 or "grid_id" not in right.columns:
        return left

    left = left.copy()
    right = right.copy()

    if "grid_id" not in left.columns:
        raise KeyError("left DataFrame missing required key: 'grid_id'")

    left["grid_id"] = left["grid_id"].astype(str)
    right["grid_id"] = right["grid_id"].astype(str)

    dup_cols = set(left.columns).intersection(set(right.columns)) - {"grid_id"}
    if dup_cols:
        right = right.drop(columns=list(dup_cols))

    return left.merge(right, on="grid_id", how=how)


def fuse_features(
    firms_features: pd.DataFrame,
    weather_features: pd.DataFrame,
    static_features: pd.DataFrame,
    execution_date: pd.Timestamp,
    resolution_km: int = 64,
    config_path: Optional[str] = None,
) -> pd.DataFrame:
    """Merge FIRMS fire, weather, and static terrain features into a unified table.

    Generates a master grid at the given resolution, left-joins each data
    source, applies fill strategies from the schema registry, and enforces
    the expected column order.

    Args:
        firms_features: Processed FIRMS fire detection features.
        weather_features: Processed weather observation features.
        static_features: Static terrain/fuel features.
        execution_date: Timestamp of the current pipeline window.
        resolution_km: Grid resolution in km (default 64).
        config_path: Optional path to schema_config.yaml.

    Returns:
        Fused DataFrame with one row per grid cell.
    """
    registry = get_registry(config_path)

    firms_features = _ensure_grid_id_df(firms_features)
    weather_features = _ensure_grid_id_df(weather_features)
    static_features = _ensure_grid_id_df(static_features)

    logger.info(
        f"Fusing features: {len(firms_features)} fire rows, "
        f"{len(weather_features)} weather rows, "
        f"{len(static_features)} static rows"
    )

    # 1) master grid
    master_grid = generate_full_grid(resolution_km)
    master_grid = master_grid[["grid_id", "latitude", "longitude"]].copy()
    master_grid["grid_id"] = master_grid["grid_id"].astype(str)
    master_grid["timestamp"] = pd.Timestamp(execution_date)
    master_grid["resolution_km"] = resolution_km

    fused = master_grid.copy()

    # 2) fire merge
    fused = _safe_merge(fused, firms_features, how="left")

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

    # 3) weather aggregate + merge (merge weather_agg, not raw weather_features)
    if (weather_features is not None) and (not weather_features.empty) and ("timestamp" in weather_features.columns):
        weather_agg = _aggregate_weather_to_window(
            weather_features,
            pd.Timestamp(execution_date),
            registry.temporal_aggregation_hours,
        )
    else:
        weather_agg = pd.DataFrame(columns=["grid_id"])

    fused = _safe_merge(fused, weather_agg, how="left")

    # 4) static merge
    fused = _safe_merge(fused, static_features, how="left")

    # 5) fill strategies
    fused = _apply_fill_strategies(fused, registry)

    # 6) quality flag
    fused["data_quality_flag"] = _compute_quality_flags(fused)

    # 7) enforce registry columns & order
    expected_columns = registry.get_feature_names()
    for col in expected_columns:
        if col not in fused.columns:
            fused[col] = None

    fused = fused.loc[:, ~fused.columns.duplicated()]
    fused = fused[[c for c in expected_columns if c in fused.columns]]

    return fused


# ---------------------------------------------------------------------------
# Temporal Lag — ML-ready variant (Plan §Problem 2)
# ---------------------------------------------------------------------------
# Fire context features that must use the PREVIOUS time window (T-1) to avoid
# data leakage.  fire_detected_binary is the prediction LABEL and stays at T.
FIRE_CONTEXT_LAG_COLS = [
    "active_fire_count",
    "mean_frp",
    "median_frp",
    "max_confidence",
    "nearest_fire_distance_km",
]


def apply_temporal_lag(
    fused: pd.DataFrame,
    prev_fire_features: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Produce an ML-ready copy of *fused* with fire context from T-1.

    - ``FIRE_CONTEXT_LAG_COLS`` are replaced by values from
      *prev_fire_features* (the previous 6-hour window).
    - ``fire_detected_binary`` is kept from the current window (T) as the
      prediction label.
    - If *prev_fire_features* is None or empty the lagged columns are filled
      with their default values (0 / 0.0 / -1.0) so the output shape is
      always stable.

    Returns a **new** DataFrame; the original *fused* is not modified.
    """
    ml = fused.copy()

    prev = _ensure_grid_id_df(prev_fire_features)

    if prev is not None and not prev.empty and "grid_id" in prev.columns:
        prev = prev.copy()
        prev["grid_id"] = prev["grid_id"].astype(str)

        # Keep only lag columns + grid_id from prev
        available = [c for c in FIRE_CONTEXT_LAG_COLS if c in prev.columns]
        if available:
            prev_subset = prev[["grid_id"] + available].copy()

            # Drop current-window fire context, merge in T-1
            ml = ml.drop(columns=available, errors="ignore")
            ml = ml.merge(prev_subset, on="grid_id", how="left")

            logger.info(
                f"Temporal lag applied: {len(available)} fire context cols "
                f"replaced with T-1 values ({len(prev_subset)} rows)"
            )
    else:
        logger.warning(
            "No previous fire features provided — filling lagged columns "
            "with defaults (no temporal lag applied)"
        )

    # Guarantee columns exist with defaults even if prev was empty
    lag_defaults = {
        "active_fire_count": 0,
        "mean_frp": 0.0,
        "median_frp": 0.0,
        "max_confidence": 0,
        "nearest_fire_distance_km": -1.0,
    }
    for col, default in lag_defaults.items():
        if col not in ml.columns:
            ml[col] = default
        else:
            ml[col] = ml[col].fillna(default)

    return ml


def fuse_features_for_ml(
    firms_features: pd.DataFrame,
    weather_features: pd.DataFrame,
    static_features: pd.DataFrame,
    execution_date: pd.Timestamp,
    prev_fire_features: Optional[pd.DataFrame] = None,
    resolution_km: int = 64,
    config_path: Optional[str] = None,
) -> pd.DataFrame:
    """Convenience wrapper: fuse + apply temporal lag for ML training.

    Returns an ML-ready DataFrame where fire context columns reflect the
    previous time window (T-1) while ``fire_detected_binary`` is the
    current-window label (T).
    """
    fused = fuse_features(
        firms_features=firms_features,
        weather_features=weather_features,
        static_features=static_features,
        execution_date=execution_date,
        resolution_km=resolution_km,
        config_path=config_path,
    )
    return apply_temporal_lag(fused, prev_fire_features)


def _aggregate_weather_to_window(
    weather_df: pd.DataFrame,
    execution_date: pd.Timestamp,
    window_hours: int,
) -> pd.DataFrame:
    """Aggregate weather data to the time window ending at execution_date.

    Filters weather rows to [execution_date - window_hours, execution_date],
    then groups by grid_id and computes mean (sum for precipitation).

    Returns an empty DataFrame with a 'grid_id' column if no weather rows
    fall within the window (Bug #2 fix: no silent fallback to all data).
    """
    weather_df = weather_df.copy()
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], errors="coerce")

    window_start = execution_date - pd.Timedelta(hours=window_hours)
    window_end = execution_date

    mask = (weather_df["timestamp"] >= window_start) & (weather_df["timestamp"] <= window_end)
    windowed = weather_df[mask]

    if windowed.empty:
        logger.warning(
            f"No weather rows in [{window_start}, {window_end}] window. "
            f"Returning empty — fusion will use fill strategies."
        )
        return pd.DataFrame(columns=["grid_id"])

    if "grid_id" not in windowed.columns:
        return pd.DataFrame(columns=["grid_id"])

    windowed = windowed.copy()
    windowed["grid_id"] = windowed["grid_id"].astype(str)

    numeric_cols = windowed.select_dtypes(include=[np.number]).columns.tolist()
    if "grid_id" in numeric_cols:
        numeric_cols.remove("grid_id")

    agg_dict = {col: "mean" for col in numeric_cols}
    if "precipitation" in agg_dict:
        agg_dict["precipitation"] = "sum"

    if not agg_dict:
        return pd.DataFrame(columns=["grid_id"])

    return windowed.groupby("grid_id").agg(agg_dict).reset_index()


def _apply_fill_strategies(df: pd.DataFrame, registry) -> pd.DataFrame:
    """Apply per-column fill strategies (zero, forward_fill, constant) from the registry."""
    fill_strategies = registry.get_fill_strategies()

    for col, strategy in fill_strategies.items():
        if col not in df.columns:
            continue

        null_count = df[col].isnull().sum()
        if null_count == 0:
            continue

        if strategy == "forward_fill":
            # single run has no history; skip
            continue
        elif strategy == "zero":
            df[col] = df[col].fillna(0)
        elif isinstance(strategy, (int, float)):
            df[col] = df[col].fillna(strategy)

    return df


def _compute_quality_flags(fused: pd.DataFrame) -> pd.Series:
    """Compute data quality flags: 0 = good, 3 = >30% nulls in feature columns."""
    flags = pd.Series(0, index=fused.index, dtype="int8")

    core_cols = [
        c for c in fused.columns
        if c not in ["grid_id", "latitude", "longitude", "timestamp", "resolution_km", "data_quality_flag"]
    ]
    if core_cols:
        null_fraction = fused[core_cols].isnull().mean(axis=1)
        flags = flags.where(null_fraction < 0.3, 3)

    return flags
