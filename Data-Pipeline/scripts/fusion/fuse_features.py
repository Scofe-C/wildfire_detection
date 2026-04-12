"""
Feature Fusion
==============
Joins processed data from all sources (FIRMS, weather, static layers)
into the unified feature table defined by schema_config.yaml.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from scripts.utils.grid_utils import generate_full_grid
from scripts.utils.schema_loader import get_registry

# Primary ingested weather columns checked by the circuit breaker (Item 6).
# Only direct API variables are included — derived/computed columns
# (fire_weather_index, vpd, soil_moisture_0_to_7cm, drought_index_proxy)
# are excluded because their absence does not indicate the weather source is down.
_WEATHER_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
]

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

    # Deduplicate right on grid_id before merging.
    # keep="first": the first concat'd source (e.g. CA weather with full 55-cell
    # grid) wins over the later region-scoped source (TX weather with 32 cells).
    # This preserves values for CA-only cells which only exist in the first source.
    if right["grid_id"].duplicated().any():
        before = len(right)
        right = right.drop_duplicates(subset="grid_id", keep="first")
        logger.debug(f"_safe_merge: deduped right {before} -> {len(right)} rows on grid_id")

    dup_cols = set(left.columns).intersection(set(right.columns)) - {"grid_id"}
    if dup_cols:
        right = right.drop(columns=list(dup_cols))

    return left.merge(right, on="grid_id", how=how)


def _apply_forward_fill(
    fused: pd.DataFrame,
    previous_fused_path: Optional[str],
    registry,
) -> pd.DataFrame:
    """Carry non-NaN values from the previous window for forward_fill columns.

    Called after the single-window fill strategies so that Open-Meteo
    outages don't leave weather columns permanently NaN.

    Args:
        fused:               Current window's fused DataFrame (modified in place).
        previous_fused_path: Path to the previous window's fused parquet.
                             If None or the file does not exist, returns fused unchanged.
        registry:            Schema registry (used to discover forward_fill columns).

    Returns:
        fused with NaN forward_fill columns patched from the previous window.
    """
    if not previous_fused_path:
        return fused

    prev_path = Path(previous_fused_path)
    if not prev_path.exists():
        logger.debug("Forward-fill: previous fused path not found (%s)", prev_path)
        return fused

    try:
        prev = pd.read_parquet(prev_path)
    except Exception as exc:
        logger.warning("Forward-fill: could not read previous fused parquet: %s", exc)
        return fused

    prev["grid_id"] = prev["grid_id"].astype(str)
    prev = prev.set_index("grid_id")

    ff_cols = [
        col for col, strategy in registry.get_fill_strategies().items()
        if strategy == "forward_fill"
        and col in fused.columns
        and col in prev.columns
    ]
    if not ff_cols:
        return fused

    filled_count = 0
    for col in ff_cols:
        null_mask = fused[col].isna()
        if not null_mask.any():
            continue
        fill_values = fused.loc[null_mask, "grid_id"].map(prev[col])
        fused.loc[null_mask, col] = fill_values
        newly_filled = fill_values.notna().sum()
        filled_count += newly_filled

    if filled_count > 0:
        logger.info(
            "Forward-fill: patched %d NaN values across %d columns from %s",
            filled_count, len(ff_cols), prev_path.name,
        )
    return fused


def check_weather_circuit_breaker(
    fused: pd.DataFrame,
    threshold: float = 0.80,
) -> None:
    """Raise ValueError if weather null rate exceeds threshold for any region.

    Called after fusion in task_fuse_features.  The DAG catches ValueError
    and re-raises as AirflowFailException to prevent export of garbage data.

    Args:
        fused:     Fused DataFrame (one row per grid cell).
        threshold: Null rate above which the circuit breaker trips (default 0.80).

    Raises:
        ValueError: With a human-readable message if any region trips the breaker.
    """
    weather_cols = [c for c in _WEATHER_COLS if c in fused.columns]
    if not weather_cols:
        return

    if "region" not in fused.columns or fused.empty:
        # Fall back to checking the whole dataset
        null_rate = fused[weather_cols].isnull().values.mean()
        if null_rate > threshold:
            raise ValueError(
                f"Circuit breaker: weather null rate {null_rate:.1%} > "
                f"{threshold:.0%}. Weather source may be down — "
                "aborting fusion export."
            )
        return

    for region, grp in fused.groupby("region", observed=True):
        null_rate = grp[weather_cols].isnull().values.mean()
        if null_rate > threshold:
            raise ValueError(
                f"Circuit breaker: weather null rate {null_rate:.1%} > "
                f"{threshold:.0%} in region '{region}'. "
                "Weather source may be down — aborting fusion export."
            )


def fuse_features(
    firms_features: pd.DataFrame,
    weather_features: pd.DataFrame,
    static_features: pd.DataFrame,
    execution_date: pd.Timestamp,
    resolution_km: int = 64,
    config_path: Optional[str] = None,
    previous_fused_path: Optional[str] = None,
    field_telemetry: Optional[pd.DataFrame] = None,
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
        previous_fused_path: Optional path to the previous window's fused
            parquet.  When provided, NaN values in ``forward_fill`` columns
            are patched with values from the previous window (Item 5).

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
    region_col = ["region"] if "region" in master_grid.columns else []
    master_grid = master_grid[["grid_id"] + region_col + ["latitude", "longitude"]].copy()
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

    # 2b) field telemetry override — ground truth has priority 1 (highest)
    #     When field observations confirm fire at a grid cell, override
    #     satellite-derived fire_detected_binary and boost confidence.
    if field_telemetry is not None and not field_telemetry.empty:
        logger.info(
            "Merging %d field telemetry observations (ground truth priority)",
            len(field_telemetry),
        )
        # Match field observations to nearest grid cell by H3 lookup
        try:
            import h3
            for _, obs in field_telemetry.iterrows():
                lat, lon = float(obs.get("latitude", 0)), float(obs.get("longitude", 0))
                if lat == 0 and lon == 0:
                    continue
                # Find the H3 cell for this observation
                h3_res_map = {64: 2, 22: 5, 10: 4, 1: 7}
                h3_res = h3_res_map.get(resolution_km, 5)
                cell_id = h3.latlng_to_cell(lat, lon, h3_res)
                # Override fire detection for this cell
                mask = fused["grid_id"] == cell_id
                if mask.any():
                    fused.loc[mask, "fire_detected_binary"] = 1
                    if "data_source_priority" not in fused.columns:
                        fused["data_source_priority"] = 2  # default: satellite
                    fused.loc[mask, "data_source_priority"] = 1  # ground truth
                    if obs.get("frp") and "mean_frp" in fused.columns:
                        # Use field FRP if higher than satellite
                        current_frp = fused.loc[mask, "mean_frp"].values[0]
                        if obs["frp"] > current_frp:
                            fused.loc[mask, "mean_frp"] = obs["frp"]
                    logger.debug("Field telemetry matched cell %s", cell_id)
        except ImportError:
            logger.warning("h3 not installed — field telemetry spatial matching skipped")
        except Exception as e:
            logger.warning("Field telemetry merge failed (non-blocking): %s", e)

    # 3) weather aggregate + merge (merge weather_agg, not raw weather_features)
    #
    # process_weather.py saves one already-aggregated row per grid cell with no
    # timestamp column — temporal windowing has already happened upstream.
    # Only call _aggregate_weather_to_window when the raw multi-row format
    # (one row per observation timestamp) is supplied instead.
    if (weather_features is not None) and (not weather_features.empty):
        if "timestamp" in weather_features.columns:
            weather_agg = _aggregate_weather_to_window(
                weather_features,
                pd.Timestamp(execution_date),
                registry.temporal_aggregation_hours,
            )
        else:
            # Already aggregated by process_weather — use directly.
            weather_agg = weather_features.copy()
            weather_agg["grid_id"] = weather_agg["grid_id"].astype(str)
            logger.info(
                f"Weather already aggregated ({len(weather_agg)} rows, "
                f"no timestamp column) — skipping temporal windowing."
            )
    else:
        # No weather data at all — produce empty frame so left-merge still
        # creates NaN-filled weather columns that fill strategies can handle.
        _weather_cols = [
            "grid_id", "temperature_2m", "relative_humidity_2m",
            "wind_speed_10m", "wind_direction_10m", "precipitation",
            "soil_moisture_0_to_7cm", "vpd", "fire_weather_index",
            "data_quality_flag", "days_since_last_precipitation",
            "cumulative_wind_run_24h", "drought_index_proxy",
        ]
        weather_agg = pd.DataFrame(columns=_weather_cols)
        logger.warning("No weather features available — weather columns will use fill strategies.")

    fused = _safe_merge(fused, weather_agg, how="left")

    # 4) static merge
    fused = _safe_merge(fused, static_features, how="left")

    # 5) fill strategies (single-window: zero/constant fills only)
    fused = _apply_fill_strategies(fused, registry)

    # 5b) cross-run forward-fill from previous window (Item 5)
    fused = _apply_forward_fill(fused, previous_fused_path, registry)

    # 6) enforce registry columns first so quality flags see all expected cols
    #    (missing static cols appear as NaN here, enabling flag 4/5 detection)
    expected_columns = registry.get_feature_names()
    for col in expected_columns:
        if col not in fused.columns:
            fused[col] = None

    # 7) quality flag (computed after all registry cols are present as NaN stubs)
    fused["data_quality_flag"] = _compute_quality_flags(fused)

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


_STATIC_STUB_COLS = {
    "fuel_model_fbfm40", "canopy_cover_pct", "vegetation_type",
    "ndvi", "elevation_m", "slope_degrees", "aspect_degrees",
    "dominant_fuel_fraction",
    # Optional LANDFIRE spread-simulation layers (NaN until rasters downloaded)
    "canopy_base_height_m", "canopy_bulk_density", "evt_national_class",
}

# Columns that are intentionally always-null until Phase 2 is implemented.
# Excluded from quality flag null-rate calculation so they don't falsely
# inflate missing-data counts.
_PHASE2_PLACEHOLDER_COLS = {"fire_weather_index", "ndvi"}

_EXCLUDE_FROM_QUALITY = {
    "grid_id", "latitude", "longitude", "timestamp",
    "resolution_km", "region", "data_quality_flag",
}


def _compute_quality_flags(fused: pd.DataFrame) -> pd.Series:
    """Compute data quality flags.

    Flag values:
      0 = good (all dynamic sources present, static sources loaded)
      3 = >30% nulls in non-static, non-placeholder dynamic columns
      4 = static source fully unavailable (all static cols NaN for this row)
      5 = partial static (some static cols NaN — boundary cell or missing tile)

    Phase 2 placeholder columns (fire_weather_index, ndvi) are excluded from
    the null-rate calculation since they are intentionally always null until
    the corresponding ingestion scripts are run.
    """
    flags = pd.Series(0, index=fused.index, dtype="int8")

    # Flag 3: >30% nulls in dynamic (non-static, non-placeholder) columns
    dynamic_cols = [
        c for c in fused.columns
        if c not in _EXCLUDE_FROM_QUALITY
        and c not in _STATIC_STUB_COLS
        and c not in _PHASE2_PLACEHOLDER_COLS
    ]
    if dynamic_cols:
        null_fraction = fused[dynamic_cols].isnull().mean(axis=1)
        flags = flags.where(null_fraction < 0.3, 3)

    # Static cols actually present in the DataFrame
    static_present = [c for c in _STATIC_STUB_COLS if c in fused.columns]
    # Exclude known Phase 2 always-null from static check
    static_checkable = [c for c in static_present if c not in _PHASE2_PLACEHOLDER_COLS]

    if static_checkable:
        null_counts = fused[static_checkable].isnull().sum(axis=1)
        all_null  = null_counts == len(static_checkable)
        some_null = (null_counts > 0) & ~all_null

        # Flag 4: all checkable static cols NaN — overrides flag 3
        flags = flags.where(~all_null, 4)
        # Flag 5: partial static NaN — overrides flag 3 (more specific diagnosis)
        flags = flags.where(~some_null, 5)

    return flags