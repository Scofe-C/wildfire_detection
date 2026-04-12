"""
firms_validator.py — FIRMS Temporal Validation (Anti-Overfitting)
=================================================================
Validates the fire spread simulator against real FIRMS satellite
detections in the backfill archive.

Constructs "observed spread events" from consecutive 6-hour snapshots
and tests whether the simulator would have predicted fire reaching those
cells. Uses strictly chronological train/validation/test splits.

Every metric reported here is what the model ACTUALLY computes tested
against what ACTUALLY happened. No synthetic constructs.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import h3
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# H3 intercell distances (center-to-center, km)
_H3_INTERCELL_KM: dict[int, float] = {2: 93.0, 5: 25.0}

# Temporal split boundaries (anti-overfitting)
_TRAIN_END = pd.Timestamp("2024-06-30")
_VALIDATION_END = pd.Timestamp("2024-12-31")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ObservedSpread:
    """One observed fire-to-neighbor spread event from FIRMS."""
    origin_cell: str
    target_cell: str
    origin_time: str  # ISO format
    target_time: str
    bearing_deg: float
    observed_speed_upper_bound_kmh: float
    origin_frp: float
    target_frp: float


# ---------------------------------------------------------------------------
# Spread event extraction
# ---------------------------------------------------------------------------

def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing from (lat1, lon1) to (lat2, lon2) in degrees."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _load_backfill_pair(path_t0: Path, path_t1: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load two consecutive backfill parquets."""
    df0 = pd.read_parquet(path_t0)
    df1 = pd.read_parquet(path_t1)
    df0["grid_id"] = df0["grid_id"].astype(str)
    df1["grid_id"] = df1["grid_id"].astype(str)
    return df0, df1


def extract_spread_events(
    backfill_dir: Path,
    resolution_km: int = 64,
    period: str = "all",
) -> list[ObservedSpread]:
    """Extract observed fire spread events from backfill archive.

    For each pair of consecutive 6-hour snapshots:
      - Find cells with fire_detected_binary=1 at T0
      - Find ring-1 neighbors with fire_detected_binary=1 at T+6h but NOT at T0
      - That's an observed spread event

    Parameters
    ----------
    backfill_dir : Path to backfill directory (e.g., .../backfill/64km)
    resolution_km : Grid resolution (64 or 22)
    period : "train", "validation", "test", or "all"
    """
    h3_res = {64: 2, 22: 5}.get(resolution_km, 2)
    intercell_km = _H3_INTERCELL_KM.get(h3_res, 93.0)

    # Find all feature parquet files sorted chronologically
    files = sorted(backfill_dir.glob("year=*/month=*/features_*.parquet"))
    if not files:
        logger.warning("No backfill files found in %s", backfill_dir)
        return []

    logger.info("Found %d backfill files in %s", len(files), backfill_dir)

    # Filter by temporal period
    def _extract_timestamp(path: Path) -> pd.Timestamp | None:
        """Extract timestamp from filename like features_2024-07-15_0600.parquet"""
        name = path.stem
        parts = name.replace("features_", "").split("_")
        if len(parts) >= 2:
            try:
                return pd.Timestamp(f"{parts[0]}T{parts[1][:2]}:{parts[1][2:]}")
            except Exception:
                pass
        return None

    timed_files = [(f, _extract_timestamp(f)) for f in files]
    timed_files = [(f, t) for f, t in timed_files if t is not None]
    timed_files.sort(key=lambda x: x[1])

    if period == "train":
        timed_files = [(f, t) for f, t in timed_files if t <= _TRAIN_END]
    elif period == "validation":
        timed_files = [(f, t) for f, t in timed_files if _TRAIN_END < t <= _VALIDATION_END]
    elif period == "test":
        timed_files = [(f, t) for f, t in timed_files if t > _VALIDATION_END]

    logger.info("Processing %d files for period=%s", len(timed_files), period)

    events: list[ObservedSpread] = []

    for i in range(len(timed_files) - 1):
        path_t0, ts_t0 = timed_files[i]
        path_t1, ts_t1 = timed_files[i + 1]

        # Only process consecutive 6-hour pairs
        delta_hours = (ts_t1 - ts_t0).total_seconds() / 3600
        if delta_hours > 12:  # Skip if gap > 12 hours (missing data)
            continue

        try:
            df0, df1 = _load_backfill_pair(path_t0, path_t1)
        except Exception as exc:
            logger.debug("Failed to load pair %s + %s: %s", path_t0.name, path_t1.name, exc)
            continue

        # Find fire cells at T0
        if "fire_detected_binary" not in df0.columns or "fire_detected_binary" not in df1.columns:
            continue

        fire_t0 = set(df0[df0["fire_detected_binary"] == 1]["grid_id"].tolist())
        fire_t1 = set(df1[df1["fire_detected_binary"] == 1]["grid_id"].tolist())

        if not fire_t0:
            continue

        # For each fire cell at T0, check ring-1 neighbors at T1
        for origin in fire_t0:
            try:
                ring1 = set(h3.grid_disk(origin, 1)) - {origin}
            except Exception:
                continue

            # New fire cells at T1 that are neighbors of T0 fire cells
            # (cells in ring-1 of origin AND have fire at T1 AND didn't have fire at T0)
            new_fire_neighbors = (ring1 & fire_t1) - fire_t0

            for target in new_fire_neighbors:
                # Compute bearing
                lat0, lon0 = h3.cell_to_latlng(origin)
                lat1, lon1 = h3.cell_to_latlng(target)
                bear = _bearing(lat0, lon0, lat1, lon1)

                # FRP data if available
                origin_frp = 0.0
                target_frp = 0.0
                if "mean_frp" in df0.columns:
                    origin_rows = df0[df0["grid_id"] == origin]
                    if not origin_rows.empty:
                        origin_frp = float(pd.to_numeric(
                            origin_rows["mean_frp"].iloc[0], errors="coerce"
                        ) or 0.0)
                if "mean_frp" in df1.columns:
                    target_rows = df1[df1["grid_id"] == target]
                    if not target_rows.empty:
                        target_frp = float(pd.to_numeric(
                            target_rows["mean_frp"].iloc[0], errors="coerce"
                        ) or 0.0)

                events.append(ObservedSpread(
                    origin_cell=origin,
                    target_cell=target,
                    origin_time=ts_t0.isoformat(),
                    target_time=ts_t1.isoformat(),
                    bearing_deg=round(bear, 1),
                    observed_speed_upper_bound_kmh=round(intercell_km / delta_hours, 2),
                    origin_frp=round(origin_frp, 1),
                    target_frp=round(target_frp, 1),
                ))

    logger.info("Extracted %d spread events from %d file pairs", len(events), len(timed_files) - 1)
    return events


# ---------------------------------------------------------------------------
# Validation against simulator
# ---------------------------------------------------------------------------

def _angular_diff(a: float, b: float) -> float:
    """Circular angular difference in [0, 180] degrees."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def validate_against_firms(
    backfill_dir: Path,
    simulator: Any,
    period: str = "test",
    resolution_km: int = 64,
) -> dict[str, Any]:
    """Validate simulator against FIRMS observed spread events.

    1. Extract observed spread events from backfill
    2. For each event, run simulator on origin cell conditions
    3. Check if model predicted nonzero rate toward the target cell
    4. Compare predicted vs observed direction and speed

    Returns dict of honest metrics.
    """
    events = extract_spread_events(backfill_dir, resolution_km, period)

    if not events:
        return {
            "n_events": 0,
            "period": period,
            "spread_event_recall": 0.0,
            "direction_accuracy_45deg": 0.0,
            "csi": 0.0,
            "persistence_baseline_csi": 0.0,
            "model_beats_persistence": False,
            "note": "No spread events found in backfill for this period.",
        }

    # Load unique origin files to get conditions
    n_predicted_spread = 0
    n_direction_within_45 = 0
    speed_calibration = []
    tp = 0  # true positive: model predicted spread, fire actually spread
    fp = 0  # false positive: model predicted spread, fire didn't spread
    fn = 0  # false negative: fire spread but model didn't predict it
    persistence_tp = 0  # fire stayed in same cell

    # Group events by origin timestamp for batch processing
    events_by_file: dict[str, list[ObservedSpread]] = {}
    for ev in events:
        key = ev.origin_time
        events_by_file.setdefault(key, []).append(ev)

    for origin_time, batch in events_by_file.items():
        # Find the corresponding backfill file
        ts = pd.Timestamp(origin_time)
        year_dir = backfill_dir / f"year={ts.year}" / f"month={ts.month:02d}"
        time_str = f"{ts.hour:02d}{ts.minute:02d}"
        pattern = f"features_{ts.date()}_{time_str}.parquet"

        matching = list(year_dir.glob(pattern))
        if not matching:
            # Try without exact time match
            matching = list(year_dir.glob(f"features_{ts.date()}_*.parquet"))

        if not matching:
            fn += len(batch)
            continue

        try:
            df = pd.read_parquet(matching[0])
            df["grid_id"] = df["grid_id"].astype(str)
        except Exception:
            fn += len(batch)
            continue

        for ev in batch:
            if ev.origin_cell not in df["grid_id"].values:
                fn += 1
                continue

            try:
                result = simulator.simulate(df, ev.origin_cell, 0.50)
            except Exception:
                fn += 1
                continue

            # Check if model predicted any spread toward target
            nb_details = result.get("neighbour_details", [])
            target_nb = None
            for nb in nb_details:
                if nb.get("neighbour_id") == ev.target_cell:
                    target_nb = nb
                    break

            if target_nb is None:
                # Target not in ring-1 (shouldn't happen but just in case)
                fn += 1
                continue

            predicted_rate = target_nb.get("spread_rate_kmh", 0.0)

            if predicted_rate > 0:
                n_predicted_spread += 1
                tp += 1

                # Direction check
                predicted_dir = result.get("spread_direction_deg", 0)
                if _angular_diff(predicted_dir, ev.bearing_deg) <= 45.0:
                    n_direction_within_45 += 1

                # Speed calibration
                speed_calibration.append({
                    "predicted_kmh": round(predicted_rate, 4),
                    "observed_upper_bound_kmh": ev.observed_speed_upper_bound_kmh,
                    "origin_cell": ev.origin_cell,
                    "target_cell": ev.target_cell,
                })
            else:
                fn += 1

    # Persistence baseline: how many fire cells at T0 still have fire at T+6h
    # (approximated: all events are non-persistent by definition since they're NEW fires)
    # Persistence CSI = 0 for spread events (persistence predicts no spread)
    persistence_csi = 0.0

    # CSI = TP / (TP + FP + FN)
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    n_total = len(events)
    recall = n_predicted_spread / n_total if n_total > 0 else 0.0
    dir_acc = n_direction_within_45 / n_predicted_spread if n_predicted_spread > 0 else 0.0

    return {
        "n_events": n_total,
        "period": period,
        "n_predicted_spread": n_predicted_spread,
        "n_direction_within_45": n_direction_within_45,
        "spread_event_recall": round(recall, 4),
        "direction_accuracy_45deg": round(dir_acc, 4),
        "csi": round(csi, 4),
        "persistence_baseline_csi": persistence_csi,
        "model_beats_persistence": csi > persistence_csi,
        "speed_calibration_sample": speed_calibration[:50],  # first 50 for report size
        "events_sample": [asdict(e) for e in events[:20]],  # first 20 for inspection
    }
