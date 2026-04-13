"""
evaluate_obj2.py — OBJ-2 Fire Spread Simulator Evaluation Framework
=====================================================================
Comprehensive evaluation against historical fires and real-time pipeline data.

Modes:
    historical  — 7 benchmark fires with documented ground truth (synthetic inputs)
    realtime    — live pipeline parquets at 64km and/or 22km (no ground truth, sanity checks)
    all         — both (default)

Usage:
    python evaluate_obj2.py                              # full evaluation
    python evaluate_obj2.py --mode historical             # benchmarks only
    python evaluate_obj2.py --mode realtime               # real-time 64km + 22km
    python evaluate_obj2.py --mode realtime --resolution 22  # 22km only
    python evaluate_obj2.py --skip-sensitivity            # skip sensitivity analysis
    python evaluate_obj2.py --fires camp_fire creek_fire  # subset of fires

Owner: OBJ-2 (fire spread model)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure UTF-8 output on Windows (cp1252 chokes on degree/le symbols from eval_metrics)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, "src")

import h3
import numpy as np
import pandas as pd

from models.obj2_spread.fire_spread_simulator import PythonFireSpreadSimulator
from models.obj2_spread.eval_metrics import (
    GroundTruth,
    compute_physics_gate,
    sanity_check_output,
)
from models.obj2_spread.spread_metrics import (
    analyze_threatened_cells,
    compute_propagation_honesty,
    compute_input_quality,
)
from models.obj2_spread.geojson_export import build_spread_geojson, save_spread_geojson

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PIPELINE = ROOT / "Data-Pipeline"

SMOKE_TEST_DIR = DATA_PIPELINE / "data/processed/smoke_test"
FUSED_64KM_LATEST = DATA_PIPELINE / "data/processed/fused/fused_features_latest.parquet"
FUSED_64KM_CA = DATA_PIPELINE / "data/processed/fused/64km/region=california/year=2026/month=03/fused_2026-03-31.parquet"
PROCESSED_22KM_DIR = DATA_PIPELINE / "data" / "processed" / "22km"
STATIC_22KM = DATA_PIPELINE / "data/static/static_features_22km.parquet"
STATIC_64KM = DATA_PIPELINE / "data/static/static_features_64km.parquet"

H3_RES = {64: 2, 22: 5}

# ---------------------------------------------------------------------------
# Prediction output directories (team lead requirement: organized subdirs)
# ---------------------------------------------------------------------------
PREDICTIONS_BASE = Path(__file__).resolve().parent / "predictions" / "obj2_spread"
PRED_HISTORICAL = PREDICTIONS_BASE / "historical"
PRED_REALTIME = PREDICTIONS_BASE / "realtime"
PRED_FIRMS = PREDICTIONS_BASE / "firms_validation"
PRED_SENSITIVITY = PREDICTIONS_BASE / "sensitivity"

# ---------------------------------------------------------------------------
# Historical fire benchmark database
# ---------------------------------------------------------------------------

def _make_df(lat: float, lon: float, conditions: dict, h3_res: int = 5) -> tuple:
    """Build synthetic DataFrame with ignition + ring-1 neighbours."""
    ignition_id = h3.latlng_to_cell(lat, lon, h3_res)
    neighbours = sorted(set(h3.grid_disk(ignition_id, 1)) - {ignition_id})
    all_ids = [ignition_id] + neighbours
    n = len(all_ids)

    rows = {col: [val] * n for col, val in conditions.items()}
    df = pd.DataFrame({"grid_id": all_ids, **rows})
    for i, cell_id in enumerate(all_ids):
        clat, clon = h3.cell_to_latlng(cell_id)
        df.loc[i, "latitude"] = clat
        df.loc[i, "longitude"] = clon
    return df, ignition_id


BENCHMARK_FIRES = [
    {
        "fire_id": "palisades_2025",
        "name": "Palisades Fire (Jan 7 2025, LA)",
        "lat": 34.05, "lon": -118.52,
        "ignition_prob": 0.90,
        "conditions": {
            "wind_speed_10m": 90.0, "wind_direction_10m": 65.0,
            "relative_humidity_2m": 8.0, "temperature_2m": 24.0,
            "days_since_last_precipitation": 45.0,
            "fuel_model_fbfm40": 147.0,
            "canopy_base_height_m": 0.8, "canopy_bulk_density": 0.08,
            "slope_degrees": 20.0, "aspect_degrees": 225.0,
            "canopy_cover_pct": 60.0, "elevation_m": 250.0,
            "vpd": 5.5, "active_fire_count": 0,
        },
        "ground_truth": GroundTruth(
            spread_direction_deg=245.0, direction_tolerance_deg=30.0,
            spread_speed_kmh_min=2.0, spread_speed_kmh_max=20.0,
            dead_fuel_moisture_pct_min=1.0, dead_fuel_moisture_pct_max=10.0,
            byram_intensity_kwm_min=1000.0,
            crown_fire_expected=["passive_crown", "active_crown"],
            source="CAL FIRE Incident Report 2025; SH7 chaparral + Santa Ana",
        ),
    },
    {
        "fire_id": "camp_fire_2018",
        "name": "Camp Fire (Nov 8 2018, Pulga CA)",
        "lat": 39.810, "lon": -121.470,
        "ignition_prob": 0.90,
        "conditions": {
            "wind_speed_10m": 85.0, "wind_direction_10m": 15.0,
            "relative_humidity_2m": 23.0, "temperature_2m": 11.0,
            "days_since_last_precipitation": 10.0,
            "fuel_model_fbfm40": 165.0,
            "canopy_base_height_m": 3.0, "canopy_bulk_density": 0.12,
            "slope_degrees": 28.0, "aspect_degrees": 220.0,
            "canopy_cover_pct": 75.0, "elevation_m": 800.0,
            "vpd": 2.5, "active_fire_count": 0,
        },
        "ground_truth": GroundTruth(
            spread_direction_deg=195.0, direction_tolerance_deg=30.0,
            spread_speed_kmh_min=4.0, spread_speed_kmh_max=15.0,
            dead_fuel_moisture_pct_min=3.0, dead_fuel_moisture_pct_max=10.0,
            byram_intensity_kwm_min=2000.0,
            crown_fire_expected=["passive_crown", "active_crown"],
            source="CAL FIRE Investigation Report 2019; TU5 timber + Diablo wind",
        ),
    },
    {
        "fire_id": "creek_fire_2020",
        "name": "Creek Fire (Sep 5 2020, Shaver Lake CA)",
        "lat": 37.10, "lon": -119.30,
        "ignition_prob": 0.85,
        "conditions": {
            "wind_speed_10m": 70.0, "wind_direction_10m": 55.0,
            "relative_humidity_2m": 10.0, "temperature_2m": 32.0,
            "days_since_last_precipitation": 60.0,
            "fuel_model_fbfm40": 165.0,
            "canopy_base_height_m": 4.0, "canopy_bulk_density": 0.15,
            "slope_degrees": 22.0, "aspect_degrees": 235.0,
            "canopy_cover_pct": 70.0, "elevation_m": 1500.0,
            "vpd": 4.5, "active_fire_count": 0,
        },
        "ground_truth": GroundTruth(
            spread_direction_deg=235.0, direction_tolerance_deg=35.0,
            spread_speed_kmh_min=3.0, spread_speed_kmh_max=18.0,
            dead_fuel_moisture_pct_min=1.0, dead_fuel_moisture_pct_max=8.0,
            byram_intensity_kwm_min=2000.0,
            crown_fire_expected=["passive_crown", "active_crown"],
            source="NIFC Incident Summary 2020; TU5 Sierra mixed conifer + Mono wind",
        ),
    },
    {
        "fire_id": "thomas_fire_2017",
        "name": "Thomas Fire (Dec 4 2017, Ventura CA)",
        "lat": 34.42, "lon": -118.88,
        "ignition_prob": 0.88,
        "conditions": {
            "wind_speed_10m": 95.0, "wind_direction_10m": 60.0,
            "relative_humidity_2m": 6.0, "temperature_2m": 26.0,
            "days_since_last_precipitation": 50.0,
            "fuel_model_fbfm40": 149.0,
            "canopy_base_height_m": 1.0, "canopy_bulk_density": 0.09,
            "slope_degrees": 25.0, "aspect_degrees": 240.0,
            "canopy_cover_pct": 65.0, "elevation_m": 350.0,
            "vpd": 6.0, "active_fire_count": 0,
        },
        "ground_truth": GroundTruth(
            spread_direction_deg=240.0, direction_tolerance_deg=30.0,
            spread_speed_kmh_min=3.0, spread_speed_kmh_max=20.0,
            dead_fuel_moisture_pct_min=1.0, dead_fuel_moisture_pct_max=8.0,
            byram_intensity_kwm_min=1500.0,
            crown_fire_expected=["passive_crown", "active_crown"],
            source="CAL FIRE Report 2017; SH9 chaparral + Santa Ana",
        ),
    },
    {
        "fire_id": "carr_fire_2018",
        "name": "Carr Fire (Jul 23 2018, Redding CA)",
        "lat": 40.65, "lon": -122.55,
        "ignition_prob": 0.85,
        "conditions": {
            "wind_speed_10m": 55.0, "wind_direction_10m": 330.0,
            "relative_humidity_2m": 12.0, "temperature_2m": 38.0,
            "days_since_last_precipitation": 55.0,
            "fuel_model_fbfm40": 109.0,
            "canopy_base_height_m": 2.5, "canopy_bulk_density": 0.07,
            "slope_degrees": 15.0, "aspect_degrees": 150.0,
            "canopy_cover_pct": 30.0, "elevation_m": 200.0,
            "vpd": 5.8, "active_fire_count": 0,
        },
        "ground_truth": GroundTruth(
            spread_direction_deg=150.0, direction_tolerance_deg=35.0,
            spread_speed_kmh_min=3.0, spread_speed_kmh_max=25.0,
            dead_fuel_moisture_pct_min=1.0, dead_fuel_moisture_pct_max=8.0,
            byram_intensity_kwm_min=800.0,
            crown_fire_expected=["surface", "passive_crown", "active_crown"],
            source="CAL FIRE Report 2018; GR9 grass-timber + extreme heat",
        ),
    },
    {
        "fire_id": "dixie_fire_2021",
        "name": "Dixie Fire (Jul 13 2021, Feather River CA)",
        "lat": 39.87, "lon": -121.39,
        "ignition_prob": 0.80,
        "conditions": {
            "wind_speed_10m": 40.0, "wind_direction_10m": 320.0,
            "relative_humidity_2m": 15.0, "temperature_2m": 34.0,
            "days_since_last_precipitation": 45.0,
            "fuel_model_fbfm40": 165.0,
            "canopy_base_height_m": 5.0, "canopy_bulk_density": 0.10,
            "slope_degrees": 18.0, "aspect_degrees": 180.0,
            "canopy_cover_pct": 65.0, "elevation_m": 1100.0,
            "vpd": 3.8, "active_fire_count": 0,
        },
        "ground_truth": GroundTruth(
            spread_direction_deg=140.0, direction_tolerance_deg=35.0,
            spread_speed_kmh_min=1.0, spread_speed_kmh_max=12.0,
            dead_fuel_moisture_pct_min=2.0, dead_fuel_moisture_pct_max=12.0,
            byram_intensity_kwm_min=500.0,
            crown_fire_expected=["surface", "passive_crown", "active_crown"],
            source="CAL FIRE Report 2021; TU5 timber + moderate NW wind",
        ),
    },
    {
        "fire_id": "woolsey_fire_2018",
        "name": "Woolsey Fire (Nov 8 2018, Malibu CA)",
        "lat": 34.23, "lon": -118.73,
        "ignition_prob": 0.88,
        "conditions": {
            "wind_speed_10m": 80.0, "wind_direction_10m": 55.0,
            "relative_humidity_2m": 10.0, "temperature_2m": 22.0,
            "days_since_last_precipitation": 40.0,
            "fuel_model_fbfm40": 147.0,
            "canopy_base_height_m": 0.8, "canopy_bulk_density": 0.08,
            "slope_degrees": 22.0, "aspect_degrees": 230.0,
            "canopy_cover_pct": 55.0, "elevation_m": 300.0,
            "vpd": 5.0, "active_fire_count": 0,
        },
        "ground_truth": GroundTruth(
            spread_direction_deg=235.0, direction_tolerance_deg=30.0,
            spread_speed_kmh_min=2.0, spread_speed_kmh_max=18.0,
            dead_fuel_moisture_pct_min=1.0, dead_fuel_moisture_pct_max=10.0,
            byram_intensity_kwm_min=1000.0,
            crown_fire_expected=["passive_crown", "active_crown"],
            source="CAL FIRE Report 2018; SH7 coastal chaparral + Santa Ana",
        ),
    },
]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

_STATIC_ENRICH_COLS = [
    "canopy_base_height_m", "canopy_bulk_density", "canopy_cover_pct",
    "fuel_model_fbfm40", "slope_degrees", "aspect_degrees", "elevation_m",
    "evt_national_class",
]

# Physically defensible California-wide medians used only when a cell has no
# static record at all (last resort — prefer real data from static parquets).
_CA_MEDIANS: dict[str, float] = {
    "slope_degrees": 12.0,
    "aspect_degrees": 180.0,   # south-facing — conservative for fire weather
    "elevation_m": 500.0,
    "canopy_cover_pct": 20.0,
    "canopy_base_height_m": 1.5,
    "canopy_bulk_density": 0.06,
    "fuel_model_fbfm40": 165.0,  # TU5 — common mixed CA woodland
}

# Hard physical clamps applied AFTER imputation to catch corrupted source values
# (e.g. negative canopy cover from bad LANDFIRE pixels, slope > 90°, etc.)
_PHYSICAL_CLAMPS: dict[str, tuple[float, float]] = {
    "canopy_cover_pct":     (0.0,   100.0),
    "slope_degrees":        (0.0,    90.0),
    "aspect_degrees":       (0.0,   360.0),
    "canopy_base_height_m": (0.0,    50.0),
    "canopy_bulk_density":  (0.0,     0.5),  # >0.5 kg/m³ is unrealistic
    "elevation_m":          (0.0,  4500.0),  # Mt Whitney = 4421m
}


def _enrich_and_impute(df: pd.DataFrame, static_path: Path) -> pd.DataFrame:
    """Step 1 — merge static columns; Step 2 — median-impute remaining NaN.

    Priority per cell:
      1. Value already in fused parquet (non-null)
      2. Value from static parquet (matched by grid_id)
      3. Median computed from non-null cells in the *same loaded dataset*
      4. California-wide fallback median (_CA_MEDIANS)
    """
    # ── Step 1: merge static where available ──────────────────────────────
    if static_path.exists():
        st = pd.read_parquet(static_path)
        st["grid_id"] = st["grid_id"].astype(str)
        available = [c for c in _STATIC_ENRICH_COLS if c in st.columns]
        if available:
            # Drop cols that exist in df so we can refill from static
            # (keep fused values where non-null by merging then combining)
            st_sub = st[["grid_id"] + available].copy()
            # Merge on a temp suffix; fill NaN in df with static values
            df = df.merge(st_sub, on="grid_id", how="left", suffixes=("", "_static"))
            for col in available:
                static_col = f"{col}_static"
                if static_col in df.columns:
                    df[col] = df[col].combine_first(df[static_col])
                    df.drop(columns=[static_col], inplace=True)
            n_cbh = df["canopy_base_height_m"].notna().sum() if "canopy_base_height_m" in df.columns else 0
            logger.info(
                "Enriched from static %s: CBH non-null=%d/%d",
                static_path.name, n_cbh, len(df),
            )
    else:
        logger.warning("Static parquet not found: %s — skipping enrich step", static_path)

    # ── Step 2: median-impute remaining NaN ───────────────────────────────
    imputed: dict[str, float] = {}
    for col, fallback in _CA_MEDIANS.items():
        if col not in df.columns:
            continue
        n_null = df[col].isna().sum()
        if n_null == 0:
            continue
        # Compute median from real (non-null) values in this dataset
        dataset_median = df[col].median()
        fill_value = dataset_median if not pd.isna(dataset_median) else fallback
        df[col] = df[col].fillna(fill_value)
        imputed[col] = round(float(fill_value), 3)

    if imputed:
        logger.info("Imputed missing values (dataset median): %s", imputed)

    # ── Step 3: hard physical clamps — catch corrupted source pixels ──────
    clamped: dict[str, str] = {}
    for col, (lo, hi) in _PHYSICAL_CLAMPS.items():
        if col not in df.columns:
            continue
        bad_mask = (df[col] < lo) | (df[col] > hi)
        n_bad = bad_mask.sum()
        if n_bad > 0:
            df[col] = df[col].clip(lower=lo, upper=hi)
            clamped[col] = f"{n_bad} values clamped to [{lo}, {hi}]"
    if clamped:
        logger.warning("Physical clamps applied (corrupted source pixels): %s", clamped)

    return df


def _load_64km() -> pd.DataFrame | None:
    """Load 64km fused parquet, enrich from static, impute remaining NaN.

    Priority: smoke_test outputs (newest first) > fused_features_latest > CA monthly.
    """
    candidates = []
    if SMOKE_TEST_DIR.exists():
        candidates += sorted(SMOKE_TEST_DIR.glob("fused_*_64km.parquet"), reverse=True)
    candidates += [FUSED_64KM_LATEST, FUSED_64KM_CA]

    for path in candidates:
        if path.exists():
            df = pd.read_parquet(path)
            df["grid_id"] = df["grid_id"].astype(str)
            logger.info("Loaded 64km: %s (%d rows)", path.name, len(df))
            df = _enrich_and_impute(df, STATIC_64KM)
            return df
    logger.warning("No 64km fused parquet found")
    return None


def _read_and_enrich_22km(path: Path) -> pd.DataFrame:
    """Read a single 22km parquet file and enrich from static."""
    df = pd.read_parquet(path)
    df["grid_id"] = df["grid_id"].astype(str)
    logger.info("Loaded 22km: %s (%d rows)", path.name, len(df))
    return _enrich_and_impute(df, STATIC_22KM)


def _load_22km() -> pd.DataFrame | None:
    """Load 22km fused parquet, enrich from static, impute remaining NaN.

    Three-tier priority — picks the freshest available data:

    Tier 1  smoke_test/fused_*_22km.parquet      (newest first)
            Written by run_pipeline_once.py — always most up-to-date.

    Tier 2  22km/region=california/…/features_*.parquet  +
            22km/region=texas/…/features_*.parquet
            Partitioned export from DAG task_export_to_parquet.
            Newest file per region; CA and TX are concatenated.

    Tier 3  22km/date=*/features.parquet
            Legacy flat format — fallback when neither tier 1 nor 2 exists.
    """
    # ── Tier 1: smoke_test ─────────────────────────────────────────────────
    if SMOKE_TEST_DIR.exists():
        candidates = sorted(SMOKE_TEST_DIR.glob("fused_*_22km.parquet"), reverse=True)
        if candidates:
            return _read_and_enrich_22km(candidates[0])

    # ── Tier 2: partitioned region=* files (CA + TX concat, newest per region)
    region_dfs: list[pd.DataFrame] = []
    for region in ("california", "texas"):
        region_dir = PROCESSED_22KM_DIR / f"region={region}"
        if region_dir.exists():
            region_files = sorted(region_dir.glob("**/features_*.parquet"), reverse=True)
            if region_files:
                part_df = pd.read_parquet(region_files[0])
                part_df["grid_id"] = part_df["grid_id"].astype(str)
                region_dfs.append(part_df)
                logger.info(
                    "Loaded 22km partitioned (%s): %s (%d rows)",
                    region, region_files[0].name, len(part_df),
                )
    if region_dfs:
        df = pd.concat(region_dfs, ignore_index=True)
        logger.info("Combined 22km CA+TX partitioned: %d rows total", len(df))
        return _enrich_and_impute(df, STATIC_22KM)

    # ── Tier 3: legacy flat date= format ────────────────────────────────────
    if PROCESSED_22KM_DIR.exists():
        flat_files = sorted(PROCESSED_22KM_DIR.glob("date=*/features.parquet"), reverse=True)
        if flat_files:
            logger.info("Falling back to flat 22km parquet: %s", flat_files[0])
            return _read_and_enrich_22km(flat_files[0])

    logger.warning(
        "No 22km processed parquet found. Run: "
        "python -m scripts.utils.run_pipeline_once --resolution-km 22"
    )
    return None


def simulate_spread_timeseries(
    df: pd.DataFrame,
    ignition_id: str,
    ignition_prob: float,
    hours: float = 1.0,
    timestep_h: float = 1.0,
    sim: "PythonFireSpreadSimulator | None" = None,
) -> dict:
    """Time-stepped fire propagation across H3 ring-1 neighbors.

    At each time step, every currently burning cell evaluates its ring-1
    neighbors. A neighbor ignites if the fire can travel the inter-cell
    distance within the timestep at the computed spread rate.

    H3 ring-1 inter-cell distance (center-to-center):
      res-2 (64km): ~93 km
      res-5 (22km): ~25 km

    Parameters
    ----------
    df           : Fused feature DataFrame (one row per grid cell).
    ignition_id  : Starting H3 cell ID.
    ignition_prob: Ignition probability scalar (0–1).
    hours        : Total simulation duration in hours.
    timestep_h   : Time step size in hours.

    Returns
    -------
    dict with:
      burned_cells    : set of all H3 cell IDs reached by fire
      timeline        : list of {t_hour, newly_ignited, burning_count}
      cell_details    : {cell_id: first result dict from simulate()}
      total_steps     : number of time steps run
    """
    # Inter-cell distance by H3 resolution (center-to-center, km)
    _H3_INTERCELL_KM = {2: 93.0, 5: 25.0}

    # Instantiate simulator if not provided
    if sim is None:
        sim = PythonFireSpreadSimulator()

    # Detect resolution from ignition cell
    try:
        res = h3.get_resolution(ignition_id)
    except Exception:
        res = 5  # default to 22km
    intercell_km = _H3_INTERCELL_KM.get(res, 25.0)

    burning: set[str] = {ignition_id}
    ever_burned: set[str] = {ignition_id}
    cell_details: dict = {}
    timeline = []
    n_steps = int(hours / timestep_h)

    for step in range(n_steps):
        t_hour = (step + 1) * timestep_h
        newly_ignited: set[str] = set()

        for cell_id in list(burning):
            # Skip cells not in dataset — no feature data to simulate
            cell_rows = df[df["grid_id"] == cell_id]
            if cell_rows.empty:
                continue

            try:
                result = sim.simulate(df, cell_id, ignition_prob)
            except Exception as exc:
                logger.warning("Propagation: sim failed for %s: %s", cell_id, exc)
                continue

            if cell_id not in cell_details:
                cell_details[cell_id] = result

            # Evaluate each threatened neighbor
            for nb in result.get("neighbour_details", []):
                nb_id = nb.get("neighbour_id")
                if nb_id in ever_burned:
                    continue  # already burning or burned

                rate_kmh = nb.get("spread_rate_kmh", 0.0)
                if rate_kmh <= 0:
                    continue

                # Time to travel inter-cell distance at this spread rate
                travel_time_h = intercell_km / rate_kmh
                if travel_time_h <= timestep_h:
                    newly_ignited.add(nb_id)

        burning = newly_ignited          # only newly ignited cells burn next step
        ever_burned |= newly_ignited

        timeline.append({
            "t_hour":         t_hour,
            "newly_ignited":  len(newly_ignited),
            "total_burning":  len(ever_burned),
            "cells":          sorted(newly_ignited),
        })

        logger.info(
            "Spread t=%.1fh: +%d new cells | total burned=%d",
            t_hour, len(newly_ignited), len(ever_burned),
        )

        if not newly_ignited:
            logger.info("Spread stopped at t=%.1fh — no new cells ignited", t_hour)
            break

    return {
        "ignition_cell":  ignition_id,
        "hours_simulated": hours,
        "timestep_h":     timestep_h,
        "burned_cells":   sorted(ever_burned),
        "total_burned":   len(ever_burned),
        "timeline":       timeline,
        "cell_details":   cell_details,
        "total_steps":    len(timeline),
    }


def _select_ignition(df: pd.DataFrame) -> tuple[str, float]:
    """Select ignition cell from FIRMS data or driest cell."""
    # Priority 1: Active fire detected
    if "active_fire_count" in df.columns:
        numeric_count = pd.to_numeric(df["active_fire_count"], errors="coerce").fillna(0)
        fire_df = df[numeric_count > 0].copy()
        if not fire_df.empty:
            fire_df["_fire_count_num"] = pd.to_numeric(fire_df["active_fire_count"], errors="coerce").fillna(0)
            best = fire_df.sort_values("_fire_count_num", ascending=False).iloc[0]
            logger.info("Ignition from FIRMS: %s (fire_count=%s)", best["grid_id"], best["active_fire_count"])
            return str(best["grid_id"]), 0.30

    # Priority 2: Driest cell
    if "relative_humidity_2m" in df.columns:
        rh = pd.to_numeric(df["relative_humidity_2m"], errors="coerce")
        valid = df[rh.notna()]
        if not valid.empty:
            driest = valid.loc[rh[valid.index].idxmin()]
            logger.info("Ignition from driest cell: %s (RH=%s%%)", driest["grid_id"], driest["relative_humidity_2m"])
            return str(driest["grid_id"]), 0.15

    # Fallback: first cell (guard against empty DataFrame)
    if df.empty:
        raise ValueError("Cannot select ignition cell: DataFrame is empty after filtering.")
    return str(df.iloc[0]["grid_id"]), 0.10


def _select_ignition_22km(
    df: pd.DataFrame,
    ignition_64km_id: str,
    ignition_prob_64km: float,
) -> tuple[str, float]:
    """Select the 22km ignition cell that zooms into the 64km ignition area.

    Flow
    ----
    1. Get all H3 res-5 children of the 64km cell (343 cells cover ~64km hex).
    2. Filter the 22km parquet to only those children.
    3. Among them, apply the same priority logic as _select_ignition():
         a) Highest active_fire_count (FIRMS hotspot within the area)
         b) Lowest relative_humidity_2m (driest — highest surface fire risk)
         c) Geographic center child (latlng_to_cell of the 64km center)
    4. Inherit the ignition probability from the 64km result — the fire risk
       score comes from OBJ-1 at 64km; the 22km run just refines the physics.

    Falls back to the full 22km parquet (_select_ignition) if no children
    are present in the parquet (e.g. a different region's data).
    """
    try:
        children = set(h3.cell_to_children(ignition_64km_id, 5))
    except Exception as exc:
        logger.warning("Could not compute 22km children of %s: %s — falling back", ignition_64km_id, exc)
        return _select_ignition(df)

    child_df = df[df["grid_id"].isin(children)].copy()

    if child_df.empty:
        logger.warning(
            "No 22km cells found inside 64km cell %s — falling back to full parquet",
            ignition_64km_id,
        )
        return _select_ignition(df)

    logger.info(
        "Zoom 64km→22km: %d child cells of %s available in parquet",
        len(child_df), ignition_64km_id,
    )

    # Ensure fuel model is numeric and valid inside the child area
    if "fuel_model_fbfm40" in child_df.columns:
        child_df["fuel_model_fbfm40"] = pd.to_numeric(child_df["fuel_model_fbfm40"], errors="coerce")
        valid_fuel = child_df[child_df["fuel_model_fbfm40"].notna() & (child_df["fuel_model_fbfm40"] > 0)]
        if not valid_fuel.empty:
            child_df = valid_fuel

    # Priority 1: active fire detected inside the zoom area
    if "active_fire_count" in child_df.columns:
        numeric_count = pd.to_numeric(child_df["active_fire_count"], errors="coerce").fillna(0)
        fire_df = child_df[numeric_count > 0].copy()
        if not fire_df.empty:
            fire_df["_fire_count_num"] = pd.to_numeric(fire_df["active_fire_count"], errors="coerce").fillna(0)
            best = fire_df.sort_values("_fire_count_num", ascending=False).iloc[0]
            logger.info(
                "22km ignition from FIRMS hotspot inside 64km area: %s (fire_count=%s)",
                best["grid_id"], best["active_fire_count"],
            )
            return str(best["grid_id"]), ignition_prob_64km

    # Priority 2: driest cell inside the zoom area
    if "relative_humidity_2m" in child_df.columns:
        rh = pd.to_numeric(child_df["relative_humidity_2m"], errors="coerce")
        valid = child_df[rh.notna()]
        if not valid.empty:
            driest = valid.loc[rh[valid.index].idxmin()]
            logger.info(
                "22km ignition from driest cell inside 64km area: %s (RH=%.1f%%)",
                driest["grid_id"], driest["relative_humidity_2m"],
            )
            return str(driest["grid_id"]), ignition_prob_64km

    # Priority 3: geographic center child of the 64km cell
    try:
        lat, lon = h3.cell_to_latlng(ignition_64km_id)
        center_child = h3.latlng_to_cell(lat, lon, 5)
        if center_child in set(child_df["grid_id"].tolist()):
            logger.info("22km ignition from geographic center child: %s", center_child)
            return center_child, ignition_prob_64km
    except Exception:
        pass

    # Fallback: first child in parquet
    fallback_id = str(child_df.iloc[0]["grid_id"])
    logger.info("22km ignition fallback (first child in parquet): %s", fallback_id)
    return fallback_id, ignition_prob_64km


# ---------------------------------------------------------------------------
# Prediction saving helpers
# ---------------------------------------------------------------------------

def _format_operational_output(
    result: dict[str, Any],
    resolution_km: int,
    ignition_prob: float,
) -> dict[str, Any]:
    """Format simulate() output into the operational spread structure.

    Produces the clean JSON structure consumed by OBJ-3 LLM and
    shown to operators:
      resolution_km / obj1_input / summary / threatened_cells
    """
    threatened = []
    for nb in result.get("neighbour_details", []):
        threatened.append({
            "neighbour_id": nb.get("neighbour_id"),
            "bearing_deg": nb.get("bearing_deg"),
            "spread_rate_kmh": nb.get("spread_rate_kmh"),
            "surface_ros_kmh": nb.get("surface_ros_kmh"),
            "head_ros_ftmin": nb.get("head_ros_ftmin"),
            "ellipse_ros_ftmin": nb.get("ellipse_ros_ftmin"),
            "slope_ros_ftmin": nb.get("slope_ros_ftmin", 0.0),
            "crown_status": nb.get("crown_status"),
            "byram_intensity_kwm": nb.get("byram_intensity_kwm"),
            "phi_slope": nb.get("phi_slope"),
            "fuel_model": nb.get("fuel_model"),
            "in_dataset": nb.get("in_dataset", False),
        })

    return {
        "resolution_km": resolution_km,
        "obj1_input": {
            "grid_id": result.get("ignition_cell"),
            "prob": round(ignition_prob, 4),
        },
        "summary": {
            "spread_direction_deg": result.get("spread_direction_deg"),
            "spread_speed_kmh": result.get("spread_speed_kmh"),
            "dead_fuel_moisture_pct": result.get("dead_fuel_moisture_pct"),
            "byram_intensity_kwm": result.get("byram_intensity_kwm"),
            "crown_fire_status": result.get("crown_fire_status"),
            "dominant_factor": result.get("dominant_factor"),
        },
        "threatened_cells": threatened,
    }


def _save_prediction(pred_dir: Path, filename: str, data: dict):
    """Save a prediction JSON to the organized directory structure."""
    pred_dir.mkdir(parents=True, exist_ok=True)
    path = pred_dir / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def _init_mlflow(mode: str, resolutions: list[int] | None, n_fires: int) -> Any:
    """Initialize MLflow tracking. Returns tracker or None if MLflow unavailable."""
    try:
        from tracking.mlflow_logger import MLflowLogger
        tracker = MLflowLogger(experiment_name="obj2-spread-evaluation")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tracker.start_run(
            run_name=f"eval_{mode}_{ts}",
            tags={
                "mode": mode,
                "resolutions": ",".join(str(r) for r in (resolutions or [])),
            },
        )
        tracker.log_params({
            "simulator": "PythonFireSpreadSimulator",
            "h3_resolutions": str(resolutions or [64, 22]),
            "n_benchmark_fires": str(n_fires),
            "wind_reduction_factor": "0.4",
        })
        logger.info("MLflow tracking initialized: obj2-spread-evaluation")
        return tracker
    except Exception as exc:
        logger.warning("MLflow not available — metrics will NOT be logged: %s", exc)
        return None


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    """Convert value to float, replacing NaN/inf with fallback.

    MLflow raises MlflowException on NaN or ±inf metrics. This guard
    ensures every logged value is a finite float. Called before every
    tracker.log_metrics() call.
    """
    try:
        v = float(value)
        return v if math.isfinite(v) else fallback
    except (TypeError, ValueError):
        return fallback


def _log_historical_metrics(tracker: Any, per_fire: list[dict], aggregate: dict):
    """Log historical evaluation metrics to MLflow."""
    if tracker is None:
        return
    try:
        tracker.log_metrics({
            "hist_gate_pass_rate": _safe_float(aggregate["overall_pass_rate"]),
            "hist_n_passed":       float(len(aggregate["fires_passing_all"])),
            "hist_n_total":        float(aggregate["n_fires"]),
        })
        for output_name, rate in aggregate.get("per_output_pass_rates", {}).items():
            tracker.log_metrics({f"hist_{output_name}_pass_rate": _safe_float(rate)})

        for i, fire in enumerate(per_fire):
            gate = fire["gate"]
            metrics: dict[str, float] = {
                "fire_gate_passed": 1.0 if gate["gate_passed"] else 0.0,
            }
            if "direction" in gate.get("per_output", {}):
                metrics["fire_direction_error"] = _safe_float(
                    gate["per_output"]["direction"].get("angular_error_deg", 0)
                )
            if "speed" in gate.get("per_output", {}):
                # log_ratio is float("-inf") when predicted speed == 0 — clamp to -3.0
                # (represents ~20× underestimate, safely finite for MLflow)
                metrics["fire_speed_log_ratio"] = _safe_float(
                    gate["per_output"]["speed"].get("log_ratio", 0), fallback=-3.0
                )
            tracker.log_metrics(metrics, step=i)
    except Exception as exc:
        logger.warning("MLflow historical logging failed: %s", exc)


def _log_realtime_metrics(tracker: Any, rt_results: dict):
    """Log real-time evaluation metrics to MLflow."""
    if tracker is None:
        return
    try:
        for res_km, data in rt_results.items():
            if data.get("error"):
                continue
            if "sanity" in data:
                n_pass = sum(1 for c in data["sanity"]["checks"] if c["passed"])
                n_total = len(data["sanity"]["checks"])
                tracker.log_metrics({
                    f"rt_{res_km}km_sanity_pass_rate": n_pass / n_total if n_total else 0,
                    f"rt_{res_km}km_speed_kmh":   _safe_float(data["result"].get("spread_speed_kmh")),
                    f"rt_{res_km}km_intensity_kwm": _safe_float(data["result"].get("byram_intensity_kwm")),
                    f"rt_{res_km}km_dfmc_pct":    _safe_float(data["result"].get("dead_fuel_moisture_pct")),
                })
            ta = data.get("threatened_analysis", {})
            if ta:
                tracker.log_metrics({
                    f"rt_{res_km}km_n_burnable":     float(ta.get("n_burnable_neighbors", 0)),
                    f"rt_{res_km}km_n_reachable_1h": float(ta.get("n_reachable_1h", 0)),
                    f"rt_{res_km}km_max_rate":       _safe_float(ta.get("max_spread_rate_kmh", 0.0)),
                    f"rt_{res_km}km_input_quality":  _safe_float(
                        data.get("input_quality", {}).get("quality_score", 0.0)
                    ),
                })
            mc = data.get("monte_carlo", {})
            if mc:
                tracker.log_metrics({
                    f"rt_{res_km}km_mc_speed_p50":      _safe_float(mc.get("spread_speed_kmh_p50")),
                    f"rt_{res_km}km_mc_speed_p90":      _safe_float(mc.get("spread_speed_kmh_p90")),
                    f"rt_{res_km}km_mc_speed_std":      _safe_float(mc.get("spread_speed_kmh_std")),
                    f"rt_{res_km}km_mc_crown_prob":     _safe_float(mc.get("crown_fire_probability")),
                    f"rt_{res_km}km_mc_max_burn_prob":  _safe_float(mc.get("max_neighbor_burn_probability")),
                    f"rt_{res_km}km_mc_dir_uncertainty": _safe_float(mc.get("direction_uncertainty_deg")),
                })
    except Exception as exc:
        logger.warning("MLflow realtime logging failed: %s", exc)


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

_PERTURBABLE = {
    # feature: (perturbation_type, values)
    "wind_speed_10m":            ("pct", [-20, -10, 10, 20]),
    "wind_direction_10m":        ("add", [-30, -15, 15, 30]),
    "relative_humidity_2m":      ("pct", [-20, -10, 10, 20]),
    "temperature_2m":            ("add", [-5, -2, 2, 5]),
    "days_since_last_precipitation": ("pct", [-20, -10, 10, 20]),
    "vpd":                       ("pct", [-20, -10, 10, 20]),
    "canopy_base_height_m":      ("pct", [-20, -10, 10, 20]),
    "canopy_bulk_density":       ("pct", [-20, -10, 10, 20]),
    "slope_degrees":             ("pct", [-20, -10, 10, 20]),
    "aspect_degrees":            ("add", [-30, -15, 15, 30]),
}

_FUEL_VARIANTS = [109, 147, 165, 185]  # GR9, SH7, TU5, TL5

_OUTPUTS = ["spread_direction_deg", "spread_speed_kmh", "dead_fuel_moisture_pct",
            "byram_intensity_kwm"]


def _run_sensitivity(
    sim: PythonFireSpreadSimulator,
    fire: dict,
) -> dict[str, Any]:
    """Run OAT sensitivity analysis for one fire."""
    base_df, ign_id = _make_df(fire["lat"], fire["lon"], fire["conditions"])
    base_result = sim.simulate(base_df, ign_id, fire["ignition_prob"])

    sensitivity = {}

    for feature, (ptype, values) in _PERTURBABLE.items():
        if feature not in fire["conditions"]:
            continue
        base_val = fire["conditions"][feature]
        if base_val == 0:
            continue

        feature_results = []
        for delta in values:
            perturbed = dict(fire["conditions"])
            if ptype == "pct":
                perturbed[feature] = base_val * (1 + delta / 100.0)
            else:
                perturbed[feature] = base_val + delta

            # Clamp to valid ranges
            if feature == "relative_humidity_2m":
                perturbed[feature] = max(1.0, min(100.0, perturbed[feature]))
            elif feature == "slope_degrees":
                perturbed[feature] = max(0.0, min(60.0, perturbed[feature]))
            elif feature in ("wind_direction_10m", "aspect_degrees"):
                perturbed[feature] = perturbed[feature] % 360

            try:
                p_df, p_id = _make_df(fire["lat"], fire["lon"], perturbed)
                p_result = sim.simulate(p_df, p_id, fire["ignition_prob"])

                entry = {"perturbation": delta, "input_value": round(perturbed[feature], 2)}
                for out_key in _OUTPUTS:
                    entry[out_key] = round(p_result.get(out_key, 0), 4)
                    # Normalized sensitivity index
                    base_out = base_result.get(out_key, 0)
                    if base_out != 0 and base_val != 0:
                        d_out = (p_result.get(out_key, 0) - base_out) / base_out
                        d_in = delta / 100.0 if ptype == "pct" else delta / base_val
                        entry[f"{out_key}_sensitivity"] = round(d_out / d_in, 4) if d_in != 0 else 0
                feature_results.append(entry)
            except Exception as exc:
                logger.warning("Sensitivity run failed: %s=%s: %s", feature, delta, exc)

        sensitivity[feature] = feature_results

    # Fuel model variants
    fuel_results = []
    for fm in _FUEL_VARIANTS:
        perturbed = dict(fire["conditions"])
        perturbed["fuel_model_fbfm40"] = float(fm)
        try:
            p_df, p_id = _make_df(fire["lat"], fire["lon"], perturbed)
            p_result = sim.simulate(p_df, p_id, fire["ignition_prob"])
            entry = {"fuel_model": fm}
            for out_key in _OUTPUTS:
                entry[out_key] = round(p_result.get(out_key, 0), 4)
            fuel_results.append(entry)
        except Exception as exc:
            logger.warning("Fuel variant %d failed: %s", fm, exc)

    sensitivity["fuel_model_fbfm40"] = fuel_results

    return sensitivity


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(per_fire: list[dict]) -> dict[str, Any]:
    """Compute summary statistics across N fires."""
    n = len(per_fire)
    if n == 0:
        return {"gate_passed": False, "overall_pass_rate": 0, "n_fires": 0}

    n_all_pass = sum(1 for f in per_fire if f["gate"]["gate_passed"])
    pass_rate = n_all_pass / n

    per_output_pass = {}
    per_output_errors = {}
    for output_name in ["direction", "speed", "moisture", "intensity", "crown_fire"]:
        passed_count = sum(
            1 for f in per_fire
            if f["gate"]["per_output"].get(output_name, {}).get("passed", False)
        )
        per_output_pass[output_name] = round(passed_count / n, 3)

        # Collect error values
        if output_name == "direction":
            errors = [f["gate"]["per_output"]["direction"]["angular_error_deg"] for f in per_fire]
        elif output_name == "speed":
            errors = [f["gate"]["per_output"]["speed"]["log_ratio"] for f in per_fire]
        elif output_name == "moisture":
            errors = [f["gate"]["per_output"]["moisture"]["abs_error_pct"] for f in per_fire]
        else:
            errors = []

        if errors:
            per_output_errors[output_name] = {
                "mean": round(float(np.mean(errors)), 2),
                "median": round(float(np.median(errors)), 2),
                "std": round(float(np.std(errors)), 2),
                "max": round(float(np.max(errors)), 2),
            }

    weakest = min(per_output_pass, key=per_output_pass.get) if per_output_pass else "unknown"

    return {
        "n_fires": n,
        "overall_pass_rate": round(pass_rate, 3),
        "gate_passed": pass_rate >= 0.80,
        "per_output_pass_rates": per_output_pass,
        "per_output_error_stats": per_output_errors,
        "fires_passing_all": [f["fire_id"] for f in per_fire if f["gate"]["gate_passed"]],
        "fires_failing": [f["fire_id"] for f in per_fire if not f["gate"]["gate_passed"]],
        "weakest_output": weakest,
    }


# ---------------------------------------------------------------------------
# Console printer
# ---------------------------------------------------------------------------

def _print_historical(per_fire: list[dict], aggregate: dict):
    """Print historical evaluation results to console."""
    print("\n" + "=" * 70)
    print("  OBJ-2 HISTORICAL FIRE EVALUATION")
    print("=" * 70)

    for f in per_fire:
        print(f"\n{'-'*70}")
        print(f"  {f['name']}")
        print(f"{'-'*70}")
        r = f["result"]
        mc = f.get("monte_carlo")

        # Hybrid output (40% det + 60% MC p90)
        DET_W, MC_W = 0.4, 0.6
        det_speed = r['spread_speed_kmh']
        if mc:
            mc_p90  = mc.get('spread_speed_kmh_p90', det_speed)
            h_speed = DET_W * det_speed + MC_W * mc_p90
            h_dir   = mc.get('dominant_direction_deg', r['spread_direction_deg'])
        else:
            mc_p90  = det_speed
            h_speed = det_speed
            h_dir   = r['spread_direction_deg']

        print(f"  approach   : HYBRID (40% det + 60% MC p90)")
        print(f"  direction  : {h_dir:.1f} deg")
        print(f"  speed      : {h_speed:.3f} km/h")
        print(f"  moisture   : {r['dead_fuel_moisture_pct']:.1f}%")
        print(f"  intensity  : {r['byram_intensity_kwm']:.1f} kW/m")
        print(f"  crown      : {r['crown_fire_status']}")
        if mc:
            print(f"  crown prob : {mc.get('crown_fire_probability', 0):.1%}")
            print(f"  speed p90  : {mc_p90:.3f} km/h (severe-scenario ceiling)")

    print("\n" + "=" * 70)
    agg = aggregate
    print(f"  AGGREGATE: {agg['overall_pass_rate']*100:.0f}% fires pass all gates "
          f"({len(agg['fires_passing_all'])}/{agg['n_fires']})")
    print("=" * 70)
    print(f"  Per-output pass rates:")
    for k, v in agg.get("per_output_pass_rates", {}).items():
        bar = "#" * int(v * 20) + "." * (20 - int(v * 20))
        print(f"    {k:15s} [{bar}] {v*100:.0f}%")

    if agg.get("per_output_error_stats"):
        print(f"\n  Error statistics:")
        for k, v in agg["per_output_error_stats"].items():
            print(f"    {k:15s}  mean={v['mean']:.2f}  median={v['median']:.2f}  "
                  f"std={v['std']:.2f}  max={v['max']:.2f}")

    print(f"\n  Weakest output: {agg['weakest_output']}")
    gate = "OVERALL GATE: PASSED" if agg["gate_passed"] else "OVERALL GATE: FAILED"
    print(f"  {gate}")
    print("=" * 70 + "\n")


def _print_realtime(results: dict):
    """Print real-time evaluation results."""
    print("\n" + "=" * 70)
    print("  OBJ-2 REAL-TIME PIPELINE EVALUATION")
    print("=" * 70)

    for res_km, data in results.items():
        print(f"\n{'-'*70}")
        print(f"  {res_km}km resolution")
        print(f"{'-'*70}")

        if data.get("error"):
            print(f"  [SKIP] {data['error']}")
            continue

        r = data["result"]
        mc = data.get("monte_carlo", {})

        # ── Hybrid output (det_weight=0.4, mc_weight=0.6 on p90) ─────
        DET_W, MC_W = 0.4, 0.6
        det_speed = r['spread_speed_kmh']
        mc_p90    = mc.get('spread_speed_kmh_p90', det_speed)
        h_speed   = DET_W * det_speed + MC_W * mc_p90
        h_dir     = mc.get('dominant_direction_deg', r['spread_direction_deg'])

        res_km_int = int(res_km)
        is_coarse  = res_km_int >= 64   # 93km intercell — spatial spread not meaningful

        print(f"  ignition cell : {r['ignition_cell']}")
        print(f"  approach      : HYBRID (40% deterministic + 60% MC p90)")
        print(f"  direction     : {h_dir:.1f} deg")
        print(f"  speed         : {h_speed:.4f} km/h")
        print(f"  moisture      : {r['dead_fuel_moisture_pct']:.1f}%")
        print(f"  intensity     : {r['byram_intensity_kwm']:.1f} kW/m")
        print(f"  crown         : {r['crown_fire_status']}")
        if mc:
            print(f"  crown prob    : {mc.get('crown_fire_probability', 0):.1%}")
            print(f"  speed CI p90  : {mc_p90:.4f} km/h (severe-scenario ceiling)")

        if is_coarse:
            # At 64km, intercell distance = 93km.
            # Fire behavior indices above are meaningful; spatial spread is not.
            print(f"\n  [NOTE] 64km resolution — fire behavior index only.")
            print(f"         Intercell distance is 93km. Use 22km for spatial spread prediction.")
        else:
            # ── Threatened cells analysis (22km only) ────────────────────
            ta = data.get("threatened_analysis", {})
            if ta:
                print(f"\n  Threatened neighbor analysis:")
                print(f"    burnable neighbors  : {ta.get('n_burnable_neighbors', '?')}/{ta.get('n_total_neighbors', '?')}")
                print(f"    reachable in 1h     : {ta.get('n_reachable_1h', 0)}")
                print(f"    max spread rate     : {ta.get('max_spread_rate_kmh', 0):.3f} km/h")
                t2n = ta.get('time_to_nearest_neighbor_h')
                print(f"    time to nearest     : {t2n:.1f}h" if t2n else "    time to nearest     : N/A")
                print(f"    spread cone         : {ta.get('spread_cone_deg', 0):.0f} deg")

            # ── Spread propagation ────────────────────────────────────────
            sp = data.get("spread_propagation")
            if sp:
                print(f"\n  1-hour spread propagation:")
                print(f"    total cells burned  : {sp['total_burned']}")
                timeline = sp.get("timeline", [])
                if timeline:
                    print(f"    {'hour':>6}  {'new cells':>10}  {'total burned':>13}")
                    for entry in timeline:
                        print(f"    {entry['t_hour']:>6.1f}  {entry['newly_ignited']:>10}  {entry['total_burning']:>13}")
                else:
                    print("    (fire did not spread beyond ignition cell)")

            # ── Neighbour burn probabilities ──────────────────────────────
            if mc:
                nb_probs = mc.get("neighbor_burn_probabilities", {})
                if nb_probs:
                    print(f"\n  Neighbour burn probabilities (MC N={mc['n_simulations']}, horizon={mc.get('horizon_hours', 1):.0f}h):")
                    sorted_nb = sorted(nb_probs.items(), key=lambda x: x[1], reverse=True)
                    for cell_id, prob in sorted_nb:
                        print(f"    {cell_id}  {prob:.1%}")

        # ── Input quality (both resolutions) ─────────────────────────────
        iq = data.get("input_quality", {})
        if iq:
            score = iq.get("quality_score", 0)
            label = "GOOD" if score > 0.75 else "MODERATE" if score > 0.5 else "LOW"
            print(f"\n  Input quality: {score:.0%} ({label})")

    print("=" * 70 + "\n")


def _print_sensitivity(sensitivity: dict):
    """Print top-3 most sensitive inputs per output."""
    print("\n" + "=" * 70)
    print("  OBJ-2 SENSITIVITY ANALYSIS (Top-3 drivers per output)")
    print("=" * 70)

    for out_key in _OUTPUTS:
        sens_key = f"{out_key}_sensitivity"
        rankings = []
        for feature, entries in sensitivity.items():
            if feature == "fuel_model_fbfm40":
                continue
            sensitivities = [abs(e.get(sens_key, 0)) for e in entries if sens_key in e]
            if sensitivities:
                rankings.append((feature, max(sensitivities)))
        rankings.sort(key=lambda x: x[1], reverse=True)

        print(f"\n  {out_key}:")
        for i, (feat, val) in enumerate(rankings[:3]):
            bar = "#" * min(int(val * 10), 30)
            print(f"    {i+1}. {feat:35s} |S|={val:.3f}  {bar}")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    mode: str = "all",
    resolutions: list[int] | None = None,
    fire_ids: list[str] | None = None,
    skip_sensitivity: bool = False,
    output_dir: str = "reports/evaluation",
) -> dict[str, Any]:
    """Run the full OBJ-2 evaluation."""
    sim = PythonFireSpreadSimulator()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report: dict[str, Any] = {
        "report_version": "2.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "simulator": "PythonFireSpreadSimulator",
    }

    # ── Initialize MLflow ────────────────────────────────────────────────
    n_fires = len(BENCHMARK_FIRES)
    if fire_ids:
        n_fires = len([f for f in BENCHMARK_FIRES if f["fire_id"] in fire_ids])
    tracker = _init_mlflow(mode, resolutions, n_fires)

    # ── Historical evaluation ────────────────────────────────────────────
    if mode in ("all", "historical"):
        fires = BENCHMARK_FIRES
        if fire_ids:
            fires = [f for f in fires if f["fire_id"] in fire_ids]

        per_fire_results = []
        all_sensitivity = {}

        for fire in fires:
            logger.info("Evaluating: %s", fire["name"])
            df, ign_id = _make_df(fire["lat"], fire["lon"], fire["conditions"])
            t0 = time.perf_counter()
            result = sim.simulate(df, ign_id, fire["ignition_prob"])
            elapsed_ms = (time.perf_counter() - t0) * 1000

            gate = compute_physics_gate(result, fire["ground_truth"])

            # Honest threatened cell analysis
            ta = analyze_threatened_cells(result)
            iq = compute_input_quality(result)

            # Monte Carlo for hybrid output
            try:
                mc_result = sim.simulate_monte_carlo(
                    df, ign_id, fire["ignition_prob"],
                    n_simulations=100,
                    horizon_hours=1.0,
                )
            except Exception as mc_exc:
                logger.warning("MC failed for %s: %s — using deterministic only", fire["fire_id"], mc_exc)
                mc_result = None

            entry = {
                "fire_id": fire["fire_id"],
                "name": fire["name"],
                "result": {k: result[k] for k in [
                    "spread_direction_deg", "spread_speed_kmh",
                    "dead_fuel_moisture_pct", "byram_intensity_kwm",
                    "crown_fire_status", "dominant_factor",
                ]},
                "gate": gate,
                "threatened_analysis": ta,
                "input_quality": iq,
                "monte_carlo": mc_result,
                "latency_ms": round(elapsed_ms, 1),
                "ground_truth": asdict(fire["ground_truth"]),
            }
            per_fire_results.append(entry)

            # Save per-fire predictions to organized directory
            fire_dir = PRED_HISTORICAL / fire["fire_id"]
            _save_prediction(fire_dir, "fire_behavior.json", result)
            _save_prediction(fire_dir, "threatened_analysis.json", ta)
            _save_prediction(fire_dir, "physics_gate.json", gate)
            _save_prediction(fire_dir, "input_quality.json", iq)

            # Sensitivity analysis
            if not skip_sensitivity:
                logger.info("  Running sensitivity for %s...", fire["fire_id"])
                sens = _run_sensitivity(sim, fire)
                all_sensitivity[fire["fire_id"]] = sens
                sens_dir = PRED_SENSITIVITY / ts
                _save_prediction(sens_dir, f"sensitivity_{fire['fire_id']}.json", sens)

        aggregate = _aggregate(per_fire_results)
        report["historical"] = {
            "per_fire": per_fire_results,
            "aggregate": aggregate,
        }
        if all_sensitivity:
            report["sensitivity"] = all_sensitivity

        _print_historical(per_fire_results, aggregate)
        _log_historical_metrics(tracker, per_fire_results, aggregate)

        if all_sensitivity:
            first_fire_id = list(all_sensitivity.keys())[0]
            _print_sensitivity(all_sensitivity[first_fire_id])

    # ── Real-time evaluation ─────────────────────────────────────────────
    if mode in ("all", "realtime"):
        # Team lead requirement: run at 22km only.
        # _select_ignition_22km() is reserved for deployment when OBJ-1
        # passes its top grid_id — at that point call:
        #   ign_id, ign_prob = _select_ignition_22km(df, obj1_grid_id, obj1_prob)
        if resolutions is None:
            resolutions = [22]

        rt_results = {}

        for res_km in resolutions:
            logger.info("Real-time evaluation: %dkm", res_km)

            if res_km == 22:
                df = _load_22km()
            elif res_km == 64:
                df = _load_64km()
            else:
                logger.warning("Resolution %dkm not supported — use 22 or 64", res_km)
                rt_results[str(res_km)] = {"error": f"Resolution {res_km}km not supported. Use 22 or 64."}
                continue

            if df is None:
                rt_results[str(res_km)] = {"error": "No data available"}
                continue

            # Filter to California for consistency
            if "region" in df.columns:
                ca = df[df["region"] == "california"]
                if not ca.empty:
                    df = ca

            # Ensure fuel model is numeric and valid
            if "fuel_model_fbfm40" in df.columns:
                df["fuel_model_fbfm40"] = pd.to_numeric(df["fuel_model_fbfm40"], errors="coerce")
                valid_fuel = df[df["fuel_model_fbfm40"].notna() & (df["fuel_model_fbfm40"] > 0)]
                if not valid_fuel.empty:
                    df = valid_fuel

            # Select ignition cell from 22km parquet
            # (FIRMS hotspot → driest cell → first cell)
            ign_id, ign_prob = _select_ignition(df)

            try:
                # ── Step 1: single-cell fire behavior ────────────────────────
                result = sim.simulate(df, ign_id, ign_prob)

                # Pass fire_detected flag so sanity check 3 is meaningful
                fire_row = df[df["grid_id"] == ign_id]
                fire_detected = (
                    bool(fire_row["fire_detected_binary"].iloc[0])
                    if not fire_row.empty and "fire_detected_binary" in fire_row.columns
                    else False
                )
                result["fire_detected"] = fire_detected
                result["input_rh_pct"] = result.get("inputs_used", {}).get("relative_humidity_pct")

                sanity = sanity_check_output(result)

                # ── Step 2: honest threatened cell analysis ──────────────────
                # horizon_hours=1.0 matches the 1-hour propagation window
                ta = analyze_threatened_cells(result, horizon_hours=1.0)
                prop_honesty = compute_propagation_honesty(result)
                iq = compute_input_quality(result)

                # ── Step 3: time-stepped propagation (1-hour horizon) ────────
                # Team lead requirement: 1-hour window at 22km.
                # At 25km intercell distance, any fire reaching a neighbor
                # within 1h would need rate ≥ 25 km/h — correctly rare.
                logger.info("Running 1-hour spread propagation from %s …", ign_id)
                spread = simulate_spread_timeseries(
                    df, ign_id, ign_prob, hours=1.0, timestep_h=1.0, sim=sim
                )

                # ── Step 4: Monte Carlo N=100 weather perturbations ──────────
                # Runs the same Rothermel physics 100 times with perturbed
                # weather (wind speed ±25%, wind direction ±25°, RH ±8%,
                # temp ±2.5°C) to produce burn probabilities per neighbour cell
                # — equivalent to Cell2Fire stochastic output, no binary needed.
                logger.info("Running Monte Carlo N=100 from %s …", ign_id)
                mc_result = sim.simulate_monte_carlo(
                    df, ign_id, ign_prob,
                    n_simulations=100,
                    horizon_hours=1.0,    # 1h window → threshold = intercell_km/1h
                )

                rt_entry = {
                    "ignition_cell": ign_id,
                    "ignition_prob": ign_prob,
                    "n_rows": len(df),
                    "result": {k: result[k] for k in [
                        "ignition_cell", "spread_direction_deg", "spread_speed_kmh",
                        "dead_fuel_moisture_pct", "byram_intensity_kwm",
                        "crown_fire_status", "dominant_factor",
                    ]},
                    "sanity": sanity,
                    "threatened_analysis": ta,
                    "propagation_honesty": prop_honesty,
                    "input_quality": iq,
                    "spread_propagation": {
                        "hours_simulated":  spread["hours_simulated"],
                        "total_burned":     spread["total_burned"],
                        "burned_cells":     spread["burned_cells"],
                        "timeline":         spread["timeline"],
                    },
                    "monte_carlo": {
                        "n_simulations":              mc_result["n_simulations"],
                        "horizon_hours":              mc_result["horizon_hours"],
                        "spread_speed_kmh_p50":       mc_result["spread_speed_kmh_p50"],
                        "spread_speed_kmh_p90":       mc_result["spread_speed_kmh_p90"],
                        "spread_speed_kmh_p95":       mc_result["spread_speed_kmh_p95"],
                        "spread_speed_kmh_mean":      mc_result["spread_speed_kmh_mean"],
                        "spread_speed_kmh_std":       mc_result["spread_speed_kmh_std"],
                        "dominant_direction_deg":     mc_result["dominant_direction_deg"],
                        "direction_uncertainty_deg":  mc_result["direction_uncertainty_deg"],
                        "crown_fire_probability":     mc_result["crown_fire_probability"],
                        "neighbor_burn_probabilities": mc_result["neighbor_burn_probabilities"],
                        "max_neighbor_burn_probability": mc_result["max_neighbor_burn_probability"],
                    },
                }
                rt_results[str(res_km)] = rt_entry

                # Save predictions to organized directory
                rt_dir = PRED_REALTIME / f"{res_km}km" / ts
                _save_prediction(rt_dir, "fire_behavior.json", result)
                _save_prediction(rt_dir, "threatened_analysis.json", ta)
                _save_prediction(rt_dir, "propagation_honesty.json", prop_honesty)
                _save_prediction(rt_dir, "sanity_checks.json", sanity)
                _save_prediction(rt_dir, "input_quality.json", iq)
                _save_prediction(rt_dir, "propagation.json", {
                    "hours_simulated": spread["hours_simulated"],
                    "total_burned": spread["total_burned"],
                    "burned_cells": spread["burned_cells"],
                    "timeline": spread["timeline"],
                })
                _save_prediction(rt_dir, "monte_carlo.json", mc_result)

                # ── Spread GeoJSON (TDD Section 6.4 + success criterion) ────
                # Produces TDD-required fields: spread_probability,
                # time_to_arrival_min, model, simulation_id, fire_intensity_kW_m
                # simulation_id ties this GeoJSON to the MLflow run (ts = run ts)
                spread_geojson = build_spread_geojson(
                    result, spread,
                    resolution_km=res_km,
                    simulation_id=ts,
                )
                save_spread_geojson(spread_geojson, rt_dir)
                rt_entry["cell2fire_geojson"] = spread_geojson["features"]

                # Clean operational output — the format shown to operators / LLM
                op_out = _format_operational_output(result, res_km, ign_prob)
                _save_prediction(rt_dir, "operational_output.json", op_out)
                rt_entry["operational_output"] = op_out

            except Exception as exc:
                logger.error("Real-time %dkm failed: %s", res_km, exc)
                rt_results[str(res_km)] = {"error": str(exc)}

        report["realtime"] = rt_results
        _print_realtime(rt_results)
        _log_realtime_metrics(tracker, rt_results)

    # ── FIRMS temporal validation ────────────────────────────────────────
    if mode == "firms_validation":
        try:
            from models.obj2_spread.firms_validator import validate_against_firms
            backfill_dir = DATA_PIPELINE / "data" / "processed" / "backfill" / "64km"
            logger.info("Running FIRMS temporal validation from %s", backfill_dir)
            firms_metrics = validate_against_firms(backfill_dir, sim)

            report["firms_validation"] = firms_metrics

            # Save to organized directory
            _save_prediction(PRED_FIRMS, "metrics_summary.json", firms_metrics)

            # Log to MLflow
            if tracker:
                try:
                    tracker.log_metrics({
                        "firms_n_events": float(firms_metrics.get("n_events", 0)),
                        "firms_spread_recall": firms_metrics.get("spread_event_recall", 0.0),
                        "firms_direction_acc_45": firms_metrics.get("direction_accuracy_45deg", 0.0),
                        "firms_csi": firms_metrics.get("csi", 0.0),
                        "firms_persistence_csi": firms_metrics.get("persistence_baseline_csi", 0.0),
                        "firms_beats_persistence": 1.0 if firms_metrics.get("model_beats_persistence") else 0.0,
                    })
                except Exception as exc:
                    logger.warning("MLflow FIRMS logging failed: %s", exc)

            # Print summary
            print("\n" + "=" * 70)
            print("  OBJ-2 FIRMS TEMPORAL VALIDATION")
            print("=" * 70)
            print(f"  Spread events found     : {firms_metrics.get('n_events', 0)}")
            print(f"  Spread event recall     : {firms_metrics.get('spread_event_recall', 0):.3f}")
            print(f"  Direction accuracy (45d) : {firms_metrics.get('direction_accuracy_45deg', 0):.3f}")
            print(f"  CSI (fire-positive)     : {firms_metrics.get('csi', 0):.3f}")
            print(f"  Persistence baseline CSI: {firms_metrics.get('persistence_baseline_csi', 0):.3f}")
            beats = firms_metrics.get("model_beats_persistence", False)
            icon = "[PASS]" if beats else "[FAIL]"
            print(f"  {icon}  Model beats persistence: {beats}")
            print("=" * 70 + "\n")

        except ImportError:
            logger.warning("firms_validator not found — skipping FIRMS validation")
        except Exception as exc:
            logger.error("FIRMS validation failed: %s", exc)
            report["firms_validation"] = {"error": str(exc)}

    # ── Save JSON report ─────────────────────────────────────────────────
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_path = out_path / f"obj2_eval_{ts}.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved: {report_path}")

    # Log report as MLflow artifact
    if tracker:
        try:
            tracker.log_artifact(str(report_path), "evaluation_reports")
            tracker.end_run()
            logger.info("MLflow run completed — metrics logged to obj2-spread-evaluation")
        except Exception as exc:
            logger.warning("MLflow finalization failed: %s", exc)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OBJ-2 Fire Spread Simulator Evaluation"
    )
    parser.add_argument(
        "--mode", choices=["all", "historical", "realtime", "firms_validation"],
        default="all", help="Evaluation mode (default: all)",
    )
    parser.add_argument(
        "--resolution", type=int, nargs="*",
        help="Resolutions for real-time mode (e.g., 64 22). Default: both.",
    )
    parser.add_argument(
        "--fires", nargs="*",
        help="Specific fire IDs for historical mode (e.g., camp_fire_2018)",
    )
    parser.add_argument(
        "--skip-sensitivity", action="store_true",
        help="Skip sensitivity analysis for faster execution",
    )
    parser.add_argument(
        "--output-dir", default="reports/evaluation",
        help="Directory for JSON report output",
    )
    args = parser.parse_args()

    report = main(
        mode=args.mode,
        resolutions=args.resolution,
        fire_ids=args.fires,
        skip_sensitivity=args.skip_sensitivity,
        output_dir=args.output_dir,
    )

    # Exit code based on historical gate (if run)
    if "historical" in report:
        sys.exit(0 if report["historical"]["aggregate"]["gate_passed"] else 1)
    sys.exit(0)
