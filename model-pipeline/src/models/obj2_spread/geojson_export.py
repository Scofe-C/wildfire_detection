"""
geojson_export.py — Fire Spread GeoJSON output
===============================================
Converts PythonFireSpreadSimulator output into a GeoJSON FeatureCollection
that can be rendered on a map and consumed by OBJ-3 reporting.

Each H3 cell is converted to its hexagon polygon boundary.
Features carry fire behavior properties (speed, direction, crown status,
intensity) so the map layer is self-contained.

Success criterion: spread.geojson written within 5 minutes of ignition trigger.
This module itself completes in < 1 second — the budget is consumed by the
upstream data pipeline (fuse_features) before simulate() is ever called.

Feature layers
--------------
  status = "ignition"   — the cell where fire started (t_hour = 0)
  status = "burned"     — cells reached by fire during propagation (t_hour = 1..N)
  status = "threatened" — ring-1 neighbors that are burnable but not yet reached
  status = "non_burnable" — ring-1 neighbors classified as non-burnable

Properties on every feature
----------------------------
  grid_id               H3 cell ID (string)
  t_hour                Hour at which the cell ignited (None for threatened)
  status                One of the four strings above
  spread_rate_kmh       Spread rate from ignition toward this cell (km/h)
  crown_fire_status     "surface" | "passive_crown" | "active_crown" | "non_burnable"
  byram_intensity_kwm   Byram fireline intensity (kW/m)
  bearing_deg           Compass bearing from ignition cell center (°, 0 = N)

Ignition cell also carries
--------------------------
  spread_direction_deg  Dominant fire-front bearing (weighted circular mean)
  spread_speed_kmh      Maximum spread rate (km/h)
  dead_fuel_moisture_pct 1-hr DFMC (%)
  dominant_factor       "wind" | "slope" | "balanced"

Owner: OBJ-2 (fire spread model)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_spread_geojson(
    result: dict[str, Any],
    spread: dict[str, Any],
    resolution_km: int = 64,
    simulation_id: str | None = None,
) -> dict[str, Any]:
    """Convert simulate() + simulate_spread_timeseries() outputs to GeoJSON.

    Produces a TDD-compliant GeoJSON FeatureCollection (Section 6.4).
    Every feature carries both the original fire behavior fields AND the
    TDD-required fields: spread_probability, time_to_arrival_min, model,
    simulation_id, fire_intensity_kW_m.

    Parameters
    ----------
    result        : Output from ``PythonFireSpreadSimulator.simulate()``.
    spread        : Output from ``simulate_spread_timeseries()``.
    resolution_km : H3 resolution in km (22 or 64). Used to derive
                    intercell distance for time_to_arrival_min.
    simulation_id : UUID / run ID linking this output to an MLflow run.
                    Auto-generated from timestamp if not provided.

    TDD fields added (Section 6.4)
    --------------------------------
    spread_probability   float [0–1]  Normalized from spread_rate_kmh.
                                      0 = no spread, 1 = physical max (50 km/h).
                                      Honest proxy — not Monte Carlo.
    time_to_arrival_min  int | None   Minutes from ignition to cell ignition.
                                      Burned cells: from propagation timeline.
                                      Threatened cells: intercell_km/rate × 60.
                                      None if spread rate is zero.
    model                string       Always "python_rothermel" — identifies
                                      this as the pure-Python Rothermel
                                      implementation, not Cell2Fire C++.
    simulation_id        string       Links to MLflow run for audit trail.
    fire_intensity_kW_m  float        Alias of byram_intensity_kwm (TDD name).
    """
    import h3
    import uuid

    # Intercell distance by resolution (H3 center-to-center, km)
    _INTERCELL_KM = {22: 25.0, 64: 93.0}
    intercell_km = _INTERCELL_KM.get(resolution_km, 93.0)

    # Physical ROS cap (50 km/h) used to normalize spread_probability
    _MAX_ROS_KMH = 50.0

    # simulation_id: use provided value or generate a stable UUID
    if not simulation_id:
        simulation_id = str(uuid.uuid4())

    ignition_cell = result.get("ignition_cell", "")
    burned_set = set(spread.get("burned_cells", []))

    # t_hour lookup: cell_id → first hour it ignited
    cell_hour: dict[str, float] = {ignition_cell: 0.0}
    for entry in spread.get("timeline", []):
        for cell_id in entry.get("cells", []):
            if cell_id not in cell_hour:
                cell_hour[cell_id] = entry["t_hour"]

    # neighbour lookup for per-cell spread rates
    nb_lookup: dict[str, dict] = {
        nb["neighbour_id"]: nb
        for nb in result.get("neighbour_details", [])
        if nb.get("neighbour_id")
    }

    def _spread_prob(rate_kmh: float | None) -> float:
        """Normalize spread rate to [0, 1] probability proxy."""
        if not rate_kmh or rate_kmh <= 0:
            return 0.0
        return round(min(rate_kmh / _MAX_ROS_KMH, 1.0), 4)

    def _arrival_min(rate_kmh: float | None, t_hour: float | None = None) -> int | None:
        """Compute time-to-arrival in minutes.

        For burned cells: use propagation timeline (t_hour × 60).
        For threatened cells: intercell_km / rate × 60.
        Returns None if rate is zero or unknown.
        """
        if t_hour is not None:
            return int(round(t_hour * 60))
        if not rate_kmh or rate_kmh <= 0:
            return None
        return int(round(intercell_km / rate_kmh * 60))

    features: list[dict] = []

    # ── 1. Ignition cell ──────────────────────────────────────────────────
    boundary = _cell_polygon(ignition_cell)
    if boundary:
        ign_rate = result.get("spread_speed_kmh", 0.0)
        ign_intensity = result.get("byram_intensity_kwm", 0.0)
        features.append({
            "type": "Feature",
            "geometry": boundary,
            "properties": {
                # TDD-required fields (Section 6.4)
                "spread_probability":  _spread_prob(ign_rate),
                "time_to_arrival_min": 0,
                "fire_intensity_kW_m": ign_intensity,
                "model":               "python_rothermel",
                "simulation_id":       simulation_id,
                # Extended fire behavior fields
                "grid_id":               ignition_cell,
                "t_hour":                0.0,
                "status":                "ignition",
                "spread_direction_deg":  result.get("spread_direction_deg"),
                "spread_speed_kmh":      ign_rate,
                "dead_fuel_moisture_pct": result.get("dead_fuel_moisture_pct"),
                "byram_intensity_kwm":   ign_intensity,
                "crown_fire_status":     result.get("crown_fire_status"),
                "dominant_factor":       result.get("dominant_factor"),
                "bearing_deg":           None,
            },
        })

    # ── 2. Burned cells (propagated from ignition) ────────────────────────
    for cell_id in sorted(burned_set):
        if cell_id == ignition_cell:
            continue
        boundary = _cell_polygon(cell_id)
        if not boundary:
            continue
        nb = nb_lookup.get(cell_id, {})
        rate = nb.get("spread_rate_kmh", 0.0)
        t_hr = cell_hour.get(cell_id)
        intensity = nb.get("byram_intensity_kwm", 0.0)
        features.append({
            "type": "Feature",
            "geometry": boundary,
            "properties": {
                # TDD-required fields
                "spread_probability":  _spread_prob(rate),
                "time_to_arrival_min": _arrival_min(rate, t_hr),
                "fire_intensity_kW_m": intensity,
                "model":               "python_rothermel",
                "simulation_id":       simulation_id,
                # Extended fields
                "grid_id":               cell_id,
                "t_hour":                t_hr,
                "status":                "burned",
                "spread_direction_deg":  None,
                "spread_speed_kmh":      rate,
                "dead_fuel_moisture_pct": result.get("dead_fuel_moisture_pct"),
                "byram_intensity_kwm":   intensity,
                "crown_fire_status":     nb.get("crown_status"),
                "dominant_factor":       None,
                "bearing_deg":           nb.get("bearing_deg"),
            },
        })

    # ── 3. Threatened and non-burnable ring-1 neighbors ───────────────────
    for nb in result.get("neighbour_details", []):
        cell_id = nb.get("neighbour_id")
        if not cell_id or cell_id in burned_set or cell_id == ignition_cell:
            continue
        boundary = _cell_polygon(cell_id)
        if not boundary:
            continue

        crown = nb.get("crown_status", "unknown")
        status = "non_burnable" if crown == "non_burnable" else "threatened"
        rate = nb.get("spread_rate_kmh", 0.0)
        intensity = nb.get("byram_intensity_kwm", 0.0)

        features.append({
            "type": "Feature",
            "geometry": boundary,
            "properties": {
                # TDD-required fields
                "spread_probability":  _spread_prob(rate),
                "time_to_arrival_min": _arrival_min(rate),
                "fire_intensity_kW_m": intensity,
                "model":               "python_rothermel",
                "simulation_id":       simulation_id,
                # Extended fields
                "grid_id":               cell_id,
                "t_hour":                None,
                "status":                status,
                "spread_direction_deg":  None,
                "spread_speed_kmh":      rate,
                "dead_fuel_moisture_pct": None,
                "byram_intensity_kwm":   intensity,
                "crown_fire_status":     crown,
                "dominant_factor":       None,
                "bearing_deg":           nb.get("bearing_deg"),
            },
        })

    geojson = {
        "type": "FeatureCollection",
        "properties": {
            # TDD-required collection-level fields
            "model":          "python_rothermel",
            "simulation_id":  simulation_id,
            "resolution_km":  resolution_km,
            # Fire behavior summary
            "ignition_cell":       ignition_cell,
            "hours_simulated":     spread.get("hours_simulated", 6.0),
            "total_burned":        spread.get("total_burned", 1),
            "spread_direction_deg": result.get("spread_direction_deg"),
            "spread_speed_kmh":    result.get("spread_speed_kmh"),
            "crown_fire_status":   result.get("crown_fire_status"),
            "byram_intensity_kwm": result.get("byram_intensity_kwm"),
            # Honest scope note — documented in TDD Section 13
            "disclaimer": (
                "Physics-based fire behavior index computed with pure-Python "
                "Rothermel (1972) at H3 resolution. Not Cell2Fire C++ output. "
                "Spatial spread predictions have high uncertainty at this resolution."
            ),
        },
        "features": features,
    }

    logger.info(
        "Built spread GeoJSON: %d features "
        "(1 ignition, %d burned, %d threatened/non-burnable)",
        len(features),
        len(burned_set) - 1,
        len(features) - len(burned_set),
    )
    return geojson


def save_spread_geojson(
    geojson: dict[str, Any],
    output_dir: Path,
    filename: str = "spread.geojson",
) -> Path:
    """Write GeoJSON to disk.

    Parameters
    ----------
    geojson     : GeoJSON dict from build_spread_geojson().
    output_dir  : Directory to write into (created if missing).
    filename    : Output filename (default: spread.geojson).

    Returns
    -------
    Path to written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2, default=str)
    logger.info("Wrote spread GeoJSON: %s (%d features)", out_path, len(geojson["features"]))
    return out_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cell_polygon(cell_id: str) -> dict | None:
    """Convert an H3 cell ID to a GeoJSON Polygon geometry.

    H3 returns boundary as [(lat, lon), ...].
    GeoJSON expects [lon, lat] and requires the ring to be closed
    (first == last coordinate).

    Returns None if the cell_id is invalid.
    """
    try:
        import h3
        boundary = h3.cell_to_boundary(cell_id)   # [(lat, lon), ...]
        coords = [[lon, lat] for lat, lon in boundary]
        coords.append(coords[0])                   # close the ring
        return {"type": "Polygon", "coordinates": [coords]}
    except Exception as exc:
        logger.warning("Could not get boundary for cell %s: %s", cell_id, exc)
        return None
