"""Bridge — converts OBJ-1 and OBJ-2 model outputs into the
``pipeline_result`` dict that OBJ-3's context builder expects.

This is the integration point between the ML/simulation models and the
LLM disaster reporting layer.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Risk level derivation
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "MODERATE": 0.25,
    "HIGH": 0.50,
    "CRITICAL": 0.75,
}


def derive_risk_level(
    max_probability: float,
    thresholds: dict[str, float] | None = None,
) -> str:
    """Derive a risk level string from the maximum ignition probability.

    Parameters
    ----------
    max_probability:
        Highest fire ignition probability across all cells (0.0–1.0).
    thresholds:
        Optional override for the boundary values.  Keys must include
        ``MODERATE``, ``HIGH``, ``CRITICAL``.

    Returns
    -------
    One of ``"LOW"``, ``"MODERATE"``, ``"HIGH"``, ``"CRITICAL"``.
    """
    t = thresholds or _DEFAULT_THRESHOLDS
    if max_probability >= t["CRITICAL"]:
        return "CRITICAL"
    if max_probability >= t["HIGH"]:
        return "HIGH"
    if max_probability >= t["MODERATE"]:
        return "MODERATE"
    return "LOW"


# ---------------------------------------------------------------------------
# Telemetry extraction
# ---------------------------------------------------------------------------

def extract_telemetry(
    obj1_input: pd.DataFrame,
    obj2_simulation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract environmental telemetry from the OBJ-1 input DataFrame.

    Handles both the actual pipeline column names (``temperature_2m``,
    ``wind_speed_10m``) and the legacy ERA5 names (``t2m``, ``u10``).

    Returns a dict suitable for ``pipeline_result["telemetry"]``.
    """
    telem: dict[str, Any] = {}

    if len(obj1_input) == 0:
        # Nothing to extract from empty data
        if obj2_simulation:
            if "dead_fuel_moisture_pct" in obj2_simulation:
                telem["dead_fuel_moisture_pct"] = obj2_simulation["dead_fuel_moisture_pct"]
            if "foliar_moisture_content_pct" in obj2_simulation:
                telem["foliar_moisture_content_pct"] = obj2_simulation["foliar_moisture_content_pct"]
        return telem

    # Temperature (°C → °F)
    if "temperature_2m" in obj1_input.columns:
        temp_c = obj1_input["temperature_2m"].max()
        telem["temperature_max"] = round(temp_c * 9 / 5 + 32, 1)
    elif "temperature_c" in obj1_input.columns:
        temp_c = obj1_input["temperature_c"].max()
        telem["temperature_max"] = round(temp_c * 9 / 5 + 32, 1)

    # Wind speed (km/h → mph)
    if "wind_speed_10m" in obj1_input.columns:
        ws_kmh = obj1_input["wind_speed_10m"].max()
        telem["wind_speed_mph"] = round(ws_kmh * 0.621371, 1)
    elif "wind_speed_m_s" in obj1_input.columns:
        ws_ms = obj1_input["wind_speed_m_s"].max()
        telem["wind_speed_mph"] = round(ws_ms * 2.23694, 1)

    # Relative humidity (%)
    if "relative_humidity_2m" in obj1_input.columns:
        telem["relative_humidity"] = round(float(obj1_input["relative_humidity_2m"].mean()), 1)
    elif "relative_humidity" in obj1_input.columns:
        telem["relative_humidity"] = round(float(obj1_input["relative_humidity"].mean()), 1)

    # Soil moisture
    if "soil_moisture_0_to_7cm" in obj1_input.columns:
        telem["soil_moisture"] = round(float(obj1_input["soil_moisture_0_to_7cm"].mean()), 4)

    # VPD
    if "vpd" in obj1_input.columns:
        telem["vpd_kpa"] = round(float(obj1_input["vpd"].mean()), 2)

    # Fire Weather Index
    if "fire_weather_index" in obj1_input.columns:
        telem["fire_weather_index"] = round(float(obj1_input["fire_weather_index"].max()), 1)

    # Fuel moisture from OBJ-2 simulation
    if obj2_simulation:
        if "dead_fuel_moisture_pct" in obj2_simulation:
            telem["dead_fuel_moisture_pct"] = obj2_simulation["dead_fuel_moisture_pct"]
        if "foliar_moisture_content_pct" in obj2_simulation:
            telem["foliar_moisture_content_pct"] = obj2_simulation["foliar_moisture_content_pct"]

    return telem


# ---------------------------------------------------------------------------
# Propagator summary from OBJ-2 simulation
# ---------------------------------------------------------------------------

def _build_propagator_summary(obj2_sim: dict[str, Any]) -> str:
    """Build a human-readable propagator summary from OBJ-2 simulation output."""
    parts: list[str] = []

    cell = obj2_sim.get("ignition_cell", "unknown")
    parts.append(f"Ignition cell: {cell}")

    speed = obj2_sim.get("spread_speed_kmh")
    direction = obj2_sim.get("spread_direction_deg")
    if speed is not None:
        parts.append(f"spreading at {speed} km/h ({round(speed * 0.621371, 1)} mph)")
    if direction is not None:
        parts.append(f"direction {direction}°")

    crown = obj2_sim.get("crown_fire_status")
    if crown:
        parts.append(f"crown fire: {crown}")

    intensity = obj2_sim.get("byram_intensity_kwm")
    if intensity is not None:
        parts.append(f"Byram intensity: {intensity} kW/m")

    factor = obj2_sim.get("dominant_factor")
    if factor:
        parts.append(f"dominant factor: {factor}")

    warnings = obj2_sim.get("warnings") or []
    if warnings:
        parts.append(f"Warnings: {', '.join(warnings)}")

    return ". ".join(parts) + "."


# ---------------------------------------------------------------------------
# Main bridge function
# ---------------------------------------------------------------------------

def build_pipeline_result(
    obj1_predictions: pd.DataFrame,
    obj1_input: pd.DataFrame,
    obj2_simulation: dict[str, Any] | None = None,
    firms_hotspots: list[dict[str, Any]] | None = None,
    bias_report: dict[str, Any] | None = None,
    source_status: dict[str, Any] | None = None,
    risk_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Convert OBJ-1 and OBJ-2 model outputs to a ``pipeline_result`` dict.

    Parameters
    ----------
    obj1_predictions:
        DataFrame from ``XGBoostFireRiskModel.predict()`` with columns
        ``prediction`` (0/1) and ``probability`` (float).
    obj1_input:
        The input DataFrame passed to OBJ-1, containing ``grid_id`` (or
        ``h3_index``), ``latitude``/``longitude``, and weather features.
    obj2_simulation:
        Optional dict from the OBJ-2 Rothermel spread simulation.
    firms_hotspots:
        Optional list of FIRMS hotspot dicts with ``lat``, ``lon``,
        ``frp``, ``confidence``, ``acq_datetime``.
    bias_report:
        Optional bias gate result dict.
    source_status:
        Optional per-source freshness status dict.
    risk_thresholds:
        Optional override for risk level boundaries.

    Returns
    -------
    dict
        Ready to pass to ``GeminiDisasterReporter.generate_report()``.
    """
    firms_hotspots = firms_hotspots or []

    # --- Merge predictions with input to get spatial info ---
    merged = obj1_input.copy()
    merged["_pred"] = obj1_predictions["prediction"].values
    merged["_prob"] = obj1_predictions["probability"].values

    # Resolve grid ID column name (pipeline uses grid_id, legacy uses h3_index)
    grid_col = "grid_id" if "grid_id" in merged.columns else "h3_index"

    # Resolve lat/lon columns
    lat_col = "latitude" if "latitude" in merged.columns else "lat"
    lon_col = "longitude" if "longitude" in merged.columns else "lon"

    # --- Build xgboost_top_cells ---
    top = merged.nlargest(20, "_prob")
    xgboost_top_cells = []
    for _, row in top.iterrows():
        xgboost_top_cells.append({
            "h3_index": str(row.get(grid_col, "")),
            "probability": round(float(row["_prob"]), 4),
            "lat": round(float(row.get(lat_col, 0)), 6),
            "lon": round(float(row.get(lon_col, 0)), 6),
        })

    # --- Derive risk level ---
    max_prob = float(merged["_prob"].max()) if len(merged) > 0 else 0.0
    risk_level = derive_risk_level(max_prob, risk_thresholds)

    # --- Telemetry ---
    telemetry = extract_telemetry(obj1_input, obj2_simulation)

    # --- Propagator summary ---
    propagator_summary = None
    if obj2_simulation:
        propagator_summary = _build_propagator_summary(obj2_simulation)

    # --- Assemble ---
    return {
        "run_id": f"bridge-{datetime.now(tz=UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
        "is_deployable": True,
        "risk_level": risk_level,
        "firms_hotspot_count": len(firms_hotspots),
        "firms_hotspots": firms_hotspots,
        "xgboost_top_cells": xgboost_top_cells,
        "cell2fire_geojson": None,
        "obj2_simulation": obj2_simulation,
        "propagator_summary": propagator_summary,
        "telemetry": telemetry or None,
        "fema_nri_tracts": [],
        "bias_report": bias_report,  # None if not provided — context builder handles absence
        "metrics": {
            "max_probability": round(max_prob, 4),
            "mean_probability": round(float(merged["_prob"].mean()), 4) if len(merged) > 0 else 0.0,
            "cells_above_50pct": int((merged["_prob"] >= 0.5).sum()) if len(merged) > 0 else 0,
        },
        "source_status": source_status,  # None if not provided — context builder handles absence
        "data_completeness": {
            "xgboost_predictions": len(xgboost_top_cells) > 0,
            "obj2_simulation": obj2_simulation is not None,
            "firms_hotspots": len(firms_hotspots) > 0,
            "telemetry": bool(telemetry),
            "fema_nri": False,  # Not yet integrated
            "bias_report": bias_report is not None,
            "source_status": source_status is not None,
        },
    }
