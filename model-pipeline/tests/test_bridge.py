"""Tests for src.pipeline.bridge — OBJ-1/OBJ-2 → OBJ-3 bridge."""

from __future__ import annotations

import pandas as pd
import pytest

from src.pipeline.bridge import (
    build_pipeline_result,
    derive_risk_level,
    extract_telemetry,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def obj1_input() -> pd.DataFrame:
    """Minimal OBJ-1 input DataFrame with pipeline column names."""
    return pd.DataFrame({
        "grid_id": ["822937fffffffff", "822987fffffffff", "8228a7fffffffff"],
        "latitude": [30.35, 36.82, 41.72],
        "longitude": [-121.79, -116.05, -117.33],
        "temperature_2m": [16.0, 27.4, 14.1],
        "relative_humidity_2m": [82.0, 12.9, 29.7],
        "wind_speed_10m": [32.6, 12.5, 9.4],
        "soil_moisture_0_to_7cm": [0.0, 0.017, 0.032],
        "vpd": [0.33, 3.31, 1.23],
        "fire_weather_index": [7.3, 16.7, 5.8],
    })


@pytest.fixture()
def obj1_predictions() -> pd.DataFrame:
    """OBJ-1 prediction output."""
    return pd.DataFrame({
        "prediction": [0, 1, 0],
        "probability": [0.12, 0.87, 0.23],
    })


@pytest.fixture()
def obj2_simulation() -> dict:
    """OBJ-2 Rothermel simulation output."""
    return {
        "ignition_cell": "822987fffffffff",
        "ignition_probability": 0.72,
        "spread_direction_deg": 77.1,
        "spread_speed_kmh": 2.27,
        "crown_fire_status": "passive_crown",
        "byram_intensity_kwm": 1446.1,
        "dead_fuel_moisture_pct": 11.3,
        "foliar_moisture_content_pct": 115.0,
        "dominant_factor": "wind",
        "inputs_used": {
            "wind_speed_10m_ms": 9.64,
            "midflame_wind_mph": 8.63,
            "ignition_cell_slope_deg": 2.2,
            "ignition_cell_fbfm40": 122.0,
        },
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# derive_risk_level
# ---------------------------------------------------------------------------

class TestDeriveRiskLevel:
    def test_low(self):
        assert derive_risk_level(0.0) == "LOW"
        assert derive_risk_level(0.24) == "LOW"

    def test_moderate(self):
        assert derive_risk_level(0.25) == "MODERATE"
        assert derive_risk_level(0.49) == "MODERATE"

    def test_high(self):
        assert derive_risk_level(0.50) == "HIGH"
        assert derive_risk_level(0.74) == "HIGH"

    def test_critical(self):
        assert derive_risk_level(0.75) == "CRITICAL"
        assert derive_risk_level(1.0) == "CRITICAL"

    def test_custom_thresholds(self):
        custom = {"MODERATE": 0.3, "HIGH": 0.6, "CRITICAL": 0.9}
        assert derive_risk_level(0.29, custom) == "LOW"
        assert derive_risk_level(0.30, custom) == "MODERATE"
        assert derive_risk_level(0.60, custom) == "HIGH"
        assert derive_risk_level(0.90, custom) == "CRITICAL"


# ---------------------------------------------------------------------------
# extract_telemetry
# ---------------------------------------------------------------------------

class TestExtractTelemetry:
    def test_pipeline_columns(self, obj1_input):
        telem = extract_telemetry(obj1_input)
        # temperature_2m max is 27.4°C → (27.4 * 9/5 + 32) = 81.3°F
        assert abs(telem["temperature_max"] - 81.3) < 0.2
        # wind_speed_10m max is 32.6 km/h → 32.6 * 0.621371 = 20.3 mph
        assert abs(telem["wind_speed_mph"] - 20.3) < 0.2
        assert "relative_humidity" in telem
        assert "soil_moisture" in telem

    def test_with_obj2_fuel_moisture(self, obj1_input, obj2_simulation):
        telem = extract_telemetry(obj1_input, obj2_simulation)
        assert telem["dead_fuel_moisture_pct"] == 11.3
        assert telem["foliar_moisture_content_pct"] == 115.0

    def test_without_obj2(self, obj1_input):
        telem = extract_telemetry(obj1_input, None)
        assert "dead_fuel_moisture_pct" not in telem

    def test_empty_dataframe(self):
        empty = pd.DataFrame(columns=["temperature_2m", "wind_speed_10m"])
        telem = extract_telemetry(empty)
        assert telem == {}


# ---------------------------------------------------------------------------
# build_pipeline_result
# ---------------------------------------------------------------------------

class TestBuildPipelineResult:
    def test_required_keys(self, obj1_predictions, obj1_input):
        result = build_pipeline_result(obj1_predictions, obj1_input)
        required_keys = {
            "run_id", "is_deployable", "risk_level", "firms_hotspot_count",
            "firms_hotspots", "xgboost_top_cells", "cell2fire_geojson",
            "obj2_simulation", "propagator_summary", "telemetry",
            "fema_nri_tracts", "bias_report", "metrics", "source_status",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_risk_level_derived(self, obj1_predictions, obj1_input):
        result = build_pipeline_result(obj1_predictions, obj1_input)
        # max prob is 0.87 → CRITICAL
        assert result["risk_level"] == "CRITICAL"

    def test_top_cells_sorted(self, obj1_predictions, obj1_input):
        result = build_pipeline_result(obj1_predictions, obj1_input)
        cells = result["xgboost_top_cells"]
        assert len(cells) == 3
        # Sorted desc by probability
        assert cells[0]["probability"] >= cells[1]["probability"]
        assert cells[1]["probability"] >= cells[2]["probability"]
        # Check h3_index is present
        assert all("h3_index" in c for c in cells)

    def test_top_cells_capped_at_20(self, obj1_input):
        # Create 30 predictions
        preds = pd.DataFrame({
            "prediction": [1] * 30,
            "probability": [i / 30 for i in range(30)],
        })
        big_input = pd.concat([obj1_input] * 10, ignore_index=True)
        result = build_pipeline_result(preds, big_input)
        assert len(result["xgboost_top_cells"]) == 20

    def test_with_obj2_simulation(self, obj1_predictions, obj1_input, obj2_simulation):
        result = build_pipeline_result(
            obj1_predictions, obj1_input, obj2_simulation=obj2_simulation,
        )
        assert result["obj2_simulation"] is obj2_simulation
        assert result["propagator_summary"] is not None
        assert "2.27 km/h" in result["propagator_summary"]
        assert "passive_crown" in result["propagator_summary"]

    def test_without_obj2(self, obj1_predictions, obj1_input):
        result = build_pipeline_result(obj1_predictions, obj1_input)
        assert result["obj2_simulation"] is None
        assert result["propagator_summary"] is None

    def test_firms_hotspots(self, obj1_predictions, obj1_input):
        hotspots = [
            {"lat": 34.12, "lon": -118.32, "frp": 85.3, "confidence": "high"},
            {"lat": 34.14, "lon": -118.30, "frp": 62.1, "confidence": "nominal"},
        ]
        result = build_pipeline_result(
            obj1_predictions, obj1_input, firms_hotspots=hotspots,
        )
        assert result["firms_hotspot_count"] == 2
        assert result["firms_hotspots"] is hotspots

    def test_grid_id_column(self, obj1_predictions, obj1_input):
        """Verify grid_id column is used and mapped to h3_index in output."""
        result = build_pipeline_result(obj1_predictions, obj1_input)
        cells = result["xgboost_top_cells"]
        assert cells[0]["h3_index"] == "822987fffffffff"  # highest prob cell

    def test_legacy_h3_index_column(self, obj1_predictions):
        """Verify fallback to h3_index column if grid_id is absent."""
        legacy_input = pd.DataFrame({
            "h3_index": ["cellA", "cellB", "cellC"],
            "lat": [30.0, 36.0, 41.0],
            "lon": [-121.0, -116.0, -117.0],
            "temperature_c": [16.0, 27.0, 14.0],
            "wind_speed_m_s": [5.0, 3.0, 2.0],
            "relative_humidity": [82.0, 13.0, 30.0],
        })
        result = build_pipeline_result(obj1_predictions, legacy_input)
        cells = result["xgboost_top_cells"]
        assert cells[0]["h3_index"] == "cellB"  # highest prob

    def test_metrics_present(self, obj1_predictions, obj1_input):
        result = build_pipeline_result(obj1_predictions, obj1_input)
        assert "max_probability" in result["metrics"]
        assert "mean_probability" in result["metrics"]
        assert "cells_above_50pct" in result["metrics"]
        assert result["metrics"]["max_probability"] == 0.87
