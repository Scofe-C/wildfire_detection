"""Unit tests for RerunEngine — operator override and re-scoring logic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline.rerun_engine import _OVERRIDE_MAP, RerunEngine


# ── Unit conversion tests (no model loading needed) ───────────────────────────

def test_fahrenheit_to_celsius():
    convert = _OVERRIDE_MAP["temperature_f"]["convert"]
    assert abs(convert(32) - 0.0) < 0.01
    assert abs(convert(212) - 100.0) < 0.01
    assert abs(convert(98.6) - 37.0) < 0.1


def test_mph_to_kmh():
    convert = _OVERRIDE_MAP["wind_speed_mph"]["convert"]
    assert abs(convert(1.0) - 1.60934) < 0.001
    assert abs(convert(60.0) - 96.56) < 0.1


def test_direct_passthrough_fields():
    for field in ("relative_humidity", "soil_moisture", "fire_weather_index"):
        convert = _OVERRIDE_MAP[field]["convert"]
        assert convert(42.0) == 42.0


# ── apply_overrides (no model required) ───────────────────────────────────────

def _make_df() -> pd.DataFrame:
    return pd.DataFrame({
        "grid_id": ["A", "B", "C"],
        "temperature_2m": [20.0, 25.0, 30.0],
        "wind_speed_10m": [10.0, 15.0, 20.0],
        "relative_humidity_2m": [40.0, 50.0, 60.0],
        "soil_moisture_0_to_7cm": [0.2, 0.3, 0.4],
        "fire_weather_index": [5.0, 8.0, 12.0],
    })


class _DummyEngine(RerunEngine):
    """Subclass that skips model loading for unit tests."""
    def __init__(self):
        self._threshold = 0.239
        self._medians = {}
        self._framework = "xgboost"
        self._model = None

    def _load_model(self, model_dir):
        return None


def test_apply_overrides_correct_cell_only():
    engine = _DummyEngine()
    df = _make_df()
    result = engine.apply_overrides(df, grid_id="B", overrides={"temperature_f": 86.0})  # 30°C
    # Only row B should change
    assert abs(result.loc[result["grid_id"] == "B", "temperature_2m"].values[0] - 30.0) < 0.1
    # Other rows unchanged
    assert result.loc[result["grid_id"] == "A", "temperature_2m"].values[0] == 20.0
    assert result.loc[result["grid_id"] == "C", "temperature_2m"].values[0] == 30.0


def test_apply_overrides_unknown_grid_id():
    engine = _DummyEngine()
    df = _make_df()
    result = engine.apply_overrides(df, grid_id="Z", overrides={"temperature_f": 100.0})
    pd.testing.assert_frame_equal(result, df)


def test_apply_overrides_unknown_field_ignored():
    engine = _DummyEngine()
    df = _make_df()
    result = engine.apply_overrides(df, grid_id="A", overrides={"nonexistent_field": 99.9})
    pd.testing.assert_frame_equal(result, df)


def test_apply_overrides_multiple_fields():
    engine = _DummyEngine()
    df = _make_df()
    result = engine.apply_overrides(
        df, grid_id="A",
        overrides={
            "temperature_f": 104.0,  # 40°C
            "wind_speed_mph": 62.14,  # ~100 km/h
            "relative_humidity": 15.0,
        },
    )
    row = result.loc[result["grid_id"] == "A"].iloc[0]
    assert abs(row["temperature_2m"] - 40.0) < 0.1
    assert abs(row["wind_speed_10m"] - 100.0) < 1.0
    assert row["relative_humidity_2m"] == 15.0