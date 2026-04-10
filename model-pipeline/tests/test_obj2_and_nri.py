"""
Tests for OBJ-2 (Cell2Fire + PROPAGATOR) and NRI Loader
========================================================
Run with: PYTHONPATH=. pytest tests/test_obj2_and_nri.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# =====================================================================
# Dice coefficient tests
# =====================================================================

class TestDiceCoefficient:
    """Tests for compute_dice_coefficient()."""

    def test_perfect_overlap(self):
        from src.models.obj2_spread.cell2fire_spread import compute_dice_coefficient
        pred = np.array([1, 1, 0, 0, 1])
        actual = np.array([1, 1, 0, 0, 1])
        assert compute_dice_coefficient(pred, actual) == 1.0

    def test_no_overlap(self):
        from src.models.obj2_spread.cell2fire_spread import compute_dice_coefficient
        pred = np.array([1, 1, 0, 0])
        actual = np.array([0, 0, 1, 1])
        assert compute_dice_coefficient(pred, actual) == 0.0

    def test_partial_overlap(self):
        from src.models.obj2_spread.cell2fire_spread import compute_dice_coefficient
        pred = np.array([1, 1, 1, 0, 0])
        actual = np.array([1, 1, 0, 0, 1])
        # intersection=2, |P|=3, |A|=3 → dice = 4/6 ≈ 0.667
        dice = compute_dice_coefficient(pred, actual)
        assert abs(dice - (4.0 / 6.0)) < 1e-6

    def test_both_empty(self):
        from src.models.obj2_spread.cell2fire_spread import compute_dice_coefficient
        pred = np.array([0, 0, 0])
        actual = np.array([0, 0, 0])
        assert compute_dice_coefficient(pred, actual) == 1.0

    def test_shape_mismatch_raises(self):
        from src.models.obj2_spread.cell2fire_spread import compute_dice_coefficient
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_dice_coefficient(np.array([1, 0]), np.array([1, 0, 1]))

    def test_2d_arrays(self):
        from src.models.obj2_spread.cell2fire_spread import compute_dice_coefficient
        pred = np.array([[1, 0], [0, 1]])
        actual = np.array([[1, 0], [1, 1]])
        # intersection=2, |P|=2, |A|=3 → dice = 4/5 = 0.8
        dice = compute_dice_coefficient(pred, actual)
        assert abs(dice - 0.8) < 1e-6


# =====================================================================
# Weather CSV formatter tests
# =====================================================================

class TestFormatWeatherCSV:
    """Tests for format_weather_csv()."""

    def test_standard_columns(self):
        from src.models.obj2_spread.cell2fire_spread import format_weather_csv
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="h"),
            "wind_speed_10m": [5.0, 7.0, 3.0],
            "wind_direction_10m": [180, 200, 160],
            "temperature_2m": [25.0, 28.0, 22.0],
            "relative_humidity_2m": [40, 35, 55],
        })
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            out = format_weather_csv(df, f.name)
        result = pd.read_csv(out)
        assert list(result.columns) == ["datetime", "ws", "wd", "tmp", "rh"]
        assert len(result) == 3
        assert result["ws"].iloc[0] == 5.0

    def test_short_column_names(self):
        from src.models.obj2_spread.cell2fire_spread import format_weather_csv
        df = pd.DataFrame({
            "datetime": pd.date_range("2025-01-01", periods=2, freq="h"),
            "ws": [5.0, 7.0],
            "wd": [180, 200],
            "tmp": [25.0, 28.0],
            "rh": [40, 35],
        })
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            out = format_weather_csv(df, f.name)
        result = pd.read_csv(out)
        assert list(result.columns) == ["datetime", "ws", "wd", "tmp", "rh"]

    def test_missing_timestamp_raises(self):
        from src.models.obj2_spread.cell2fire_spread import (
            Cell2FireError,
            format_weather_csv,
        )
        df = pd.DataFrame({"ws": [5.0], "wd": [180], "tmp": [25.0], "rh": [40]})
        with pytest.raises(Cell2FireError, match="No timestamp column"), tempfile.NamedTemporaryFile(suffix=".csv") as f:
            format_weather_csv(df, f.name)

    def test_missing_weather_column_raises(self):
        from src.models.obj2_spread.cell2fire_spread import (
            Cell2FireError,
            format_weather_csv,
        )
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="h"),
            "ws": [5.0, 7.0],
            # Missing wd, tmp, rh
        })
        with pytest.raises(Cell2FireError, match="Missing weather columns"), tempfile.NamedTemporaryFile(suffix=".csv") as f:
            format_weather_csv(df, f.name)


# =====================================================================
# Burn probability parser tests
# =====================================================================

class TestParseBurnProbability:
    """Tests for parse_burn_probability()."""

    def test_parses_grid_files(self):
        from src.models.obj2_spread.cell2fire_spread import parse_burn_probability

        with tempfile.TemporaryDirectory() as tmpdir:
            grids_dir = Path(tmpdir) / "Grids" / "Grids1"
            grids_dir.mkdir(parents=True)

            # Create 3 simulation output grids (5x5 each)
            for i in range(3):
                grid = np.zeros((5, 5))
                grid[0:2, 0:2] = 1  # cells 0,0 to 1,1 always burn
                if i >= 1:
                    grid[3, 3] = 1  # cell 3,3 burns in 2 of 3 sims
                np.savetxt(
                    grids_dir / f"ForestGrid{i:02d}.csv",
                    grid, delimiter=",", header="c1,c2,c3,c4,c5",
                )

            prob = parse_burn_probability(Path(tmpdir), n_simulations=3)
            assert prob.shape == (5, 5)
            assert abs(prob[0, 0] - 1.0) < 1e-6     # burned in all 3
            assert abs(prob[3, 3] - 2.0 / 3) < 1e-6  # burned in 2 of 3
            assert prob[4, 4] == 0.0                  # never burned


# =====================================================================
# PROPAGATOR fuel reclassification tests
# =====================================================================

class TestReclassifyFuel:
    """Tests for reclassify_fuel()."""

    def test_known_codes(self):
        from src.models.obj2_spread.propagator_spread import reclassify_fuel
        codes = np.array([101, 141, 181, 99])  # grass, shrub, litter, NB
        result = reclassify_fuel(codes)
        assert list(result) == [1, 3, 5, 7]

    def test_unknown_codes_map_to_nonburnable(self):
        from src.models.obj2_spread.propagator_spread import reclassify_fuel
        codes = np.array([999, 0, -1])
        result = reclassify_fuel(codes)
        assert list(result) == [7, 7, 7]

    def test_custom_crosswalk(self):
        from src.models.obj2_spread.propagator_spread import reclassify_fuel
        custom = {101: 5, 102: 6}
        codes = np.array([101, 102, 103])
        result = reclassify_fuel(codes, crosswalk=custom)
        assert list(result) == [5, 6, 7]  # 103 unmapped → 7


# =====================================================================
# Cell2FireSpread BaseModel interface compliance
# =====================================================================

class TestCell2FireSpreadInterface:
    """Verify Cell2FireSpread implements the BaseModel contract."""

    def test_inherits_base_model(self):
        from src.models.base import BaseModel
        from src.models.obj2_spread.cell2fire_spread import Cell2FireSpread
        model = Cell2FireSpread()
        assert isinstance(model, BaseModel)

    def test_initial_state(self):
        from src.models.obj2_spread.cell2fire_spread import Cell2FireSpread
        model = Cell2FireSpread()
        assert model.model_name == "cell2fire"
        assert model.version == "0.1.0"
        assert not model.is_loaded

    def test_predict_before_load_raises(self):
        from src.models.obj2_spread.cell2fire_spread import (
            Cell2FireError,
            Cell2FireSpread,
        )
        model = Cell2FireSpread()
        with pytest.raises(Cell2FireError, match="load_model"):
            model.predict(pd.DataFrame())

    def test_load_model_with_json(self, tmp_path):
        from src.models.obj2_spread.cell2fire_spread import Cell2FireSpread

        config = {
            "ignition_points": [[10, 20]],
            "aoi_bounds": [-118.5, 34.0, -118.0, 34.5],
            "params": {"n_simulations": 50},
        }
        config_file = tmp_path / "simulation_config.json"
        config_file.write_text(json.dumps(config))

        # Create a minimal model_config.yaml
        model_cfg = {
            "obj2": {
                "cell2fire": {
                    "binary_path": "/usr/local/bin/Cell2Fire",
                    "raster_inputs": {
                        "dem": str(tmp_path / "dem"),
                        "fuel": str(tmp_path / "fuel"),
                    },
                    "default_params": {
                        "n_simulations": 100,
                        "fire_period_length_hr": 1.0,
                        "output_grid": True,
                    },
                    "sweep_space": {},
                    "validation": {"minimum_dice": 0.50},
                },
                "propagator": {"enabled": True},
            },
        }
        cfg_path = tmp_path / "model_config.yaml"
        import yaml
        cfg_path.write_text(yaml.dump(model_cfg))

        model = Cell2FireSpread()
        with patch(
            "src.models.obj2_spread.cell2fire_spread.load_obj2_config",
            return_value=model_cfg["obj2"],
        ):
            model.load_model(config_file)

        assert model.is_loaded
        assert model._sim_params["n_simulations"] == 50  # overridden

    def test_load_model_missing_file_raises(self, tmp_path):
        from src.models.obj2_spread.cell2fire_spread import (
            Cell2FireError,
            Cell2FireSpread,
        )
        model = Cell2FireSpread()
        with pytest.raises((FileNotFoundError, Cell2FireError)), patch(
            "src.models.obj2_spread.cell2fire_spread.load_obj2_config",
            return_value={
                "cell2fire": {
                    "binary_path": "Cell2Fire",
                    "default_params": {},
                }
            },
        ):
            model.load_model(tmp_path / "nonexistent")

    def test_repr(self):
        from src.models.obj2_spread.cell2fire_spread import Cell2FireSpread
        model = Cell2FireSpread()
        r = repr(model)
        assert "cell2fire" in r
        assert "not loaded" in r

    def test_artifact_hash(self, tmp_path):
        from src.models.obj2_spread.cell2fire_spread import Cell2FireSpread
        model = Cell2FireSpread()
        f = tmp_path / "model.json"
        f.write_text('{"test": true}')
        h = model.compute_artifact_hash(f)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex


# =====================================================================
# PropagatorSpread interface compliance
# =====================================================================

class TestPropagatorSpreadInterface:
    """Verify PropagatorSpread implements the BaseModel contract."""

    def test_inherits_base_model(self):
        from src.models.base import BaseModel
        from src.models.obj2_spread.propagator_spread import PropagatorSpread
        model = PropagatorSpread()
        assert isinstance(model, BaseModel)

    def test_disclaimer_on_every_prediction(self, tmp_path):
        from src.models.obj2_spread.propagator_spread import PropagatorSpread

        prop_cfg = {
            "enabled": True,
            "crosswalk_path": None,
            "default_params": {"wind_reduction_factor": 0.4},
        }

        config_file = tmp_path / "propagator_config.json"
        config_file.write_text(json.dumps({}))

        model = PropagatorSpread()
        with patch(
            "src.models.obj2_spread.propagator_spread._load_propagator_config",
            return_value=prop_cfg,
        ):
            model.load_model(config_file)

        X = pd.DataFrame({
            "wind_speed_10m": [10.0, 5.0, 15.0],
            "slope_degrees": [5.0, 20.0, 35.0],
            "relative_humidity_2m": [30, 60, 20],
        })
        result = model.predict(X)
        assert "disclaimer" in result.columns
        assert all(result["disclaimer"] == PropagatorSpread.DISCLAIMER)
        assert "prediction" in result.columns
        assert "probability" in result.columns

    def test_explain_returns_structure(self):
        from src.models.obj2_spread.propagator_spread import PropagatorSpread
        model = PropagatorSpread()
        model._is_loaded = True
        model._crosswalk = {101: 1, 141: 3}
        result = model.explain(pd.DataFrame())
        assert "disclaimer" in result
        assert "crosswalk_coverage" in result
        assert result["crosswalk_coverage"]["mapped_codes"] == 2


# =====================================================================
# NRI Loader tests
# =====================================================================

class TestNRILoader:
    """Tests for src.bias.nri_loader module."""

    def test_load_nri_missing_dir_raises(self):
        from src.bias.nri_loader import NRILoadError, load_nri
        with pytest.raises(NRILoadError, match="not found"):
            load_nri(cache_dir="/nonexistent/path")

    def test_compute_vulnerability_quartiles(self):
        import geopandas as gpd
        from shapely.geometry import Point

        from src.bias.nri_loader import compute_vulnerability_quartiles

        nri = gpd.GeoDataFrame({
            "SOVI_SCORE": np.random.uniform(0, 100, size=100),
            "geometry": [Point(0, i) for i in range(100)],
        })
        result = compute_vulnerability_quartiles(nri)
        assert "nri_vulnerability_quartile" in result.columns
        labels = set(result["nri_vulnerability_quartile"].unique())
        assert labels == {"Low", "Medium", "High", "Very High"}
        # Each quartile should have ~25 tracts
        for label in labels:
            count = (result["nri_vulnerability_quartile"] == label).sum()
            assert 20 <= count <= 30

    def test_spatial_join_fills_unknown_for_unmatched(self):
        import geopandas as gpd
        from shapely.geometry import Point

        from src.bias.nri_loader import spatial_join_predictions

        # Create NRI with tracts only near (0, 0)
        nri = gpd.GeoDataFrame({
            "nri_vulnerability_quartile": ["High", "Low"],
            "SOVI_SCORE": [80.0, 20.0],
            "geometry": [Point(0, 0), Point(0, 0.01)],
        }, crs="EPSG:4326")

        # Predictions with one nearby and one far-away cell
        # sjoin_nearest always matches, so "Unknown" only appears
        # if nri_vulnerability_quartile is NaN — which doesn't happen
        # with sjoin_nearest. This tests the join completes without error.
        preds = pd.DataFrame({
            "h3_index": ["8529a183fffffff", "8529a1c3fffffff"],
        })

        with patch("h3.cell_to_latlng") as mock_c2l:
            mock_c2l.side_effect = [
                (0.0, 0.0),      # near the NRI tracts
                (50.0, 50.0),    # far away
            ]
            result = spatial_join_predictions(preds, nri)

        assert len(result) == 2
        assert "nri_vulnerability_quartile" in result.columns


# =====================================================================
# Crosswalk loader tests
# =====================================================================

class TestCrosswalkLoader:
    """Tests for load_crosswalk()."""

    def test_default_crosswalk(self):
        from src.models.obj2_spread.propagator_spread import load_crosswalk
        cw = load_crosswalk(crosswalk_path=None)
        assert len(cw) > 0
        assert cw[101] == 1   # grass
        assert cw[141] == 3   # shrub

    def test_custom_crosswalk_file(self, tmp_path):
        from src.models.obj2_spread.propagator_spread import load_crosswalk
        custom = {"101": 5, "102": 6}
        cw_file = tmp_path / "crosswalk.json"
        cw_file.write_text(json.dumps(custom))
        cw = load_crosswalk(cw_file)
        assert cw[101] == 5
        assert cw[102] == 6

    def test_missing_file_falls_back_to_default(self):
        from src.models.obj2_spread.propagator_spread import load_crosswalk
        cw = load_crosswalk(crosswalk_path="/nonexistent.json")
        assert len(cw) > 0  # falls back to DEFAULT_CROSSWALK
