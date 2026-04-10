"""
Tests for src/models/obj2_spread/cell2fire_spread.py

These tests use mocking to avoid requiring the Cell2Fire C++ binary
or real GeoTIFF files during CI.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.obj2_spread.cell2fire_spread import Cell2FireSpread
from src.models.obj2_spread.exceptions import Cell2FireError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sim_config(tmp_path):
    """Write a minimal simulation_config.json for testing."""
    config = {
        "fire_name": "test_fire",
        "ignition_points": [[10, 10]],
        "aoi_bounds": [-118.72, 33.97, -118.37, 34.20],
        "raster_paths": {
            "dem":  str(tmp_path / "elevation.tif"),
            "fuel": str(tmp_path / "fbfm40.tif"),
        },
        "weather_csv": str(tmp_path / "weather.csv"),
        "params": {
            "n_simulations": 10,
            "fire_period_length_hr": 1.0,
        },
    }
    config_path = tmp_path / "simulation_config.json"
    config_path.write_text(json.dumps(config))
    return config_path


@pytest.fixture
def mock_obj2_config():
    """Minimal obj2 config dict."""
    return {
        "cell2fire": {
            "binary_path": "Cell2Fire",
            "default_params": {
                "n_simulations": 10,
                "fire_period_length_hr": 1.0,
                "output_grid": True,
            },
            "raster_inputs": {},
            "sweep_space": {
                "n_simulations": [10, 50],
            },
            "validation": {
                "minimum_dice": 0.50,
            },
        }
    }


@pytest.fixture
def sample_feature_df():
    """Minimal feature DataFrame with h3_index and weather columns."""
    return pd.DataFrame({
        "h3_index": ["8529a183fffffff", "8529a1c3fffffff"],
        "timestamp": pd.date_range("2025-01-07 18:00", periods=2, freq="h"),
        "wind_speed_10m": [22.5, 25.0],
        "wind_direction_10m": [315, 310],
        "temperature_2m": [19.5, 21.0],
        "relative_humidity_2m": [10, 8],
    })


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

class TestLoadModel:

    def test_raises_if_json_not_found(self, tmp_path):
        model = Cell2FireSpread()
        with patch(
            "src.models.obj2_spread.cell2fire_spread.load_obj2_config",
            return_value={"cell2fire": {"binary_path": "Cell2Fire", "default_params": {}}}
        ), pytest.raises(FileNotFoundError):
            model.load_model(tmp_path / "nonexistent.json")

    def test_raises_on_wrong_extension(self, tmp_path, mock_obj2_config):
        model = Cell2FireSpread()
        bad_file = tmp_path / "config.yaml"
        bad_file.write_text("key: value")
        with patch(
            "src.models.obj2_spread.cell2fire_spread.load_obj2_config",
            return_value=mock_obj2_config,
        ), pytest.raises(Cell2FireError, match="Expected .json"):
            model.load_model(bad_file)

    def test_loads_successfully(self, sim_config, mock_obj2_config):
        model = Cell2FireSpread()
        with patch(
            "src.models.obj2_spread.cell2fire_spread.load_obj2_config",
            return_value=mock_obj2_config,
        ), patch("shutil.which", return_value="/usr/local/bin/Cell2Fire"):
            model.load_model(sim_config)
        assert model._is_loaded is True

    def test_sim_params_overridden_by_config(self, sim_config, mock_obj2_config):
        model = Cell2FireSpread()
        with patch(
            "src.models.obj2_spread.cell2fire_spread.load_obj2_config",
            return_value=mock_obj2_config,
        ), patch("shutil.which", return_value="/usr/local/bin/Cell2Fire"):
            model.load_model(sim_config)
        # Config sets n_simulations=10, should override default
        assert model._sim_params["n_simulations"] == 10


# ---------------------------------------------------------------------------
# predict (mocked binary)
# ---------------------------------------------------------------------------

class TestPredict:

    def _load_model(self, model, sim_config, mock_obj2_config, tmp_path):
        """Helper to load model with mocked config and fake rasters."""
        import rasterio
        from rasterio.transform import from_bounds

        # Create fake DEM and fuel TIFs
        for name in ("elevation.tif", "fbfm40.tif"):
            path = tmp_path / name
            transform = from_bounds(-118.72, 33.97, -118.37, 34.20, 50, 50)
            data = np.ones((50, 50), dtype=np.int16)
            with rasterio.open(
                path, "w", driver="GTiff",
                height=50, width=50, count=1,
                dtype=data.dtype, crs="EPSG:4326", transform=transform,
            ) as dst:
                dst.write(data, 1)

        # Create fake weather CSV
        weather = pd.DataFrame({
            "datetime": pd.date_range("2025-01-07 18:00", periods=3, freq="h"),
            "ws": [22.5, 25.0, 24.0],
            "wd": [315, 310, 318],
            "tmp": [19.5, 21.0, 22.0],
            "rh": [10, 8, 8],
        })
        weather.to_csv(tmp_path / "weather.csv", index=False)

        with patch(
            "src.models.obj2_spread.cell2fire_spread.load_obj2_config",
            return_value=mock_obj2_config,
        ), patch("shutil.which", return_value="/usr/local/bin/Cell2Fire"):
            model.load_model(sim_config)

    def test_raises_if_not_loaded(self, sample_feature_df):
        model = Cell2FireSpread()
        with pytest.raises(Cell2FireError, match="load_model"):
            model.predict(sample_feature_df)

    def test_returns_prediction_and_probability_columns(
        self, sim_config, mock_obj2_config, sample_feature_df, tmp_path
    ):
        model = Cell2FireSpread()
        self._load_model(model, sim_config, mock_obj2_config, tmp_path)

        fake_burn_prob = np.random.rand(50, 50)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            with patch(
                "src.models.obj2_spread.cell2fire_spread.parse_burn_probability",
                return_value=fake_burn_prob,
            ):
                result = model.predict(sample_feature_df)

        assert "prediction" in result.columns
        assert "probability" in result.columns
        assert len(result) == len(sample_feature_df)

    def test_prediction_is_binary(
        self, sim_config, mock_obj2_config, sample_feature_df, tmp_path
    ):
        model = Cell2FireSpread()
        self._load_model(model, sim_config, mock_obj2_config, tmp_path)

        fake_burn_prob = np.random.rand(50, 50)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            with patch(
                "src.models.obj2_spread.cell2fire_spread.parse_burn_probability",
                return_value=fake_burn_prob,
            ):
                result = model.predict(sample_feature_df)

        assert set(result["prediction"].unique()).issubset({0, 1})

    def test_raises_on_binary_failure(
        self, sim_config, mock_obj2_config, sample_feature_df, tmp_path
    ):
        model = Cell2FireSpread()
        self._load_model(model, sim_config, mock_obj2_config, tmp_path)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stderr="Segmentation fault"
            )
            with pytest.raises(Cell2FireError, match="exited with code 1"):
                model.predict(sample_feature_df)
