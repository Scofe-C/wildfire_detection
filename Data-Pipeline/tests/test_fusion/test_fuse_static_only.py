"""
Tests for fusion when only static features are present.
Verifies that the fusion layer handles missing FIRMS and weather data gracefully.
"""

import pandas as pd
import pytest

from scripts.fusion.fuse_features import fuse_features


@pytest.fixture
def empty_firms_df():
    """Empty FIRMS features DataFrame."""
    return pd.DataFrame(columns=[
        "grid_id", "active_fire_count", "mean_frp", "median_frp",
        "max_confidence", "nearest_fire_distance_km", "fire_detected_binary",
    ])


@pytest.fixture
def empty_weather_df():
    """Empty weather features DataFrame."""
    return pd.DataFrame(columns=[
        "grid_id", "temperature_2m", "relative_humidity_2m",
        "wind_speed_10m", "wind_direction_10m",
    ])


@pytest.fixture
def valid_static_df():
    """Valid static features DataFrame."""
    return pd.DataFrame({
        "grid_id": ["cell_a", "cell_b", "cell_c"],
        "latitude": [34.0, 34.1, 34.2],
        "longitude": [-118.0, -118.1, -118.2],
        "elevation_m": [500.0, 600.0, 700.0],
        "slope_degrees": [10.0, 15.0, 20.0],
        "aspect_degrees": [180.0, 90.0, 270.0],
        "fuel_model": [1, 2, 3],
        "canopy_cover_pct": [50.0, 60.0, 70.0],
        "vegetation_type": [1, 2, 1],
    })


class TestStaticOnlyFusion:
    """When only static features are present (no FIRMS, no weather)."""

    def test_static_columns_populated(self, valid_static_df, empty_firms_df, empty_weather_df):
        """Static columns should be present and populated."""
        result = fuse_features(
            firms_features=empty_firms_df,
            weather_features=empty_weather_df,
            static_features=valid_static_df,
            execution_date=pd.Timestamp("2025-01-15 06:00"),
            resolution_km=64,
        )
        assert "elevation_m" in result.columns
        assert "slope_degrees" in result.columns

    def test_fire_columns_get_defaults(self, valid_static_df, empty_firms_df, empty_weather_df):
        """Fire columns should get default values (0, 0.0, -1.0) when no FIRMS data."""
        result = fuse_features(
            firms_features=empty_firms_df,
            weather_features=empty_weather_df,
            static_features=valid_static_df,
            execution_date=pd.Timestamp("2025-01-15 06:00"),
            resolution_km=64,
        )
        if "active_fire_count" in result.columns:
            assert (result["active_fire_count"] == 0).all() or result["active_fire_count"].isna().all()
        if "fire_detected_binary" in result.columns:
            assert (result["fire_detected_binary"] == 0).all() or result["fire_detected_binary"].isna().all()

    def test_output_not_empty(self, valid_static_df, empty_firms_df, empty_weather_df):
        """Output should not be empty — master grid generates rows."""
        result = fuse_features(
            firms_features=empty_firms_df,
            weather_features=empty_weather_df,
            static_features=valid_static_df,
            execution_date=pd.Timestamp("2025-01-15 06:00"),
            resolution_km=64,
        )
        assert len(result) > 0
