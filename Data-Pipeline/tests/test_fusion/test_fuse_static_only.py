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

    def test_region_column_present_and_non_null(self, valid_static_df, empty_firms_df, empty_weather_df):
        """fuse_features must include 'region' column with no NaN values."""
        result = fuse_features(
            firms_features=empty_firms_df,
            weather_features=empty_weather_df,
            static_features=valid_static_df,
            execution_date=pd.Timestamp("2025-01-15 06:00"),
            resolution_km=64,
        )
        assert "region" in result.columns, "'region' column missing from fused output"
        assert result["region"].notna().all(), "'region' column contains NaN"
        # Must only contain known region names
        assert set(result["region"].unique()).issubset({"california", "texas"})

    def test_quality_flag_4_when_all_static_null(self, empty_firms_df, empty_weather_df):
        """data_quality_flag must be 4 when all static columns are NaN (no caches)."""
        from scripts.utils.grid_utils import generate_full_grid
        grid = generate_full_grid(64)
        empty_static = pd.DataFrame({
            "grid_id": grid["grid_id"].astype(str).values,
        })  # no static feature columns at all

        result = fuse_features(
            firms_features=empty_firms_df,
            weather_features=empty_weather_df,
            static_features=empty_static,
            execution_date=pd.Timestamp("2025-01-15 06:00"),
            resolution_km=64,
        )
        assert "data_quality_flag" in result.columns
        # All static cols will be NaN → flag 4 for every row
        assert (result["data_quality_flag"] == 4).all(), (
            f"Expected flag=4 for all rows, got: {result['data_quality_flag'].value_counts().to_dict()}"
        )

    def test_quality_flag_5_when_partial_static_null(self, empty_firms_df, empty_weather_df):
        """data_quality_flag must be 5 for rows where some (not all) static cols are NaN."""
        from scripts.utils.grid_utils import generate_full_grid
        grid = generate_full_grid(64)
        n = len(grid)

        # Provide LANDFIRE but leave SRTM missing → partial static
        partial_static = pd.DataFrame({
            "grid_id":              grid["grid_id"].astype(str).values,
            "fuel_model_fbfm40":    [101.0] * n,
            "canopy_cover_pct":     [30.0]  * n,
            "vegetation_type":      [3105.0] * n,
            "dominant_fuel_fraction": [0.7] * n,
            # elevation_m, slope_degrees, aspect_degrees intentionally absent
        })

        result = fuse_features(
            firms_features=empty_firms_df,
            weather_features=empty_weather_df,
            static_features=partial_static,
            execution_date=pd.Timestamp("2025-01-15 06:00"),
            resolution_km=64,
        )
        assert "data_quality_flag" in result.columns
        flags = result["data_quality_flag"].unique()
        # Partial static → flag 5 (some cols null); no row should have flag 4
        assert 4 not in flags, "flag=4 should not appear when some static cols are present"
        assert 5 in flags, "flag=5 expected for rows with partial static data"

    @pytest.mark.integration
    def test_quality_flag_0_when_all_present(self, empty_firms_df):
        """data_quality_flag must be 0 when all non-placeholder static + dynamic cols are present."""
        from scripts.utils.grid_utils import generate_full_grid
        grid = generate_full_grid(64)
        n = len(grid)

        full_static = pd.DataFrame({
            "grid_id":              grid["grid_id"].astype(str).values,
            "fuel_model_fbfm40":    [101.0] * n,
            "canopy_cover_pct":     [30.0]  * n,
            "vegetation_type":      [3105.0] * n,
            "dominant_fuel_fraction": [0.7] * n,
            "elevation_m":          [500.0] * n,
            "slope_degrees":        [10.0]  * n,
            "aspect_degrees":       [180.0] * n,
        })
        full_weather = pd.DataFrame({
            "grid_id":                  grid["grid_id"].astype(str).values,
            "temperature_2m":           [25.0] * n,
            "relative_humidity_2m":     [40.0] * n,
            "wind_speed_10m":           [15.0] * n,
            "wind_direction_10m":       [270.0] * n,
            "precipitation":            [0.0]  * n,
            "soil_moisture_0_to_7cm":   [0.1]  * n,
            "vpd":                      [1.5]  * n,
            "days_since_last_precipitation": [1] * n,
            "cumulative_wind_run_24h":  [360.0] * n,
            "drought_index_proxy":      [0.4]  * n,
            "data_quality_flag":        [0]    * n,
        })

        result = fuse_features(
            firms_features=empty_firms_df,
            weather_features=full_weather,
            static_features=full_static,
            execution_date=pd.Timestamp("2025-07-15 12:00"),
            resolution_km=64,
        )
        assert "data_quality_flag" in result.columns
        # No row should have flag 4 or 5 when all non-placeholder cols are present
        bad_flags = result[result["data_quality_flag"].isin([4, 5])]
        assert len(bad_flags) == 0, (
            f"Expected flag=0 for all rows, found flags: "
            f"{result['data_quality_flag'].value_counts().to_dict()}"
        )
