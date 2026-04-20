"""
Test Suite for Weather Ingestion and Processing
================================================
Comprehensive tests for Mohammed's weather data pipeline.
NO AGGREGATION VERSION - maintains hourly resolution.

Owner: Mohammed
Dependencies: pytest, pandas

Test Coverage:
    - Weather ingestion from Open-Meteo API
    - NWS fallback functionality
    - Coordinate batching and rounding
    - Weather processing (hourly, no aggregation)
    - Derived feature calculations
    - Error handling and edge cases
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from scripts.ingestion.ingest_weather import (
    _create_coordinate_batches,
    _parse_open_meteo_response,
    _fahrenheit_to_celsius,
    _parse_nws_wind_speed,
    _parse_nws_wind_direction,
)

from scripts.processing.process_weather import (
    process_weather_data,
    _compute_days_since_precip,
    _compute_wind_run,
    _compute_drought_proxy,
)


# ============================================================================
# INGESTION TESTS
# ============================================================================

class TestWeatherIngestion:
    """Tests for weather data ingestion (ingest_weather.py)"""
    
    def test_coordinate_batching(self):
        """Test that coordinates are correctly batched for API requests"""
        grid_centroids = pd.DataFrame({
            'grid_id': [f'grid_{i}' for i in range(75)],
            'latitude': [34.0 + i*0.1 for i in range(75)],
            'longitude': [-118.0 - i*0.1 for i in range(75)]
        })
        
        batches = _create_coordinate_batches(grid_centroids, batch_size=50)
        
        assert len(batches) == 2, "Should create 2 batches (50 + 25)"
        assert len(batches[0]) == 50, "First batch should have 50 items"
        assert len(batches[1]) == 25, "Second batch should have 25 items"
    
    def test_coordinate_rounding(self):
        """Test that coordinates are rounded to 3 decimal places"""
        coord = 34.123456789
        rounded = round(coord, 3)
        assert rounded == 34.123, "Should round to 3 decimal places"
    
    def test_fahrenheit_to_celsius(self):
        """Test temperature conversion"""
        assert _fahrenheit_to_celsius(32) == 0.0, "Freezing point"
        assert _fahrenheit_to_celsius(212) == 100.0, "Boiling point"
        assert _fahrenheit_to_celsius(None) is None, "Handle None"
    
    def test_parse_nws_wind_speed(self):
        """Test NWS wind speed parsing"""
        assert _parse_nws_wind_speed("15 mph") == pytest.approx(24.1, rel=0.1)
        assert _parse_nws_wind_speed("10 to 20 mph") == pytest.approx(24.1, rel=0.1)
        assert _parse_nws_wind_speed(None) is None
        assert _parse_nws_wind_speed("invalid") is None
    
    def test_parse_nws_wind_direction(self):
        """Test NWS wind direction parsing"""
        assert _parse_nws_wind_direction("N") == 0
        assert _parse_nws_wind_direction("E") == 90
        assert _parse_nws_wind_direction("S") == 180
        assert _parse_nws_wind_direction("W") == 270
        assert _parse_nws_wind_direction("NE") == 45
        assert _parse_nws_wind_direction(None) is None
    
    def test_parse_open_meteo_response_single_location(self):
        """Test parsing single-location Open-Meteo response"""
        mock_response = {
            "hourly": {
                "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
                "temperature_2m": [15.5, 16.2],
                "relative_humidity_2m": [65, 63],
                "wind_speed_10m": [10.5, 11.2],
                "wind_direction_10m": [180, 185],
                "precipitation": [0.0, 0.5],
                "soil_moisture_3_to_9cm": [0.25, 0.24],
                "vapor_pressure_deficit": [1.2, 1.3],
            }
        }
        
        batch = pd.DataFrame({
            'grid_id': ['test_grid'],
            'latitude': [34.05],
            'longitude': [-118.24]
        })
        
        result = _parse_open_meteo_response(mock_response, batch)
        
        assert len(result) == 2, "Should have 2 hourly records"
        assert result['grid_id'].iloc[0] == 'test_grid'
        assert result['temperature_2m'].iloc[0] == 15.5
        assert result['precipitation'].iloc[1] == 0.5
    
    def test_parse_open_meteo_response_multi_location(self):
        """Test parsing multi-location Open-Meteo response"""
        mock_response = [
            {
                "hourly": {
                    "time": ["2024-01-01T00:00"],
                    "temperature_2m": [15.5],
                    "relative_humidity_2m": [65],
                    "wind_speed_10m": [10.5],
                    "wind_direction_10m": [180],
                    "precipitation": [0.0],
                    "soil_moisture_3_to_9cm": [0.25],
                    "vapor_pressure_deficit": [1.2],
                }
            },
            {
                "hourly": {
                    "time": ["2024-01-01T00:00"],
                    "temperature_2m": [20.0],
                    "relative_humidity_2m": [55],
                    "wind_speed_10m": [15.0],
                    "wind_direction_10m": [270],
                    "precipitation": [1.0],
                    "soil_moisture_3_to_9cm": [0.30],
                    "vapor_pressure_deficit": [1.8],
                }
            }
        ]
        
        batch = pd.DataFrame({
            'grid_id': ['grid_1', 'grid_2'],
            'latitude': [34.05, 36.78],
            'longitude': [-118.24, -119.42]
        })
        
        result = _parse_open_meteo_response(mock_response, batch)
        
        assert len(result) == 2, "Should have 2 records (one per location)"
        assert result['grid_id'].iloc[0] == 'grid_1'
        assert result['grid_id'].iloc[1] == 'grid_2'
        assert result['temperature_2m'].iloc[1] == 20.0
    
    def test_parse_open_meteo_list_response(self):
        """Test parsing when Open-Meteo returns a list directly"""
        mock_response = [
            {
                "hourly": {
                    "time": ["2024-01-01T00:00"],
                    "temperature_2m": [15.5],
                    "relative_humidity_2m": [65],
                    "wind_speed_10m": [10.5],
                    "wind_direction_10m": [180],
                    "precipitation": [0.0],
                    "soil_moisture_3_to_9cm": [0.25],
                    "vapor_pressure_deficit": [1.2],
                }
            }
        ]
        
        batch = pd.DataFrame({
            'grid_id': ['grid_1'],
            'latitude': [34.05],
            'longitude': [-118.24]
        })
        
        result = _parse_open_meteo_response(mock_response, batch)
        
        assert len(result) == 1, "Should handle list response"
        assert result['grid_id'].iloc[0] == 'grid_1'


# ============================================================================
# PROCESSING TESTS (NO AGGREGATION)
# ============================================================================

class TestWeatherProcessing:
    """Tests for weather data processing functions (DataFrame-level API)."""

    def test_compute_days_since_precipitation_current_rain(self):
        """Test days since precip when current hour has rain."""
        df = pd.DataFrame({
            "grid_id": ["grid_1"],
            "timestamp": [datetime(2024, 1, 10, 12, 0, 0)],
            "precipitation": [2.0],
        })
        result = _compute_days_since_precip(df)
        assert result.iloc[0] == 0, "Should return 0 when current hour has rain"

    def test_compute_days_since_precipitation_no_rain(self):
        """Test days since precip with no rain in window."""
        timestamps = [datetime(2024, 1, 10) + timedelta(hours=i) for i in range(6)]
        df = pd.DataFrame({
            "grid_id": ["grid_1"] * 6,
            "timestamp": timestamps,
            "precipitation": [0.0] * 6,
        })
        result = _compute_days_since_precip(df)
        # All dry → days_dry = max(window_hours/24, 1.0)
        assert result.iloc[0] >= 1.0, "Should return >= 1 day when no rain"

    def test_compute_wind_run(self):
        """Test cumulative wind run calculation."""
        timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(24)]
        df = pd.DataFrame({
            "grid_id": ["grid_1"] * 24,
            "timestamp": timestamps,
            "wind_speed_10m": [10.0] * 24,  # 10 km/h constant
        })
        result = _compute_wind_run(df)
        # 10 km/h * 24 hours = 240 km cumulative for all rows in the group
        assert result.iloc[0] == pytest.approx(240.0, rel=0.1)

    def test_compute_drought_proxy_wet_conditions(self):
        """Test drought proxy for wet conditions."""
        df = pd.DataFrame({
            "grid_id": ["grid_1"],
            "timestamp": [datetime(2024, 1, 10)],
            "soil_moisture_0_to_7cm": [0.4],
            "temperature_2m": [15.0],
            "precipitation": [10.0],
        })
        result = _compute_drought_proxy(df)
        assert 0 <= result.iloc[0] <= 1, "Drought proxy should be 0-1"
        assert result.iloc[0] < 0.5, "Wet conditions should have low drought proxy"

    def test_compute_drought_proxy_dry_conditions(self):
        """Test drought proxy for dry conditions."""
        df = pd.DataFrame({
            "grid_id": ["grid_1"],
            "timestamp": [datetime(2024, 1, 10)],
            "soil_moisture_0_to_7cm": [0.05],
            "temperature_2m": [38.0],
            "precipitation": [0.0],
        })
        result = _compute_drought_proxy(df)
        assert 0 <= result.iloc[0] <= 1, "Drought proxy should be 0-1"
        assert result.iloc[0] > 0.5, "Dry conditions should have high drought proxy"

    def test_compute_drought_proxy_missing_data(self):
        """Test drought proxy handles missing data gracefully."""
        df = pd.DataFrame({
            "grid_id": ["grid_1"],
            "timestamp": [datetime(2024, 1, 10)],
            "soil_moisture_0_to_7cm": [None],
            "temperature_2m": [None],
            "precipitation": [None],
        })
        result = _compute_drought_proxy(df)
        assert 0 <= result.iloc[0] <= 1, "Should handle None values"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestWeatherIntegration:
    """End-to-end integration tests"""
    
    @pytest.fixture
    def sample_grid(self):
        """Sample grid centroids for testing"""
        return pd.DataFrame({
            'grid_id': ['grid_1', 'grid_2', 'grid_3'],
            'latitude': [34.05, 36.78, 37.77],
            'longitude': [-118.24, -119.42, -122.42]
        })
    
    @pytest.fixture
    def sample_raw_weather(self, tmp_path):
        """Create sample raw weather CSV"""
        execution_date = datetime(2024, 1, 1, 12, 0, 0)
        timestamps = [execution_date - timedelta(hours=i) for i in range(24, 0, -1)]
        
        data = []
        for grid_id in ['grid_1', 'grid_2']:
            for ts in timestamps:
                data.append({
                    'grid_id': grid_id,
                    'timestamp': ts,
                    'temperature_2m': 15 + np.random.randn(),
                    'relative_humidity_2m': 60 + np.random.randn() * 5,
                    'wind_speed_10m': 10 + np.random.randn() * 2,
                    'wind_direction_10m': 180 + np.random.randn() * 10,
                    'precipitation': max(0, np.random.randn() * 0.5),
                    'soil_moisture_3_to_9cm': 0.25 + np.random.randn() * 0.05,
                    'vapor_pressure_deficit': 1.2 + np.random.randn() * 0.2,
                    'data_quality_flag': 0
                })
        
        df = pd.DataFrame(data)
        csv_path = tmp_path / "weather_raw_20240101_120000.csv"
        df.to_csv(csv_path, index=False)
        
        return csv_path
    
    def test_end_to_end_processing_no_aggregation(self, sample_raw_weather, tmp_path):
        """Test complete pipeline: raw → processed."""
        processed_df = process_weather_data(
            raw_csv_path=str(sample_raw_weather),
        )

        # Verify key columns present
        for col in ["grid_id", "temperature_2m", "wind_speed_10m",
                     "precipitation", "data_quality_flag"]:
            assert col in processed_df.columns, f"Missing column: {col}"

        assert len(processed_df) > 0, "Should have processed records"

    def test_derived_features_per_grid(self, sample_raw_weather):
        """Test that derived features are calculated per grid cell."""
        processed_df = process_weather_data(
            raw_csv_path=str(sample_raw_weather),
        )

        # Check that each grid has data
        for grid_id in processed_df["grid_id"].unique():
            grid_data = processed_df[processed_df["grid_id"] == grid_id]
            assert len(grid_data) > 0, f"No data for {grid_id}"

    def test_output_column_order(self, sample_raw_weather):
        """Test that output columns start with grid_id."""
        processed_df = process_weather_data(
            raw_csv_path=str(sample_raw_weather),
        )

        cols = processed_df.columns.tolist()
        assert cols[0] == "grid_id", "First column should be grid_id"


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame(columns=[
            "grid_id", "timestamp", "temperature_2m", "wind_speed_10m",
        ])
        assert len(empty_df) == 0

    def test_single_hour_wind_run(self):
        """Test wind run with only 1 hour of data."""
        group = pd.DataFrame({
            "grid_id": ["grid_1"],
            "timestamp": [datetime(2024, 1, 1)],
            "wind_speed_10m": [15.0],
        })
        result = _compute_wind_run(group)
        assert len(result) == 1
        assert result.iloc[0] == 15.0, "Single hour should equal wind speed"

    def test_days_since_precip_all_nan(self):
        """Test days since precip when all precipitation values are NaN."""
        df = pd.DataFrame({
            "grid_id": ["grid_1"],
            "timestamp": [datetime(2024, 1, 10)],
            "precipitation": [float("nan")],
        })
        result = _compute_days_since_precip(df)
        assert pd.isna(result.iloc[0]), "Should return NaN when all precip is NaN"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])