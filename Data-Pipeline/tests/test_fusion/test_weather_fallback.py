"""
Tests for Bug #2 fix: silent weather window fallback.
Verifies that _aggregate_weather_to_window returns empty instead of
using all weather data when no rows match the time window.
"""

import pandas as pd
import pytest

from scripts.fusion.fuse_features import _aggregate_weather_to_window


@pytest.fixture
def weather_df():
    """Weather DataFrame with timestamps only in January 2025."""
    return pd.DataFrame({
        "grid_id": ["cell_a", "cell_a", "cell_b", "cell_b"],
        "timestamp": pd.to_datetime([
            "2025-01-15 00:00",
            "2025-01-15 06:00",
            "2025-01-15 00:00",
            "2025-01-15 06:00",
        ]),
        "temperature_2m": [10.0, 15.0, 12.0, 17.0],
        "relative_humidity_2m": [40.0, 35.0, 45.0, 38.0],
        "wind_speed_10m": [5.0, 8.0, 6.0, 9.0],
    })


class TestWeatherWindowFallback:
    """Bug #2 fix: empty window no longer falls back to all data."""

    def test_matching_window_returns_data(self, weather_df):
        """When data matches the window, aggregated result is returned."""
        # execution_date = window end; window_hours = window width
        # Window: [2025-01-15 00:00, 2025-01-15 06:00] → should match data
        result = _aggregate_weather_to_window(
            weather_df,
            execution_date=pd.Timestamp("2025-01-15 06:00"),
            window_hours=6,
        )
        assert not result.empty
        assert "grid_id" in result.columns

    def test_empty_window_returns_empty(self, weather_df):
        """When no data matches the window, return empty DataFrame."""
        # Window: [2025-06-01 00:00, 2025-06-01 06:00] → no matching data
        result = _aggregate_weather_to_window(
            weather_df,
            execution_date=pd.Timestamp("2025-06-01 06:00"),
            window_hours=6,
        )
        assert result.empty or (len(result) == 0)
        assert "grid_id" in result.columns

    def test_empty_window_does_not_return_all_data(self, weather_df):
        """Critical: empty window must NOT silently use all weather rows."""
        result = _aggregate_weather_to_window(
            weather_df,
            execution_date=pd.Timestamp("2025-06-01 06:00"),
            window_hours=6,
        )
        # If bug is still present, result would have 2 rows (per grid_id aggregation)
        assert len(result) == 0, (
            "Bug #2 regression: empty window should NOT fall back to all weather data"
        )
