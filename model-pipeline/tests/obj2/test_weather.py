"""
Tests for src/models/obj2_spread/weather.py
"""
from pathlib import Path

import pandas as pd
import pytest

from src.models.obj2_spread.weather import (
    format_weather_csv,
    validate_weather_df,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_weather_df():
    """Minimal valid weather DataFrame using pipeline column names."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-07 18:00", periods=6, freq="h"),
        "wind_speed_10m":       [18.0, 22.5, 25.0, 24.0, 21.0, 19.0],
        "wind_direction_10m":   [320,  315,  310,  318,  322,  325],
        "temperature_2m":       [18.0, 19.5, 21.0, 22.0, 21.5, 20.5],
        "relative_humidity_2m": [12,   10,    8,    8,    9,   10],
    })


@pytest.fixture
def short_named_df():
    """Weather DataFrame already using short column names (ws/wd/tmp/rh)."""
    return pd.DataFrame({
        "datetime": pd.date_range("2025-01-07 18:00", periods=3, freq="h"),
        "ws":  [18.0, 22.5, 25.0],
        "wd":  [320,  315,  310],
        "tmp": [18.0, 19.5, 21.0],
        "rh":  [12,   10,    8],
    })


# ---------------------------------------------------------------------------
# format_weather_csv
# ---------------------------------------------------------------------------

class TestFormatWeatherCsv:

    def test_writes_correct_columns(self, valid_weather_df, tmp_path):
        out = tmp_path / "weather.csv"
        format_weather_csv(valid_weather_df, out)

        result = pd.read_csv(out)
        assert set(result.columns) == {"datetime", "ws", "wd", "tmp", "rh"}

    def test_row_count_preserved(self, valid_weather_df, tmp_path):
        out = tmp_path / "weather.csv"
        format_weather_csv(valid_weather_df, out)
        result = pd.read_csv(out)
        assert len(result) == len(valid_weather_df)

    def test_sorted_by_datetime(self, tmp_path):
        df = pd.DataFrame({
            "timestamp": ["2025-01-07 20:00", "2025-01-07 18:00", "2025-01-07 19:00"],
            "wind_speed_10m": [20, 18, 19],
            "wind_direction_10m": [310, 320, 315],
            "temperature_2m": [21, 18, 19],
            "relative_humidity_2m": [8, 12, 10],
        })
        out = tmp_path / "weather.csv"
        format_weather_csv(df, out)
        result = pd.read_csv(out, parse_dates=["datetime"])
        assert result["datetime"].is_monotonic_increasing

    def test_accepts_short_column_names(self, short_named_df, tmp_path):
        out = tmp_path / "weather.csv"
        format_weather_csv(short_named_df, out)
        result = pd.read_csv(out)
        assert set(result.columns) == {"datetime", "ws", "wd", "tmp", "rh"}

    def test_raises_on_missing_timestamp(self, tmp_path):
        df = pd.DataFrame({
            "wind_speed_10m": [18.0],
            "wind_direction_10m": [320],
            "temperature_2m": [18.0],
            "relative_humidity_2m": [12],
        })
        with pytest.raises(ValueError, match="No timestamp column"):
            format_weather_csv(df, tmp_path / "weather.csv")

    def test_raises_on_missing_weather_columns(self, tmp_path):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-07", periods=1, freq="h"),
            "wind_speed_10m": [18.0],
            # missing temperature and RH
        })
        with pytest.raises(ValueError, match="Missing weather columns"):
            format_weather_csv(df, tmp_path / "weather.csv")

    def test_returns_path(self, valid_weather_df, tmp_path):
        out = tmp_path / "weather.csv"
        result = format_weather_csv(valid_weather_df, out)
        assert isinstance(result, Path)
        assert result == out


# ---------------------------------------------------------------------------
# validate_weather_df
# ---------------------------------------------------------------------------

class TestValidateWeatherDf:

    def test_valid_df_returns_no_warnings(self, valid_weather_df):
        warnings = validate_weather_df(valid_weather_df)
        assert warnings == []

    def test_warns_on_missing_timestamp(self):
        df = pd.DataFrame({
            "wind_speed_10m": [18.0],
            "wind_direction_10m": [320],
            "temperature_2m": [18.0],
            "relative_humidity_2m": [12],
        })
        warnings = validate_weather_df(df)
        assert any("timestamp" in w for w in warnings)

    def test_warns_on_null_values(self, valid_weather_df):
        df = valid_weather_df.copy()
        df.loc[0, "wind_speed_10m"] = None
        warnings = validate_weather_df(df)
        assert any("null" in w for w in warnings)

    def test_warns_on_high_wind_speed(self, valid_weather_df):
        df = valid_weather_df.copy()
        df["wind_speed_10m"] = 60.0  # unrealistically high m/s
        warnings = validate_weather_df(df)
        assert any("high" in w.lower() for w in warnings)
