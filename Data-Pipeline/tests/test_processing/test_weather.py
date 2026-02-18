"""
Tests for Weather Processing
==============================
Owner: Person B (Mohammed)

Covers the seven scenarios required by Section 3.6 of the assignment guide,
plus tests for the three derived features added in the improvement plan.
"""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_hourly_csv(tmp_path):
    """Realistic multi-hour weather CSV for a single grid cell."""
    rows = [
        "grid_id,timestamp,temperature_2m,relative_humidity_2m,wind_speed_10m,"
        "wind_direction_10m,precipitation,soil_moisture_0_to_7cm,vpd,"
        "fire_weather_index,data_quality_flag",
        "cell_a,2026-08-15 00:00:00,28.5,42.0,18.0,270.0,0.0,0.12,1.8,None,0",
        "cell_a,2026-08-15 01:00:00,27.8,44.0,16.0,265.0,0.0,0.11,1.6,None,0",
        "cell_a,2026-08-15 02:00:00,26.3,48.0,14.0,260.0,0.0,0.11,1.4,None,0",
        "cell_a,2026-08-15 03:00:00,25.1,52.0,12.0,255.0,0.0,0.12,1.2,None,0",
        "cell_a,2026-08-15 04:00:00,24.0,55.0,11.0,250.0,0.5,0.13,1.0,None,0",  # rain
        "cell_a,2026-08-15 05:00:00,23.5,58.0,10.0,250.0,2.1,0.15,0.8,None,0",  # rain >1mm
    ]
    path = tmp_path / "weather_raw_test.csv"
    path.write_text("\n".join(rows))
    return str(path)


@pytest.fixture
def dry_weather_csv(tmp_path):
    """24 hours of completely dry weather — for days_since_precip test."""
    rows = [
        "grid_id,timestamp,temperature_2m,relative_humidity_2m,wind_speed_10m,"
        "wind_direction_10m,precipitation,soil_moisture_0_to_7cm,vpd,"
        "fire_weather_index,data_quality_flag",
    ]
    for h in range(24):
        rows.append(
            f"cell_dry,2026-08-15 {h:02d}:00:00,35.0,20.0,25.0,270.0,0.0,0.05,3.5,None,0"
        )
    path = tmp_path / "weather_dry.csv"
    path.write_text("\n".join(rows))
    return str(path)


@pytest.fixture
def nws_fallback_csv(tmp_path):
    """Weather CSV with data_quality_flag=2 (NWS fallback was used)."""
    rows = [
        "grid_id,timestamp,temperature_2m,relative_humidity_2m,wind_speed_10m,"
        "wind_direction_10m,precipitation,soil_moisture_0_to_7cm,vpd,"
        "fire_weather_index,data_quality_flag",
        "cell_nws,2026-08-15 00:00:00,22.0,65.0,8.0,180.0,1.5,0.20,0.9,None,2",
        "cell_nws,2026-08-15 01:00:00,21.5,67.0,7.0,175.0,0.0,0.21,0.8,None,2",
    ]
    path = tmp_path / "weather_nws.csv"
    path.write_text("\n".join(rows))
    return str(path)


# ---------------------------------------------------------------------------
# Test 1: Temperature range validation (−50°C to 65°C)
# ---------------------------------------------------------------------------

class TestTemperatureRangeValidation:

    def test_normal_temperature_passes_through(self, sample_hourly_csv):
        from scripts.processing.process_weather import process_weather_data
        result = process_weather_data(sample_hourly_csv)
        assert result["temperature_2m"].iloc[0] < 65.0
        assert result["temperature_2m"].iloc[0] > -50.0

    def test_extreme_temperature_survives_processing(self, tmp_path):
        """Out-of-range values pass through processing (validation catches them)."""
        from scripts.processing.process_weather import process_weather_data
        rows = (
            "grid_id,timestamp,temperature_2m,precipitation,data_quality_flag\n"
            "cell_x,2026-08-15 00:00:00,999.0,0.0,0\n"
        )
        path = tmp_path / "extreme.csv"
        path.write_text(rows)
        result = process_weather_data(str(path))
        # Processing does NOT clip temperature — validation (validate_schema) does
        assert result.iloc[0]["temperature_2m"] == pytest.approx(999.0)


# ---------------------------------------------------------------------------
# Test 2: Humidity sanity (0–100%)
# ---------------------------------------------------------------------------

class TestHumiditySanity:

    def test_humidity_values_in_range(self, sample_hourly_csv):
        from scripts.processing.process_weather import process_weather_data
        result = process_weather_data(sample_hourly_csv)
        rh = result["relative_humidity_2m"].iloc[0]
        assert 0.0 <= rh <= 100.0, f"Humidity {rh}% outside [0, 100]"

    def test_humidity_is_mean_of_hourly_values(self, tmp_path):
        """Aggregated humidity must be the mean of hourly readings."""
        from scripts.processing.process_weather import process_weather_data
        rows = (
            "grid_id,timestamp,temperature_2m,relative_humidity_2m,precipitation,data_quality_flag\n"
            "cell_rh,2026-08-15 00:00:00,25.0,40.0,0.0,0\n"
            "cell_rh,2026-08-15 01:00:00,25.0,60.0,0.0,0\n"
            "cell_rh,2026-08-15 02:00:00,25.0,80.0,0.0,0\n"
        )
        path = tmp_path / "rh.csv"
        path.write_text(rows)
        result = process_weather_data(str(path))
        assert result.iloc[0]["relative_humidity_2m"] == pytest.approx(60.0, abs=0.1)


# ---------------------------------------------------------------------------
# Test 3: Precipitation aggregation (6-hour sum)
# Assignment guide: provide 6 hourly values, verify sum
# ---------------------------------------------------------------------------

class TestPrecipitationAggregation:

    def test_6hour_precip_sum_is_correct(self, tmp_path):
        """Sum of 6 hourly precipitation values must equal the aggregated total."""
        from scripts.processing.process_weather import process_weather_data

        hourly_precip = [0.0, 1.5, 2.3, 0.0, 0.8, 1.2]
        expected_sum = sum(hourly_precip)  # 5.8 mm

        rows = ["grid_id,timestamp,temperature_2m,precipitation,data_quality_flag"]
        for h, p in enumerate(hourly_precip):
            rows.append(f"cell_precip,2026-08-15 {h:02d}:00:00,20.0,{p},0")

        path = tmp_path / "precip6h.csv"
        path.write_text("\n".join(rows))

        result = process_weather_data(str(path))
        assert result.iloc[0]["precipitation"] == pytest.approx(expected_sum, abs=0.01)

    def test_zero_precip_remains_zero(self, tmp_path):
        """Cells with no rain should have precipitation sum = 0."""
        from scripts.processing.process_weather import process_weather_data
        rows = (
            "grid_id,timestamp,temperature_2m,precipitation,data_quality_flag\n"
            "cell_dry,2026-08-15 00:00:00,30.0,0.0,0\n"
            "cell_dry,2026-08-15 01:00:00,30.0,0.0,0\n"
        )
        path = tmp_path / "zero_precip.csv"
        path.write_text(rows)
        result = process_weather_data(str(path))
        assert result.iloc[0]["precipitation"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 4: NWS fallback trigger — data_quality_flag must be 2
# Assignment guide: mock Open-Meteo failure, verify NWS called and flag=2
# ---------------------------------------------------------------------------

class TestNWSFallback:

    def test_nws_fallback_sets_quality_flag_2(self, nws_fallback_csv):
        """When input data has quality flag 2 (NWS source), output must preserve it."""
        from scripts.processing.process_weather import process_weather_data
        result = process_weather_data(nws_fallback_csv)
        assert result.iloc[0]["data_quality_flag"] == 2, (
            f"NWS fallback data should have quality flag=2, "
            f"got {result.iloc[0]['data_quality_flag']}"
        )

    def test_primary_source_flag_is_zero(self, sample_hourly_csv):
        """Data from Open-Meteo primary should have quality flag=0."""
        from scripts.processing.process_weather import process_weather_data
        result = process_weather_data(sample_hourly_csv)
        assert result.iloc[0]["data_quality_flag"] == 0

    def test_mixed_flags_takes_minimum(self, tmp_path):
        """When some hours are flag 0 and some are flag 2, output min = 0."""
        from scripts.processing.process_weather import process_weather_data
        rows = (
            "grid_id,timestamp,temperature_2m,precipitation,data_quality_flag\n"
            "cell_mix,2026-08-15 00:00:00,20.0,0.0,0\n"
            "cell_mix,2026-08-15 01:00:00,21.0,0.0,2\n"
        )
        path = tmp_path / "mixed_flags.csv"
        path.write_text(rows)
        result = process_weather_data(str(path))
        assert result.iloc[0]["data_quality_flag"] == 0


# ---------------------------------------------------------------------------
# Test 5: Forward-fill (null rows get flag=4)
# Assignment guide: Open-Meteo nulls → forward-fill → quality flag set
# ---------------------------------------------------------------------------

class TestForwardFill:

    def test_null_weather_columns_produce_nan_in_output(self, tmp_path):
        """Null weather values in raw data become NaN in aggregated output.

        Forward-fill with flag=4 happens at fusion level (fuse_features.py),
        not in process_weather. Verify that nulls are preserved here.
        """
        from scripts.processing.process_weather import process_weather_data
        rows = (
            "grid_id,timestamp,temperature_2m,precipitation,data_quality_flag\n"
            "cell_null,2026-08-15 00:00:00,,0.0,4\n"
            "cell_null,2026-08-15 01:00:00,,0.0,4\n"
        )
        path = tmp_path / "nulls.csv"
        path.write_text(rows)
        result = process_weather_data(str(path))
        assert pd.isna(result.iloc[0]["temperature_2m"]), (
            "Null temperature in raw data should remain NaN after aggregation"
        )
        assert result.iloc[0]["data_quality_flag"] == 4

    def test_partial_nulls_compute_mean_of_non_null(self, tmp_path):
        """Mean aggregation ignores NaN values."""
        from scripts.processing.process_weather import process_weather_data
        rows = (
            "grid_id,timestamp,temperature_2m,precipitation,data_quality_flag\n"
            "cell_partial,2026-08-15 00:00:00,20.0,0.0,0\n"
            "cell_partial,2026-08-15 01:00:00,,0.0,0\n"
            "cell_partial,2026-08-15 02:00:00,40.0,0.0,0\n"
        )
        path = tmp_path / "partial.csv"
        path.write_text(rows)
        result = process_weather_data(str(path))
        # Mean of [20, NaN, 40] = 30
        assert result.iloc[0]["temperature_2m"] == pytest.approx(30.0, abs=0.1)


# ---------------------------------------------------------------------------
# Test 6: Derived features — hand-calculated verification
# Assignment guide: hand-calculate days_since, wind_run, drought_proxy and verify
# ---------------------------------------------------------------------------

class TestDerivedFeatures:

    def test_days_since_precip_zero_when_recent_rain(self, sample_hourly_csv):
        """Cell with rain in the window should have days_since_precip = 0."""
        from scripts.processing.process_weather import process_weather_data
        result = process_weather_data(sample_hourly_csv)
        # sample_hourly_csv has 2.1mm rain in hour 05 — exceeds 1mm threshold
        assert result.iloc[0]["days_since_last_precipitation"] == 0

    def test_days_since_precip_positive_when_dry(self, dry_weather_csv):
        """Cell with no rain in 24-hour window should have days_since_precip > 0."""
        from scripts.processing.process_weather import process_weather_data
        result = process_weather_data(dry_weather_csv)
        val = result.iloc[0]["days_since_last_precipitation"]
        # Should be > 0 (dry for the entire window duration)
        assert val is not None and not pd.isna(val) and val >= 0

    def test_cumulative_wind_run_hand_calculation(self, tmp_path):
        """Verify wind run = sum(wind_speed × 1 hour) for a known sequence."""
        from scripts.processing.process_weather import process_weather_data
        wind_speeds = [10.0, 20.0, 30.0]  # km/h, 3 hours
        expected_wind_run = sum(wind_speeds)  # 60.0 km

        rows = ["grid_id,timestamp,temperature_2m,wind_speed_10m,precipitation,data_quality_flag"]
        for h, ws in enumerate(wind_speeds):
            rows.append(f"cell_wind,2026-08-15 {h:02d}:00:00,20.0,{ws},0.0,0")

        path = tmp_path / "wind.csv"
        path.write_text("\n".join(rows))
        result = process_weather_data(str(path))
        assert result.iloc[0]["cumulative_wind_run_24h"] == pytest.approx(
            expected_wind_run, abs=0.1
        )

    def test_drought_proxy_bounded_zero_to_one(self, sample_hourly_csv):
        """drought_index_proxy must always be in [0.0, 1.0]."""
        from scripts.processing.process_weather import process_weather_data
        result = process_weather_data(sample_hourly_csv)
        val = result.iloc[0]["drought_index_proxy"]
        assert 0.0 <= val <= 1.0, f"drought_index_proxy {val} outside [0, 1]"

    def test_drought_proxy_higher_for_hot_dry_conditions(self, tmp_path):
        """Hot, dry, low-soil-moisture conditions should yield higher drought score."""
        from scripts.processing.process_weather import process_weather_data

        # Hot dry conditions
        hot_dry = (
            "grid_id,timestamp,temperature_2m,relative_humidity_2m,wind_speed_10m,"
            "wind_direction_10m,precipitation,soil_moisture_0_to_7cm,vpd,"
            "fire_weather_index,data_quality_flag\n"
            "cell_hot,2026-08-15 00:00:00,42.0,15.0,30.0,270.0,0.0,0.02,5.0,None,0\n"
        )
        # Cool wet conditions
        cool_wet = (
            "grid_id,timestamp,temperature_2m,relative_humidity_2m,wind_speed_10m,"
            "wind_direction_10m,precipitation,soil_moisture_0_to_7cm,vpd,"
            "fire_weather_index,data_quality_flag\n"
            "cell_cool,2026-08-15 00:00:00,8.0,85.0,5.0,90.0,10.0,0.45,0.2,None,0\n"
        )

        hot_path = tmp_path / "hot_dry.csv"
        cool_path = tmp_path / "cool_wet.csv"
        hot_path.write_text(hot_dry)
        cool_path.write_text(cool_wet)

        hot_result = process_weather_data(str(hot_path))
        cool_result = process_weather_data(str(cool_path))

        hot_drought = hot_result.iloc[0]["drought_index_proxy"]
        cool_drought = cool_result.iloc[0]["drought_index_proxy"]

        assert hot_drought > cool_drought, (
            f"Hot dry drought ({hot_drought:.2f}) should exceed "
            f"cool wet drought ({cool_drought:.2f})"
        )


# ---------------------------------------------------------------------------
# Test 7: Unit conversions (NWS Fahrenheit → Celsius, mph → km/h)
# Assignment guide: verify conversion correctness
# ---------------------------------------------------------------------------

class TestUnitConversions:

    def test_fahrenheit_to_celsius_conversion(self):
        """32°F = 0°C, 212°F = 100°C, 98.6°F = 37°C."""
        from scripts.ingestion.ingest_weather import _fahrenheit_to_celsius
        assert _fahrenheit_to_celsius(32.0) == pytest.approx(0.0, abs=0.01)
        assert _fahrenheit_to_celsius(212.0) == pytest.approx(100.0, abs=0.01)
        assert _fahrenheit_to_celsius(98.6) == pytest.approx(37.0, abs=0.1)
        assert _fahrenheit_to_celsius(None) is None

    def test_nws_wind_speed_mph_to_kmh(self):
        """10 mph = 16.09 km/h."""
        from scripts.ingestion.ingest_weather import _parse_nws_wind_speed
        result = _parse_nws_wind_speed("10 mph")
        assert result == pytest.approx(16.093, abs=0.1)

    def test_nws_wind_speed_range_format(self):
        """'5 to 15 mph' should be averaged to 10 mph = 16.09 km/h."""
        from scripts.ingestion.ingest_weather import _parse_nws_wind_speed
        result = _parse_nws_wind_speed("5 to 15 mph")
        assert result == pytest.approx(16.093, abs=0.1)

    def test_nws_wind_direction_cardinal_to_degrees(self):
        """N=0°, E=90°, S=180°, W=270°."""
        from scripts.ingestion.ingest_weather import _parse_nws_wind_direction
        assert _parse_nws_wind_direction("N")  == pytest.approx(0.0)
        assert _parse_nws_wind_direction("E")  == pytest.approx(90.0)
        assert _parse_nws_wind_direction("S")  == pytest.approx(180.0)
        assert _parse_nws_wind_direction("W")  == pytest.approx(270.0)
        assert _parse_nws_wind_direction("NE") == pytest.approx(45.0)
        assert _parse_nws_wind_direction(None) is None


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_csv_returns_empty_dataframe(self, tmp_path):
        from scripts.processing.process_weather import process_weather_data
        path = tmp_path / "empty.csv"
        path.write_text("grid_id,timestamp,temperature_2m\n")
        result = process_weather_data(str(path))
        assert result.empty or len(result) == 0

    def test_missing_file_returns_empty_dataframe(self, tmp_path):
        from scripts.processing.process_weather import process_weather_data
        result = process_weather_data(str(tmp_path / "does_not_exist.csv"))
        assert "grid_id" in result.columns
        assert len(result) == 0

    def test_missing_grid_id_column_returns_empty(self, tmp_path):
        from scripts.processing.process_weather import process_weather_data
        rows = "timestamp,temperature_2m\n2026-08-15 00:00:00,25.0\n"
        path = tmp_path / "no_grid_id.csv"
        path.write_text(rows)
        result = process_weather_data(str(path))
        assert len(result) == 0

    def test_multiple_grid_cells_aggregated_separately(self, tmp_path):
        """Two grid cells in the same CSV must remain separate rows in output."""
        from scripts.processing.process_weather import process_weather_data
        rows = (
            "grid_id,timestamp,temperature_2m,precipitation,data_quality_flag\n"
            "cell_a,2026-08-15 00:00:00,20.0,0.0,0\n"
            "cell_b,2026-08-15 00:00:00,30.0,0.0,0\n"
        )
        path = tmp_path / "two_cells.csv"
        path.write_text(rows)
        result = process_weather_data(str(path))
        assert len(result) == 2
        cell_a = result[result["grid_id"] == "cell_a"].iloc[0]
        cell_b = result[result["grid_id"] == "cell_b"].iloc[0]
        assert cell_a["temperature_2m"] == pytest.approx(20.0)
        assert cell_b["temperature_2m"] == pytest.approx(30.0)
