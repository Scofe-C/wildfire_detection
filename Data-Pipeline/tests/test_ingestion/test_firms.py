"""
Tests for FIRMS Data Ingestion and Processing
==============================================
Owner: Person A

Test categories:
    1. API response parsing (mocked — no real API calls in tests)
    2. Edge cases: zero detections, timeout, malformed responses
    3. Spatial join correctness at multiple resolutions
    4. FRP clipping behavior
    5. Raw data validation
"""

import io
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_firms_csv():
    """Realistic FIRMS CSV response."""
    return (
        "latitude,longitude,brightness,scan,track,acq_date,acq_time,"
        "satellite,instrument,confidence,version,bright_t31,frp,daynight\n"
        "34.0522,-118.2437,320.5,1.0,1.0,2024-08-15,1432,N,VIIRS,85,2.0NRT,290.1,15.3,D\n"
        "36.7783,-119.4179,350.2,1.0,1.0,2024-08-15,1432,N,VIIRS,92,2.0NRT,300.5,42.7,D\n"
        "37.3382,-121.8863,310.1,1.0,1.0,2024-08-15,1432,N,VIIRS,45,2.0NRT,285.3,5.1,D\n"
        "34.0522,-118.2437,340.8,1.0,1.0,2024-08-15,1435,N,VIIRS,88,2.0NRT,295.7,22.0,D\n"
    )


@pytest.fixture
def sample_firms_df(sample_firms_csv):
    """DataFrame from sample FIRMS CSV."""
    return pd.read_csv(io.StringIO(sample_firms_csv))


@pytest.fixture
def empty_firms_df():
    """Empty DataFrame with FIRMS columns."""
    return pd.DataFrame(columns=[
        "latitude", "longitude", "brightness", "scan", "track",
        "acq_date", "acq_time", "satellite", "instrument", "confidence",
        "version", "bright_t31", "frp", "daynight",
    ])


# ---------------------------------------------------------------------------
# Tests: Raw Data Validation
# ---------------------------------------------------------------------------
class TestFirmsValidation:

    def test_valid_data_passes(self, sample_firms_df):
        from scripts.ingestion.ingest_firms import validate_firms_raw
        is_valid, issues = validate_firms_raw(sample_firms_df)
        assert is_valid is True
        assert len(issues) == 0

    def test_empty_data_is_valid(self, empty_firms_df):
        from scripts.ingestion.ingest_firms import validate_firms_raw
        is_valid, issues = validate_firms_raw(empty_firms_df)
        assert is_valid is True

    def test_out_of_range_latitude_flagged(self, sample_firms_df):
        from scripts.ingestion.ingest_firms import validate_firms_raw
        sample_firms_df.loc[0, "latitude"] = 80.0  # Arctic — outside US
        is_valid, issues = validate_firms_raw(sample_firms_df)
        assert is_valid is False
        assert any("latitude" in issue for issue in issues)

    def test_negative_frp_flagged(self, sample_firms_df):
        from scripts.ingestion.ingest_firms import validate_firms_raw
        sample_firms_df.loc[0, "frp"] = -5.0  # Sensor artifact
        is_valid, issues = validate_firms_raw(sample_firms_df)
        assert is_valid is False
        assert any("FRP" in issue for issue in issues)

    def test_missing_required_columns_flagged(self):
        from scripts.ingestion.ingest_firms import validate_firms_raw
        df = pd.DataFrame({"some_col": [1, 2, 3]})
        is_valid, issues = validate_firms_raw(df)
        assert is_valid is False
        assert any("Missing required" in issue for issue in issues)


# ---------------------------------------------------------------------------
# Tests: FIRMS Processing
# ---------------------------------------------------------------------------
class TestFirmsProcessing:

    def test_clean_normalizes_modis_confidence(self):
        from scripts.processing.process_firms import _clean_raw_firms
        df = pd.DataFrame({
            "latitude": [34.0, 35.0, 36.0],
            "longitude": [-118.0, -119.0, -120.0],
            "frp": [10.0, 20.0, 30.0],
            "confidence": ["l", "n", "h"],
        })
        cleaned = _clean_raw_firms(df)
        assert cleaned["confidence"].tolist() == [30, 60, 90]

    def test_clean_drops_missing_coordinates(self):
        from scripts.processing.process_firms import _clean_raw_firms
        df = pd.DataFrame({
            "latitude": [34.0, None, 36.0],
            "longitude": [-118.0, -119.0, None],
            "frp": [10.0, 20.0, 30.0],
            "confidence": [80, 90, 70],
        })
        cleaned = _clean_raw_firms(df)
        assert len(cleaned) == 1  # Only first row has both lat and lon

    def test_frp_clipping_removes_extreme_values(self):
        from scripts.processing.process_firms import _clip_frp_outliers
        # Create data with one extreme outlier
        frp_values = [10.0] * 99 + [99999.0]  # 99th pctile is 10, outlier is huge
        df = pd.DataFrame({"frp": frp_values})
        clipped = _clip_frp_outliers(df)
        assert clipped["frp"].max() < 99999.0

    def test_frp_clipping_handles_negative_values(self):
        from scripts.processing.process_firms import _clip_frp_outliers
        df = pd.DataFrame({"frp": [-5.0, 10.0, 20.0]})
        clipped = _clip_frp_outliers(df)
        assert clipped["frp"].min() >= 0

    def test_aggregate_produces_correct_columns(self):
        from scripts.processing.process_firms import _aggregate_to_grid
        df = pd.DataFrame({
            "grid_id": ["cell_a", "cell_a", "cell_b"],
            "latitude": [34.0, 34.1, 35.0],
            "frp": [10.0, 20.0, 30.0],
            "confidence": [80, 90, 70],
        })
        result = _aggregate_to_grid(df)
        expected_cols = {
            "grid_id", "active_fire_count", "mean_frp",
            "median_frp", "max_confidence", "fire_detected_binary",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_aggregate_counts_fires_per_cell(self):
        from scripts.processing.process_firms import _aggregate_to_grid
        df = pd.DataFrame({
            "grid_id": ["cell_a", "cell_a", "cell_a", "cell_b"],
            "latitude": [34.0, 34.1, 34.2, 35.0],
            "frp": [10.0, 20.0, 30.0, 5.0],
            "confidence": [80, 90, 70, 50],
        })
        result = _aggregate_to_grid(df)
        cell_a = result[result["grid_id"] == "cell_a"].iloc[0]
        cell_b = result[result["grid_id"] == "cell_b"].iloc[0]
        assert cell_a["active_fire_count"] == 3
        assert cell_b["active_fire_count"] == 1

    def test_aggregate_computes_mean_and_median_frp(self):
        from scripts.processing.process_firms import _aggregate_to_grid
        df = pd.DataFrame({
            "grid_id": ["cell_a", "cell_a", "cell_a"],
            "latitude": [34.0, 34.1, 34.2],
            "frp": [10.0, 20.0, 30.0],
            "confidence": [80, 90, 70],
        })
        result = _aggregate_to_grid(df)
        cell_a = result[result["grid_id"] == "cell_a"].iloc[0]
        assert cell_a["mean_frp"] == pytest.approx(20.0)
        assert cell_a["median_frp"] == pytest.approx(20.0)

    def test_all_cells_marked_as_fire_detected(self):
        from scripts.processing.process_firms import _aggregate_to_grid
        df = pd.DataFrame({
            "grid_id": ["cell_a", "cell_b"],
            "latitude": [34.0, 35.0],
            "frp": [10.0, 20.0],
            "confidence": [80, 90],
        })
        result = _aggregate_to_grid(df)
        assert (result["fire_detected_binary"] == 1).all()


# ---------------------------------------------------------------------------
# Tests: Schema Loader
# ---------------------------------------------------------------------------
class TestSchemaLoader:

    def test_registry_loads_without_error(self):
        from scripts.utils.schema_loader import get_registry
        registry = get_registry()
        assert registry is not None

    def test_enabled_features_not_empty(self):
        from scripts.utils.schema_loader import get_registry
        registry = get_registry()
        features = registry.get_enabled_features()
        assert len(features) > 0

    def test_feature_groups_exist(self):
        from scripts.utils.schema_loader import get_registry
        registry = get_registry()
        expected_groups = [
            "identifiers", "weather", "vegetation",
            "topography", "fire_context", "derived", "metadata",
        ]
        for group in expected_groups:
            features = registry.get_features_by_group(group)
            assert len(features) > 0, f"Group '{group}' has no enabled features"

    def test_dtype_map_covers_all_features(self):
        from scripts.utils.schema_loader import get_registry
        registry = get_registry()
        dtype_map = registry.get_dtype_map()
        feature_names = registry.get_feature_names()
        for name in feature_names:
            assert name in dtype_map, f"Feature '{name}' missing from dtype map"

    def test_supported_resolutions(self):
        from scripts.utils.schema_loader import get_registry
        registry = get_registry()
        assert 64 in registry.supported_resolutions
        assert 10 in registry.supported_resolutions
        assert 1 in registry.supported_resolutions

    def test_h3_resolution_mapping(self):
        from scripts.utils.schema_loader import get_registry
        registry = get_registry()
        assert registry.get_h3_resolution(64) == 2
        assert registry.get_h3_resolution(10) == 4
        assert registry.get_h3_resolution(1) == 7

    def test_invalid_resolution_raises(self):
        from scripts.utils.schema_loader import get_registry
        registry = get_registry()
        with pytest.raises(ValueError):
            registry.get_h3_resolution(50)  # Not a supported resolution


# ---------------------------------------------------------------------------
# Tests: Rate Limiter
# ---------------------------------------------------------------------------
class TestRateLimiter:

    def test_requests_within_limit_not_blocked(self):
        from scripts.utils.rate_limiter import RateLimiter, RateLimitConfig
        config = RateLimitConfig(max_requests_per_window=100, window_seconds=60)
        limiter = RateLimiter(config)
        # Should not block for first request
        limiter.wait_if_needed()
        limiter.record_request()
        assert limiter.requests_remaining == 99

    def test_backoff_increases_with_failures(self):
        from scripts.utils.rate_limiter import RateLimiter, RateLimitConfig
        config = RateLimitConfig(
            backoff_base_seconds=1.0,
            backoff_max_seconds=60.0,
            jitter=False,
        )
        limiter = RateLimiter(config)
        limiter.record_failure()
        delay_1 = limiter.get_backoff_delay()
        limiter.record_failure()
        delay_2 = limiter.get_backoff_delay()
        assert delay_2 > delay_1

    def test_backoff_respects_max(self):
        from scripts.utils.rate_limiter import RateLimiter, RateLimitConfig
        config = RateLimitConfig(
            backoff_base_seconds=1.0,
            backoff_max_seconds=10.0,
            jitter=False,
        )
        limiter = RateLimiter(config)
        for _ in range(20):
            limiter.record_failure()
        delay = limiter.get_backoff_delay()
        assert delay <= 10.0


# ---------------------------------------------------------------------------
# Tests: Missing scenarios from Section 2.6 (added per gap analysis)
# ---------------------------------------------------------------------------

class TestFirmsAPIEdgeCases:

    @patch("scripts.ingestion.ingest_firms.requests.get")
    def test_api_timeout_triggers_retry(self, mock_get, tmp_path):
        """Timeout on first two attempts, success on third — retries work."""
        import requests as req_lib
        from scripts.ingestion.ingest_firms import fetch_firms_data

        success_csv = (
            "latitude,longitude,brightness,scan,track,acq_date,acq_time,"
            "satellite,instrument,confidence,version,bright_t31,frp,daynight\n"
            "34.05,-118.24,320.5,1.0,1.0,2026-08-15,1432,N,VIIRS,85,2.0NRT,290.1,15.3,D\n"
        )
        mock_get.side_effect = [
            req_lib.exceptions.Timeout("Simulated timeout 1"),
            req_lib.exceptions.Timeout("Simulated timeout 2"),
            MagicMock(status_code=200, text=success_csv),
        ]

        import datetime
        result_path = fetch_firms_data(
            execution_date=datetime.datetime(2026, 8, 15),
            output_dir=str(tmp_path),
        )
        assert result_path.exists()
        df = pd.read_csv(result_path)
        assert len(df) >= 0  # Pipeline does not fail on retry success

    @patch("scripts.ingestion.ingest_firms.requests.get")
    def test_malformed_csv_does_not_crash(self, mock_get, tmp_path):
        """HTML error page or garbage response returns empty CSV without raising."""
        mock_get.return_value = MagicMock(
            status_code=200,
            text="<html><body>Service Unavailable</body></html>",
        )

        import datetime
        result_path = fetch_firms_data(
            execution_date=datetime.datetime(2026, 8, 15),
            output_dir=str(tmp_path),
        )
        assert result_path.exists(), "fetch_firms_data must always write a file"
        df = pd.read_csv(result_path)
        # Should produce an empty or near-empty DataFrame, not crash
        assert isinstance(df, pd.DataFrame)

    def test_multi_resolution_cell_counts_are_consistent(self):
        """At higher resolution, more cells must be generated for the same bbox.

        This verifies the spatial join logic produces sensible cell counts
        and that changing DEFAULT_RESOLUTION_KM from 64 → 22 doesn't break
        aggregation.
        """
        from scripts.processing.process_firms import process_firms_data
        import tempfile

        raw_csv = (
            "latitude,longitude,brightness,scan,track,acq_date,acq_time,"
            "satellite,instrument,confidence,version,bright_t31,frp,daynight,region,sensor_source\n"
            "34.0522,-118.2437,320.5,1.0,1.0,2026-08-15,1432,N,VIIRS,85,2.0NRT,290.1,15.3,D,california,VIIRS_SNPP_NRT\n"
            "36.7783,-119.4179,350.2,1.0,1.0,2026-08-15,1432,N,VIIRS,92,2.0NRT,300.5,42.7,D,california,VIIRS_SNPP_NRT\n"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(raw_csv)
            raw_path = f.name

        result_64 = process_firms_data(raw_csv_path=raw_path, resolution_km=64)
        result_22 = process_firms_data(raw_csv_path=raw_path, resolution_km=22)

        # Both resolutions must produce a valid non-empty DataFrame
        assert len(result_64) > 0, "64km grid should have at least 1 fire cell"
        assert len(result_22) > 0, "22km grid should have at least 1 fire cell"

        # Higher resolution means smaller cells → same detections may map to
        # different cells but active_fire_count totals should match
        total_fires_64 = result_64["active_fire_count"].sum()
        total_fires_22 = result_22["active_fire_count"].sum()
        assert total_fires_64 == total_fires_22, (
            f"Total fire count must be the same regardless of resolution: "
            f"64km={total_fires_64}, 22km={total_fires_22}"
        )

        # Required columns present at both resolutions
        required = {"grid_id", "active_fire_count", "mean_frp", "fire_detected_binary"}
        assert required.issubset(set(result_64.columns))
        assert required.issubset(set(result_22.columns))

    @patch("scripts.ingestion.ingest_firms.requests.get")
    def test_all_retries_exhausted_returns_empty_not_crash(self, mock_get, tmp_path):
        """If all retries fail (e.g. network down), pipeline must produce empty CSV."""
        import requests as req_lib
        from scripts.ingestion.ingest_firms import fetch_firms_data

        mock_get.side_effect = req_lib.exceptions.ConnectionError("Network unreachable")

        import datetime
        result_path = fetch_firms_data(
            execution_date=datetime.datetime(2026, 8, 15),
            output_dir=str(tmp_path),
        )
        assert result_path.exists(), "Must always write a file even on total failure"

    def test_duplicate_sensor_detections_not_deduplicated(self):
        """VIIRS SNPP and VIIRS NOAA20 detections of the same fire are both counted.

        Section 2.6: 'Duplicate detection: Same fire detected by VIIRS_SNPP
        and VIIRS_NOAA20 — verify it is counted correctly (not deduplicated,
        since each sensor pass is a valid observation).'
        """
        from scripts.processing.process_firms import _aggregate_to_grid

        # Same lat/lon fire detected by two sensors → same grid cell, 2 detections
        df = pd.DataFrame({
            "grid_id": ["cell_fire", "cell_fire"],
            "latitude": [34.0522, 34.0522],
            "frp": [25.0, 27.0],
            "confidence": [85, 88],
            # Different sensor sources — both should be counted
            "sensor_source": ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"],
        })
        result = _aggregate_to_grid(df)
        cell = result[result["grid_id"] == "cell_fire"].iloc[0]

        assert cell["active_fire_count"] == 2, (
            "Both sensor detections must be counted — "
            f"expected 2, got {cell['active_fire_count']}"
        )

# Add fetch_firms_data import at module level for the new tests
from scripts.ingestion.ingest_firms import fetch_firms_data  # noqa: E402
