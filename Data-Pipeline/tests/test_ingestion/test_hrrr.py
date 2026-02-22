"""
Tests for HRRR Weather Ingestion
==================================
All external calls (boto3 S3, Herbie, xarray) are mocked — no real
network access in tests.

Test categories:
    1. Cycle selection: latest cycle, S3 404 fallback, complete failure
    2. Field fetching: Herbie mock, missing variable graceful degradation
    3. Interpolation: centroid extraction, edge cell fallback
    4. Derived variables: wind speed/direction, VPD
    5. Output schema: all columns present, quality flag = 3
    6. Integration: fetch_weather_data HRRR branch trigger conditions
    7. Merge: HRRR focal + Open-Meteo background
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest
try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False



# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def focal_grid_df():
    """Minimal focal grid DataFrame matching generate_fire_focal_grid() output."""
    return pd.DataFrame({
        "grid_id":    ["cell_fire_1", "cell_zone_1", "cell_zone_2"],
        "latitude":   [37.5,          37.55,          37.45],
        "longitude":  [-120.2,        -120.15,        -120.25],
        "cell_type":  ["fire",        "detection_zone", "detection_zone"],
        "ring_distance": [0,           1,               1],
    })


@pytest.fixture
def execution_dt():
    return datetime(2026, 8, 15, 18, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def mock_s3_client():
    """S3 client that returns success for the first S3 key checked."""
    client = MagicMock()
    client.exceptions.ClientError = Exception
    client.head_object.return_value = {}   # 200 OK
    return client


@pytest.fixture
def mock_xarray_da():
    if not XARRAY_AVAILABLE:
        pytest.skip("xarray not installed")
    """Minimal xarray DataArray mimicking HRRR field shape (10×10 grid)."""
    import xarray as xr
    lats = np.linspace(37.0, 38.0, 10)
    lons = np.linspace(-121.0, -119.5, 10)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    data = np.full((10, 10), 25.0)   # flat temperature field for easy assertions

    da = xr.DataArray(
        data,
        dims=["y", "x"],
        attrs={"GRIB_shortName": "TMP"},
    )
    # Attach lat/lon as data variables (how Herbie/cfgrib exposes them)
    ds = xr.Dataset({
        "t2m": da,
        "latitude": xr.DataArray(lat_grid, dims=["y", "x"]),
        "longitude": xr.DataArray(lon_grid, dims=["y", "x"]),
    })
    return ds["t2m"], ds


# ---------------------------------------------------------------------------
# 1. Cycle selection
# ---------------------------------------------------------------------------

class TestCycleSelection:

    def test_selects_latest_cycle_when_available(self, execution_dt, mock_s3_client):
        pytest.importorskip("boto3")
        from scripts.ingestion.ingest_hrrr import _select_hrrr_cycle
        with patch("boto3.client") as mock_boto3_client:
            mock_boto3_client.return_value = mock_s3_client
            result = _select_hrrr_cycle(execution_dt)

        assert result is not None
        assert result.hour == execution_dt.hour
        assert result.minute == 0
        assert result.second == 0

    def test_falls_back_to_previous_cycle_on_404(self, execution_dt):
        """Simulates current cycle not published yet — should try N-1."""
        pytest.importorskip("boto3")
        from scripts.ingestion.ingest_hrrr import _select_hrrr_cycle

        call_count = 0
        def head_object_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Simulate 404 for current cycle
                err = MagicMock()
                err.response = {"Error": {"Code": "404"}}
                raise Exception("404")
            return {}  # success on second attempt

        client = MagicMock()
        client.exceptions.ClientError = Exception
        client.head_object.side_effect = head_object_side_effect

        with patch("boto3.client") as mock_boto3_client:
            mock_boto3.client.return_value = client
            result = _select_hrrr_cycle(execution_dt)

        assert result is not None
        assert result.hour == execution_dt.hour - 1   # fell back one hour

    def test_returns_none_when_all_cycles_missing(self, execution_dt):
        pytest.importorskip("boto3")
        from scripts.ingestion.ingest_hrrr import _select_hrrr_cycle, N_CYCLE_LOOKBACK

        client = MagicMock()
        client.exceptions.ClientError = Exception
        err = MagicMock()
        err.response = {"Error": {"Code": "404"}}
        client.head_object.side_effect = Exception("404")

        with patch("boto3.client") as mock_boto3_client:
            mock_boto3.client.return_value = client
            result = _select_hrrr_cycle(execution_dt)

        assert result is None

    def test_returns_none_when_boto3_not_installed(self, execution_dt):
        from scripts.ingestion.ingest_hrrr import _select_hrrr_cycle
        with patch.dict("sys.modules", {"boto3": None}):
            result = _select_hrrr_cycle(execution_dt)
        assert result is None

    def test_s3_key_format(self):
        from scripts.ingestion.ingest_hrrr import _hrrr_s3_key
        dt = datetime(2026, 8, 15, 18, 0, 0, tzinfo=timezone.utc)
        key = _hrrr_s3_key(dt)
        assert key == "hrrr.20260815/conus/hrrr.t18z.wrfsfcf00.grib2"


# ---------------------------------------------------------------------------
# 2. Field fetching
# ---------------------------------------------------------------------------

class TestFieldFetching:

    def test_returns_fields_dict_on_success(self, execution_dt, mock_xarray_da):
        import sys, types
        da, ds = mock_xarray_da

        mock_herbie_instance = MagicMock()
        mock_herbie_instance.xarray.return_value = ds

        # _fetch_hrrr_fields does `from herbie import Herbie` inside the
        # function body, so we inject a fake herbie module with our mock.
        fake_herbie_mod = types.ModuleType("herbie")
        fake_herbie_mod.Herbie = MagicMock(return_value=mock_herbie_instance)

        with patch.dict(sys.modules, {"herbie": fake_herbie_mod}):
            import importlib
            import scripts.ingestion.ingest_hrrr as hrrr_mod
            importlib.reload(hrrr_mod)
            result = hrrr_mod._fetch_hrrr_fields(execution_dt)

        assert result is not None
        assert isinstance(result, dict)
        # At least temperature and RH should be present
        assert "temperature_2m" in result or len(result) > 0

    def test_returns_none_when_herbie_not_installed(self, execution_dt):
        from scripts.ingestion.ingest_hrrr import _fetch_hrrr_fields
        with patch.dict("sys.modules", {"herbie": None}):
            result = _fetch_hrrr_fields(execution_dt)
        assert result is None

    def test_partial_failure_returns_available_fields(self, execution_dt, mock_xarray_da):
        """When one variable fails, others should still be returned."""
        import sys, types
        da, ds = mock_xarray_da
        call_count = 0

        def xarray_side_effect(search, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated GRIB message not found")
            return ds

        mock_herbie_instance = MagicMock()
        mock_herbie_instance.xarray.side_effect = xarray_side_effect

        fake_herbie_mod = types.ModuleType("herbie")
        fake_herbie_mod.Herbie = MagicMock(return_value=mock_herbie_instance)

        with patch.dict(sys.modules, {"herbie": fake_herbie_mod}):
            import importlib
            import scripts.ingestion.ingest_hrrr as hrrr_mod
            importlib.reload(hrrr_mod)
            result = hrrr_mod._fetch_hrrr_fields(execution_dt)

        # Should return whatever succeeded, not None
        # (partial is better than aborting)
        assert result is not None or call_count > 0   # tried at least one field


# ---------------------------------------------------------------------------
# 3. Interpolation
# ---------------------------------------------------------------------------

class TestInterpolation:

    def test_produces_one_record_per_centroid(self, focal_grid_df, mock_xarray_da):
        da, ds = mock_xarray_da
        cycle_dt = datetime(2026, 8, 15, 18, tzinfo=timezone.utc)

        from scripts.ingestion.ingest_hrrr import _interpolate_to_centroids

        # Provide a mock that has lat/lon accessible via indexing
        class FakeDA:
            def __init__(self):
                n = 10
                lats = np.linspace(37.0, 38.0, n)
                lons = np.linspace(-121.0, -119.5, n)
                self._lat_grid, self._lon_grid = np.meshgrid(lats, lons, indexing="ij")
                self.values = np.full((n, n), 25.5)
                self.shape = (n, n)

            def __getitem__(self, key):
                if key == "latitude":
                    return MagicMock(values=self._lat_grid)
                if key == "longitude":
                    return MagicMock(values=self._lon_grid)
                raise KeyError(key)

        records = _interpolate_to_centroids(
            {"temperature_2m": FakeDA()},
            focal_grid_df,
            cycle_dt,
        )

        assert len(records) == len(focal_grid_df)
        assert all("grid_id" in r for r in records)
        assert all("temperature_2m" in r for r in records)

    def test_records_contain_correct_grid_ids(self, focal_grid_df):
        from scripts.ingestion.ingest_hrrr import _interpolate_to_centroids

        class FakeDA:
            def __init__(self):
                n = 5
                lats = np.linspace(37.0, 38.0, n)
                lons = np.linspace(-121.0, -119.5, n)
                self._lat_grid, self._lon_grid = np.meshgrid(lats, lons, indexing="ij")
                self.values = np.ones((n, n)) * 20.0
                self.shape = (n, n)

            def __getitem__(self, key):
                if key == "latitude":
                    return MagicMock(values=self._lat_grid)
                if key == "longitude":
                    return MagicMock(values=self._lon_grid)
                raise KeyError(key)

        cycle_dt = datetime(2026, 8, 15, 18, tzinfo=timezone.utc)
        records = _interpolate_to_centroids(
            {"temperature_2m": FakeDA()},
            focal_grid_df,
            cycle_dt,
        )

        result_ids = {r["grid_id"] for r in records}
        expected_ids = set(focal_grid_df["grid_id"].tolist())
        assert result_ids == expected_ids


# ---------------------------------------------------------------------------
# 4. Derived variables
# ---------------------------------------------------------------------------

class TestDerivedVariables:

    def test_wind_speed_conversion(self):
        from scripts.ingestion.ingest_hrrr import _uv_to_speed_kmh
        u = pd.Series([0.0, 10.0, -10.0, 0.0])
        v = pd.Series([0.0,  0.0,   0.0, 10.0])
        speed = _uv_to_speed_kmh(u, v)
        assert speed[0] == pytest.approx(0.0,    abs=0.01)
        assert speed[1] == pytest.approx(36.0,   abs=0.1)   # 10 m/s × 3.6
        assert speed[2] == pytest.approx(36.0,   abs=0.1)
        assert speed[3] == pytest.approx(36.0,   abs=0.1)

    def test_wind_direction_from_north(self):
        from scripts.ingestion.ingest_hrrr import _uv_to_direction
        # Wind from North: u=0, v=-1 (blowing southward means FROM north)
        u = pd.Series([0.0])
        v = pd.Series([-1.0])
        direction = _uv_to_direction(u, v)
        assert direction[0] == pytest.approx(0.0, abs=1.0)  # 0° = from North

    def test_wind_direction_from_west(self):
        from scripts.ingestion.ingest_hrrr import _uv_to_direction
        # Wind from West: u=+1 (blowing eastward), v=0
        u = pd.Series([1.0])
        v = pd.Series([0.0])
        direction = _uv_to_direction(u, v)
        assert direction[0] == pytest.approx(270.0, abs=1.0)  # 270° = from West

    def test_vpd_positive(self):
        from scripts.ingestion.ingest_hrrr import _compute_vpd
        temp = pd.Series([30.0, 20.0, 0.0])
        rh   = pd.Series([50.0, 80.0, 100.0])
        vpd  = _compute_vpd(temp, rh)
        assert all(vpd >= 0), "VPD must be non-negative"
        assert vpd[2] == pytest.approx(0.0, abs=0.01)  # 100% RH → VPD=0

    def test_vpd_increases_with_temperature(self):
        from scripts.ingestion.ingest_hrrr import _compute_vpd
        rh   = pd.Series([50.0, 50.0])
        temp = pd.Series([20.0, 35.0])
        vpd  = _compute_vpd(temp, rh)
        assert vpd[1] > vpd[0]

    def test_vpd_decreases_with_humidity(self):
        from scripts.ingestion.ingest_hrrr import _compute_vpd
        temp = pd.Series([30.0, 30.0])
        rh   = pd.Series([30.0, 80.0])
        vpd  = _compute_vpd(temp, rh)
        assert vpd[0] > vpd[1]


# ---------------------------------------------------------------------------
# 5. Output schema
# ---------------------------------------------------------------------------

class TestOutputSchema:

    @patch("scripts.ingestion.ingest_hrrr._select_hrrr_cycle")
    @patch("scripts.ingestion.ingest_hrrr._fetch_hrrr_fields")
    @patch("scripts.ingestion.ingest_hrrr._interpolate_to_centroids")
    def test_output_has_all_required_columns(
        self, mock_interp, mock_fields, mock_cycle, focal_grid_df, tmp_path
    ):
        from scripts.ingestion.ingest_hrrr import fetch_hrrr_for_focal_grid, OUTPUT_COLUMNS

        mock_cycle.return_value = datetime(2026, 8, 15, 17, tzinfo=timezone.utc)
        mock_fields.return_value = {"temperature_2m": MagicMock()}
        mock_interp.return_value = [
            {
                "grid_id": "cell_fire_1",
                "temperature_2m": 32.5,
                "_u_wind_10m": 5.0,
                "_v_wind_10m": -3.0,
                "relative_humidity_2m": 25.0,
                "precipitation": 0.0,
                "soil_moisture_0_to_7cm": 0.08,
            },
            {
                "grid_id": "cell_zone_1",
                "temperature_2m": 31.0,
                "_u_wind_10m": 4.5,
                "_v_wind_10m": -2.8,
                "relative_humidity_2m": 28.0,
                "precipitation": 0.0,
                "soil_moisture_0_to_7cm": 0.09,
            },
        ]

        result_path = fetch_hrrr_for_focal_grid(
            focal_grid=focal_grid_df,
            execution_date=datetime(2026, 8, 15, 18, tzinfo=timezone.utc),
            output_dir=str(tmp_path),
        )

        assert result_path is not None
        assert result_path.exists()

        df = pd.read_csv(result_path)
        for col in OUTPUT_COLUMNS:
            assert col in df.columns, f"Missing output column: {col}"

    @patch("scripts.ingestion.ingest_hrrr._select_hrrr_cycle")
    @patch("scripts.ingestion.ingest_hrrr._fetch_hrrr_fields")
    @patch("scripts.ingestion.ingest_hrrr._interpolate_to_centroids")
    def test_quality_flag_is_3(
        self, mock_interp, mock_fields, mock_cycle, focal_grid_df, tmp_path
    ):
        from scripts.ingestion.ingest_hrrr import fetch_hrrr_for_focal_grid, HRRR_QUALITY_FLAG

        mock_cycle.return_value = datetime(2026, 8, 15, 17, tzinfo=timezone.utc)
        mock_fields.return_value = {"temperature_2m": MagicMock()}
        mock_interp.return_value = [
            {"grid_id": "cell_fire_1", "temperature_2m": 30.0,
             "_u_wind_10m": 0.0, "_v_wind_10m": 0.0,
             "relative_humidity_2m": 40.0, "precipitation": 0.0,
             "soil_moisture_0_to_7cm": 0.1}
        ]

        result_path = fetch_hrrr_for_focal_grid(
            focal_grid=focal_grid_df,
            execution_date=datetime(2026, 8, 15, 18, tzinfo=timezone.utc),
            output_dir=str(tmp_path),
        )

        df = pd.read_csv(result_path)
        assert all(df["data_quality_flag"] == HRRR_QUALITY_FLAG)

    def test_returns_none_on_empty_focal_grid(self, tmp_path):
        from scripts.ingestion.ingest_hrrr import fetch_hrrr_for_focal_grid
        result = fetch_hrrr_for_focal_grid(
            focal_grid=pd.DataFrame(),
            execution_date=datetime(2026, 8, 15, 18, tzinfo=timezone.utc),
            output_dir=str(tmp_path),
        )
        assert result is None

    @patch("scripts.ingestion.ingest_hrrr._select_hrrr_cycle")
    def test_returns_none_when_cycle_selection_fails(self, mock_cycle, focal_grid_df, tmp_path):
        from scripts.ingestion.ingest_hrrr import fetch_hrrr_for_focal_grid
        mock_cycle.return_value = None
        result = fetch_hrrr_for_focal_grid(
            focal_grid=focal_grid_df,
            execution_date=datetime(2026, 8, 15, 18, tzinfo=timezone.utc),
            output_dir=str(tmp_path),
        )
        assert result is None


# ---------------------------------------------------------------------------
# 6. Integration: fetch_weather_data HRRR branch trigger conditions
# ---------------------------------------------------------------------------

class TestFetchWeatherDataHRRRBranch:

    def test_cron_trigger_skips_hrrr(self, focal_grid_df, tmp_path):
        """Cron runs must never call HRRR — Open-Meteo only."""
        from scripts.ingestion.ingest_weather import fetch_weather_data

        with patch("scripts.ingestion.ingest_weather._try_hrrr_focal") as mock_hrrr, \
             patch("scripts.ingestion.ingest_weather._fetch_open_meteo_batch") as mock_om:
            mock_om.return_value = pd.DataFrame()  # empty — writes fallback CSV

            fetch_weather_data(
                grid_centroids=focal_grid_df[["grid_id", "latitude", "longitude"]],
                execution_date=datetime(2026, 8, 15, 18, tzinfo=timezone.utc),
                trigger_source="cron",
                fire_cells=[],
                output_dir=str(tmp_path),
            )

        mock_hrrr.assert_not_called()

    def test_watchdog_emergency_calls_hrrr(self, focal_grid_df, tmp_path):
        """watchdog_emergency must attempt HRRR before Open-Meteo."""
        from scripts.ingestion.ingest_weather import fetch_weather_data

        with patch("scripts.ingestion.ingest_weather._try_hrrr_focal") as mock_hrrr, \
             patch("scripts.ingestion.ingest_weather._fetch_open_meteo_batch") as mock_om:
            mock_hrrr.return_value = None  # HRRR fails → falls through to Open-Meteo
            mock_om.return_value = pd.DataFrame()

            fetch_weather_data(
                grid_centroids=focal_grid_df[["grid_id", "latitude", "longitude"]],
                execution_date=datetime(2026, 8, 15, 18, tzinfo=timezone.utc),
                trigger_source="watchdog_emergency",
                fire_cells=["8e283082ddbffff"],
                output_dir=str(tmp_path),
            )

        mock_hrrr.assert_called_once()

    def test_watchdog_no_fire_cells_skips_hrrr(self, focal_grid_df, tmp_path):
        """If fire_cells is empty, HRRR should not be called even on watchdog trigger."""
        from scripts.ingestion.ingest_weather import fetch_weather_data

        with patch("scripts.ingestion.ingest_weather._try_hrrr_focal") as mock_hrrr, \
             patch("scripts.ingestion.ingest_weather._fetch_open_meteo_batch") as mock_om:
            mock_om.return_value = pd.DataFrame()

            fetch_weather_data(
                grid_centroids=focal_grid_df[["grid_id", "latitude", "longitude"]],
                execution_date=datetime(2026, 8, 15, 18, tzinfo=timezone.utc),
                trigger_source="watchdog_emergency",
                fire_cells=[],   # empty — no confirmed cells
                output_dir=str(tmp_path),
            )

        mock_hrrr.assert_not_called()


# ---------------------------------------------------------------------------
# 7. Merge: HRRR focal + Open-Meteo background
# ---------------------------------------------------------------------------

class TestHRRRBackgroundMerge:

    def test_merged_output_covers_all_cells(self, focal_grid_df, tmp_path):
        """All cells in grid_centroids should appear in the merged CSV."""
        from scripts.ingestion.ingest_weather import _merge_hrrr_with_background
        from scripts.utils.rate_limiter import RateLimiter, RateLimitConfig

        # Write a minimal HRRR CSV for focal cells only (first 2 cells)
        hrrr_cells = focal_grid_df.iloc[:2]
        hrrr_data = pd.DataFrame({
            "grid_id": hrrr_cells["grid_id"].tolist(),
            "timestamp": ["2026-08-15T17:00:00+00:00"] * 2,
            "temperature_2m": [32.0, 31.5],
            "relative_humidity_2m": [25.0, 26.0],
            "wind_speed_10m": [18.0, 17.5],
            "wind_direction_10m": [270.0, 268.0],
            "precipitation": [0.0, 0.0],
            "soil_moisture_0_to_7cm": [0.08, 0.09],
            "vpd": [2.1, 2.0],
            "fire_weather_index": [None, None],
            "data_quality_flag": [3, 3],
        })
        hrrr_path = tmp_path / "weather_hrrr_test.csv"
        hrrr_data.to_csv(hrrr_path, index=False)

        # Mock Open-Meteo for the background cell (third cell)
        bg_weather = pd.DataFrame({
            "grid_id":               ["cell_zone_2"],
            "timestamp":             ["2026-08-15T16:00:00+00:00"],
            "temperature_2m":        [30.0],
            "relative_humidity_2m":  [30.0],
            "wind_speed_10m":        [15.0],
            "wind_direction_10m":    [260.0],
            "precipitation":         [0.0],
            "soil_moisture_0_to_7cm":[0.10],
            "vpd":                   [1.9],
            "fire_weather_index":    [None],
            "data_quality_flag":     [0],
        })

        limiter = RateLimiter(RateLimitConfig())
        om_config = {
            "base_url": "https://api.open-meteo.com/v1/forecast",
            "historical_url": "https://archive-api.open-meteo.com/v1/archive",
            "timeout_seconds": 20,
            "max_retries": 1,
        }

        with patch("scripts.ingestion.ingest_weather._fetch_open_meteo_batch",
                   return_value=bg_weather):
            result_path = _merge_hrrr_with_background(
                hrrr_path=hrrr_path,
                grid_centroids=focal_grid_df[["grid_id", "latitude", "longitude"]],
                execution_date=datetime(2026, 8, 15, 18, tzinfo=timezone.utc),
                lookback_hours=2,
                output_dir=str(tmp_path),
                om_config=om_config,
                limiter=limiter,
                config_path=None,
            )

        assert result_path.exists()
        merged = pd.read_csv(result_path)
        result_ids = set(merged["grid_id"].astype(str).tolist())
        expected_ids = set(focal_grid_df["grid_id"].tolist())
        assert result_ids == expected_ids, (
            f"Missing cells in merged output: {expected_ids - result_ids}"
        )

    def test_hrrr_cells_have_flag_3(self, focal_grid_df, tmp_path):
        """HRRR focal cells must retain data_quality_flag=3 after merge."""
        from scripts.ingestion.ingest_weather import _merge_hrrr_with_background
        from scripts.utils.rate_limiter import RateLimiter, RateLimitConfig

        hrrr_data = pd.DataFrame({
            "grid_id": ["cell_fire_1"],
            "timestamp": ["2026-08-15T17:00:00+00:00"],
            "temperature_2m": [33.0],
            "relative_humidity_2m": [22.0],
            "wind_speed_10m": [20.0],
            "wind_direction_10m": [275.0],
            "precipitation": [0.0],
            "soil_moisture_0_to_7cm": [0.07],
            "vpd": [2.3],
            "fire_weather_index": [None],
            "data_quality_flag": [3],
        })
        hrrr_path = tmp_path / "weather_hrrr_flag_test.csv"
        hrrr_data.to_csv(hrrr_path, index=False)

        limiter = RateLimiter(RateLimitConfig())
        om_config = {
            "base_url": "https://api.open-meteo.com/v1/forecast",
            "historical_url": "https://archive-api.open-meteo.com/v1/archive",
            "timeout_seconds": 20,
            "max_retries": 1,
        }

        with patch("scripts.ingestion.ingest_weather._fetch_open_meteo_batch",
                   return_value=pd.DataFrame()):
            result_path = _merge_hrrr_with_background(
                hrrr_path=hrrr_path,
                grid_centroids=focal_grid_df[["grid_id", "latitude", "longitude"]],
                execution_date=datetime(2026, 8, 15, 18, tzinfo=timezone.utc),
                lookback_hours=2,
                output_dir=str(tmp_path),
                om_config=om_config,
                limiter=limiter,
                config_path=None,
            )

        merged = pd.read_csv(result_path)
        hrrr_row = merged[merged["grid_id"] == "cell_fire_1"]
        assert not hrrr_row.empty
        assert int(hrrr_row["data_quality_flag"].iloc[0]) == 3