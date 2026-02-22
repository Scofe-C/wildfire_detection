# tests/conftest.py
"""
Shared pytest fixtures available to all test modules.
Place this file at tests/conftest.py — pytest discovers it automatically.
"""
import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# Fix Airflow SQLite path on Windows — must be set before airflow is imported
_db = os.path.join(tempfile.gettempdir(), "airflow_test.db").replace("\\", "/")
if not _db.startswith("/"):
    _db = "/" + _db
os.environ.setdefault("AIRFLOW__DATABASE__SQL_ALCHEMY_CONN", f"sqlite:///{_db}")
os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "false")
os.environ.setdefault("AIRFLOW__CORE__UNIT_TEST_MODE", "true")
os.environ.setdefault("GCS_BUCKET_NAME", "my-wildfire-1")
os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS",
    str(Path(__file__).resolve().parent.parent / "gcp-key.json"),
)
os.environ.setdefault("FIRMS_MAP_KEY", "test-key")


@pytest.fixture(scope="session")
def project_root():
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def execution_date_fire_season():
    return datetime(2026, 8, 15, 18, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def execution_date_off_season():
    return datetime(2026, 2, 15, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def minimal_fused_df():
    """Minimal valid fused DataFrame — usable by any downstream test."""
    return pd.DataFrame({
        "grid_id":    ["cell_a", "cell_b", "cell_c"],
        "latitude":   [37.5,     37.6,     36.0],
        "longitude":  [-120.2,   -120.3,   -119.0],
        "timestamp":  pd.Timestamp("2026-08-15T18:00:00", tz="UTC"),
        "resolution_km": 64,
        "active_fire_count":       [5, 0, 2],
        "mean_frp":                [120.0, 0.0, 45.0],
        "median_frp":              [100.0, 0.0, 40.0],
        "max_confidence":          [90, 0, 75],
        "nearest_fire_distance_km": [0.0, 50.0, 10.0],
        "fire_detected_binary":    [1, 0, 1],
        "temperature_2m":          [32.0, 30.0, 28.0],
        "relative_humidity_2m":    [22.0, 35.0, 40.0],
        "wind_speed_10m":          [18.0, 12.0, 8.0],
        "wind_direction_10m":      [270.0, 245.0, 180.0],
        "data_quality_flag":       [0, 0, 0],
        "data_source_priority":    [2, 2, 2],
    })


@pytest.fixture
def dummy_registry():
    """Minimal registry stub for tests that need a registry but not a real config."""
    class DummyRegistry:
        anomaly_config = {"monitored_features": ["temperature_2m", "wind_speed_10m"]}
        fire_season_months = [6, 7, 8, 9, 10, 11]
        temporal_aggregation_hours = 6

        def get_z_threshold(self, month):
            return 4.0 if month in self.fire_season_months else 3.5

        def get_feature_names(self):
            return list(self.__class__.__dict__.get("_feature_names", []))

    return DummyRegistry()