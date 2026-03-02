"""
Tests for schema_loader utility
================================
Covers the two previously untested public functions:
  - load_config()
  - get_empty_dataframe()

The FeatureRegistry class itself is exercised indirectly through
both functions; its individual methods are used throughout the suite
so they are not duplicated here.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_config(tmp_path) -> Path:
    """Write a minimal valid schema_config.yaml to a temp dir."""
    cfg = {
        "pipeline": {
            "name": "test_pipeline",
            "default_resolution_km": 64,
            "supported_resolutions_km": [64],
            "h3_resolution_map": {64: 2},
            "temporal_aggregation_hours": 6,
            "geographic_scope": {
                "california": {"bbox": [-124.48, 32.53, -114.13, 42.01]},
            },
        },
        "features": {
            "identifiers": [
                {"name": "grid_id",    "type": "string",  "enabled": True, "nullable": False},
                {"name": "latitude",   "type": "float32", "enabled": True, "nullable": False},
                {"name": "longitude",  "type": "float32", "enabled": True, "nullable": False},
            ],
            "weather": [
                {"name": "temperature_2m",       "type": "float32", "enabled": True,  "nullable": True},
                {"name": "relative_humidity_2m", "type": "float32", "enabled": False, "nullable": True},
            ],
        },
        "data_sources": {
            "firms": {
                "base_url": "https://firms.example.com",
                "api_key_env_var": "FIRMS_MAP_KEY",
                "rate_limit": {
                    "max_requests_per_10min": 100,
                    "backoff_base_seconds": 1,
                    "backoff_max_seconds": 10,
                    "jitter": True,
                },
                "max_retries": 2,
            },
            "open_meteo": {
                "base_url": "https://api.open-meteo.com/v1/forecast",
                "rate_limit": {
                    "max_requests_per_day": 500,
                    "backoff_base_seconds": 1,
                    "backoff_max_seconds": 30,
                },
                "max_retries": 2,
            },
        },
        "anomaly_detection": {
            "monitored_features": ["temperature_2m"],
            "z_score_threshold": {"fire_season": 4.0, "off_season": 3.5},
            "seasons": {
                "fire_season": {"months": [6, 7, 8, 9, 10, 11]},
                "off_season": {"months": [12, 1, 2, 3, 4, 5]},
            },
        },
        "validation": {"max_null_rate": 0.15, "row_count_tolerance_pct": 5},
        "storage": {"gcs_bucket_env_var": "GCS_BUCKET_NAME"},
        "watchdog": {"gcs_paths": {}},
    }
    path = tmp_path / "schema_config.yaml"
    path.write_text(yaml.dump(cfg))
    return path


@pytest.fixture
def bad_yaml(tmp_path) -> Path:
    """Write a deliberately malformed YAML file."""
    path = tmp_path / "bad_config.yaml"
    path.write_text("stages:\n  - [unclosed bracket\n  bad: : indent")
    return path


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:

    def test_loads_valid_config(self, minimal_config):
        """load_config must return a dict with top-level keys."""
        from scripts.utils.schema_loader import load_config
        cfg = load_config(str(minimal_config))
        assert isinstance(cfg, dict)
        assert "features" in cfg
        assert "pipeline" in cfg

    def test_missing_file_raises_file_not_found(self, tmp_path):
        """load_config must raise FileNotFoundError for a non-existent path."""
        from scripts.utils.schema_loader import load_config
        with pytest.raises(FileNotFoundError, match="Schema config not found"):
            load_config(str(tmp_path / "does_not_exist.yaml"))

    def test_malformed_yaml_raises(self, bad_yaml):
        """load_config must raise yaml.YAMLError for malformed YAML."""
        import yaml as _yaml
        from scripts.utils.schema_loader import load_config
        with pytest.raises(_yaml.YAMLError):
            load_config(str(bad_yaml))

    def test_default_path_is_used_when_none(self):
        """Calling load_config() with no args must load the real project config."""
        from scripts.utils.schema_loader import load_config
        cfg = load_config()
        # Real config must have at least a features and pipeline section
        assert "features" in cfg
        assert "pipeline" in cfg

    def test_returned_dict_contains_features(self, minimal_config):
        """The loaded config must contain the features section as a dict."""
        from scripts.utils.schema_loader import load_config
        cfg = load_config(str(minimal_config))
        assert isinstance(cfg["features"], dict)

    def test_config_path_override_takes_precedence(self, minimal_config, tmp_path):
        """An explicit config_path must be used instead of the default."""
        # Write a second config with a sentinel value
        other_cfg = {"pipeline": {"name": "other"}, "features": {}, "data_sources": {},
                     "anomaly_detection": {}, "validation": {}, "storage": {}, "watchdog": {}}
        other_path = tmp_path / "other.yaml"
        other_path.write_text(yaml.dump(other_cfg))

        from scripts.utils.schema_loader import load_config
        cfg = load_config(str(other_path))
        assert cfg["pipeline"]["name"] == "other"


# ---------------------------------------------------------------------------
# get_empty_dataframe
# ---------------------------------------------------------------------------

class TestGetEmptyDataframe:

    def test_returns_dataframe(self, minimal_config):
        """get_empty_dataframe must return a pandas DataFrame."""
        from scripts.utils.schema_loader import get_empty_dataframe
        df = get_empty_dataframe(str(minimal_config))
        assert isinstance(df, pd.DataFrame)

    def test_dataframe_is_empty(self, minimal_config):
        """Returned DataFrame must have zero rows."""
        from scripts.utils.schema_loader import get_empty_dataframe
        df = get_empty_dataframe(str(minimal_config))
        assert len(df) == 0

    def test_columns_match_enabled_features_only(self, minimal_config):
        """Only enabled features must appear as columns.

        The minimal_config has:
          identifiers: grid_id (enabled), latitude (enabled), longitude (enabled)
          weather: temperature_2m (enabled), relative_humidity_2m (DISABLED)
        """
        from scripts.utils.schema_loader import get_empty_dataframe
        df = get_empty_dataframe(str(minimal_config))
        assert "grid_id" in df.columns
        assert "latitude" in df.columns
        assert "temperature_2m" in df.columns
        # Disabled feature must NOT appear
        assert "relative_humidity_2m" not in df.columns

    def test_columns_match_real_schema(self):
        """get_empty_dataframe with default config must include all enabled features."""
        from scripts.utils.schema_loader import get_empty_dataframe, get_registry
        df = get_empty_dataframe()
        registry = get_registry()
        expected_cols = set(registry.get_feature_names())
        assert set(df.columns) == expected_cols

    def test_dtype_applied_to_numeric_columns(self, minimal_config):
        """Numeric columns must have their declared dtype applied."""
        from scripts.utils.schema_loader import get_empty_dataframe
        df = get_empty_dataframe(str(minimal_config))
        # latitude declared as float32
        assert df["latitude"].dtype == "float32"

    def test_returns_new_instance_each_call(self, minimal_config):
        """Two calls must return independent DataFrame objects."""
        from scripts.utils.schema_loader import get_empty_dataframe
        df1 = get_empty_dataframe(str(minimal_config))
        df2 = get_empty_dataframe(str(minimal_config))
        assert df1 is not df2