"""
Schema Loader Utility
=====================
Reads schema_config.yaml and provides a programmatic interface for all
pipeline components to discover enabled features, their types, validation
rules, and fill strategies.

This module is the ONLY place that reads schema_config.yaml. All other
modules import from here rather than parsing the YAML themselves.

Owner: Person D (Fusion + Validation)
Consumers: All team members
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
import pandas as pd


# ---------------------------------------------------------------------------
# Config path resolution
# ---------------------------------------------------------------------------
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "schema_config.yaml"


def load_config(config_path: Optional[str] = None) -> dict:
    """Load the full schema configuration from YAML.

    Args:
        config_path: Override path. If None, uses the default config location.

    Returns:
        Parsed YAML as a nested dict.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If config file is malformed.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Schema config not found at {path}. "
            f"Ensure configs/schema_config.yaml exists."
        )
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Feature Registry
# ---------------------------------------------------------------------------
class FeatureRegistry:
    """Provides typed access to the feature schema.

    Usage:
        registry = FeatureRegistry()
        enabled = registry.get_enabled_features()
        dtypes = registry.get_dtype_map()
        weather_cols = registry.get_features_by_group("weather")
    """

    def __init__(self, config_path: Optional[str] = None):
        self._config = load_config(config_path)
        self._features = self._config.get("features", {})
        self._pipeline = self._config.get("pipeline", {})
        self._sources = self._config.get("data_sources", {})
        self._anomaly = self._config.get("anomaly_detection", {})
        self._validation = self._config.get("validation", {})
        self._storage = self._config.get("storage", {})

    # --- Feature Queries ---

    def get_all_features(self) -> list[dict]:
        """Return all features across all groups as a flat list."""
        all_features = []
        for group_name, features in self._features.items():
            for feat in features:
                feat_copy = feat.copy()
                feat_copy["group"] = group_name
                all_features.append(feat_copy)
        return all_features

    def get_enabled_features(self) -> list[dict]:
        """Return only features where enabled: true."""
        return [f for f in self.get_all_features() if f.get("enabled", True)]

    def get_features_by_group(self, group: str) -> list[dict]:
        """Return enabled features for a specific group.

        Args:
            group: One of 'identifiers', 'weather', 'vegetation',
                   'topography', 'fire_context', 'derived', 'metadata'.
        """
        return [
            f for f in self.get_enabled_features()
            if f.get("group") == group
        ]

    def get_features_by_source(self, source: str) -> list[dict]:
        """Return enabled features from a specific data source.

        Args:
            source: One of 'firms', 'open_meteo', 'landfire', 'srtm',
                    'modis', 'computed'.
        """
        return [
            f for f in self.get_enabled_features()
            if f.get("source") == source
        ]

    def get_feature_names(self, group: Optional[str] = None) -> list[str]:
        """Return column names for enabled features, optionally filtered by group."""
        if group:
            features = self.get_features_by_group(group)
        else:
            features = self.get_enabled_features()
        return [f["name"] for f in features]

    def get_dtype_map(self) -> dict[str, str]:
        """Return a dict mapping column name to pandas dtype string.

        Used for enforcing schema on DataFrames:
            df = df.astype(registry.get_dtype_map())
        """
        return {
            f["name"]: f["type"]
            for f in self.get_enabled_features()
            if "type" in f
        }

    def get_validation_rules(self) -> dict[str, dict]:
        """Return validation rules keyed by column name.

        Returns:
            Dict of {column_name: {min, max, allowed_values, ...}}
        """
        return {
            f["name"]: f["validation"]
            for f in self.get_enabled_features()
            if "validation" in f
        }

    def get_fill_strategies(self) -> dict[str, str]:
        """Return fill strategies for nullable columns.

        Returns:
            Dict of {column_name: strategy} where strategy is one of
            'forward_fill', 'zero', or a default value.
        """
        strategies = {}
        for f in self.get_enabled_features():
            if "fill_strategy" in f:
                strategies[f["name"]] = f["fill_strategy"]
            elif "default_value" in f:
                strategies[f["name"]] = f["default_value"]
        return strategies

    def get_nullable_columns(self) -> list[str]:
        """Return names of columns that allow nulls."""
        return [
            f["name"]
            for f in self.get_enabled_features()
            if f.get("nullable", True)
        ]

    def get_non_nullable_columns(self) -> list[str]:
        """Return names of columns that must never be null."""
        return [
            f["name"]
            for f in self.get_enabled_features()
            if not f.get("nullable", True)
        ]

    # --- Pipeline Configuration ---

    @property
    def default_resolution_km(self) -> int:
        return self._pipeline.get("default_resolution_km", 64)

    @property
    def supported_resolutions(self) -> list[int]:
        return self._pipeline.get("supported_resolutions_km", [64, 10, 1])

    @property
    def h3_resolution_map(self) -> dict[int, int]:
        """Map from km resolution to H3 resolution level."""
        raw = self._pipeline.get("h3_resolution_map", {64: 2, 10: 4, 1: 7})
        return {int(k): int(v) for k, v in raw.items()}

    def get_h3_resolution(self, km: int) -> int:
        """Get H3 resolution for a given km resolution."""
        h3_map = self.h3_resolution_map
        if km not in h3_map:
            raise ValueError(
                f"Resolution {km} km not supported. "
                f"Supported: {list(h3_map.keys())}"
            )
        return h3_map[km]

    @property
    def temporal_aggregation_hours(self) -> int:
        return self._pipeline.get("temporal_aggregation_hours", 6)

    @property
    def geographic_bboxes(self) -> dict[str, list[float]]:
        """Return bounding boxes for each geographic scope region."""
        scope = self._pipeline.get("geographic_scope", {})
        return {
            region: info["bbox"]
            for region, info in scope.items()
        }

    # --- Data Source Configuration ---

    def get_source_config(self, source: str) -> dict:
        """Return full configuration for a data source."""
        if source not in self._sources:
            raise KeyError(
                f"Unknown data source '{source}'. "
                f"Available: {list(self._sources.keys())}"
            )
        return self._sources[source]

    # --- Anomaly Detection Configuration ---

    @property
    def anomaly_config(self) -> dict:
        return self._anomaly

    @property
    def fire_season_months(self) -> list[int]:
        return self._anomaly.get("seasons", {}).get(
            "fire_season", {}
        ).get("months", [6, 7, 8, 9, 10, 11])

    @property
    def off_season_months(self) -> list[int]:
        return self._anomaly.get("seasons", {}).get(
            "off_season", {}
        ).get("months", [12, 1, 2, 3, 4, 5])

    def get_z_threshold(self, month: int) -> float:
        """Return the z-score anomaly threshold for a given month."""
        thresholds = self._anomaly.get("z_score_threshold", {})
        if month in self.fire_season_months:
            return thresholds.get("fire_season", 4.0)
        return thresholds.get("off_season", 3.5)

    # --- Storage Configuration ---

    @property
    def gcs_bucket(self) -> str:
        env_var = self._storage.get("gcs_bucket_env_var", "GCS_BUCKET_NAME")
        bucket = os.environ.get(env_var)
        if not bucket:
            raise EnvironmentError(
                f"Environment variable {env_var} not set. "
                f"Set it to your GCS bucket name."
            )
        return bucket

    def get_gcs_path(self, path_key: str) -> str:
        """Return GCS path for a storage category.

        Args:
            path_key: One of 'raw', 'processed', 'static', 'models'.
        """
        paths = self._storage.get("paths", {})
        if path_key not in paths:
            raise KeyError(f"Unknown storage path key '{path_key}'.")
        return f"gs://{self.gcs_bucket}/{paths[path_key]}"

    # --- Validation Configuration ---

    @property
    def max_null_rate(self) -> float:
        return self._validation.get("max_null_rate", 0.15)

    @property
    def row_count_tolerance_pct(self) -> int:
        return self._validation.get("row_count_tolerance_pct", 5)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------
def get_registry(config_path: Optional[str] = None) -> FeatureRegistry:
    """Factory function for obtaining a FeatureRegistry instance."""
    return FeatureRegistry(config_path)


def get_empty_dataframe(config_path: Optional[str] = None) -> pd.DataFrame:
    """Create an empty DataFrame with the correct schema.

    Useful for initializing DataFrames in processing tasks and for tests.
    """
    registry = get_registry(config_path)
    columns = registry.get_feature_names()
    dtypes = registry.get_dtype_map()
    df = pd.DataFrame(columns=columns)
    # Apply dtypes where possible (some require data to cast)
    for col, dtype in dtypes.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (TypeError, ValueError):
                pass  # Will be cast when data is present
    return df
