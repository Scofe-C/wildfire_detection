from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import DataLoadError, load_backfill, split_features_target
from src.data.schema import FeatureSchema, FeatureSpec, load_schema, validate_dataframe


@pytest.fixture
def schema():
    return FeatureSchema(
        version="1.0.0",
        index_columns=[
            {"name": "h3_index", "dtype": "string"},
            {"name": "timestamp", "dtype": "datetime64[ns]"},
        ],
        target={"name": "fire_detected", "dtype": "int8"},
        features=[
            FeatureSpec("frp_max", "float32", "firms", (0.0, 5000.0), True),
            FeatureSpec("temperature_max", "float32", "weather", (200.0, 350.0), True),
            FeatureSpec("soil_moisture_0_5cm", "float32", "smap", (0.0, 1.0), False),
        ],
        normalization_stats_path="",
    )


@pytest.fixture
def valid_df():
    n = 100
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "h3_index": ["8828308281fffff"] * n,
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="6h"),
        "fire_detected": rng.integers(0, 2, size=n).astype(np.int8),
        "frp_max": rng.uniform(0, 100, size=n).astype(np.float32),
        "temperature_max": rng.uniform(270, 320, size=n).astype(np.float32),
    })


class TestSchemaValidation:
    def test_valid_passes(self, schema, valid_df):
        assert validate_dataframe(valid_df, schema) == []

    def test_missing_required_feature(self, schema, valid_df):
        df = valid_df.drop(columns=["frp_max"])
        errors = validate_dataframe(df, schema)
        assert any("frp_max" in e for e in errors)

    def test_missing_optional_ok(self, schema, valid_df):
        assert validate_dataframe(valid_df, schema) == []

    def test_out_of_bounds(self, schema, valid_df):
        df = valid_df.copy()
        df.loc[0, "frp_max"] = 99999.0
        errors = validate_dataframe(df, schema)
        assert any("frp_max" in e and "out of bounds" in e for e in errors)

    def test_missing_target(self, schema, valid_df):
        df = valid_df.drop(columns=["fire_detected"])
        assert any("fire_detected" in e for e in validate_dataframe(df, schema))

    def test_missing_index(self, schema, valid_df):
        df = valid_df.drop(columns=["h3_index"])
        assert any("h3_index" in e for e in validate_dataframe(df, schema))


class TestDataLoader:
    def test_missing_dir(self, tmp_path):
        with pytest.raises(DataLoadError, match="not found"):
            load_backfill(tmp_path / "nope")

    def test_empty_dir(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(DataLoadError, match="No parquet"):
            load_backfill(d)

    def test_load_valid(self, tmp_path, valid_df, schema):
        d = tmp_path / "backfill"
        d.mkdir()
        valid_df.to_parquet(d / "test.parquet")
        df = load_backfill(d, schema=schema)
        assert len(df) == len(valid_df)

    def test_split(self, valid_df, schema):
        X, y, meta = split_features_target(valid_df, schema)
        assert "frp_max" in X.columns
        assert "fire_detected" not in X.columns
        assert "h3_index" in meta.columns
        assert len(y) == len(valid_df)


class TestSchemaLoading:
    def test_load_config(self):
        cfg = Path(__file__).resolve().parents[1] / "configs" / "feature_schema.yaml"
        if cfg.exists():
            s = load_schema(cfg)
            assert s.version == "2.0.0"
            assert len(s.features) > 0

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_schema("/no/such/file.yaml")
