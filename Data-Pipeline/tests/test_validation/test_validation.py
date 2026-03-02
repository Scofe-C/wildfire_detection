import pandas as pd

from scripts.utils.schema_loader import FeatureRegistry
from scripts.validation.validate_schema import run_validation


def _get_enabled_column_names(registry):
    feats = registry.get_enabled_features()
    if len(feats) == 0:
        return []
    if isinstance(feats[0], dict):
        return [f["name"] for f in feats if "name" in f]
    return list(feats)


def _choose_default_value(col: str, dtype: str, rules: dict | None):
    rules = rules or {}

    # If allowed_values exists, pick first allowed value
    if "allowed_values" in rules and rules["allowed_values"]:
        return rules["allowed_values"][0]

    # Special-case geo coordinates to satisfy min/max bounds
    if col == "latitude":
        return 34.0
    if col == "longitude":
        return -118.0

    # Special-case categorical code ranges (schema min/max)
    if col == "fuel_model_fbfm40":
        return 91  # min in your schema

    dtype = str(dtype)
    if dtype in ("float32", "float64"):
        return 0.0
    if dtype in ("int8", "int16", "int32", "int64"):
        return 0
    if dtype == "bool":
        return False
    return "x"


def _make_minimal_df(registry, cols, n_rows=2):
    dtype_map = registry.get_dtype_map()
    rules_map = registry.get_validation_rules()

    data = {}
    for c in cols:
        if c == "grid_id":
            data[c] = [f"cell_{i}" for i in range(n_rows)]
            continue
        dtype = dtype_map.get(c, "float32")
        rules = rules_map.get(c, {})
        v = _choose_default_value(c, dtype, rules)
        data[c] = [v] * n_rows

    df = pd.DataFrame(data)

    # Enforce numeric dtypes
    for c in cols:
        if c == "grid_id":
            continue
        dtype = str(dtype_map.get(c, ""))
        if dtype in ("float32", "float64"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        elif dtype in ("int8", "int16", "int32", "int64"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        elif dtype == "bool":
            df[c] = df[c].astype(bool)

    return df


def test_validation_passes_on_minimal_valid_df():
    registry = FeatureRegistry("configs/schema_config.yaml")
    cols = _get_enabled_column_names(registry)

    df = _make_minimal_df(registry, cols, n_rows=2)

    passed, results = run_validation(df, registry, resolution_km=10, enforce_row_count=False)
    assert passed is True, results.get("issues", [])


def test_validation_catches_out_of_range_values():
    registry = FeatureRegistry("configs/schema_config.yaml")
    cols = _get_enabled_column_names(registry)

    df = _make_minimal_df(registry, cols, n_rows=1)

    if "temperature_2m" in df.columns:
        df["temperature_2m"] = [999.0]

    passed, results = run_validation(df, registry, resolution_km=10, enforce_row_count=False)
    assert passed is False
    assert len(results.get("issues", [])) > 0


def test_validation_catches_null_non_nullable():
    registry = FeatureRegistry("configs/schema_config.yaml")
    cols = _get_enabled_column_names(registry)

    df = _make_minimal_df(registry, cols, n_rows=1)

    if "grid_id" in df.columns:
        df["grid_id"] = [None]

    passed, results = run_validation(df, registry, resolution_km=10, enforce_row_count=False)
    assert passed is False
    assert len(results.get("issues", [])) > 0
