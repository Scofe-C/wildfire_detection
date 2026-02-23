from __future__ import annotations

from typing import Any, Dict, List, Tuple

import great_expectations as ge
import pandas as pd
import great_expectations as gx
from scripts.utils.grid_utils import generate_full_grid


_TYPE_MAP = {
    "float32": "float",
    "float64": "float",
    "int8": "int",
    "int16": "int",
    "int32": "int",
    "int64": "int",
    "string": "str",
    "bool": "bool",
}

_EXPECTED_GRID_COUNT_CACHE: Dict[int, int] = {}


def _get_expected_row_count(resolution_km: int) -> int:
    if resolution_km in _EXPECTED_GRID_COUNT_CACHE:
        return _EXPECTED_GRID_COUNT_CACHE[resolution_km]
    grid_gdf = generate_full_grid(resolution_km=resolution_km)
    expected = int(len(grid_gdf))
    _EXPECTED_GRID_COUNT_CACHE[resolution_km] = expected
    return expected


def run_validation(
    df: pd.DataFrame,
    registry,
    resolution_km: int,
    enforce_row_count: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Great Expectations validation gate.
    Returns (passed, results_dict) where results_dict contains issues list.
    """
    # Old (removed):
    validator = ge.from_pandas(df)

    feature_names: List[str] = registry.get_feature_names()
    dtype_map: Dict[str, str] = registry.get_dtype_map()
    rules_map: Dict[str, Dict[str, Any]] = registry.get_validation_rules()
    non_nullable: List[str] = registry.get_non_nullable_columns()

    max_null_rate: float = float(getattr(registry, "max_null_rate", 0.15))
    tol_pct: float = float(getattr(registry, "row_count_tolerance_pct", 5)) / 100.0

    # 1) column existence
    for col in feature_names:
        validator.expect_column_to_exist(col)

    # 2) types
    for col, dtype in dtype_map.items():
        if col not in feature_names:
            continue
        ge_type = _TYPE_MAP.get(str(dtype))
        if ge_type:
            validator.expect_column_values_to_be_of_type(col, ge_type)

    # 3) rules (min/max, allowed values)
    for col, rules in rules_map.items():
        if col not in feature_names:
            continue

        if ("min" in rules) or ("max" in rules):
            # Skip between-check for non-numeric columns (avoids str vs int TypeError)
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                validator.expect_column_values_to_be_between(
                    col,
                    min_value=rules.get("min"),
                    max_value=rules.get("max"),
                )

        if "allowed_values" in rules:
            validator.expect_column_values_to_be_in_set(
                col,
                list(rules["allowed_values"]),
            )

    # 4) null constraints
    for col in non_nullable:
        if col in feature_names:
            validator.expect_column_values_to_not_be_null(col, mostly=1.0)

    mostly = max(0.0, min(1.0, 1.0 - max_null_rate))
    for col in feature_names:
        if col in non_nullable:
            continue
        if col in df.columns:
            validator.expect_column_values_to_not_be_null(col, mostly=mostly)

    # 5) grid_id uniqueness (avoid GE error when all-null)
    if "grid_id" in df.columns:
        if df["grid_id"].notna().any():
            validator.expect_column_proportion_of_unique_values_to_be_between(
                "grid_id",
                min_value=0.99,
                max_value=1.0,
            )

    # 6) row count bounds (production gate; disable in unit tests)
    if enforce_row_count:
        expected = _get_expected_row_count(resolution_km)
        lo = int(expected * (1.0 - tol_pct))
        hi = int(expected * (1.0 + tol_pct))
        validator.expect_table_row_count_to_be_between(lo, hi)

    result = validator.validate(result_format="SUMMARY")
    passed = bool(result.get("success", False))

    issues: List[str] = []
    for r in result.get("results", []):
        if not r.get("success", True):
            exp = r.get("expectation_config", {}).get("expectation_type", "unknown_expectation")
            kw = r.get("expectation_config", {}).get("kwargs", {})
            col = kw.get("column", "")
            issues.append(f"{exp} failed column={col} kwargs={kw}")

    return passed, {"passed": passed, "issues": issues, "ge_summary": result}


# ---------------------------------------------------------------------------
# CLI entry point — used by dvc repro and for ad-hoc manual runs
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import logging
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Run Great Expectations schema validation on the fused feature dataset."
    )
    parser.add_argument(
        "--input",
        default="data/processed/fused",
        help="Path to Parquet file or directory of fused features.",
    )
    parser.add_argument(
        "--resolution-km",
        type=int,
        default=64,
        help="Grid resolution in km — used for row count validation (default: 64).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/baselines",
        help="Directory to write statistics summary JSON (default: data/processed/baselines).",
    )
    parser.add_argument(
        "--no-row-count",
        action="store_true",
        help="Disable row count enforcement (useful for partial/test datasets).",
    )
    args = parser.parse_args()

    import pandas as pd
    from scripts.utils.schema_loader import get_registry

    registry = get_registry()
    input_path = Path(args.input)

    if input_path.is_dir():
        parts = list(input_path.rglob("*.parquet"))
        if not parts:
            log.error(f"No Parquet files found in {input_path}")
            sys.exit(1)
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    else:
        df = pd.read_parquet(input_path)

    log.info(f"Validating {len(df):,} rows at resolution {args.resolution_km} km")

    passed, results = run_validation(
        df=df,
        registry=registry,
        resolution_km=args.resolution_km,
        enforce_row_count=not args.no_row_count,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write statistics summary alongside the GE results
    stats_path = output_dir / "stats_latest.json"
    summary = {
        "run_at": __import__("datetime").datetime.utcnow().isoformat(),
        "row_count": len(df),
        "resolution_km": args.resolution_km,
        "passed": passed,
        "issue_count": len(results.get("issues", [])),
        "issues": results.get("issues", []),
        "column_null_rates": {
            col: float(df[col].isna().mean())
            for col in df.columns
        },
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"Validation {'PASSED' if passed else 'FAILED'} — stats written to {stats_path}")

    if not passed:
        log.warning(f"Issues: {results.get('issues', [])[:5]}")
        # Non-zero exit so DVC marks the stage as failed, but pipeline continues
        # (Airflow task uses trigger_rule='all_done' on detect_anomalies)
        sys.exit(1)