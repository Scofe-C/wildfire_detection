from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd
from scripts.utils.grid_utils import generate_full_grid


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
    """Pandas-based schema validation. Returns (passed, results_dict)."""

    feature_names: List[str] = registry.get_feature_names()
    rules_map: Dict[str, Dict[str, Any]] = registry.get_validation_rules()
    non_nullable: List[str] = registry.get_non_nullable_columns()

    max_null_rate: float = float(getattr(registry, "max_null_rate", 0.15))
    tol_pct: float = float(getattr(registry, "row_count_tolerance_pct", 5)) / 100.0

    errors: List[str] = []
    warnings: List[str] = []

    # 1) Column existence — ERROR: missing columns break downstream tasks
    for col in feature_names:
        if col not in df.columns:
            errors.append(f"missing_column: {col}")

    # 2) Null constraints on required columns — ERROR
    for col in non_nullable:
        if col not in df.columns:
            continue
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            errors.append(f"non_nullable_has_nulls: column={col} null_count={null_count}")

    # 3) Null rate on optional columns — WARNING (real data often has gaps)
    _SKIP_NULL_CHECK = {"fire_weather_index", "ndvi"}
    for col in feature_names:
        if col in non_nullable or col in _SKIP_NULL_CHECK:
            continue
        if col not in df.columns:
            continue
        null_rate = float(df[col].isna().mean())
        if null_rate > max_null_rate:
            warnings.append(
                f"high_null_rate: column={col} null_rate={null_rate:.2%} threshold={max_null_rate:.2%}"
            )

    # 4) Range rules (min/max) — WARNING (outliers shouldn't block the pipeline)
    for col, rules in rules_map.items():
        if col not in feature_names or col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        if "min" in rules and float(series.min()) < float(rules["min"]):
            warnings.append(
                f"below_min: column={col} min_found={series.min()} min_allowed={rules['min']}"
            )
        if "max" in rules and float(series.max()) > float(rules["max"]):
            warnings.append(
                f"above_max: column={col} max_found={series.max()} max_allowed={rules['max']}"
            )
        if "allowed_values" in rules:
            invalid = set(df[col].dropna().unique()) - set(rules["allowed_values"])
            if invalid:
                warnings.append(f"invalid_values: column={col} values={invalid}")

    # 5) grid_id uniqueness — WARNING
    if "grid_id" in df.columns and df["grid_id"].notna().any():
        total = len(df)
        unique = df["grid_id"].nunique()
        if total > 0 and (unique / total) < 0.99:
            warnings.append(
                f"low_grid_id_uniqueness: unique={unique} total={total} ratio={unique/total:.2%}"
            )

    # 6) Row count bounds — WARNING (tolerance already generous)
    if enforce_row_count:
        expected = _get_expected_row_count(resolution_km)
        lo = int(expected * (1.0 - tol_pct))
        hi = int(expected * (1.0 + tol_pct))
        actual = len(df)
        if not (lo <= actual <= hi):
            warnings.append(
                f"row_count_out_of_bounds: actual={actual} expected={expected} "
                f"allowed=[{lo}, {hi}]"
            )

    # Only structural errors (missing columns, non-nullable nulls) block the pipeline.
    # Range/null-rate/row-count issues are warnings — logged but not blocking.
    passed = len(errors) == 0
    return passed, {"passed": passed, "errors": errors, "warnings": warnings,
                     "issues": errors + warnings}


# ---------------------------------------------------------------------------
# CLI entry point
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
        description="Run pandas schema validation on the fused feature dataset."
    )
    parser.add_argument("--input", default="data/processed/fused")
    parser.add_argument("--resolution-km", type=int, default=64)
    parser.add_argument("--output-dir", default="data/processed/baselines")
    parser.add_argument("--no-row-count", action="store_true")
    args = parser.parse_args()

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
    stats_path = output_dir / "stats_latest.json"
    errors = results.get("errors", [])
    warnings = results.get("warnings", [])
    summary = {
        "run_at": __import__("datetime").datetime.utcnow().isoformat(),
        "row_count": len(df),
        "resolution_km": args.resolution_km,
        "passed": passed,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "issues": errors + warnings,
        "column_null_rates": {col: float(df[col].isna().mean()) for col in df.columns},
    }
    with open(stats_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"Validation {'PASSED' if passed else 'FAILED'} — stats written to {stats_path}")
    if errors:
        log.error(f"Errors (blocking): {errors[:5]}")
        sys.exit(1)
    if warnings:
        log.warning(f"Warnings (non-blocking): {warnings[:5]}")
