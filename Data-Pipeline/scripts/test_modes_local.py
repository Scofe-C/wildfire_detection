#!/usr/bin/env python3
"""
Local Pipeline Mode Tester
==========================
Test quiet, active, and emergency modes WITHOUT Airflow.
Calls the same ingestion/processing/fusion functions the DAG uses.

Usage:
    python scripts/test_modes_local.py --mode quiet
    python scripts/test_modes_local.py --mode active
    python scripts/test_modes_local.py --mode emergency
    python scripts/test_modes_local.py --mode all

Each mode runs: FIRMS → Weather → Process FIRMS → Process Weather → Fuse → Validate
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is in Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mode_tester")

# ---------------------------------------------------------------------------
# Paths (mirrors wildfire_dag.py)
# ---------------------------------------------------------------------------
DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
STATIC_DIR    = DATA_DIR / "static"

REGIONS = {
    "california": {"bbox": [-124.48, 32.53, -114.13, 42.01]},
    "texas":      {"bbox": [-106.65, 25.84,  -93.51, 36.50]},
}

# ---------------------------------------------------------------------------
# Mode configurations (mirrors DAG params)
# ---------------------------------------------------------------------------
MODE_CONFIGS = {
    "quiet": {
        "trigger_source": "cron",
        "mode": "quiet",
        "resolution_km": 64,
        "fire_cells": [],
        "fire_frp_mw": 0.0,
        "weather_lookback_hours": 24,
        "regions": [],  # all regions
    },
    "active": {
        "trigger_source": "watchdog_active",
        "mode": "active",
        "resolution_km": 22,
        "fire_cells": ["8e283082ddbffff"],  # sample H3 cell (California)
        "fire_frp_mw": 45.0,
        "weather_lookback_hours": 2,
        "regions": ["california"],
    },
    "emergency": {
        "trigger_source": "watchdog_emergency",
        "mode": "emergency",
        "resolution_km": 22,
        "fire_cells": ["8e283082ddbffff"],
        "fire_frp_mw": 120.0,
        "weather_lookback_hours": 2,
        "regions": ["california"],
    },
}


def run_pipeline(mode_name: str) -> dict:
    """Run the full pipeline locally for a given mode.

    Returns dict with timing and status for each step.
    """
    config = MODE_CONFIGS[mode_name]
    results = {}
    execution_date = datetime.now(timezone.utc)
    resolution_km = config["resolution_km"]
    regions = config["regions"] or list(REGIONS.keys())

    logger.info("=" * 60)
    logger.info(f"MODE: {mode_name.upper()}")
    logger.info(f"  trigger_source: {config['trigger_source']}")
    logger.info(f"  resolution_km:  {resolution_km}")
    logger.info(f"  regions:        {regions}")
    logger.info(f"  fire_cells:     {config['fire_cells']}")
    logger.info("=" * 60)

    total_start = time.time()

    # --- Step 1: FIRMS Ingestion ---
    for region in regions:
        step_name = f"firms_{region}"
        logger.info(f"\n--- Step: FIRMS ingestion ({region}) ---")
        t0 = time.time()
        try:
            from scripts.ingestion.ingest_firms import fetch_firms_data

            firms_path = fetch_firms_data(
                execution_date=execution_date,
                resolution_km=resolution_km,
                lookback_hours=24,
                output_dir=str(RAW_DIR / "firms"),
                region=region,
            )
            results[step_name] = {"status": "OK", "time": time.time() - t0, "path": str(firms_path)}
            logger.info(f"  ✓ {step_name}: {time.time() - t0:.1f}s → {firms_path}")
        except Exception as e:
            results[step_name] = {"status": "FAIL", "time": time.time() - t0, "error": str(e)}
            logger.error(f"  ✗ {step_name}: {e}")

    # --- Step 2: Weather Ingestion ---
    for region in regions:
        step_name = f"weather_{region}"
        logger.info(f"\n--- Step: Weather ingestion ({region}) ---")
        t0 = time.time()
        try:
            from scripts.ingestion.ingest_weather import fetch_weather_data
            from scripts.utils.grid_utils import generate_grid_for_bbox

            bbox = REGIONS[region]["bbox"]
            grid = generate_grid_for_bbox(bbox, resolution_km)
            grid_centroids = grid[["grid_id", "latitude", "longitude"]]

            weather_path = fetch_weather_data(
                grid_centroids=grid_centroids,
                execution_date=execution_date,
                lookback_hours=config["weather_lookback_hours"],
                output_dir=str(RAW_DIR / "weather"),
                trigger_source=config["trigger_source"],
                fire_cells=config["fire_cells"] or None,
                h3_ring_max=5,
                region=region,
            )
            results[step_name] = {"status": "OK", "time": time.time() - t0, "path": str(weather_path)}
            logger.info(f"  ✓ {step_name}: {time.time() - t0:.1f}s → {weather_path}")
        except Exception as e:
            results[step_name] = {"status": "FAIL", "time": time.time() - t0, "error": str(e)}
            logger.error(f"  ✗ {step_name}: {e}")

    # --- Step 3: Process FIRMS ---
    for region in regions:
        step_name = f"process_firms_{region}"
        logger.info(f"\n--- Step: Process FIRMS ({region}) ---")
        t0 = time.time()
        try:
            from scripts.processing.process_firms import process_firms_data
            import shutil

            firms_result = results.get(f"firms_{region}", {})
            raw_path = firms_result.get("path")
            if not raw_path:
                raise FileNotFoundError("No FIRMS raw data available")

            firms_features = process_firms_data(
                raw_csv_path=raw_path,
                resolution_km=resolution_km,
            )

            latest_path = PROCESSED_DIR / "firms" / f"firms_features_{region}_latest.parquet"
            previous_path = PROCESSED_DIR / "firms" / f"firms_features_{region}_previous.parquet"
            latest_path.parent.mkdir(parents=True, exist_ok=True)

            if latest_path.exists():
                shutil.copy2(str(latest_path), str(previous_path))

            firms_features.to_parquet(latest_path, index=False)
            results[step_name] = {"status": "OK", "time": time.time() - t0, "rows": len(firms_features)}
            logger.info(f"  ✓ {step_name}: {time.time() - t0:.1f}s, {len(firms_features)} rows")
        except Exception as e:
            results[step_name] = {"status": "FAIL", "time": time.time() - t0, "error": str(e)}
            logger.error(f"  ✗ {step_name}: {e}")

    # --- Step 4: Process Weather ---
    for region in regions:
        step_name = f"process_weather_{region}"
        logger.info(f"\n--- Step: Process Weather ({region}) ---")
        t0 = time.time()
        try:
            from scripts.processing.process_weather import process_weather_data

            weather_result = results.get(f"weather_{region}", {})
            raw_path = weather_result.get("path")
            if not raw_path:
                raise FileNotFoundError("No weather raw data available")

            weather_features = process_weather_data(
                raw_csv_path=raw_path,
                resolution_km=resolution_km,
            )

            output_path = PROCESSED_DIR / "weather" / f"weather_features_{region}_latest.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            weather_features.to_parquet(output_path, index=False)

            results[step_name] = {"status": "OK", "time": time.time() - t0, "rows": len(weather_features)}
            logger.info(f"  ✓ {step_name}: {time.time() - t0:.1f}s, {len(weather_features)} rows")
        except Exception as e:
            results[step_name] = {"status": "FAIL", "time": time.time() - t0, "error": str(e)}
            logger.error(f"  ✗ {step_name}: {e}")

    # --- Step 5: Fusion ---
    logger.info("\n--- Step: Feature Fusion ---")
    t0 = time.time()
    try:
        from scripts.fusion.fuse_features import fuse_features
        import pandas as pd

        firms_dfs, weather_dfs = [], []
        for region in regions:
            fp = PROCESSED_DIR / "firms" / f"firms_features_{region}_latest.parquet"
            wp = PROCESSED_DIR / "weather" / f"weather_features_{region}_latest.parquet"
            if fp.exists():
                df = pd.read_parquet(fp)
                df["region"] = region
                firms_dfs.append(df)
            if wp.exists():
                weather_dfs.append(pd.read_parquet(wp))

        firms_df   = pd.concat(firms_dfs, ignore_index=True)   if firms_dfs   else pd.DataFrame()
        weather_df = pd.concat(weather_dfs, ignore_index=True) if weather_dfs else pd.DataFrame()

        static_path = STATIC_DIR / f"static_features_{resolution_km}km.parquet"
        static_df = pd.read_parquet(static_path) if static_path.exists() else pd.DataFrame()

        fused = fuse_features(
            firms_features=firms_df,
            weather_features=weather_df,
            static_features=static_df,
            execution_date=pd.Timestamp(execution_date),
            resolution_km=resolution_km,
        )

        output_path = PROCESSED_DIR / "fused" / "fused_features_latest.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fused.to_parquet(output_path, index=False)

        results["fusion"] = {"status": "OK", "time": time.time() - t0, "rows": len(fused)}
        logger.info(f"  ✓ fusion: {time.time() - t0:.1f}s, {len(fused)} rows → {output_path}")
    except Exception as e:
        results["fusion"] = {"status": "FAIL", "time": time.time() - t0, "error": str(e)}
        logger.error(f"  ✗ fusion: {e}")

    # --- Step 6: Schema Validation ---
    logger.info("\n--- Step: Schema Validation ---")
    t0 = time.time()
    try:
        fused_path = PROCESSED_DIR / "fused" / "fused_features_latest.parquet"
        if fused_path.exists():
            import pandas as pd
            from scripts.utils.schema_loader import get_registry
            from scripts.validation.validate_schema import run_validation

            fused_df = pd.read_parquet(fused_path)
            registry = get_registry()
            passed, validation_results = run_validation(fused_df, registry, resolution_km=resolution_km)

            status = "OK" if passed else "WARN"
            issues = validation_results.get("issues", [])
            results["validation"] = {"status": status, "time": time.time() - t0, "issues": len(issues)}
            logger.info(f"  {'✓' if passed else '⚠'} validation: {time.time() - t0:.1f}s, {len(issues)} issues")
        else:
            results["validation"] = {"status": "SKIP", "time": 0, "reason": "No fused data"}
    except Exception as e:
        results["validation"] = {"status": "FAIL", "time": time.time() - t0, "error": str(e)}
        logger.error(f"  ✗ validation: {e}")

    total_time = time.time() - total_start

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info(f"MODE {mode_name.upper()} COMPLETE — {total_time:.1f}s total")
    logger.info("=" * 60)

    ok_count   = sum(1 for v in results.values() if v["status"] == "OK")
    fail_count = sum(1 for v in results.values() if v["status"] == "FAIL")
    warn_count = sum(1 for v in results.values() if v["status"] == "WARN")

    for step, info in results.items():
        icon = {"OK": "✓", "FAIL": "✗", "WARN": "⚠", "SKIP": "–"}.get(info["status"], "?")
        logger.info(f"  {icon} {step}: {info['status']} ({info.get('time', 0):.1f}s)")

    logger.info(f"\n  Result: {ok_count} OK, {warn_count} warnings, {fail_count} failures")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test pipeline modes locally (no Airflow)")
    parser.add_argument(
        "--mode", choices=["quiet", "active", "emergency", "all"],
        default="quiet",
        help="Which mode to test (default: quiet)",
    )
    args = parser.parse_args()

    if args.mode == "all":
        all_results = {}
        for mode in ["quiet", "active", "emergency"]:
            all_results[mode] = run_pipeline(mode)
            logger.info("\n")

        # Final summary across all modes
        logger.info("=" * 60)
        logger.info("ALL MODES SUMMARY")
        logger.info("=" * 60)
        for mode, results in all_results.items():
            ok    = sum(1 for v in results.values() if v["status"] == "OK")
            fail  = sum(1 for v in results.values() if v["status"] == "FAIL")
            total = sum(v.get("time", 0) for v in results.values())
            icon = "✓" if fail == 0 else "✗"
            logger.info(f"  {icon} {mode.upper():12s}: {ok} OK, {fail} FAIL, {total:.1f}s")
    else:
        run_pipeline(args.mode)


if __name__ == "__main__":
    main()
