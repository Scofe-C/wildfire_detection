#!/usr/bin/env python3
"""
Combine Historical Data into Region CSVs
=========================================
Reads all partitioned Parquet files from historical_data/64km/ and writes:
  historical_data/california_historical.csv
  historical_data/texas_historical.csv

Sorted by timestamp ascending (oldest → newest) — required for time-based
train/test split in the model pipeline.

Usage:
    python combine_historical_data.py
    python combine_historical_data.py --resolution 64
    python combine_historical_data.py --resolution 64 --output-dir /some/other/path
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("combine")

SCRIPT_DIR     = Path(__file__).resolve().parent
MODEL_PIPELINE = SCRIPT_DIR.parent
HISTORICAL_DIR = MODEL_PIPELINE / "historical_data"
REGIONS        = ["california", "texas"]


def combine_region(region: str, resolution_km: int, output_dir: Path) -> Path:
    region_root = HISTORICAL_DIR / f"{resolution_km}km" / f"region={region}"

    if not region_root.exists():
        logger.warning("No data found for %s at %s", region, region_root)
        return None

    parquet_files = sorted(region_root.glob("**/*.parquet"))
    if not parquet_files:
        logger.warning("No parquet files under %s", region_root)
        return None

    logger.info("[%s] Found %d parquet files", region, len(parquet_files))

    dfs = []
    for f in parquet_files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            logger.warning("  Skipping %s: %s", f.name, e)

    if not dfs:
        logger.error("[%s] All files failed to read", region)
        return None

    combined = pd.concat(dfs, ignore_index=True)

    # Drop rows belonging to the other region that leaked in due to the full
    # grid being passed to fuse_features — these rows have null weather data
    # because weather was only fetched for this region's grid cells.
    weather_cols = [
        "temperature_2m", "relative_humidity_2m",
        "wind_speed_10m", "wind_direction_10m",
    ]
    present_weather = [c for c in weather_cols if c in combined.columns]
    if present_weather:
        before_filter = len(combined)
        combined = combined.dropna(subset=present_weather, how="all").reset_index(drop=True)
        dropped_cross = before_filter - len(combined)
        if dropped_cross:
            logger.info(
                "[%s] Dropped %d cross-region rows (null weather)", region, dropped_cross
            )

    # Sort by timestamp — critical for time-based train/test split
    if "timestamp" in combined.columns:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"])
        combined = combined.sort_values("timestamp").reset_index(drop=True)

    # Drop duplicate rows (same grid_id + timestamp from overlapping chunks)
    before = len(combined)
    dedup_cols = [c for c in ["grid_id", "timestamp"] if c in combined.columns]
    if dedup_cols:
        combined = combined.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    dropped = before - len(combined)
    if dropped:
        logger.info("[%s] Dropped %d duplicate rows", region, dropped)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{region}_historical.csv"
    combined.to_csv(out_path, index=False)

    fire_rows = int(combined["fire_detected_binary"].sum()) if "fire_detected_binary" in combined.columns else "?"
    fire_pct  = 100.0 * fire_rows / len(combined) if isinstance(fire_rows, int) else "?"

    logger.info(
        "[%s] Written: %d rows × %d cols | fire+ rows: %s (%.2f%%) → %s",
        region, len(combined), len(combined.columns),
        fire_rows, fire_pct, out_path,
    )

    if "timestamp" in combined.columns:
        logger.info(
            "[%s] Date range: %s → %s",
            region,
            combined["timestamp"].min().date(),
            combined["timestamp"].max().date(),
        )

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine historical parquet files into region CSVs")
    parser.add_argument("--resolution", default=64, type=int, help="Grid resolution in km (default: 64)")
    parser.add_argument("--output-dir", default=str(HISTORICAL_DIR), help="Output directory for CSVs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results = {}

    for region in REGIONS:
        path = combine_region(region, args.resolution, output_dir)
        results[region] = path

    logger.info("")
    logger.info("Output files:")
    for region, path in results.items():
        if path:
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info("  %-12s → %s  (%.1f MB)", region, path.name, size_mb)
        else:
            logger.info("  %-12s → SKIPPED (no data)", region)
