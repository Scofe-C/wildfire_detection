"""
Quick end-to-end pipeline smoke test
=====================================
Runs one 6-hour window of the full pipeline (ingest → process → fuse) and
prints a summary of the fused feature table.

Usage:
    python -m scripts.utils.run_pipeline_once
    python -m scripts.utils.run_pipeline_once --date 2024-07-15T12:00 --resolution-km 64
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline_smoke_test")

BASE_DIR   = Path(__file__).resolve().parents[2]
STATIC_DIR = BASE_DIR / "data" / "static"
OUT_DIR    = BASE_DIR / "data" / "processed" / "smoke_test"


def run(date_str: str = "2024-07-15T12:00", resolution_km: int = 64) -> pd.DataFrame:
    execution_date = pd.Timestamp(date_str)
    logger.info("=== Pipeline smoke test | %s | %d km ===", execution_date, resolution_km)

    from scripts.utils.grid_utils import generate_full_grid
    from scripts.ingestion.ingest_firms import fetch_firms_data
    from scripts.ingestion.ingest_weather import fetch_weather_data
    from scripts.processing.process_firms import process_firms_data
    from scripts.processing.process_weather import process_weather_data
    from scripts.processing.process_static import load_and_process_static

    # Build grid centroids (needed by weather fetcher)
    grid = generate_full_grid(resolution_km)
    grid_centroids = grid[["grid_id", "latitude", "longitude"]]

    # ── 1. Ingest ──────────────────────────────────────────────────────────────
    logger.info("Fetching FIRMS data …")
    firms_csv = fetch_firms_data(execution_date, resolution_km)

    logger.info("Fetching weather data …")
    weather_csv = fetch_weather_data(grid_centroids, execution_date)

    # ── 2. Process ─────────────────────────────────────────────────────────────
    logger.info("Processing FIRMS …")
    firms_feat = process_firms_data(str(firms_csv), resolution_km)

    logger.info("Processing weather …")
    weather_feat = process_weather_data(str(weather_csv), resolution_km)

    logger.info("Loading static features …")
    static_path = load_and_process_static(resolution_km, str(STATIC_DIR))
    static_feat = pd.read_parquet(static_path)

    # ── 3. Fuse ────────────────────────────────────────────────────────────────
    from scripts.fusion.fuse_features import fuse_features

    logger.info("Fusing all features …")
    fused = fuse_features(
        firms_feat, weather_feat, static_feat,
        execution_date, resolution_km,
    )

    # ── 4. Output ──────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"fused_{execution_date:%Y%m%dT%H%M}_{resolution_km}km.parquet"
    fused.to_parquet(out_path, index=False)
    logger.info("Wrote fused output: %s", out_path)

    # ── 5. Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  FUSED FEATURE TABLE  |  {execution_date}  |  {resolution_km} km")
    print("=" * 70)
    print(f"  Shape        : {fused.shape[0]} rows × {fused.shape[1]} columns")
    print(f"  Columns      : {list(fused.columns)}")
    print()

    # NaN audit per column
    nan_pct = fused.isnull().mean().mul(100).round(1)
    print("  NaN % per column:")
    for col, pct in nan_pct.items():
        flag = "  ** MISSING" if pct == 100 else ("  ** partial" if pct > 0 else "")
        print(f"    {col:<35} {pct:5.1f}%{flag}")

    print()
    print("  Numeric summary (non-NaN columns):")
    numeric_cols = fused.select_dtypes("number").columns
    filled = [c for c in numeric_cols if fused[c].notna().any()]
    if filled:
        print(fused[filled].describe().T[["mean", "min", "max"]].to_string())

    print()
    print("  Fire detections this window:")
    fire_cells = fused[fused["fire_detected_binary"] == 1]
    print(f"    {len(fire_cells)} cell(s) with active fire")
    if not fire_cells.empty:
        print(fire_cells[["grid_id", "latitude", "longitude",
                           "active_fire_count", "max_frp" if "max_frp" in fused.columns else "mean_frp"]
                         ].to_string(index=False))

    print("=" * 70)
    print(f"  Output saved to: {out_path}")
    print("=" * 70 + "\n")

    return fused


def _parse_args():
    p = argparse.ArgumentParser(description="End-to-end pipeline smoke test")
    p.add_argument("--date", default="2024-07-15T12:00",
                   help="Execution timestamp (default: 2024-07-15T12:00)")
    p.add_argument("--resolution-km", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    fused = run(args.date, args.resolution_km)
    sys.exit(0)
