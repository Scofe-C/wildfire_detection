"""
Historical Backfill Script
===========================
Generates 2 years of historical fused feature data for ML model training.

Loops through 6-hour windows from 2023-01-01 to 2025-01-31, calling the
existing fusion pipeline with each execution_date.  Supports resume by
skipping windows that already have output files.

Usage:
    python -m scripts.backfill.historical_backfill
    python -m scripts.backfill.historical_backfill --start 2024-01-01 --end 2024-06-30
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# Project root (Data-Pipeline/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Defaults from schema_config.yaml ml_training.backfill
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2025-01-31"
DEFAULT_FREQ_HOURS = 6
DEFAULT_RESOLUTION_KM = 64
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "backfill"


def generate_backfill_dates(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    freq_hours: int = DEFAULT_FREQ_HOURS,
) -> list[pd.Timestamp]:
    """Generate list of 6-hour window timestamps for backfill."""
    return list(pd.date_range(start=start, end=end, freq=f"{freq_hours}h"))


def _output_path_for_window(
    execution_date: pd.Timestamp,
    output_dir: Path,
    resolution_km: int,
) -> Path:
    """Deterministic output path for a single 6-hour window."""
    date_str = execution_date.strftime("%Y-%m-%d")
    hour_str = execution_date.strftime("%H")
    year = execution_date.strftime("%Y")
    month = execution_date.strftime("%m")

    out_dir = output_dir / f"{resolution_km}km" / f"year={year}" / f"month={month}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"features_{date_str}_{hour_str}00.parquet"


def run_backfill(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    freq_hours: int = DEFAULT_FREQ_HOURS,
    resolution_km: int = DEFAULT_RESOLUTION_KM,
    output_dir: Optional[str] = None,
    skip_existing: bool = True,
) -> dict:
    """Run the historical backfill pipeline.

    For each 6-hour window in [start, end]:
      1. Call ingest + process for FIRMS (historical CSVs)
      2. Call ingest + process for weather (Open-Meteo archive API)
      3. Fuse features
      4. Apply temporal lag (ML-ready output)
      5. Write parquet

    Args:
        start: Backfill start date (YYYY-MM-DD).
        end: Backfill end date (YYYY-MM-DD).
        freq_hours: Window frequency in hours.
        resolution_km: Grid resolution.
        output_dir: Override output directory.
        skip_existing: Skip windows that already have output files.

    Returns:
        Dict with counts of processed, skipped, and failed windows.
    """
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    dates = generate_backfill_dates(start, end, freq_hours)

    stats = {"total": len(dates), "processed": 0, "skipped": 0, "failed": 0}
    logger.info(
        f"Backfill: {stats['total']} windows from {start} to {end} "
        f"({freq_hours}h intervals, {resolution_km}km resolution)"
    )

    for i, exec_date in enumerate(dates):
        out_path = _output_path_for_window(exec_date, out_dir, resolution_km)

        # Resume support: skip existing files
        if skip_existing and out_path.exists():
            stats["skipped"] += 1
            continue

        try:
            _run_single_window(exec_date, resolution_km, out_path)
            stats["processed"] += 1

            if (i + 1) % 100 == 0:
                logger.info(
                    f"Progress: {i + 1}/{stats['total']} "
                    f"(processed={stats['processed']}, "
                    f"skipped={stats['skipped']}, "
                    f"failed={stats['failed']})"
                )
        except Exception as e:
            stats["failed"] += 1
            logger.error(f"Window {exec_date} failed: {e}")

    logger.info(f"Backfill complete: {stats}")
    return stats


def _run_single_window(
    execution_date: pd.Timestamp,
    resolution_km: int,
    output_path: Path,
) -> None:
    """Process a single 6-hour window through the full pipeline.

    This calls the existing pipeline components (ingest → process → fuse)
    with the given execution_date.  In production this reads from the
    Open-Meteo historical archive API and FIRMS annual CSVs.

    Note:
        This is a skeleton that delegates to the existing pipeline modules.
        Full implementation requires FIRMS annual CSV downloads and
        Open-Meteo archive API integration (see ml_llm_readiness_and_plan.md
        Part 2, Steps 1-4).
    """
    from scripts.fusion.fuse_features import fuse_features_for_ml

    logger.debug(f"Processing window: {execution_date}")

    # --- Placeholder: load historical data for this window ---
    # In production, this would:
    #   1. Filter FIRMS annual CSV to this 6h window
    #   2. Call Open-Meteo archive API for weather
    #   3. Load static features from cache
    # For now, create empty DataFrames as stubs.
    firms_df = pd.DataFrame(columns=[
        "grid_id", "active_fire_count", "mean_frp", "median_frp",
        "max_confidence", "nearest_fire_distance_km", "fire_detected_binary",
    ])
    weather_df = pd.DataFrame(columns=["grid_id", "timestamp"])
    static_df = pd.DataFrame(columns=["grid_id"])

    # Previous window fire features for temporal lag
    prev_date = execution_date - pd.Timedelta(hours=6)
    prev_fire_df = None  # Would load from previous window's output

    ml_fused = fuse_features_for_ml(
        firms_features=firms_df,
        weather_features=weather_df,
        static_features=static_df,
        execution_date=execution_date,
        prev_fire_features=prev_fire_df,
        resolution_km=resolution_km,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ml_fused.to_parquet(output_path, index=False)
    logger.debug(f"Wrote {len(ml_fused)} rows → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run historical backfill for ML training data"
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=DEFAULT_END, help="End date (YYYY-MM-DD)")
    parser.add_argument("--freq", type=int, default=DEFAULT_FREQ_HOURS, help="Window freq (hours)")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION_KM, help="Grid resolution (km)")
    parser.add_argument("--output-dir", default=None, help="Output directory override")
    parser.add_argument("--no-skip", action="store_true", help="Reprocess existing windows")

    args = parser.parse_args()
    run_backfill(
        start=args.start,
        end=args.end,
        freq_hours=args.freq,
        resolution_km=args.resolution,
        output_dir=args.output_dir,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()
