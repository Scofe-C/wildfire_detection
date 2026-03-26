"""
Historical Backfill
===================
Replays the wildfire detection pipeline over a historical date range to
produce the training dataset used for ML model development.

For each 6-hour window (configurable), the script:
  1. Fetches FIRMS active fire detections (VIIRS + MODIS, region-sharded)
  2. Fetches Open-Meteo weather for all grid centroids
  3. Processes FIRMS detections into grid-level fire features
  4. Processes weather into grid-level weather features
  5. Loads pre-computed static features (LANDFIRE, SRTM, NDVI)
  6. Fuses all features into the unified schema
  7. Writes partitioned parquet to data/processed/backfill/

Resume support:
  With --resume (default), windows whose output files already exist on disk
  are skipped.  Delete or move the output directory to force a full rerun.

DVC tracking:
  After a successful backfill run, add DVC stages for historical_backfill,
  bias_analysis, and train_ignition_model as described in
  missing_sources_and_todo.md §4.

Usage:
    python -m scripts.backfill.historical_backfill \\
        --start 2020-01-01 \\
        --end   2025-12-31 \\
        --resolution-km 64

    # Dry run (print windows without fetching data):
    python -m scripts.backfill.historical_backfill \\
        --start 2024-01-01 \\
        --end   2024-01-02 \\
        --dry-run
"""

from __future__ import annotations

import argparse
import logging
import time
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Default settings (match schema_config.yaml backfill section)
DEFAULT_START        = "2023-01-01"
DEFAULT_END          = "2025-01-31"
DEFAULT_FREQ_HOURS   = 6
DEFAULT_RESOLUTION   = 64
DEFAULT_OUTPUT_DIR   = "data/processed/backfill"
DEFAULT_REGIONS      = ["california", "texas"]
DEFAULT_RAW_DIR      = "data/raw"
DEFAULT_STATIC_DIR   = "data/static"


# ---------------------------------------------------------------------------
# Output path helper
# ---------------------------------------------------------------------------

def _window_output_paths(
    output_dir: Path,
    t: pd.Timestamp,
    regions: list[str],
) -> list[Path]:
    """Return the expected parquet paths for all regions at time t."""
    return [
        output_dir
        / f"region={region}"
        / f"year={t.year}"
        / f"month={t.month:02d}"
        / f"features_{t:%Y%m%dT%H%M}.parquet"
        for region in regions
    ]


def _all_exist(paths: list[Path]) -> bool:
    return all(p.exists() for p in paths)


# ---------------------------------------------------------------------------
# Single window processing
# ---------------------------------------------------------------------------

def _process_window(
    t: pd.Timestamp,
    resolution_km: int,
    regions: list[str],
    output_dir: Path,
    raw_dir: str,
    static_dir: str,
) -> None:
    """Fetch, process, fuse, and write one 6-hour window.

    Raises on any unrecoverable error so the caller can track failures.
    """
    from scripts.ingestion.ingest_firms import fetch_firms_data
    from scripts.ingestion.ingest_weather import fetch_weather_data
    from scripts.processing.process_firms import process_firms_data
    from scripts.processing.process_weather import process_weather_data
    from scripts.processing.process_static import load_and_process_static
    from scripts.fusion.fuse_features import fuse_features
    from scripts.utils.grid_utils import generate_full_grid

    # Load static features (shared across regions; cached)
    static_path = load_and_process_static(resolution_km, static_dir)
    static_df   = pd.read_parquet(static_path)

    # Load full grid for weather centroids
    grid    = generate_full_grid(resolution_km)
    grid_df = grid[["grid_id", "latitude", "longitude"]]

    firms_dfs:   list[pd.DataFrame] = []
    weather_dfs: list[pd.DataFrame] = []

    for region in regions:
        # ── FIRMS ────────────────────────────────────────────────────────────
        firms_raw_path = fetch_firms_data(
            execution_date=t,
            resolution_km=resolution_km,
            lookback_hours=DEFAULT_FREQ_HOURS,
            output_dir=str(Path(raw_dir) / "firms"),
            region=region,
        )
        firms_df = process_firms_data(str(firms_raw_path), resolution_km)
        firms_df["region"] = region
        firms_dfs.append(firms_df)

        # ── Weather ──────────────────────────────────────────────────────────
        region_grid = grid_df[grid_df["grid_id"].isin(
            grid[grid["region"] == region]["grid_id"]
        )] if "region" in grid.columns else grid_df

        weather_raw_path = fetch_weather_data(
            grid_centroids=region_grid,
            execution_date=t,
            lookback_hours=DEFAULT_FREQ_HOURS,
            output_dir=str(Path(raw_dir) / "weather"),
            trigger_source="cron",
        )
        weather_df = process_weather_data(str(weather_raw_path), resolution_km)
        weather_df["region"] = region
        weather_dfs.append(weather_df)

    all_firms   = pd.concat(firms_dfs,   ignore_index=True) if firms_dfs   else pd.DataFrame()
    all_weather = pd.concat(weather_dfs, ignore_index=True) if weather_dfs else pd.DataFrame()

    fused = fuse_features(
        firms_features=all_firms,
        weather_features=all_weather,
        static_features=static_df,
        execution_date=t,
        resolution_km=resolution_km,
    )

    # ── Write partitioned output ──────────────────────────────────────────────
    for region in regions:
        region_mask = fused["region"] == region if "region" in fused.columns else pd.Series(True, index=fused.index)
        region_df   = fused[region_mask]
        if region_df.empty:
            logger.debug("No rows for region '%s' at %s — skipping write", region, t)
            continue

        out_path = (
            output_dir
            / f"region={region}"
            / f"year={t.year}"
            / f"month={t.month:02d}"
            / f"features_{t:%Y%m%dT%H%M}.parquet"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        region_df.to_parquet(out_path, index=False)

    logger.debug("Window %s written (%d rows total)", t, len(fused))


# ---------------------------------------------------------------------------
# Main backfill loop
# ---------------------------------------------------------------------------

def run_backfill(
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    frequency_hours: int = DEFAULT_FREQ_HOURS,
    resolution_km: int = DEFAULT_RESOLUTION,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    resume: bool = True,
    regions: list[str] = DEFAULT_REGIONS,
    raw_dir: str = DEFAULT_RAW_DIR,
    static_dir: str = DEFAULT_STATIC_DIR,
    dry_run: bool = False,
) -> dict[str, int]:
    """Replay the pipeline over a historical date range.

    Args:
        start_date:      ISO date string for the first window ("2020-01-01").
        end_date:        ISO date string for the last window ("2025-12-31").
        frequency_hours: Spacing between windows in hours (default: 6).
        resolution_km:   H3 grid resolution.
        output_dir:      Root directory for output parquet files.
        resume:          Skip windows whose output files already exist.
        regions:         List of region names to process.
        raw_dir:         Directory for raw API fetches (FIRMS, weather).
        static_dir:      Directory containing pre-computed static parquet caches.
        dry_run:         Print windows without fetching or writing any data.

    Returns:
        Summary dict with counts: processed, skipped, failed.
    """
    out_root = Path(output_dir)
    windows  = pd.date_range(
        start=start_date,
        end=end_date,
        freq=f"{frequency_hours}h",
        tz="UTC",
    )

    n_total    = len(windows)
    n_skipped  = 0
    n_processed = 0
    n_failed   = 0

    logger.info(
        "Backfill: %d windows from %s → %s "
        "(freq=%dh, res=%dkm, regions=%s, resume=%s)",
        n_total, start_date, end_date, frequency_hours,
        resolution_km, regions, resume,
    )

    if dry_run:
        logger.info("DRY RUN — printing windows only, no data fetched")
        for i, t in enumerate(windows):
            paths = _window_output_paths(out_root, t, regions)
            status = "EXISTS" if _all_exist(paths) else "PENDING"
            print(f"  [{i+1:>6}/{n_total}] {t}  {status}")
        return {"processed": 0, "skipped": n_total, "failed": 0}

    t0 = time.monotonic()

    for i, t in enumerate(windows, 1):
        out_paths = _window_output_paths(out_root, t, regions)

        if resume and _all_exist(out_paths):
            n_skipped += 1
            if i % 100 == 0:
                logger.info(
                    "[%d/%d] Skipping %s (output exists)", i, n_total, t
                )
            continue

        try:
            _process_window(
                t=t,
                resolution_km=resolution_km,
                regions=regions,
                output_dir=out_root,
                raw_dir=raw_dir,
                static_dir=static_dir,
            )
            n_processed += 1

        except Exception:
            n_failed += 1
            logger.error(
                "[%d/%d] Window %s FAILED:\n%s",
                i, n_total, t, traceback.format_exc(),
            )
            # Continue to next window; partial failures are expected
            # (e.g. FIRMS API outage on a specific date)

        # Progress log every 50 processed windows
        if n_processed % 50 == 0 and n_processed > 0:
            elapsed  = time.monotonic() - t0
            rate     = n_processed / elapsed if elapsed > 0 else float("nan")
            remaining = (n_total - i) / rate / 3600 if rate > 0 else float("nan")
            logger.info(
                "[%d/%d] Progress: %d processed, %d skipped, %d failed — "
                "%.1f windows/s — ~%.1f h remaining",
                i, n_total, n_processed, n_skipped, n_failed, rate, remaining,
            )

    elapsed = time.monotonic() - t0
    summary = {"processed": n_processed, "skipped": n_skipped, "failed": n_failed}
    logger.info(
        "Backfill complete in %.0f s — processed=%d, skipped=%d, failed=%d",
        elapsed, n_processed, n_skipped, n_failed,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay the wildfire pipeline over a historical date range."
    )
    p.add_argument("--start",          default=DEFAULT_START,
                   help=f"Start date ISO (default: {DEFAULT_START})")
    p.add_argument("--end",            default=DEFAULT_END,
                   help=f"End date ISO (default: {DEFAULT_END})")
    p.add_argument("--frequency-hours", type=int, default=DEFAULT_FREQ_HOURS,
                   help=f"Window frequency in hours (default: {DEFAULT_FREQ_HOURS})")
    p.add_argument("--resolution-km",  type=int, default=DEFAULT_RESOLUTION,
                   help=f"H3 resolution in km (default: {DEFAULT_RESOLUTION})")
    p.add_argument("--output-dir",     default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--raw-dir",        default=DEFAULT_RAW_DIR,
                   help="Directory for raw API outputs (default: data/raw)")
    p.add_argument("--static-dir",     default=DEFAULT_STATIC_DIR,
                   help="Directory containing static feature caches (default: data/static)")
    p.add_argument("--regions",        nargs="+", default=DEFAULT_REGIONS)
    p.add_argument("--no-resume",      dest="resume", action="store_false",
                   help="Reprocess all windows, ignoring existing outputs")
    p.add_argument("--dry-run",        action="store_true",
                   help="Print windows without fetching any data")
    p.add_argument("--log-level",      default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    summary = run_backfill(
        start_date=args.start,
        end_date=args.end,
        frequency_hours=args.frequency_hours,
        resolution_km=args.resolution_km,
        output_dir=args.output_dir,
        resume=args.resume,
        regions=args.regions,
        raw_dir=args.raw_dir,
        static_dir=args.static_dir,
        dry_run=args.dry_run,
    )
    print(f"Summary: {summary}")
