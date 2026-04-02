#!/usr/bin/env python3
"""
Historical Data Backfill for OBJ-1 Training
============================================
Fetches FIRMS SP (Standard Processing) + OpenMeteo archive data for a date
range and runs it through the same processing/fusion pipeline as the live
Airflow DAG.  Output is stored in model-pipeline/historical_data/ in the
same partitioned Parquet format the ML loader expects.

Usage:
    python backfill_historical_data.py
    python backfill_historical_data.py --start 2024-06-01 --end 2025-01-31
    python backfill_historical_data.py --start 2024-06-01 --end 2025-01-31 --resolution 64

Output layout:
    model-pipeline/historical_data/{resolution_km}km/
        region=california/year=2024/month=06/features_2024-06-01.parquet
        region=california/year=2024/month=06/features_2024-06-02.parquet
        ...
        region=texas/year=2025/month=01/features_2025-01-31.parquet

Why FIRMS SP not NRT:
    NRT sensors (VIIRS_SNPP_NRT etc.) only hold a rolling 10-day window.
    SP (Standard Processing) sensors archive all historical data and accept
    a date parameter:
        /api/area/csv/{key}/VIIRS_SNPP_SP/{bbox}/{days}/{end_date}

Target column note:
    fire_detected_binary is derived automatically by process_firms_data.
    A grid cell gets fire_detected_binary=1 if ANY FIRMS detection (frp > 0)
    falls within that H3 cell during the day.  No manual labelling needed —
    FIRMS IS the ground truth for fire occurrence.

Environment variables required:
    FIRMS_MAP_KEY  — NASA FIRMS API key (same key used by the Airflow DAG)
"""

import argparse
import io
import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Path setup — make Data-Pipeline scripts importable
# ---------------------------------------------------------------------------
SCRIPT_DIR    = Path(__file__).resolve().parent          # model-pipeline/scripts/
MODEL_PIPELINE = SCRIPT_DIR.parent                       # model-pipeline/
WILDFIRE_ROOT  = MODEL_PIPELINE.parent                   # wildfire_detection/
DATA_PIPELINE  = WILDFIRE_ROOT / "Data-Pipeline"         # Data-Pipeline/
OUTPUT_BASE    = MODEL_PIPELINE / "historical_data"

sys.path.insert(0, str(DATA_PIPELINE))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backfill")

# ---------------------------------------------------------------------------
# Constants — mirrors wildfire_dag.py + schema_config.yaml
# ---------------------------------------------------------------------------
REGIONS = {
    "california": [-124.48, 32.53, -114.13, 42.01],
    "texas":      [-106.65, 25.84,  -93.51, 36.50],
}

# SP = Standard Processing: full historical archive, requires a date parameter.
# NRT sensors only hold the last 10 days and cannot be used for backfill.
FIRMS_SP_SENSORS = ["VIIRS_SNPP_SP", "VIIRS_NOAA20_SP", "MODIS_SP"]
FIRMS_BASE_URL   = "https://firms.modaps.eosdis.nasa.gov/api/area"
FIRMS_CHUNK_DAYS = 5    # SP API maximum days per request (API limit: 1–5)


# ===========================================================================
# FIRMS SP historical fetch
# ===========================================================================

def _fetch_firms_sp_chunk(
    api_key: str,
    sensor: str,
    bbox: list,
    end_date: date,
    days: int,
    timeout: int = 30,
    max_retries: int = 3,
) -> Optional[pd.DataFrame]:
    """Fetch up to `days` days of FIRMS SP detections ending on `end_date`."""
    west, south, east, north = bbox
    bbox_str = f"{west},{south},{east},{north}"
    date_str = end_date.strftime("%Y-%m-%d")
    url = f"{FIRMS_BASE_URL}/csv/{api_key}/{sensor}/{bbox_str}/{days}/{date_str}"

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout)

            if resp.status_code == 200:
                content = resp.text.strip()
                if (not content
                        or content.lower().startswith("no data")
                        or content.lower().startswith("error")):
                    return None
                df = pd.read_csv(io.StringIO(content))
                return df if len(df) > 0 else None

            if resp.status_code == 429:
                wait = 5 * (2 ** attempt)
                logger.warning("FIRMS rate limited — sleeping %ds (attempt %d)", wait, attempt + 1)
                time.sleep(wait)
                continue

            # 4xx (except 429) are non-retryable
            if 400 <= resp.status_code < 500:
                logger.error(
                    "FIRMS %s non-retryable HTTP %d: %s",
                    sensor, resp.status_code, resp.text[:200],
                )
                return None

            # 5xx — retry
            logger.warning("FIRMS HTTP %d for %s — retrying", resp.status_code, sensor)
            time.sleep(2 ** attempt)

        except requests.RequestException as exc:
            logger.warning("FIRMS request error attempt %d: %s", attempt + 1, exc)
            time.sleep(2 ** attempt)

    return None


def fetch_firms_for_range(
    api_key: str,
    bbox: list,
    start: date,
    end: date,
    region: str,
) -> pd.DataFrame:
    """
    Fetch all FIRMS SP detections for a date range, chunked into 10-day windows.
    Returns a combined DataFrame with an `acq_date` column (as date objects).
    """
    all_dfs = []
    chunk_end = end

    while chunk_end >= start:
        chunk_start = max(start, chunk_end - timedelta(days=FIRMS_CHUNK_DAYS - 1))
        days_in_chunk = (chunk_end - chunk_start).days + 1

        for sensor in FIRMS_SP_SENSORS:
            df = _fetch_firms_sp_chunk(api_key, sensor, bbox, chunk_end, days_in_chunk)
            if df is not None:
                df["region"] = region
                df["sensor_source"] = sensor
                all_dfs.append(df)
                logger.info(
                    "  [%s] %s: %d detections (%s – %s)",
                    region, sensor, len(df), chunk_start, chunk_end,
                )
            time.sleep(0.3)  # polite pacing

        chunk_end = chunk_start - timedelta(days=1)

    if not all_dfs:
        logger.info("[%s] No FIRMS detections found for %s – %s", region, start, end)
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Deduplicate — same fire pixel can appear in multiple sensors
    dedup_cols = [c for c in ["latitude", "longitude", "acq_date", "acq_time"] if c in combined.columns]
    if dedup_cols:
        combined = combined.drop_duplicates(subset=dedup_cols)

    # Normalise acq_date to Python date objects for groupby
    combined["acq_date"] = pd.to_datetime(combined["acq_date"]).dt.date

    logger.info(
        "[%s] FIRMS total after dedup: %d detections (%s – %s)",
        region, len(combined), start, end,
    )
    return combined


# ===========================================================================
# Per-day processing — mirrors the DAG task sequence
# ===========================================================================

def process_one_day(
    day: date,
    firms_day_df: pd.DataFrame,
    weather_csv_path: Path,
    static_df: pd.DataFrame,
    resolution_km: int,
    prev_fused_path: Optional[Path],
    tmp_dir: Path,
) -> Optional[pd.DataFrame]:
    """
    Run one day through the full Data-Pipeline processing stack:
      process_firms → process_weather → fuse_features → apply_temporal_lag

    Returns the ML-ready fused DataFrame (same schema as the live pipeline),
    or None if processing fails.
    """
    from scripts.processing.process_firms import process_firms_data
    from scripts.processing.process_weather import process_weather_data
    from scripts.fusion.fuse_features import fuse_features, apply_temporal_lag

    day_str = day.strftime("%Y-%m-%d")
    execution_ts = pd.Timestamp(day_str, tz="UTC")

    # --- Write FIRMS day slice to temp CSV for process_firms_data ---
    firms_csv = tmp_dir / f"firms_{day_str}.csv"
    if firms_day_df.empty:
        pd.DataFrame(
            columns=["latitude", "longitude", "acq_date", "acq_time", "frp", "confidence"]
        ).to_csv(firms_csv, index=False)
    else:
        firms_day_df.to_csv(firms_csv, index=False)

    firms_features = process_firms_data(str(firms_csv), resolution_km=resolution_km)
    weather_features = process_weather_data(str(weather_csv_path), resolution_km=resolution_km)

    prev_str = str(prev_fused_path) if (prev_fused_path and prev_fused_path.exists()) else ""

    fused = fuse_features(
        firms_features=firms_features,
        weather_features=weather_features,
        static_features=static_df,
        execution_date=execution_ts,
        resolution_km=resolution_km,
        previous_fused_path=prev_str,
    )

    if fused is None or len(fused) == 0:
        logger.warning("[%s] Fusion returned empty result", day_str)
        return None

    # Apply temporal lag columns — same as fuse_features task in the DAG
    try:
        prev_fire_df = firms_features if not firms_features.empty else None
        ml_fused = apply_temporal_lag(fused, prev_fire_df)
    except Exception as exc:
        logger.warning("apply_temporal_lag failed (%s) — using plain fused", exc)
        ml_fused = fused

    return ml_fused


# ===========================================================================
# Main backfill loop
# ===========================================================================

def run_backfill(
    start: date,
    end: date,
    resolution_km: int = 64,
    skip_existing: bool = True,
) -> None:
    """
    Iterate over all days in [start, end] for each region, fetch data, process,
    and write partitioned Parquet files to model-pipeline/historical_data/.
    """
    from scripts.processing.process_static import load_and_process_static
    from scripts.utils.grid_utils import generate_full_grid
    from scripts.ingestion.ingest_weather import fetch_weather_data

    firms_api_key = os.environ.get("FIRMS_MAP_KEY")
    if not firms_api_key:
        logger.error("FIRMS_MAP_KEY is not set. Export it before running:")
        logger.error("  export FIRMS_MAP_KEY=your_key_here")
        sys.exit(1)

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", OUTPUT_BASE)
    logger.info("Date range: %s → %s (%d days)", start, end, (end - start).days + 1)
    logger.info("Resolution: %d km", resolution_km)

    # ------------------------------------------------------------------
    # Static features — load once, reused for every day/region
    # ------------------------------------------------------------------
    static_cache = DATA_PIPELINE / "data" / "static" / f"static_features_{resolution_km}km.parquet"
    if static_cache.exists():
        logger.info("Using cached static features: %s", static_cache)
        static_df = pd.read_parquet(static_cache)
    else:
        logger.info("Static cache not found — building (may take a few minutes)...")
        out = load_and_process_static(
            resolution_km=resolution_km,
            output_dir=str(DATA_PIPELINE / "data" / "static"),
        )
        static_df = pd.read_parquet(out)
    logger.info("Static features loaded: %d rows", len(static_df))

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)

        for region, bbox in REGIONS.items():
            logger.info("")
            logger.info("=" * 60)
            logger.info("REGION: %s  (%s → %s)", region.upper(), start, end)
            logger.info("=" * 60)

            # Grid centroids for weather fetch
            full_grid = generate_full_grid(resolution_km)
            grid_centroids = (
                full_grid[full_grid["region"] == region][["grid_id", "latitude", "longitude"]]
            )

            # ------------------------------------------------------------------
            # Fetch all FIRMS SP for this region + date range at once
            # (chunked internally to stay within the 10-day API limit)
            # ------------------------------------------------------------------
            logger.info("[%s] Fetching FIRMS SP data...", region)
            firms_all = fetch_firms_for_range(firms_api_key, bbox, start, end, region)

            # Index by acquisition date for O(1) day lookups
            if not firms_all.empty and "acq_date" in firms_all.columns:
                firms_by_day: dict[date, pd.DataFrame] = {
                    d: grp.reset_index(drop=True)
                    for d, grp in firms_all.groupby("acq_date")
                }
            else:
                firms_by_day = {}

            logger.info(
                "[%s] FIRMS days with detections: %d / %d",
                region, len(firms_by_day), (end - start).days + 1,
            )

            # ------------------------------------------------------------------
            # Day-by-day loop
            # ------------------------------------------------------------------
            prev_fused_path: Optional[Path] = None
            current = start
            days_processed = 0
            days_skipped = 0
            days_failed = 0

            while current <= end:
                day_str   = current.strftime("%Y-%m-%d")
                year_str  = current.strftime("%Y")
                month_str = current.strftime("%m")

                out_dir  = (
                    OUTPUT_BASE
                    / f"{resolution_km}km"
                    / f"region={region}"
                    / f"year={year_str}"
                    / f"month={month_str}"
                )
                out_path = out_dir / f"features_{day_str}.parquet"

                # Resume support — skip days already written
                if skip_existing and out_path.exists():
                    logger.debug("[%s] %s already exists — skipping", region, day_str)
                    prev_fused_path = out_path
                    days_skipped += 1
                    current += timedelta(days=1)
                    continue

                # --- Fetch weather for this specific day ---
                # ingest_weather auto-selects archive endpoint for historical dates
                weather_csv = tmp_dir / f"weather_{region}_{day_str}.csv"
                if not weather_csv.exists():
                    try:
                        execution_dt = datetime(
                            current.year, current.month, current.day, tzinfo=timezone.utc
                        )
                        fetched_path = fetch_weather_data(
                            grid_centroids=grid_centroids,
                            execution_date=execution_dt,
                            lookback_hours=24,
                            output_dir=str(tmp_dir),
                            region=region,
                        )
                        shutil.copy(str(fetched_path), str(weather_csv))
                    except Exception as exc:
                        logger.warning(
                            "[%s] %s weather fetch failed: %s — skipping day",
                            region, day_str, exc,
                        )
                        days_failed += 1
                        current += timedelta(days=1)
                        continue

                # --- FIRMS for this day (may be empty on low-fire days) ---
                firms_day = firms_by_day.get(current, pd.DataFrame())

                # --- Full processing + fusion ---
                try:
                    result = process_one_day(
                        day=current,
                        firms_day_df=firms_day,
                        weather_csv_path=weather_csv,
                        static_df=static_df,
                        resolution_km=resolution_km,
                        prev_fused_path=prev_fused_path,
                        tmp_dir=tmp_dir,
                    )
                except Exception as exc:
                    logger.error(
                        "[%s] %s processing error: %s", region, day_str, exc, exc_info=True
                    )
                    days_failed += 1
                    current += timedelta(days=1)
                    continue

                if result is not None and len(result) > 0:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    result["date"] = day_str
                    result["region"] = region
                    result.to_parquet(out_path, index=False)

                    fire_cells = (
                        int(result["fire_detected_binary"].sum())
                        if "fire_detected_binary" in result.columns
                        else "?"
                    )
                    logger.info(
                        "[%s] %s → %d rows | %s fire cells → %s",
                        region, day_str, len(result), fire_cells, out_path.name,
                    )
                    prev_fused_path = out_path
                    days_processed += 1
                else:
                    logger.warning("[%s] %s produced no output", region, day_str)
                    days_failed += 1

                current += timedelta(days=1)

            logger.info(
                "[%s] Done — processed: %d | skipped: %d | failed: %d",
                region, days_processed, days_skipped, days_failed,
            )

    _print_summary(OUTPUT_BASE, resolution_km)


# ===========================================================================
# Summary report
# ===========================================================================

def _print_summary(output_base: Path, resolution_km: int) -> None:
    parquet_files = sorted(output_base.glob(f"{resolution_km}km/**/*.parquet"))
    if not parquet_files:
        logger.info("No output files found in %s", output_base)
        return

    total_rows = 0
    total_fire = 0
    for f in parquet_files:
        try:
            df = pd.read_parquet(f, columns=["fire_detected_binary"])
            total_rows += len(df)
            total_fire += int(df["fire_detected_binary"].sum())
        except Exception:
            pass

    logger.info("")
    logger.info("=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info("  Parquet files : %d", len(parquet_files))
    logger.info("  Total rows    : %d", total_rows)
    logger.info(
        "  Fire+ rows    : %d (%.2f%%)",
        total_fire, 100.0 * total_fire / max(1, total_rows),
    )
    logger.info("  Output dir    : %s", output_base)
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next step — point the model loader at this directory:")
    logger.info("  backfill_dir: %s/%skm", output_base, resolution_km)


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill historical wildfire training data (Jun 2024 – Jan 2025 default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--start", default="2024-06-01",
        help="Start date YYYY-MM-DD (default: 2024-06-01)",
    )
    parser.add_argument(
        "--end", default="2025-01-31",
        help="End date YYYY-MM-DD (default: 2025-01-31)",
    )
    parser.add_argument(
        "--resolution", default=64, type=int,
        help="Grid resolution in km (default: 64)",
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Re-process dates that already have output files",
    )
    args = parser.parse_args()

    run_backfill(
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        resolution_km=args.resolution,
        skip_existing=not args.no_skip,
    )
