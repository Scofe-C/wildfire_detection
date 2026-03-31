"""
Static Feature Processing
=========================
Generates static terrain and fuel features for the H3 grid.

Real data is loaded from pre-computed parquet caches in ``data/static/``.
Each cache is produced by the corresponding ingestion script:

  LANDFIRE (fuel, canopy, vegetation):
      python -m scripts.ingestion.ingest_landfire --resolution-km <N> --output-dir data/static

  SRTM (elevation, slope, aspect):
      python -m scripts.ingestion.ingest_srtm --resolution-km <N> --output-dir data/static

If a cache file is missing, the corresponding columns fall back to NaN stubs
and a warning is logged.  The pipeline continues gracefully; the downstream
``data_quality_flag`` will be set to 4 for those cells.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.utils.grid_utils import generate_full_grid

logger = logging.getLogger(__name__)

STATIC_COLUMNS = [
    "grid_id",
    "latitude",
    "longitude",
    "fuel_model_fbfm40",
    "canopy_cover_pct",
    "vegetation_type",
    "elevation_m",
    "slope_degrees",
    "aspect_degrees",
    "dominant_fuel_fraction",
]

# Columns supplied by each source
_LANDFIRE_COLS = ["fuel_model_fbfm40", "canopy_cover_pct", "vegetation_type", "dominant_fuel_fraction"]
_SRTM_COLS     = ["elevation_m", "slope_degrees", "aspect_degrees"]


def load_and_process_static(
    resolution_km: int,
    output_dir: str,
    force_rebuild: bool = False,
) -> Path:
    """Generate static feature Parquet for the full H3 grid.

    Reads pre-computed per-source parquet caches from output_dir and joins
    them onto the H3 grid.  Missing caches fall back to NaN stubs.

    Args:
        resolution_km: Grid resolution in km (maps to H3 level).
        output_dir: Directory containing the source caches and where the
                    fused static parquet is written.
        force_rebuild: If True, regenerate even if cache exists.

    Returns:
        Path to the written Parquet file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"static_features_{resolution_km}km.parquet"

    if out_path.exists() and not force_rebuild:
        logger.info("Static cache hit: %s", out_path)
        return out_path

    # Build base grid
    logger.info("Generating full grid at %d km resolution", resolution_km)
    grid = generate_full_grid(resolution_km)
    df = grid[["grid_id", "latitude", "longitude"]].copy()
    df["grid_id"] = df["grid_id"].astype(str)

    sources_loaded = []

    # ── LANDFIRE ──────────────────────────────────────────────────────────────
    lf_path = out_dir / f"landfire_features_{resolution_km}km.parquet"
    if lf_path.exists():
        try:
            lf = pd.read_parquet(lf_path)
            lf["grid_id"] = lf["grid_id"].astype(str)
            lf = lf[["grid_id"] + _LANDFIRE_COLS].drop_duplicates("grid_id")
            df = df.merge(lf, on="grid_id", how="left")
            sources_loaded.append("LANDFIRE")
            logger.info("Loaded LANDFIRE features from %s", lf_path)
        except Exception as exc:
            logger.error("LANDFIRE load failed (%s) — using NaN stubs", exc)
            for col in _LANDFIRE_COLS:
                df[col] = np.nan
    else:
        logger.warning(
            "LANDFIRE cache not found (%s). "
            "Run: python -m scripts.ingestion.ingest_landfire "
            "--resolution-km %d --output-dir %s",
            lf_path, resolution_km, output_dir,
        )
        for col in _LANDFIRE_COLS:
            df[col] = np.nan

    # ── SRTM ─────────────────────────────────────────────────────────────────
    srtm_path = out_dir / f"srtm_features_{resolution_km}km.parquet"
    if srtm_path.exists():
        try:
            srtm = pd.read_parquet(srtm_path)
            srtm["grid_id"] = srtm["grid_id"].astype(str)
            srtm = srtm[["grid_id"] + _SRTM_COLS].drop_duplicates("grid_id")
            df = df.merge(srtm, on="grid_id", how="left")
            sources_loaded.append("SRTM")
            logger.info("Loaded SRTM features from %s", srtm_path)
        except Exception as exc:
            logger.error("SRTM load failed (%s) — using NaN stubs", exc)
            for col in _SRTM_COLS:
                df[col] = np.nan
    else:
        logger.warning(
            "SRTM cache not found (%s). "
            "Run: python -m scripts.ingestion.ingest_srtm "
            "--resolution-km %d --output-dir %s",
            srtm_path, resolution_km, output_dir,
        )
        for col in _SRTM_COLS:
            df[col] = np.nan

    # Log overall stub status
    if not sources_loaded:
        logger.warning(
            "All static features are NaN stubs — no source caches found in %s. "
            "See scripts/ingestion/ingest_landfire.py, ingest_srtm.py.",
            output_dir,
        )
    else:
        logger.info("Static features loaded from: %s", ", ".join(sources_loaded))
        missing = [s for s in ("LANDFIRE", "SRTM") if s not in sources_loaded]
        if missing:
            logger.warning("Still using NaN stubs for: %s", ", ".join(missing))

    # Ensure all expected columns exist (fill any still-missing ones)
    for col in STATIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[STATIC_COLUMNS]

    df.to_parquet(out_path, index=False)
    logger.info("Wrote static features: %s (%d rows)", out_path, len(df))
    return out_path
