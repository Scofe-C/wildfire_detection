"""
Static Feature Processing
=========================
Generates static terrain and fuel features for the H3 grid.
Supports 64 km and 22 km resolutions for California + Texas.

Real data is loaded from pre-computed parquet caches in ``data/static/``.
Each cache is produced by the corresponding ingestion script:

  LANDFIRE (fuel, canopy, vegetation, CBH, CBD, EVT-CNC):
      python -m scripts.ingestion.ingest_landfire --resolution-km 64 22 --output-dir data/static

  SRTM (elevation, slope, aspect):
      python -m scripts.ingestion.ingest_srtm --resolution-km 64 22 --output-dir data/static

Then fuse both resolutions in one run:
      python -m scripts.processing.process_static --resolution-km 64 22 --output-dir data/static

If a cache file is missing, the corresponding columns fall back to NaN stubs
and a warning is logged.  The pipeline continues gracefully; the downstream
``data_quality_flag`` will be set to 4 for those cells.
"""

from __future__ import annotations

import argparse
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
    "dominant_fuel_fraction",
    # Fire spread simulation features (optional LANDFIRE layers)
    "canopy_base_height_m",   # LF2020 CBH ÷ 10  → metres; crown fire initiation
    "canopy_bulk_density",    # LF2024 CBD ÷ 100 → kg/m³; crown fire spread rate
    "evt_national_class",     # LF2020 EVT-CNC code; vegetation class for fuel moisture
    # Topography
    "elevation_m",
    "slope_degrees",
    "aspect_degrees",
]

# Columns supplied by each source
_LANDFIRE_COLS = [
    "fuel_model_fbfm40",
    "canopy_cover_pct",
    "vegetation_type",
    "dominant_fuel_fraction",
    "canopy_base_height_m",
    "canopy_bulk_density",
    "evt_national_class",
]
_SRTM_COLS = ["elevation_m", "slope_degrees", "aspect_degrees"]


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fuse LANDFIRE + SRTM feature caches into static_features_<N>km.parquet "
            "for each requested resolution. Covers California + Texas only. "
            "Run ingest_landfire and ingest_srtm first to populate the source caches."
        )
    )
    p.add_argument(
        "--resolution-km",
        type=int,
        nargs="+",
        default=[64, 22],
        metavar="KM",
        help="One or more grid resolutions in km. Default: 64 22 (both CA/TX grids).",
    )
    p.add_argument(
        "--output-dir",
        default="data/static",
        help="Directory containing source parquet caches and where output is written.",
    )
    p.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Recompute parquet even if cached file already exists.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    resolutions = args.resolution_km
    logger.info(
        "Running process_static for resolutions: %s  |  regions: California + Texas",
        resolutions,
    )

    outputs = []
    for res_km in resolutions:
        logger.info("─── Resolution %d km ───────────────────────────────────", res_km)
        out = load_and_process_static(
            resolution_km=res_km,
            output_dir=args.output_dir,
            force_rebuild=args.force_rebuild,
        )
        outputs.append(out)

    print("\n=== process_static complete ===")
    for o in outputs:
        print(f"  {o}")
