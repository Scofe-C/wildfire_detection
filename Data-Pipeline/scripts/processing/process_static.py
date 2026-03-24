"""
Static Feature Processing
=========================
Generates static terrain and fuel features for the H3 grid.

Currently outputs NaN stubs for all static columns because real LANDFIRE
and SRTM data sources are not yet wired. See missing_sources_and_todo.md
for download URLs and integration steps.

Once LANDFIRE/SRTM/MODIS ingestion is implemented, replace the NaN stubs
with actual raster-to-grid aggregation.
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
    "ndvi",
    "elevation_m",
    "slope_degrees",
    "aspect_degrees",
    "dominant_fuel_fraction",
]


def load_and_process_static(
    resolution_km: int,
    output_dir: str,
    force_rebuild: bool = False,
) -> Path:
    """Generate static feature Parquet for the full H3 grid.

    Args:
        resolution_km: Grid resolution in km (maps to H3 level).
        output_dir: Directory to write the output Parquet file.
        force_rebuild: If True, regenerate even if cache exists.

    Returns:
        Path to the written Parquet file.
    """
    # 1) paths — define output location first
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"static_features_{resolution_km}km.parquet"

    # 2) cache check
    if out_path.exists() and not force_rebuild:
        logger.info("Static cache hit: %s", out_path)
        return out_path

    # 3) build grid
    logger.info("Generating full grid at %d km resolution", resolution_km)
    grid = generate_full_grid(resolution_km)
    df = grid[["grid_id", "latitude", "longitude"]].copy()
    df["grid_id"] = df["grid_id"].astype(str)

    # 4) Static stubs — all NaN until real LANDFIRE/SRTM/MODIS is wired.
    # These NaN values signal to downstream ML that the data source is
    # unavailable, rather than producing misleading zeros.
    logger.warning(
        "Static features are stubs (NaN). "
        "Wire LANDFIRE/SRTM/MODIS download for real data. "
        "See missing_sources_and_todo.md."
    )
    df["elevation_m"]           = np.nan
    df["slope_degrees"]         = np.nan
    df["aspect_degrees"]        = np.nan
    df["fuel_model_fbfm40"]     = np.nan
    df["canopy_cover_pct"]      = np.nan
    df["vegetation_type"]       = np.nan
    df["ndvi"]                  = np.nan
    df["dominant_fuel_fraction"] = np.nan

    df = df[STATIC_COLUMNS]

    # 5) write
    df.to_parquet(out_path, index=False)
    logger.info("Wrote static features: %s (%d rows)", out_path, len(df))
    return out_path
