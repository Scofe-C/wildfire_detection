"""
LANDFIRE 2022 Ingestion
=======================
Reads manually downloaded LANDFIRE 2022 GeoTIFF rasters and computes zonal
statistics over the H3 hexagonal grid.

--- HOW TO OBTAIN THE DATA ---

Go to: https://landfire.gov/data/FullExtentDownloads
Select: LF 2022 → CONUS → Fuel

Download these three ZIPs (~1-2 GB each):
    FBFM40  (fuel model)     : LF2022_FBFM40_230_CONUS.zip
    CC      (canopy cover)   : LF2022_CC_230_CONUS.zip
    EVT     (vegetation type): LF2022_EVT_230_CONUS.zip

Extract each ZIP. Each contains a .tif raster. Place them in:
    data/static/landfire_raw/
        LC22_F40_230.tif     ← fuel model  (rename if needed)
        LC22_CC_230.tif      ← canopy cover
        LC22_EVT_230.tif     ← vegetation type

Any .tif whose filename contains "F40" or "FBFM40" is picked up as fuel model;
"_CC_" as canopy cover; "EVT" as vegetation type. Exact names don't matter as
long as one keyword matches.

--- USAGE ---

    python -m scripts.ingestion.ingest_landfire \\
        --resolution-km 64 \\
        --output-dir data/static

Output:
    data/static/landfire_features_<N>km.parquet  — per-H3-cell feature table
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
import rasterio.windows
from rasterstats import zonal_stats
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as _transform_from_bounds

from scripts.utils.grid_utils import generate_full_grid

logger = logging.getLogger(__name__)

_WGS84 = CRS.from_epsg(4326)

# Target resolution for the reprojected raster (~1 km).  LANDFIRE source data
# is 30 m, but 64-km H3 zonal stats only need ~1 km detail.  This keeps the
# clipped output at ~7 k × 5 k pixels instead of 90 k × 100 k.
_TARGET_RES_DEG = 0.01  # degrees ≈ 1.1 km


# ---------------------------------------------------------------------------
# Raster discovery
# ---------------------------------------------------------------------------

def _find_raster(raw_dir: Path, keywords: list[str]) -> Optional[Path]:
    """Return the first .tif in raw_dir whose name contains any keyword."""
    for tif in sorted(raw_dir.glob("*.tif")):
        nl = tif.name.lower()
        if any(kw.lower() in nl for kw in keywords):
            return tif
    return None


def _locate_rasters(raw_dir: Path) -> dict[str, Path]:
    """Find the three required rasters in raw_dir.

    Raises FileNotFoundError if any layer is missing.
    """
    mapping = {
        "fbfm40": ["f40", "fbfm40"],
        "cc":     ["_cc_", "_cc."],
        "evt":    ["evt"],
    }
    found: dict[str, Path] = {}
    missing: list[str] = []
    for layer, keywords in mapping.items():
        path = _find_raster(raw_dir, keywords)
        if path:
            found[layer] = path
            logger.info("Found LANDFIRE %s: %s", layer.upper(), path.name)
        else:
            missing.append(layer.upper())

    if missing:
        raise FileNotFoundError(
            f"Missing LANDFIRE rasters in {raw_dir}: {missing}\n"
            "Download from https://landfire.gov/data/FullExtentDownloads "
            "(LF 2022 → CONUS → Fuel) and place extracted .tif files there.\n"
            "Expected keywords in filenames: F40/FBFM40, _CC_, EVT."
        )
    return found


# ---------------------------------------------------------------------------
# Reprojection helper
# ---------------------------------------------------------------------------

def _reproject_to_wgs84(
    raster_path: Path,
    clip_bounds_wgs84: tuple[float, float, float, float] | None = None,
) -> str:
    """Reproject to WGS84, clipping and resampling to _TARGET_RES_DEG.

    LANDFIRE source data is 30 m (EPSG:5070).  Full CONUS at 30 m is ~90k×100k
    pixels — too large to allocate.  We clip to the CA+TX bbox and resample to
    ~1 km (_TARGET_RES_DEG), which is ample precision for 64-km H3 cells.
    The result is a small temp _wgs84.tif written next to the original.
    """
    with rasterio.open(str(raster_path)) as src:
        src_crs = src.crs

        # Determine output bounds in WGS84
        if clip_bounds_wgs84 is not None:
            west, south, east, north = clip_bounds_wgs84
        else:
            west, south, east, north = rasterio.warp.transform_bounds(
                src_crs, _WGS84, *src.bounds
            )

        # Compute output grid at _TARGET_RES_DEG resolution
        dst_width  = max(1, int(round((east  - west)  / _TARGET_RES_DEG)))
        dst_height = max(1, int(round((north - south) / _TARGET_RES_DEG)))
        dst_transform = _transform_from_bounds(west, south, east, north,
                                               dst_width, dst_height)

        logger.info(
            "Reprojecting %s → WGS84 at %.3f° [%d×%d px] …",
            raster_path.name, _TARGET_RES_DEG, dst_width, dst_height,
        )

        meta = src.meta.copy()
        meta.update(
            crs=_WGS84, transform=dst_transform,
            width=dst_width, height=dst_height,
        )

        tmp = str(raster_path).replace(".tif", "_wgs84.tif")
        with rasterio.open(tmp, "w", **meta) as dst:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=_WGS84,
                    resampling=Resampling.nearest,  # categorical rasters
                )
        return tmp


# ---------------------------------------------------------------------------
# Zonal statistics
# ---------------------------------------------------------------------------

def _zonal_mode(raster_wgs84: str, geoms: list) -> list[dict]:
    """Per-geometry mode and dominant fraction for a categorical raster.

    Filters out known nodata sentinels (-9999, 0, int16-max=32767) before
    computing the mode so they do not pollute results.
    """
    with rasterio.open(raster_wgs84) as src:
        nd = src.nodata

    # Build a set of pixel values to exclude (nodata sentinels)
    _NODATA_SENTINELS = {-9999, 0, 32767, 32768, 65535}
    if nd is not None:
        _NODATA_SENTINELS.add(int(nd))

    stats = zonal_stats(geoms, raster_wgs84, categorical=True,
                        nodata=nd, all_touched=True)
    results = []
    for stat in stats:
        if not stat:
            results.append({"mode": None, "frac": None})
            continue
        # Drop nodata sentinel keys
        stat = {k: v for k, v in stat.items() if int(k) not in _NODATA_SENTINELS}
        if not stat:
            results.append({"mode": None, "frac": None})
            continue
        total = sum(stat.values())
        if total == 0:
            results.append({"mode": None, "frac": None})
            continue
        mode_val = max(stat, key=stat.__getitem__)
        results.append({"mode": int(mode_val), "frac": float(stat[mode_val] / total)})
    return results


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute_landfire_features(
    grid: gpd.GeoDataFrame,
    raster_paths: dict[str, Path],
    output_dir: str | Path,
    resolution_km: int,
) -> Path:
    """Compute per-H3-cell LANDFIRE features via zonal statistics.

    Args:
        grid:          Full H3 GeoDataFrame (geometry in EPSG:4326).
        raster_paths:  {"fbfm40": Path, "cc": Path, "evt": Path}.
        output_dir:    Directory to write the output parquet.
        resolution_km: Grid resolution (used in the output filename).

    Returns:
        Path to landfire_features_<N>km.parquet.
    """
    out_path = Path(output_dir) / f"landfire_features_{resolution_km}km.parquet"

    geoms = list(grid.geometry)
    logger.info("Computing LANDFIRE zonal statistics (%d H3 cells) …", len(geoms))

    # Clip rasters to the grid's bounding box — avoids processing the full
    # CONUS extent (~1–2 GB per raster) when we only need CA+TX pixels.
    bounds = grid.total_bounds  # [minx, miny, maxx, maxy] in WGS84
    clip = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
    logger.info("Clipping rasters to grid bbox: W=%.2f S=%.2f E=%.2f N=%.2f", *clip)

    fbfm40_wgs84 = _reproject_to_wgs84(raster_paths["fbfm40"], clip)
    cc_wgs84     = _reproject_to_wgs84(raster_paths["cc"],     clip)
    evt_wgs84    = _reproject_to_wgs84(raster_paths["evt"],    clip)

    fbfm_stats = _zonal_mode(fbfm40_wgs84, geoms)

    # Read CC nodata from the reprojected raster (may be 32767 or -9999)
    with rasterio.open(cc_wgs84) as _cc_src:
        cc_nd = _cc_src.nodata if _cc_src.nodata is not None else -9999
    cc_stats   = zonal_stats(geoms, cc_wgs84, stats=["mean"],
                             nodata=cc_nd, all_touched=True)

    evt_stats  = _zonal_mode(evt_wgs84, geoms)

    df = grid[["grid_id", "latitude", "longitude"]].copy()
    df["fuel_model_fbfm40"]      = [s["mode"] for s in fbfm_stats]
    df["canopy_cover_pct"]       = [s.get("mean") for s in cc_stats]
    df["vegetation_type"]        = [s["mode"] for s in evt_stats]
    df["dominant_fuel_fraction"] = [s["frac"] for s in fbfm_stats]

    # Clean up reprojected temp files
    for tmp in [fbfm40_wgs84, cc_wgs84, evt_wgs84]:
        if tmp.endswith("_wgs84.tif"):
            Path(tmp).unlink(missing_ok=True)

    df.to_parquet(out_path, index=False)
    logger.info("Wrote LANDFIRE features: %s (%d rows)", out_path, len(df))
    return out_path


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run(
    resolution_km: int = 64,
    output_dir: str | Path = "data/static",
    force_rebuild: bool = False,
) -> Path:
    """Read LANDFIRE rasters from data/static/landfire_raw/ and compute features.

    Args:
        resolution_km:  H3 grid resolution.
        output_dir:     Directory containing landfire_raw/ and where the output
                        parquet is written.
        force_rebuild:  Recompute even if the parquet already exists.

    Returns:
        Path to landfire_features_<N>km.parquet.
    """
    out_path = Path(output_dir) / f"landfire_features_{resolution_km}km.parquet"
    if out_path.exists() and not force_rebuild:
        logger.info("LANDFIRE cache hit: %s", out_path)
        return out_path

    raw_dir = Path(output_dir) / "landfire_raw"
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"LANDFIRE raw directory not found: {raw_dir}\n"
            "Download rasters from https://landfire.gov/data/FullExtentDownloads "
            "and place extracted .tif files there."
        )

    raster_paths = _locate_rasters(raw_dir)
    grid = generate_full_grid(resolution_km)
    return compute_landfire_features(grid, raster_paths, output_dir, resolution_km)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute H3 LANDFIRE features from manually downloaded rasters "
            "in data/static/landfire_raw/. "
            "See module docstring for download instructions."
        )
    )
    p.add_argument("--resolution-km", type=int, default=64)
    p.add_argument("--output-dir", default="data/static")
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    out = run(
        resolution_km=args.resolution_km,
        output_dir=args.output_dir,
        force_rebuild=args.force_rebuild,
    )
    print(f"Output: {out}")
