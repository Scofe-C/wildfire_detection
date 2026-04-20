"""
SRTM 30m Terrain Ingestion
===========================
Reads manually downloaded SRTM GeoTIFF tiles (from OpenTopography), mosaics
them per region, derives slope and aspect via numpy.gradient, then computes
per-H3-cell zonal statistics (elevation mean, slope mean, aspect circular mean).

--- HOW TO OBTAIN THE DATA ---

Register for a free API key at https://opentopography.org/developers, then
download 7 tiles (CA requires 3, TX requires 4 to stay under the 450,000 km²
per-request limit):

  California — 3 tiles:
    south  south=32.53 north=36.0  west=-124.48 east=-114.13
    mid    south=36.0  north=39.0  west=-124.48 east=-114.13
    north  south=39.0  north=42.01 west=-124.48 east=-114.13

  Texas — 4 tiles (2×2):
    nw     south=31.17 north=36.50 west=-106.65 east=-100.08
    ne     south=31.17 north=36.50 west=-100.08 east=-93.51
    sw     south=25.84 north=31.17 west=-106.65 east=-100.08
    se     south=25.84 north=31.17 west=-100.08 east=-93.51

  URL template:
    https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1
      &south=<S>&north=<N>&west=<W>&east=<E>&outputFormat=GTiff&API_Key=<KEY>

Place downloaded files in data/static/srtm_raw/ with these exact names:
    srtm_california_south.tif
    srtm_california_mid.tif
    srtm_california_north.tif
    srtm_texas_nw.tif
    srtm_texas_ne.tif
    srtm_texas_sw.tif
    srtm_texas_se.tif

--- USAGE ---

    python -m scripts.ingestion.ingest_srtm \\
        --resolution-km 64 \\
        --output-dir data/static

Output:
    data/static/srtm_features_<N>km.parquet  — per-H3-cell elevation/slope/aspect
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.merge
import rasterio.warp
from rasterstats import zonal_stats
from rasterio.crs import CRS
from rasterio.enums import Resampling

from scripts.utils.grid_utils import generate_full_grid

logger = logging.getLogger(__name__)

_WGS84 = CRS.from_epsg(4326)

# Target resolution for the mosaicked DEM (~1 km).  SRTM source data is 30 m
# but 64-km H3 zonal stats only need ~1 km detail.  Passing this to
# rasterio.merge keeps the in-memory array at ~1 k × 1 k pixels per region
# instead of ~37 k × 34 k pixels, preventing OOM errors.
_TARGET_RES_DEG = 0.01  # degrees ≈ 1.1 km

# Tile filename prefixes per region — all files matching any prefix are mosaicked.
# Accepts common download typos ("strm_", "strn_") alongside the canonical "srtm_".
_REGION_TILE_PREFIXES = {
    "california": ["srtm_california_", "strm_california_", "strn_california_"],
    "texas":      ["srtm_texas_",      "strm_texas_",      "strn_texas_"],
}


# ---------------------------------------------------------------------------
# Mosaic helpers
# ---------------------------------------------------------------------------

def _find_tiles(tile_dir: Path, region: str) -> list[Path]:
    """Return all .tif tiles belonging to a region, sorted by name."""
    prefixes = _REGION_TILE_PREFIXES.get(region, [f"srtm_{region}_"])
    tiles: list[Path] = []
    for prefix in prefixes:
        tiles.extend(sorted(tile_dir.glob(f"{prefix}*.tif")))
    if not tiles:
        # Fallback: single file named srtm_<region>.tif
        single = tile_dir / f"srtm_{region}.tif"
        if single.exists():
            tiles = [single]
    return tiles


def _mosaic_region(tile_paths: list[Path]) -> tuple[np.ndarray, object, CRS]:
    """Mosaic multiple GeoTIFF tiles into one array, resampled to _TARGET_RES_DEG.

    SRTM tiles are 30 m (~0.000278°).  Passing res=_TARGET_RES_DEG to merge
    resamples during the mosaic step, keeping the result at ~1 k × 1 k pixels
    per region instead of ~37 k × 34 k, which prevents OOM errors.

    Returns:
        (elevation_array shape (H,W), transform, crs)
    """
    datasets = [rasterio.open(str(p)) for p in tile_paths]
    mosaic, transform = rasterio.merge.merge(
        datasets, nodata=-9999, res=_TARGET_RES_DEG, resampling=Resampling.bilinear
    )
    crs = datasets[0].crs
    for ds in datasets:
        ds.close()
    return mosaic[0].astype(float), transform, crs


# ---------------------------------------------------------------------------
# Terrain derivation
# ---------------------------------------------------------------------------

def _compute_slope_aspect(
    elevation: np.ndarray,
    transform,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive slope (degrees) and aspect (degrees 0-360) from a DEM array.

    Uses numpy.gradient to compute rise/run.  Cell size is approximated from
    the transform and mid-latitude of the raster.
    """
    elev = elevation.copy()
    elev[elev == -9999] = np.nan

    res_x_deg = abs(transform.a)
    res_y_deg = abs(transform.e)
    lat_mid   = transform.f + (elev.shape[0] / 2) * transform.e

    cell_x_m = res_x_deg * 111_320 * math.cos(math.radians(lat_mid))
    cell_y_m = res_y_deg * 111_000

    gy, gx  = np.gradient(elev, cell_y_m, cell_x_m)
    slope   = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
    aspect  = np.degrees(np.arctan2(-gx, gy)) % 360

    return slope, aspect


def _circular_mean(angles: np.ndarray) -> float:
    """Circular mean of an array of angles in degrees."""
    valid = angles[~np.isnan(angles)]
    if len(valid) == 0:
        return float("nan")
    r = np.deg2rad(valid)
    return float(np.degrees(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))) % 360)


# ---------------------------------------------------------------------------
# Zonal statistics
# ---------------------------------------------------------------------------

def _write_tmp_tif(arr: np.ndarray, transform, crs: CRS, path: str) -> None:
    """Write a float32 array to a temporary GeoTIFF for rasterstats."""
    h, w = arr.shape
    with rasterio.open(path, "w", driver="GTiff", dtype="float32",
                       width=w, height=h, count=1, crs=crs,
                       transform=transform, nodata=float("nan")) as dst:
        dst.write(arr.astype("float32"), 1)


def _aspect_zonal(geoms: list, aspect_tif: str) -> list[Optional[float]]:
    """Circular mean aspect per geometry using raster pixel values."""
    stats = zonal_stats(geoms, aspect_tif, stats=[], raster_out=True,
                        nodata=float("nan"), all_touched=True)
    results: list[Optional[float]] = []
    for s in stats:
        mini = s.get("mini_raster_array")
        if mini is None or not hasattr(mini, "compressed"):
            results.append(None)
        else:
            results.append(_circular_mean(mini.compressed().astype(float)))
    return results


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute_srtm_features(
    grid: gpd.GeoDataFrame,
    tile_dir: str | Path,
    output_dir: str | Path,
    resolution_km: int,
) -> Path:
    """Mosaic SRTM tiles and compute per-H3-cell elevation, slope, aspect.

    Args:
        grid:          Full H3 GeoDataFrame (geometry in EPSG:4326).
        tile_dir:      Directory containing srtm_<region>_*.tif files.
        output_dir:    Directory to write the output parquet.
        resolution_km: Grid resolution (used in the output filename).

    Returns:
        Path to srtm_features_<N>km.parquet.
    """
    out_path = Path(output_dir) / f"srtm_features_{resolution_km}km.parquet"
    tile_dir = Path(tile_dir)
    tmp_dir  = Path(output_dir) / "srtm_tmp"
    tmp_dir.mkdir(exist_ok=True)

    records: list[dict] = []

    for region in _REGION_TILE_PREFIXES:
        tiles = _find_tiles(tile_dir, region)
        if not tiles:
            logger.warning(
                "No SRTM tiles found for region '%s' in %s — "
                "expected files like srtm_%s_*.tif. Skipping.",
                region, tile_dir, region,
            )
            continue

        region_grid = grid[grid["region"] == region] if "region" in grid.columns else grid
        if region_grid.empty:
            logger.warning("No H3 cells for region '%s' — skipping", region)
            continue

        logger.info("Mosaicking %d SRTM tile(s) for '%s' …", len(tiles), region)
        elevation, transform, crs = _mosaic_region(tiles)

        logger.info("Deriving slope and aspect for '%s' …", region)
        slope, aspect = _compute_slope_aspect(elevation, transform)

        # Replace nodata in elevation
        elevation[elevation == -9999] = np.nan

        # Write temp GeoTIFFs for rasterstats
        elev_tif   = str(tmp_dir / f"elev_{region}.tif")
        slope_tif  = str(tmp_dir / f"slope_{region}.tif")
        aspect_tif = str(tmp_dir / f"aspect_{region}.tif")
        _write_tmp_tif(elevation, transform, crs, elev_tif)
        _write_tmp_tif(slope,     transform, crs, slope_tif)
        _write_tmp_tif(aspect,    transform, crs, aspect_tif)

        geoms = list(region_grid.geometry)
        logger.info("Computing terrain zonal stats: %d H3 cells for '%s' …",
                    len(geoms), region)

        elev_stats  = zonal_stats(geoms, elev_tif,  stats=["mean"],
                                  nodata=float("nan"), all_touched=True)
        slope_stats = zonal_stats(geoms, slope_tif, stats=["mean"],
                                  nodata=float("nan"), all_touched=True)
        aspect_vals = _aspect_zonal(geoms, aspect_tif)

        for i, row in enumerate(region_grid.itertuples()):
            records.append({
                "grid_id":        row.grid_id,
                "latitude":       row.latitude,
                "longitude":      row.longitude,
                "elevation_m":    elev_stats[i].get("mean"),
                "slope_degrees":  slope_stats[i].get("mean"),
                "aspect_degrees": aspect_vals[i],
            })

        for p in [elev_tif, slope_tif, aspect_tif]:
            Path(p).unlink(missing_ok=True)

    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False)
    logger.info("Wrote SRTM features: %s (%d rows)", out_path, len(df))
    return out_path


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run(
    resolution_km: int = 64,
    output_dir: str | Path = "data/static",
    force_rebuild: bool = False,
) -> Path:
    """Read SRTM tiles from data/static/srtm_raw/ and compute H3 terrain features.

    Args:
        resolution_km:  H3 grid resolution.
        output_dir:     Directory containing srtm_raw/ and where the output
                        parquet is written.
        force_rebuild:  Recompute even if the parquet already exists.

    Returns:
        Path to srtm_features_<N>km.parquet.
    """
    out_path = Path(output_dir) / f"srtm_features_{resolution_km}km.parquet"
    if out_path.exists() and not force_rebuild:
        logger.info("SRTM cache hit: %s", out_path)
        return out_path

    tile_dir = Path(output_dir) / "srtm_raw"
    if not tile_dir.exists():
        raise FileNotFoundError(
            f"SRTM tile directory not found: {tile_dir}\n"
            "Download tiles from OpenTopography and place them there.\n"
            "See the module docstring for download URLs and expected filenames."
        )

    grid = generate_full_grid(resolution_km)
    return compute_srtm_features(grid, tile_dir, output_dir, resolution_km)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute H3 terrain features (elevation, slope, aspect) from "
            "manually downloaded SRTM GeoTIFF tiles in data/static/srtm_raw/."
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
