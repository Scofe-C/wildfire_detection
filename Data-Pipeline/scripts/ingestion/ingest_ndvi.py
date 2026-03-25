"""
MODIS NDVI Ingestion (MOD13A2 v061)
=====================================
Downloads MODIS 16-day NDVI composites for California and Texas using the
earthaccess library (NASA Earthdata) and computes per-H3-cell zonal means.

Product: MOD13A2 v061 (Terra NDVI, 1 km, 16-day composite)
Band used: "1 km 16 days NDVI" (int16, scale factor 0.0001)

Prerequisites:
    pip install earthaccess
    # Then authenticate once:
    python -c "import earthaccess; earthaccess.login(persist=True)"
    # Or set env vars: EARTHDATA_USER, EARTHDATA_PASSWORD

Usage (CLI):
    python -m scripts.ingestion.ingest_ndvi \\
        --resolution-km 64 \\
        --output-dir data/static \\
        --start-date 2024-05-01 \\
        --end-date 2024-06-30

Output files:
    ndvi_raw/              — downloaded HDF/GeoTIFF files (cached)
    ndvi_features_<N>km.parquet — per-H3-cell NDVI values [0.0, 1.0]
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
from rasterstats import zonal_stats
from rasterio.crs import CRS
from rasterio.enums import Resampling

from scripts.utils.grid_utils import generate_full_grid

logger = logging.getLogger(__name__)

_WGS84 = CRS.from_epsg(4326)

# MOD13A2 product short name and version
_MODIS_SHORT_NAME = "MOD13A2"
_MODIS_VERSION    = "061"

# MODIS Sinusoidal CRS (official WKT from NASA)
_MODIS_SINU_PROJ4 = (
    "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 "
    "+a=6371007.181 +b=6371007.181 +units=m +no_defs"
)

# Bounding boxes: (W, S, E, N) — earthaccess expects (lower_left, upper_right)
_REGION_BBOXES = {
    "california": (-124.48, 32.53, -114.13, 42.01),
    "texas":      (-106.65, 25.84,  -93.51, 36.50),
}

# Combined bbox for a single search covering both regions
_COMBINED_BBOX = (
    min(b[0] for b in _REGION_BBOXES.values()),  # min lon
    min(b[1] for b in _REGION_BBOXES.values()),  # min lat
    max(b[2] for b in _REGION_BBOXES.values()),  # max lon
    max(b[3] for b in _REGION_BBOXES.values()),  # max lat
)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_ndvi(
    output_dir: str | Path,
    start_date: str,
    end_date: str,
    regions: list[str] | None = None,
    force: bool = False,
) -> list[Path]:
    """Download MOD13A2 NDVI granules for the given date range and regions.

    Args:
        output_dir:  Root directory; files stored in ``output_dir/ndvi_raw/``.
        start_date:  ISO date string, e.g. "2024-05-01".
        end_date:    ISO date string, e.g. "2024-06-30".
        regions:     Regions to cover (default: california + texas).
        force:       Re-download files that already exist.

    Returns:
        List of Paths to downloaded files (HDF4 or GeoTIFF depending on
        what earthaccess returns for the requested product).
    """
    try:
        import earthaccess
    except ImportError:
        raise ImportError(
            "earthaccess is required for MODIS NDVI download. "
            "Install it with: pip install earthaccess"
        )

    raw_dir = Path(output_dir) / "ndvi_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Authenticate (uses persisted token, env vars, or .netrc)
    _earthdata_login(earthaccess)

    # Determine search bounding box
    if regions is None:
        regions = list(_REGION_BBOXES.keys())

    if len(regions) == 1 and regions[0] in _REGION_BBOXES:
        bbox = _REGION_BBOXES[regions[0]]
    else:
        # Union of all requested regions
        lons_min = [_REGION_BBOXES[r][0] for r in regions if r in _REGION_BBOXES]
        lats_min = [_REGION_BBOXES[r][1] for r in regions if r in _REGION_BBOXES]
        lons_max = [_REGION_BBOXES[r][2] for r in regions if r in _REGION_BBOXES]
        lats_max = [_REGION_BBOXES[r][3] for r in regions if r in _REGION_BBOXES]
        bbox = (min(lons_min), min(lats_min), max(lons_max), max(lats_max))

    logger.info(
        "Searching MOD13A2 v%s for %s → %s over bbox %s …",
        _MODIS_VERSION, start_date, end_date, bbox
    )

    results = earthaccess.search_data(
        short_name=_MODIS_SHORT_NAME,
        version=_MODIS_VERSION,
        temporal=(start_date, end_date),
        bounding_box=bbox,
    )

    if not results:
        logger.warning(
            "No MOD13A2 granules found for %s → %s. "
            "NDVI will remain NaN.",
            start_date, end_date,
        )
        return []

    logger.info("Found %d MOD13A2 granules — downloading …", len(results))

    downloaded_paths: list[Path] = []
    for granule in results:
        # earthaccess returns objects with .data_links() or similar
        links = granule.data_links(access="direct") or granule.data_links()
        for link in links:
            filename = Path(link).name
            dest = raw_dir / filename
            if dest.exists() and not force:
                downloaded_paths.append(dest)
                continue
            try:
                earthaccess.download([granule], local_path=str(raw_dir))
                if dest.exists():
                    downloaded_paths.append(dest)
            except Exception as exc:
                logger.warning("Failed to download %s: %s", filename, exc)
            break  # one file per granule

    logger.info("Downloaded %d NDVI files to %s", len(downloaded_paths), raw_dir)
    return downloaded_paths


def _earthdata_login(earthaccess) -> None:
    """Authenticate with NASA Earthdata, preferring env vars."""
    user = os.environ.get("EARTHDATA_USER")
    pwd  = os.environ.get("EARTHDATA_PASSWORD")
    if user and pwd:
        earthaccess.login(strategy="environment")
    else:
        earthaccess.login(strategy="netrc")


# ---------------------------------------------------------------------------
# HDF/GeoTIFF → WGS84 reprojection
# ---------------------------------------------------------------------------

def _open_ndvi_band(hdf_path: Path) -> tuple[np.ndarray, object, CRS]:
    """Open the NDVI band from a MOD13A2 HDF file.

    MOD13A2 HDF4 files contain subdatasets. The NDVI band is named
    'HDF4_EOS:EOS_GRID:<file>:MOD_Grid_16DAY_1km_VI:1 km 16 days NDVI'.

    Returns:
        (data_int16, transform, src_crs)
    """
    with rasterio.open(str(hdf_path)) as src:
        # Check if this is already a single-band GeoTIFF
        if src.count >= 1 and src.driver != "HDF4Image":
            data = src.read(1)
            return data, src.transform, src.crs

        # HDF4: list subdatasets and pick the NDVI one
        subdatasets = src.subdatasets

    ndvi_ds = None
    for sd in subdatasets:
        if "NDVI" in sd and "16" in sd:
            ndvi_ds = sd
            break

    if ndvi_ds is None:
        # Fallback: use first subdataset
        ndvi_ds = subdatasets[0] if subdatasets else str(hdf_path)
        logger.warning("Could not identify NDVI subdataset; using: %s", ndvi_ds)

    with rasterio.open(ndvi_ds) as src:
        data = src.read(1)
        return data, src.transform, src.crs


def _reproject_ndvi_to_wgs84(
    data_int16: np.ndarray,
    src_transform,
    src_crs: CRS,
    fill_value: int = -3000,
) -> tuple[np.ndarray, object]:
    """Reproject NDVI array from Sinusoidal to WGS84.

    Returns:
        (reprojected_float32_array, dst_transform)
        Values outside valid NDVI range (-2000..10000) are set to NaN.
    """
    # Mask fill / nodata values before reprojection
    data_f = data_int16.astype("float32")
    data_f[data_int16 == fill_value] = np.nan
    data_f[data_int16 < -2000]       = np.nan

    dst_crs = _WGS84
    dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
        src_crs, dst_crs,
        data_int16.shape[1], data_int16.shape[0],
        *rasterio.transform.array_bounds(
            data_int16.shape[0], data_int16.shape[1], src_transform
        ),
    )

    dst = np.full((dst_height, dst_width), np.nan, dtype="float32")

    rasterio.warp.reproject(
        source=data_f,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    return dst, dst_transform


# ---------------------------------------------------------------------------
# Zonal statistics
# ---------------------------------------------------------------------------

def compute_ndvi_features(
    grid: gpd.GeoDataFrame,
    ndvi_files: list[Path],
    output_dir: str | Path,
    resolution_km: int,
) -> Path:
    """Compute per-H3-cell NDVI from MOD13A2 files.

    When multiple files/tiles cover the same area, values are averaged.
    Raw int16 values are scaled by 0.0001 → float [0.0, 1.0].

    Args:
        grid:          GeoDataFrame with geometry column (H3 hexagons, EPSG:4326).
        ndvi_files:    List of downloaded HDF or GeoTIFF paths.
        output_dir:    Directory to cache the output parquet.
        resolution_km: Used in the output filename.

    Returns:
        Path to ndvi_features_<N>km.parquet.
    """
    out_path = Path(output_dir) / f"ndvi_features_{resolution_km}km.parquet"

    if not ndvi_files:
        logger.warning("No NDVI files provided — writing NaN parquet")
        df = grid[["grid_id", "latitude", "longitude"]].copy()
        df["ndvi"] = np.nan
        df.to_parquet(out_path, index=False)
        return out_path

    geoms = list(grid.geometry)
    ndvi_tmp = Path(output_dir) / "ndvi_tmp"
    ndvi_tmp.mkdir(exist_ok=True)

    # Accumulate per-file zonal means, then average across tiles
    ndvi_accum: list[list[Optional[float]]] = []

    for hdf_path in ndvi_files:
        try:
            data_int16, transform, crs = _open_ndvi_band(hdf_path)
        except Exception as exc:
            logger.warning("Could not open %s: %s", hdf_path, exc)
            continue

        # Reproject to WGS84
        data_wgs84, dst_transform = _reproject_ndvi_to_wgs84(
            data_int16, transform, crs
        )

        # Scale: raw int16 values are -2000..10000 (fill=-3000)
        # Apply scale factor to get actual NDVI [-0.2..1.0]
        data_scaled = data_wgs84 * 0.0001

        # Write temp GeoTIFF for rasterstats
        h, w = data_scaled.shape
        tmp_tif = str(ndvi_tmp / f"{hdf_path.stem}_ndvi.tif")
        profile = {
            "driver":    "GTiff",
            "dtype":     "float32",
            "width":     w,
            "height":    h,
            "count":     1,
            "crs":       _WGS84,
            "transform": dst_transform,
            "nodata":    float("nan"),
        }
        with rasterio.open(tmp_tif, "w", **profile) as dst:
            dst.write(data_scaled, 1)

        stats = zonal_stats(
            geoms, tmp_tif,
            stats=["mean"],
            nodata=float("nan"),
            all_touched=True,
        )
        ndvi_accum.append([s.get("mean") for s in stats])
        Path(tmp_tif).unlink(missing_ok=True)

    # Clean up tmp dir
    try:
        ndvi_tmp.rmdir()
    except OSError:
        pass

    if not ndvi_accum:
        ndvi_vals = [None] * len(grid)
    else:
        # Per-cell mean across all tiles
        arr = np.array(ndvi_accum, dtype=float)  # shape (n_files, n_cells)
        with np.errstate(all="ignore"):
            ndvi_vals = np.nanmean(arr, axis=0).tolist()

    df = grid[["grid_id", "latitude", "longitude"]].copy()
    df["ndvi"] = ndvi_vals

    df.to_parquet(out_path, index=False)
    logger.info("Wrote NDVI features: %s (%d rows)", out_path, len(df))
    return out_path


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run(
    resolution_km: int = 64,
    output_dir: str | Path = "data/static",
    start_date: str = "2024-01-01",
    end_date: str | None = None,
    regions: list[str] | None = None,
    force_rebuild: bool = False,
) -> Path:
    """Download MOD13A2 NDVI (if needed) and compute H3 cell features.

    Args:
        resolution_km:  H3 grid resolution.
        output_dir:     Directory for output parquet and cached HDF files.
        start_date:     Start of NDVI composite search window (ISO date).
        end_date:       End of search window (default: today).
        regions:        Regions to cover (default: all configured).
        force_rebuild:  Re-download and recompute even if parquet exists.

    Returns:
        Path to ndvi_features_<N>km.parquet.
    """
    import datetime

    out_path = Path(output_dir) / f"ndvi_features_{resolution_km}km.parquet"
    if out_path.exists() and not force_rebuild:
        logger.info("NDVI cache hit: %s", out_path)
        return out_path

    if end_date is None:
        end_date = datetime.date.today().isoformat()

    ndvi_files = download_ndvi(
        output_dir,
        start_date=start_date,
        end_date=end_date,
        regions=regions,
        force=force_rebuild,
    )

    grid = generate_full_grid(resolution_km)
    return compute_ndvi_features(grid, ndvi_files, output_dir, resolution_km)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    import datetime

    p = argparse.ArgumentParser(
        description="Download MOD13A2 NDVI composites and compute H3 zonal features."
    )
    p.add_argument("--resolution-km", type=int, default=64)
    p.add_argument("--output-dir", default="data/static")
    p.add_argument("--start-date", default="2024-01-01",
                   help="Start of NDVI composite search (ISO date, default: 2024-01-01)")
    p.add_argument("--end-date", default=datetime.date.today().isoformat(),
                   help="End of NDVI composite search (ISO date, default: today)")
    p.add_argument("--regions", nargs="+", default=None)
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
        start_date=args.start_date,
        end_date=args.end_date,
        regions=args.regions,
        force_rebuild=args.force_rebuild,
    )
    print(f"Output: {out}")
