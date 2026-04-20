"""
LANDFIRE Ingestion
==================
Reads manually downloaded LANDFIRE GeoTIFF rasters and computes zonal
statistics over the H3 hexagonal grid.

--- HOW TO OBTAIN THE DATA ---

Go to: https://landfire.gov/data/FullExtentDownloads

REQUIRED (LF 2022 → CONUS → Fuel):
    FBFM40  (fuel model)     : LF2022_FBFM40_230_CONUS.zip  → LC22_F40_230.tif
    CC      (canopy cover)   : LF2022_CC_230_CONUS.zip       → LC22_CC_230.tif
    EVT     (vegetation type): LF2022_EVT_230_CONUS.zip      → LC22_EVT_230.tif

OPTIONAL — fire spread simulation features:
    CBH (canopy base height)  : LF2020_CBH_200_CONUS.zip     → US_200CBH.tif
        Raw units: tenths of meters (÷ 10 = meters).
        Critical for crown fire initiation threshold (Van Wagner model).
    CBD (canopy bulk density) : LF2024_CBD_240_CONUS.zip     → US_240CBD.tif
        Raw units: kg per 100 m³ (÷ 100 = kg/m³).
        Controls active crown fire spread rate (Scott & Reinhardt 2001).
    EVT-CNC (national class)  : LF2020_EVT_200_CONUS.zip     → US_200EVT.tif
        Updated EVT with NVC-aligned classes for foliar moisture estimation.

Extract each ZIP and place .tif files in:
    data/static/landfire_raw/

Keyword matching (case-insensitive, in filename):
    FBFM40: "f40" or "fbfm40"
    CC    : "_cc_" or "_cc."
    EVT   : "evt" (but NOT "cbh", "cbd")
    CBH   : "cbh"
    CBD   : "cbd"
    EVT-CNC: "evtc" or "200evt" or "us_200evt"

The pipeline continues gracefully if optional rasters are absent —
their columns will be NaN in the output parquet.

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
    """Find required and optional LANDFIRE rasters in raw_dir.

    Required layers (FileNotFoundError if missing):
        fbfm40, cc, evt

    Optional layers (logged warning if missing, omitted from returned dict):
        cbh       — Canopy Base Height (LF2020, raw ×10 = tenths of metres)
        cbd       — Canopy Bulk Density (LF2024, raw ×100 = kg per 100 m³)
        evt_cnc   — EVT National Canopy Class (LF2020)

    EVT-CNC uses the same LF2020 EVT product as `evt` but sourced from the
    LF2020 vintage.  If both an LF2022 and LF2020 EVT file are present, the
    keyword search returns the first alphabetical match; rename files to
    disambiguate (e.g. prefix with "LF2022_" vs "LF2020_").
    """
    # NOTE: FBFM40, CC, EVT are preferred but optional if CBH/CBD are available
    # (for fire spread crown fire simulation without fuel reclassification).
    preferred_mapping = {
        "fbfm40": ["f40", "fbfm40"],
        "cc":     ["_cc_", "_cc."],
        "evt":    ["evt"],
    }
    optional_mapping = {
        "cbh":     ["cbh"],
        "cbd":     ["cbd"],
        "evt_cnc": ["evtc", "200evt", "us_200evt"],
    }

    found: dict[str, Path] = {}
    missing_preferred: list[str] = []

    for layer, keywords in preferred_mapping.items():
        path = _find_raster(raw_dir, keywords)
        if path:
            found[layer] = path
            logger.info("Found LANDFIRE %s: %s", layer.upper(), path.name)
        else:
            missing_preferred.append(layer.upper())

    if missing_preferred:
        logger.warning(
            "Missing preferred LANDFIRE rasters in %s: %s\n"
            "If available, download from https://landfire.gov/data/FullExtentDownloads "
            "(LF 2022 → CONUS → Fuel) and place extracted .tif files there.\n"
            "Expected keywords in filenames: F40/FBFM40, _CC_, EVT.\n"
            "Proceeding with optional layers only (e.g. CBH/CBD for crown fire).",
            raw_dir, missing_preferred
        )

    for layer, keywords in optional_mapping.items():
        path = _find_raster(raw_dir, keywords)
        if path:
            found[layer] = path
            logger.info("Found optional LANDFIRE %s: %s", layer.upper(), path.name)
        else:
            logger.info(
                "Optional LANDFIRE %s not found in %s — column will be NaN. "
                "See module docstring for download instructions.",
                layer.upper(), raw_dir,
            )

    return found


# ---------------------------------------------------------------------------
# Reprojection helper
# ---------------------------------------------------------------------------

def _reproject_to_wgs84(
    raster_path: Path,
    clip_bounds_wgs84: tuple[float, float, float, float] | None = None,
    resampling: Resampling = Resampling.nearest,
) -> str:
    """Reproject to WGS84, clipping and resampling to _TARGET_RES_DEG.

    LANDFIRE source data is 30 m (EPSG:5070).  Full CONUS at 30 m is ~90k×100k
    pixels — too large to allocate.  We clip to the CA+TX bbox and resample to
    ~1 km (_TARGET_RES_DEG), which is ample precision for 64-km H3 cells.
    The result is a small temp _wgs84.tif written next to the original.

    Args:
        raster_path:       Source GeoTIFF (EPSG:5070, 30 m).
        clip_bounds_wgs84: Optional (west, south, east, north) clip window.
        resampling:        Resampling algorithm.  Use ``Resampling.nearest``
                           for categorical rasters (FBFM40, EVT) and
                           ``Resampling.bilinear`` for continuous ones (CBH, CBD, CC).
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
            "Reprojecting %s → WGS84 at %.3f° [%d×%d px] (resampling=%s) …",
            raster_path.name, _TARGET_RES_DEG, dst_width, dst_height,
            resampling.name,
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
                    resampling=resampling,
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

    Required layers in raster_paths: fbfm40, cc, evt.
    Optional layers (produce NaN column if absent): cbh, cbd, evt_cnc.

    CBH raw-to-output conversion  : value ÷ 10  → canopy_base_height_m
    CBD raw-to-output conversion  : value ÷ 100 → canopy_bulk_density (kg/m³)
    EVT-CNC                       : mode code  → evt_national_class

    Args:
        grid:          Full H3 GeoDataFrame (geometry in EPSG:4326).
        raster_paths:  Layer-name → Path mapping from _locate_rasters().
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

    tmp_files: list[str] = []

    # ── Required layers ────────────────────────────────────────────────────────
    fbfm40_wgs84 = _reproject_to_wgs84(raster_paths["fbfm40"], clip,
                                        resampling=Resampling.nearest)
    cc_wgs84     = _reproject_to_wgs84(raster_paths["cc"],     clip,
                                        resampling=Resampling.bilinear)
    evt_wgs84    = _reproject_to_wgs84(raster_paths["evt"],    clip,
                                        resampling=Resampling.nearest)
    tmp_files.extend([fbfm40_wgs84, cc_wgs84, evt_wgs84])

    fbfm_stats = _zonal_mode(fbfm40_wgs84, geoms)

    with rasterio.open(cc_wgs84) as _cc_src:
        cc_nd = _cc_src.nodata if _cc_src.nodata is not None else -9999
    cc_stats  = zonal_stats(geoms, cc_wgs84, stats=["mean"],
                            nodata=cc_nd, all_touched=True)
    evt_stats = _zonal_mode(evt_wgs84, geoms)

    df = grid[["grid_id", "latitude", "longitude"]].copy()
    df["fuel_model_fbfm40"]      = [s["mode"] for s in fbfm_stats]
    df["canopy_cover_pct"]       = [s.get("mean") for s in cc_stats]
    df["vegetation_type"]        = [s["mode"] for s in evt_stats]
    df["dominant_fuel_fraction"] = [s["frac"] for s in fbfm_stats]

    # ── Optional: CBH — Canopy Base Height ────────────────────────────────────
    # Raw values are tenths of meters (e.g. 15 → 1.5 m).  Nodata = -9999 / 0.
    if "cbh" in raster_paths:
        cbh_wgs84 = _reproject_to_wgs84(raster_paths["cbh"], clip,
                                         resampling=Resampling.bilinear)
        tmp_files.append(cbh_wgs84)
        with rasterio.open(cbh_wgs84) as _cbh_src:
            cbh_nd = _cbh_src.nodata if _cbh_src.nodata is not None else -9999
        cbh_stats = zonal_stats(geoms, cbh_wgs84, stats=["mean"],
                                nodata=cbh_nd, all_touched=True)
        _CBH_NODATA = {-9999, 0, 32767, 32768, 65535}
        df["canopy_base_height_m"] = [
            round(s["mean"] / 10.0, 2)
            if s.get("mean") is not None and float(s["mean"]) not in _CBH_NODATA
            else None
            for s in cbh_stats
        ]
        logger.info("CBH zonal stats complete (canopy_base_height_m)")
    else:
        df["canopy_base_height_m"] = np.nan
        logger.info("CBH not available — canopy_base_height_m set to NaN")

    # ── Optional: CBD — Canopy Bulk Density ───────────────────────────────────
    # Raw values are kg per 100 m³ (e.g. 8 → 0.08 kg/m³).  Nodata = -9999 / 0.
    if "cbd" in raster_paths:
        cbd_wgs84 = _reproject_to_wgs84(raster_paths["cbd"], clip,
                                         resampling=Resampling.bilinear)
        tmp_files.append(cbd_wgs84)
        with rasterio.open(cbd_wgs84) as _cbd_src:
            cbd_nd = _cbd_src.nodata if _cbd_src.nodata is not None else -9999
        cbd_stats = zonal_stats(geoms, cbd_wgs84, stats=["mean"],
                                nodata=cbd_nd, all_touched=True)
        _CBD_NODATA = {-9999, 0, 32767, 32768, 65535}
        df["canopy_bulk_density"] = [
            round(s["mean"] / 100.0, 4)
            if s.get("mean") is not None and float(s["mean"]) not in _CBD_NODATA
            else None
            for s in cbd_stats
        ]
        logger.info("CBD zonal stats complete (canopy_bulk_density kg/m³)")
    else:
        df["canopy_bulk_density"] = np.nan
        logger.info("CBD not available — canopy_bulk_density set to NaN")

    # ── Optional: EVT-CNC — National Canopy Class ─────────────────────────────
    if "evt_cnc" in raster_paths:
        evt_cnc_wgs84 = _reproject_to_wgs84(raster_paths["evt_cnc"], clip,
                                              resampling=Resampling.nearest)
        tmp_files.append(evt_cnc_wgs84)
        evt_cnc_stats = _zonal_mode(evt_cnc_wgs84, geoms)
        df["evt_national_class"] = [s["mode"] for s in evt_cnc_stats]
        logger.info("EVT-CNC zonal stats complete (evt_national_class)")
    else:
        df["evt_national_class"] = np.nan
        logger.info("EVT-CNC not available — evt_national_class set to NaN")

    # ── Cleanup temp files ─────────────────────────────────────────────────────
    for tmp in tmp_files:
        if tmp.endswith("_wgs84.tif"):
            Path(tmp).unlink(missing_ok=True)

    df.to_parquet(out_path, index=False)
    logger.info("Wrote LANDFIRE features: %s (%d rows, %d columns)",
                out_path, len(df), len(df.columns))
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
            "Compute H3 LANDFIRE features from manually downloaded rasters. "
            "Supports multiple resolutions in one run (e.g. --resolution-km 64 22). "
            "Rasters are read from --raw-dir (default: <output-dir>/landfire_raw/). "
            "See module docstring for download instructions."
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
    p.add_argument("--output-dir", default="data/static",
                   help="Directory containing landfire_raw/ and where output parquets are written.")
    p.add_argument(
        "--raw-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory containing the raw LANDFIRE .tif files. "
            "Defaults to <output-dir>/landfire_raw/. "
            "Use this to point directly at a custom download folder."
        ),
    )
    p.add_argument("--force-rebuild", action="store_true",
                   help="Recompute parquet even if cached file already exists.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    resolutions = args.resolution_km  # list of ints
    raw_dir_override = Path(args.raw_dir) if args.raw_dir else None

    logger.info(
        "Running LANDFIRE ingest for resolutions: %s  |  regions: California + Texas",
        resolutions,
    )

    outputs = []
    for res_km in resolutions:
        logger.info("─── Resolution %d km ───────────────────────────────────", res_km)

        out_path = Path(args.output_dir) / f"landfire_features_{res_km}km.parquet"
        if out_path.exists() and not args.force_rebuild:
            logger.info("Cache hit — skipping (use --force-rebuild to overwrite): %s", out_path)
            outputs.append(out_path)
            continue

        # Determine raw raster directory
        raw_dir = raw_dir_override if raw_dir_override else Path(args.output_dir) / "landfire_raw"
        if not raw_dir.exists():
            raise FileNotFoundError(
                f"LANDFIRE raw directory not found: {raw_dir}\n"
                "Download rasters from https://landfire.gov/data/FullExtentDownloads "
                "and place extracted .tif files there, or use --raw-dir to point "
                "to your download folder."
            )

        raster_paths = _locate_rasters(raw_dir)
        grid = generate_full_grid(res_km)
        out = compute_landfire_features(grid, raster_paths, args.output_dir, res_km)
        outputs.append(out)

    print("\n=== LANDFIRE ingest complete ===")
    for o in outputs:
        print(f"  {o}")
