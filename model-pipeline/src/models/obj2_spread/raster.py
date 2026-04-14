"""
Raster utilities for fire spread simulation.

Handles GeoTIFF clipping to AOI and parsing output grids
into burn probability arrays.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def clip_raster_to_aoi(
    raster_path: str | Path,
    bounds: tuple[float, float, float, float],
    output_path: str | Path,
    target_resolution: float | None = None,
) -> Path:
    """Clip a GeoTIFF to a bounding box, optionally resampling.

    Parameters
    ----------
    raster_path : path
        Source GeoTIFF.
    bounds : (west, south, east, north)
        Bounding box in the raster's CRS.
    output_path : path
        Where to write the clipped raster.
    target_resolution : float | None
        If provided, resample to this cell size in CRS units.
        For EPSG:4326 use degrees (~0.00027 per 30m).
        For projected rasters use metres (e.g. 30.0).

    Returns
    -------
    Path to the clipped GeoTIFF.

    Raises
    ------
    ValueError
        If the clip produces an empty raster.
    """
    import rasterio
    from rasterio.windows import from_bounds

    raster_path = Path(raster_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(raster_path) as src:
        window = from_bounds(*bounds, transform=src.transform)
        data = src.read(1, window=window)

        if data.size == 0 or data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError(
                f"Clip produced empty raster. "
                f"Bounds {bounds} may be outside raster extent {src.bounds}. "
                f"Raster CRS: {src.crs}."
            )

        win_transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(
            height=data.shape[0],
            width=data.shape[1],
            transform=win_transform,
        )

        if target_resolution is not None:
            from rasterio.enums import Resampling
            from rasterio.transform import from_bounds as tfm_from_bounds

            scale_x = abs(src.res[0]) / target_resolution
            scale_y = abs(src.res[1]) / target_resolution
            new_h = max(1, int(data.shape[0] * scale_y))
            new_w = max(1, int(data.shape[1] * scale_x))

            data = src.read(
                1,
                window=window,
                out_shape=(new_h, new_w),
                resampling=Resampling.bilinear,
            )
            profile.update(
                height=new_h,
                width=new_w,
                transform=tfm_from_bounds(*bounds, new_w, new_h),
            )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)

    logger.info(
        "Clipped %s → %s (%dx%d cells)",
        raster_path.name, output_path.name,
        data.shape[1], data.shape[0],
    )
    return output_path


def parse_burn_probability(
    output_dir: Path,
    n_simulations: int,
) -> np.ndarray:
    """Read simulation output grids and compute per-cell burn probability.

    Expects one CSV grid per simulation under output/Grids/Grids1/.
    Burn probability = fraction of simulations where each cell burned.

    Parameters
    ----------
    output_dir : Path
        Simulation --output-folder path.
    n_simulations : int
        Number of simulations run (denominator).

    Returns
    -------
    np.ndarray
        2D float array of burn probabilities in [0, 1].

    Raises
    ------
    ValueError
        If no output grid files are found.
    """
    grids_dir = output_dir / "Grids" / "Grids1"
    if not grids_dir.exists():
        grids_dir = output_dir / "Grids"

    grid_files = sorted(grids_dir.glob("ForestGrid*.csv"))
    if not grid_files:
        raise ValueError(
            f"No output grids found in {grids_dir}. "
            "Check that the simulation ran with --grids flag."
        )

    logger.info("Parsing %d grid files from %s", len(grid_files), grids_dir)

    burn_count: np.ndarray | None = None
    for gf in grid_files:
        try:
            grid = np.loadtxt(gf, delimiter=",", skiprows=1)
        except Exception as e:
            logger.warning("Failed to parse grid file %s: %s", gf.name, e)
            continue

        burned = (grid > 0).astype(float)
        if burn_count is None:
            burn_count = burned
        else:
            if burned.shape != burn_count.shape:
                logger.warning(
                    "Grid shape mismatch: %s vs %s — skipping %s",
                    burned.shape, burn_count.shape, gf.name,
                )
                continue
            burn_count += burned

    if burn_count is None:
        raise ValueError("All grid files failed to parse.")

    burn_prob = burn_count / max(n_simulations, 1)
    logger.info(
        "Burn probability — shape: %dx%d | mean: %.3f | max: %.3f | "
        "pct_burned: %.1f%%",
        burn_prob.shape[1], burn_prob.shape[0],
        burn_prob.mean(), burn_prob.max(),
        (burn_prob > 0).mean() * 100,
    )
    return burn_prob
