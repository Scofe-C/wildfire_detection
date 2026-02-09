"""
H3 Grid Utilities
=================
Provides the shared spatial grid system used by all pipeline components.
All coordinate-to-grid-cell conversions, grid generation, and spatial
lookups go through this module.

Owner: Person C (Static Layers + Grid System)
Consumers: All team members

Dependencies:
    pip install h3 geopandas shapely numpy
"""

import logging
from typing import Optional

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from scripts.utils.schema_loader import get_registry

logger = logging.getLogger(__name__)


def km_to_h3_resolution(km: int, config_path: Optional[str] = None) -> int:
    """Convert a km resolution to the corresponding H3 resolution level.

    Args:
        km: Target resolution in kilometers (64, 10, or 1).
        config_path: Optional path to schema config override.

    Returns:
        H3 resolution integer.

    Raises:
        ValueError: If km is not a supported resolution.
    """
    registry = get_registry(config_path)
    return registry.get_h3_resolution(km)


def generate_grid_for_bbox(
    bbox: list[float],
    resolution_km: int = 64,
    config_path: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Generate H3 hexagonal grid cells covering a bounding box.

    Args:
        bbox: [west, south, east, north] in WGS84 degrees.
        resolution_km: Grid resolution in km (64, 10, or 1).
        config_path: Optional schema config override.

    Returns:
        GeoDataFrame with columns: grid_id, latitude, longitude, geometry.
        CRS is EPSG:4326.
    """
    h3_res = km_to_h3_resolution(resolution_km, config_path)
    west, south, east, north = bbox

    # Create polygon from bbox for h3.polyfill
    bbox_polygon = Polygon([
        (west, south),
        (east, south),
        (east, north),
        (west, north),
        (west, south),
    ])

    # h3.polyfill_geojson expects GeoJSON-like dict
    geojson = {
        "type": "Polygon",
        "coordinates": [list(bbox_polygon.exterior.coords)],
    }

    # Generate all H3 cells that intersect the bounding box
    h3_cells = h3.polyfill_geojson(geojson, h3_res)
    logger.info(
        f"Generated {len(h3_cells)} H3 cells at resolution {h3_res} "
        f"(~{resolution_km} km) for bbox {bbox}"
    )

    if not h3_cells:
        logger.warning(f"No H3 cells generated for bbox {bbox} at res {h3_res}")
        return gpd.GeoDataFrame(
            columns=["grid_id", "latitude", "longitude", "geometry"],
            crs="EPSG:4326",
        )

    # Build GeoDataFrame with cell centroids and boundaries
    records = []
    for cell_id in h3_cells:
        lat, lon = h3.h3_to_geo(cell_id)
        boundary = h3.h3_to_geo_boundary(cell_id, geo_json=True)
        polygon = Polygon(boundary)
        records.append({
            "grid_id": cell_id,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "geometry": polygon,
        })

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    return gdf


def generate_full_grid(
    resolution_km: int = 64,
    config_path: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Generate the complete grid covering all configured geographic regions.

    Merges grids for California and Texas (and any future regions) into
    a single GeoDataFrame with duplicates removed.

    Args:
        resolution_km: Grid resolution in km.
        config_path: Optional schema config override.

    Returns:
        GeoDataFrame covering the full project scope.
    """
    registry = get_registry(config_path)
    bboxes = registry.geographic_bboxes

    grids = []
    for region_name, bbox in bboxes.items():
        logger.info(f"Generating grid for {region_name}: {bbox}")
        grid = generate_grid_for_bbox(bbox, resolution_km, config_path)
        grid["region"] = region_name
        grids.append(grid)

    if not grids:
        raise ValueError("No geographic regions configured in schema_config.yaml")

    full_grid = pd.concat(grids, ignore_index=True)

    # Remove duplicate cells (border regions between CA and TX won't overlap
    # at these distances, but this future-proofs for adjacent regions)
    before_dedup = len(full_grid)
    full_grid = full_grid.drop_duplicates(subset="grid_id", keep="first")
    after_dedup = len(full_grid)
    if before_dedup != after_dedup:
        logger.info(
            f"Removed {before_dedup - after_dedup} duplicate cells at region borders"
        )

    logger.info(
        f"Full grid: {len(full_grid)} cells at {resolution_km} km resolution"
    )
    return full_grid


def point_to_grid_id(
    lat: float,
    lon: float,
    resolution_km: int = 64,
    config_path: Optional[str] = None,
) -> str:
    """Convert a lat/lon point to its containing H3 grid cell ID.

    Args:
        lat: Latitude in WGS84 degrees.
        lon: Longitude in WGS84 degrees.
        resolution_km: Grid resolution in km.

    Returns:
        H3 cell index string.
    """
    h3_res = km_to_h3_resolution(resolution_km, config_path)
    return h3.geo_to_h3(lat, lon, h3_res)


def points_to_grid_ids(
    lats: np.ndarray,
    lons: np.ndarray,
    resolution_km: int = 64,
    config_path: Optional[str] = None,
) -> np.ndarray:
    """Vectorized conversion of lat/lon arrays to H3 grid cell IDs.

    Args:
        lats: Array of latitudes.
        lons: Array of longitudes.
        resolution_km: Grid resolution in km.

    Returns:
        Array of H3 cell index strings.
    """
    h3_res = km_to_h3_resolution(resolution_km, config_path)

    # h3 library does not support vectorized operations natively,
    # so we use a list comprehension (fast enough for FIRMS data volumes)
    grid_ids = np.array([
        h3.geo_to_h3(lat, lon, h3_res)
        for lat, lon in zip(lats, lons)
    ])
    return grid_ids


def get_cell_area_km2(grid_id: str) -> float:
    """Return the approximate area of an H3 cell in square kilometers.

    Args:
        grid_id: H3 cell index string.

    Returns:
        Area in km².
    """
    res = h3.h3_get_resolution(grid_id)
    return h3.hex_area(res, unit="km^2")


def get_cell_neighbors(grid_id: str, ring_size: int = 1) -> list[str]:
    """Return neighboring H3 cells within a given ring distance.

    Useful for computing nearest_fire_distance_km when the nearest fire
    is in an adjacent cell.

    Args:
        grid_id: Center cell H3 index.
        ring_size: Number of rings (1 = immediate neighbors, ~6 cells).

    Returns:
        List of neighbor H3 cell index strings (excludes center cell).
    """
    return list(h3.k_ring(grid_id, ring_size) - {grid_id})


def get_parent_cell(grid_id: str, parent_resolution_km: int,
                     config_path: Optional[str] = None) -> str:
    """Get the parent cell at a coarser resolution.

    Enables drill-down: a 1 km cell can report which 10 km or 64 km cell
    it belongs to.

    Args:
        grid_id: Child cell H3 index.
        parent_resolution_km: Target coarser resolution in km.

    Returns:
        Parent H3 cell index string.
    """
    parent_h3_res = km_to_h3_resolution(parent_resolution_km, config_path)
    return h3.h3_to_parent(grid_id, parent_h3_res)
