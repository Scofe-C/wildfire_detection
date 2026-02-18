"""
H3 Grid Utilities
=================
Shared spatial grid system used by all pipeline components.

Improvements applied:
  1c. prune_non_vegetated_cells() added — removes truly non-burnable cells
      while preserving wildland-urban interface and agriculture (TX grass fires).
      Controlled per-region via spatial_pruning_enabled in schema_config.yaml.

Cross-platform compat:
  - h3-py 3.x and 4.x supported via _polyfill_geojson_compat() and
    _grid_disk_compat(). Do not call h3 functions directly outside these wrappers.
  - get_cell_neighbors() uses h3.k_ring (3.x) or h3.grid_disk (4.x) — both work.
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


# ---------------------------------------------------------------------------
# Internal h3 version compatibility helpers
# ---------------------------------------------------------------------------

def _polyfill_geojson_compat(geojson_poly: dict, res: int) -> list[str]:
    """Polyfill a GeoJSON polygon with H3 cells.

    Supports h3-py 3.x (polyfill_geojson) and 4.x+ (polygon_to_cells).
    """
    if hasattr(h3, "polyfill_geojson"):
        # h3-py 3.x
        return list(h3.polyfill_geojson(geojson_poly, res))

    if hasattr(h3, "polygon_to_cells"):
        # h3-py 4.x+: expects LatLngPoly with (lat, lon) tuples
        coords = geojson_poly["coordinates"][0]   # outer ring (lon, lat)
        outer = [(lat, lon) for (lon, lat) in coords]  # flip to (lat, lon)
        try:
            from h3 import LatLngPoly
            poly = LatLngPoly(outer)
        except ImportError:
            try:
                from h3.api.basic_str import LatLngPoly  # type: ignore
                poly = LatLngPoly(outer)
            except ImportError as e:
                raise RuntimeError(f"Cannot import LatLngPoly: {e}")
        return list(h3.polygon_to_cells(poly, res))

    if hasattr(h3, "geo_to_cells"):
        return list(h3.geo_to_cells(geojson_poly, res))

    raise AttributeError("Unsupported h3 version: no polyfill function found")


def _cell_to_latlng_compat(cell_id: str) -> tuple[float, float]:
    """Return (lat, lon) for a cell. Supports h3 3.x and 4.x."""
    if hasattr(h3, "h3_to_geo"):
        return h3.h3_to_geo(cell_id)  # 3.x → (lat, lon)
    return h3.cell_to_latlng(cell_id)  # 4.x → (lat, lon)


def _cell_to_boundary_compat(cell_id: str) -> list[tuple]:
    """Return boundary coords as GeoJSON-compatible list. Supports h3 3.x and 4.x."""
    if hasattr(h3, "h3_to_geo_boundary"):
        return h3.h3_to_geo_boundary(cell_id, geo_json=True)  # 3.x
    return h3.cell_to_boundary(cell_id, geo_json=True)  # 4.x


def _grid_disk_compat(cell_id: str, k: int) -> set[str]:
    """Return cell + all cells within k rings. Supports h3 3.x and 4.x.

    h3-py 3.x: h3.k_ring(cell_id, k)
    h3-py 4.x: h3.grid_disk(cell_id, k)
    Both return the center cell plus neighbors.
    """
    if hasattr(h3, "grid_disk"):
        return set(h3.grid_disk(cell_id, k))  # 4.x
    return h3.k_ring(cell_id, k)              # 3.x


def _geo_to_h3_compat(lat: float, lon: float, res: int) -> str:
    """Convert lat/lon to H3 cell. Supports h3 3.x and 4.x."""
    if hasattr(h3, "geo_to_h3"):
        return h3.geo_to_h3(lat, lon, res)   # 3.x
    return h3.latlng_to_cell(lat, lon, res)  # 4.x


def _h3_get_resolution_compat(cell_id: str) -> int:
    """Get H3 resolution of a cell. Supports h3 3.x and 4.x."""
    if hasattr(h3, "h3_get_resolution"):
        return h3.h3_get_resolution(cell_id)  # 3.x
    return h3.get_resolution(cell_id)          # 4.x


def _h3_to_parent_compat(cell_id: str, res: int) -> str:
    """Get parent cell at coarser resolution. Supports h3 3.x and 4.x."""
    if hasattr(h3, "h3_to_parent"):
        return h3.h3_to_parent(cell_id, res)  # 3.x
    return h3.cell_to_parent(cell_id, res)    # 4.x


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def km_to_h3_resolution(km: int, config_path: Optional[str] = None) -> int:
    """Convert a km resolution to the corresponding H3 resolution level."""
    registry = get_registry(config_path)
    return registry.get_h3_resolution(km)


def generate_grid_for_bbox(
    bbox: list[float],
    resolution_km: int = 22,
    config_path: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Generate H3 hexagonal grid cells covering a bounding box.

    Args:
        bbox: [west, south, east, north] in WGS84 degrees.
        resolution_km: Grid resolution in km.
        config_path: Optional schema config override.

    Returns:
        GeoDataFrame with columns: grid_id, latitude, longitude, geometry. CRS EPSG:4326.
    """
    h3_res = km_to_h3_resolution(resolution_km, config_path)
    west, south, east, north = bbox

    bbox_polygon = Polygon([
        (west, south), (east, south), (east, north), (west, north), (west, south),
    ])
    geojson = {
        "type": "Polygon",
        "coordinates": [list(bbox_polygon.exterior.coords)],
    }

    h3_cells = _polyfill_geojson_compat(geojson, h3_res)

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

    records = []
    for cell_id in h3_cells:
        lat, lon = _cell_to_latlng_compat(cell_id)
        boundary = _cell_to_boundary_compat(cell_id)
        polygon = Polygon(boundary)
        records.append({
            "grid_id": str(cell_id),
            "latitude": round(float(lat), 6),
            "longitude": round(float(lon), 6),
            "geometry": polygon,
        })

    return gpd.GeoDataFrame(records, crs="EPSG:4326")


def prune_non_vegetated_cells(
    grid: gpd.GeoDataFrame,
    static_features: pd.DataFrame,
    region_name: Optional[str] = None,
    config_path: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Remove grid cells with no wildfire-relevant vegetation.

    --- Improvement 1c ---
    Discards only truly non-burnable cells (urban core, snow, water, barren).
    Keeps:
      - Agriculture (FBFM code 93) — TX grass fires start in ag land
      - Urban-wildland interface cells — most destructive fires burn into suburbs
        (Paradise 2018, Marshall 2021, Maui 2023)

    A cell is only discarded if BOTH:
      1. Its fuel model is in the non-burnable set (91, 92, 98, 99)
      2. ALL its H3 ring-1 neighbors also have non-burnable fuel models

    CA note: FBFM99 (barren) in Mojave/Owens Valley sits adjacent to real WUI
    cells — the neighbor check guards against false pruning, but if you see
    unexpected cell loss in desert areas, set spatial_pruning_enabled: false for
    california in schema_config.yaml.

    Args:
        grid: GeoDataFrame from generate_full_grid().
        static_features: DataFrame with 'grid_id' and 'fuel_model_fbfm40' columns.
        region_name: Optional region key to check spatial_pruning_enabled config.
        config_path: Optional schema config override.

    Returns:
        Pruned GeoDataFrame with non-burnable interior cells removed.
    """
    # Check per-region pruning flag
    if region_name is not None:
        registry = get_registry(config_path)
        scope = registry.config.get("pipeline", {}).get("geographic_scope", {})
        region_cfg = scope.get(region_name, {})
        if not region_cfg.get("spatial_pruning_enabled", True):
            logger.info(f"Spatial pruning disabled for region '{region_name}' — skipping")
            return grid

    # Non-burnable FBFM40 codes. 93 (Agriculture) intentionally excluded.
    NON_BURNABLE = {91, 92, 98, 99}

    if static_features is None or static_features.empty:
        logger.info("Spatial pruning skipped: no static features available")
        return grid

    if "fuel_model_fbfm40" not in static_features.columns:
        logger.info("Spatial pruning skipped: 'fuel_model_fbfm40' column absent")
        return grid

    fuel_map = (
        static_features
        .dropna(subset=["fuel_model_fbfm40"])
        .set_index("grid_id")["fuel_model_fbfm40"]
        .astype(int)
        .to_dict()
    )

    non_burnable_cells = {
        gid for gid, fuel in fuel_map.items() if fuel in NON_BURNABLE
    }

    safe_to_discard = set()
    for cell_id in non_burnable_cells:
        try:
            neighbors = _grid_disk_compat(cell_id, 1)  # center + ring-1
            neighbor_fuels = {
                fuel_map[n] for n in neighbors if n in fuel_map
            }
            # Only discard if every neighbor is also non-burnable
            if neighbor_fuels and neighbor_fuels.issubset(NON_BURNABLE):
                safe_to_discard.add(cell_id)
        except Exception:
            continue  # On any error: keep cell (safe default)

    before = len(grid)
    grid = grid[~grid["grid_id"].isin(safe_to_discard)].copy()
    after = len(grid)
    interface_kept = len(non_burnable_cells) - len(safe_to_discard)

    logger.info(
        f"Spatial pruning [{region_name or 'all'}]: "
        f"{before} → {after} cells "
        f"({before - after} non-burnable interior removed, "
        f"{interface_kept} WUI interface cells kept, "
        f"{(before - after) / before:.0%} savings)"
    )
    return grid


def generate_full_grid(
    resolution_km: int = 22,
    config_path: Optional[str] = None,
    static_features: Optional[pd.DataFrame] = None,
) -> gpd.GeoDataFrame:
    """Generate the complete grid covering all configured geographic regions.

    Merges CA and TX grids. Deduplicates border cells (keep first = CA).
    If static_features is provided, applies spatial pruning per-region.

    Args:
        resolution_km: Grid resolution in km (default now 22, was 64).
        config_path: Optional schema config override.
        static_features: Optional DataFrame for spatial pruning (improvement 1c).

    Returns:
        GeoDataFrame with columns: grid_id, latitude, longitude, geometry, region.
    """
    registry = get_registry(config_path)
    bboxes = registry.geographic_bboxes

    grids = []
    for region_name, bbox in bboxes.items():
        logger.info(f"Generating grid for {region_name}: {bbox}")
        grid = generate_grid_for_bbox(bbox, resolution_km, config_path)
        grid["region"] = region_name

        # --- Improvement 1c: spatial pruning per region ---
        if static_features is not None and not static_features.empty:
            # Filter static features to this region's cells only
            region_cell_ids = set(grid["grid_id"])
            region_static = static_features[
                static_features["grid_id"].isin(region_cell_ids)
            ]
            grid = prune_non_vegetated_cells(
                grid, region_static,
                region_name=region_name,
                config_path=config_path,
            )

        grids.append(grid)

    if not grids:
        raise ValueError("No geographic regions configured in schema_config.yaml")

    full_grid = pd.concat(grids, ignore_index=True)

    before_dedup = len(full_grid)
    full_grid = full_grid.drop_duplicates(subset="grid_id", keep="first")
    after_dedup = len(full_grid)
    if before_dedup != after_dedup:
        logger.info(f"Removed {before_dedup - after_dedup} duplicate cells at region borders")

    logger.info(f"Full grid: {len(full_grid)} cells at {resolution_km} km resolution")
    return full_grid


def point_to_grid_id(
    lat: float,
    lon: float,
    resolution_km: int = 22,
    config_path: Optional[str] = None,
) -> str:
    """Convert a lat/lon point to its containing H3 grid cell ID."""
    h3_res = km_to_h3_resolution(resolution_km, config_path)
    return _geo_to_h3_compat(lat, lon, h3_res)


def points_to_grid_ids(
    lats: np.ndarray,
    lons: np.ndarray,
    resolution_km: int = 22,
    config_path: Optional[str] = None,
) -> np.ndarray:
    """Vectorized conversion of lat/lon arrays to H3 grid cell IDs."""
    h3_res = km_to_h3_resolution(resolution_km, config_path)
    return np.array([
        _geo_to_h3_compat(lat, lon, h3_res)
        for lat, lon in zip(lats, lons)
    ])


def get_cell_area_km2(grid_id: str) -> float:
    """Return the approximate area of an H3 cell in square kilometers."""
    res = _h3_get_resolution_compat(grid_id)
    return h3.hex_area(res, unit="km^2")


def get_cell_neighbors(grid_id: str, ring_size: int = 1) -> list[str]:
    """Return neighboring H3 cells within a given ring distance.

    Uses _grid_disk_compat to support both h3 3.x and 4.x.
    Excludes the center cell.
    """
    return list(_grid_disk_compat(grid_id, ring_size) - {grid_id})


def get_parent_cell(
    grid_id: str,
    parent_resolution_km: int,
    config_path: Optional[str] = None,
) -> str:
    """Get the parent cell at a coarser resolution."""
    parent_h3_res = km_to_h3_resolution(parent_resolution_km, config_path)
    return _h3_to_parent_compat(grid_id, parent_h3_res)


def generate_fire_focal_grid(
    fire_cell_ids: list[str],
    ring_min: int = 1,
    ring_max: int = 5,
    config_path: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Generate a focused analysis grid around confirmed fire cells.

    Used when the watchdog confirms an active fire: instead of running
    the full regional grid (~800-1000 cells for CA), we generate a
    dense analysis zone covering the fire + its detection boundary.

    Detection boundary ring sizes at H3 res 5 (~5.1 km edge):
        ring 1 ≈  5 km  (immediate perimeter)
        ring 3 ≈ 15 km  (mid-range)
        ring 5 ≈ 25 km  (outer detection boundary)

    Args:
        fire_cell_ids: List of H3 cell IDs at res 5 confirmed to have fire.
        ring_min: Inner ring (default 1 ≈ 5 km). Cells closer than this
                  are the fire cells themselves.
        ring_max: Outer ring (default 5 ≈ 25 km). Defines detection perimeter.
        config_path: Optional schema config override.

    Returns:
        GeoDataFrame with columns: grid_id, latitude, longitude, geometry,
        cell_type ('fire' | 'detection_zone'), ring_distance.
    """
    if not fire_cell_ids:
        logger.warning("generate_fire_focal_grid: empty fire_cell_ids — returning empty grid")
        return gpd.GeoDataFrame(
            columns=["grid_id", "latitude", "longitude", "geometry", "cell_type", "ring_distance"],
            crs="EPSG:4326",
        )

    fire_set = set(fire_cell_ids)
    all_cells: dict[str, dict] = {}

    for cell_id in fire_cell_ids:
        # The fire cell itself
        if cell_id not in all_cells:
            try:
                lat, lon = _cell_to_latlng_compat(cell_id)
                boundary = _cell_to_boundary_compat(cell_id)
                all_cells[cell_id] = {
                    "grid_id": cell_id,
                    "latitude": round(float(lat), 6),
                    "longitude": round(float(lon), 6),
                    "geometry": Polygon(boundary),
                    "cell_type": "fire",
                    "ring_distance": 0,
                }
            except Exception as e:
                logger.debug(f"Could not process fire cell {cell_id}: {e}")
                continue

        # Detection zone rings (ring_min to ring_max)
        for ring_k in range(ring_min, ring_max + 1):
            ring_cells = _grid_disk_compat(cell_id, ring_k) - _grid_disk_compat(cell_id, ring_k - 1)
            for neighbor_id in ring_cells:
                if neighbor_id in all_cells:
                    # Keep minimum ring distance if cell already seen from another fire cell
                    if all_cells[neighbor_id]["ring_distance"] > ring_k:
                        all_cells[neighbor_id]["ring_distance"] = ring_k
                    continue
                try:
                    lat, lon = _cell_to_latlng_compat(neighbor_id)
                    boundary = _cell_to_boundary_compat(neighbor_id)
                    all_cells[neighbor_id] = {
                        "grid_id": neighbor_id,
                        "latitude": round(float(lat), 6),
                        "longitude": round(float(lon), 6),
                        "geometry": Polygon(boundary),
                        "cell_type": "fire" if neighbor_id in fire_set else "detection_zone",
                        "ring_distance": ring_k,
                    }
                except Exception:
                    continue

    if not all_cells:
        return gpd.GeoDataFrame(
            columns=["grid_id", "latitude", "longitude", "geometry", "cell_type", "ring_distance"],
            crs="EPSG:4326",
        )

    focal_grid = gpd.GeoDataFrame(list(all_cells.values()), crs="EPSG:4326")

    fire_count = (focal_grid["cell_type"] == "fire").sum()
    zone_count = (focal_grid["cell_type"] == "detection_zone").sum()
    logger.info(
        f"Fire focal grid: {fire_count} fire cells + {zone_count} detection zone cells "
        f"(rings {ring_min}–{ring_max}, total {len(focal_grid)} cells)"
    )
    return focal_grid
