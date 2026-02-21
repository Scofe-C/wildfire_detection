"""
Spatial Grid Export — Track B
==============================
Exports fused features as spatial grid arrays (NumPy .npz) for
CNN/GCN-based wildfire models.

Output per time window:
  - spatial_grid_{date}.npz: 3D array (H × W × C) of feature channels
  - adjacency_{date}.npz:    Sparse COO adjacency matrix for GCN

Grid construction:
  - H3-indexed cells → 2D lat/lon grid at the pipeline resolution
  - Each cell = pixel with N feature channels
  - Missing cells filled with NaN sentinel
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature channels for spatial grid (order matters)
SPATIAL_FEATURE_CHANNELS = [
    "active_fire_count",
    "mean_frp",
    "max_confidence",
    "nearest_fire_distance_km",
    "fire_detected_binary",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "elevation_m",
    "slope_degrees",
    "aspect_degrees",
    "fuel_model",
    "canopy_cover_pct",
]


def _build_grid_indices(
    fused_df: pd.DataFrame,
    resolution_km: float = 22.0,
) -> tuple[np.ndarray, np.ndarray, dict, int, int]:
    """Build 2D grid index mapping from lat/lon coordinates.

    Returns:
        (row_indices, col_indices, cell_to_pixel_map, n_rows, n_cols)
    """
    lats = fused_df["latitude"].values
    lons = fused_df["longitude"].values

    # Quantize to grid cells at the given resolution
    lat_step = resolution_km / 111.0  # ~111 km per degree latitude
    lon_step = resolution_km / (111.0 * np.cos(np.radians(np.mean(lats))))

    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()

    row_indices = ((lats - lat_min) / lat_step).astype(int)
    col_indices = ((lons - lon_min) / lon_step).astype(int)

    n_rows = row_indices.max() + 1
    n_cols = col_indices.max() + 1

    cell_to_pixel = {}
    for i in range(len(fused_df)):
        grid_id = fused_df.iloc[i].get("grid_id", f"cell_{i}")
        cell_to_pixel[grid_id] = (row_indices[i], col_indices[i])

    return row_indices, col_indices, cell_to_pixel, n_rows, n_cols


def export_spatial_grid(
    fused_df: pd.DataFrame,
    output_dir: str,
    resolution_km: float = 22.0,
    date_str: Optional[str] = None,
) -> Path:
    """Export fused features as a spatial grid (H × W × C) NumPy array.

    Args:
        fused_df: Fused features DataFrame with lat/lon and feature columns.
        output_dir: Directory to write the .npz file.
        resolution_km: Pipeline grid resolution.
        date_str: Optional date string for filename.

    Returns:
        Path to the written .npz file.
    """
    if fused_df.empty:
        logger.warning("Empty DataFrame — skipping spatial grid export.")
        return Path(output_dir) / "spatial_grid_empty.npz"

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    row_idx, col_idx, cell_map, n_rows, n_cols = _build_grid_indices(
        fused_df, resolution_km
    )

    # Determine available feature channels
    available_channels = [c for c in SPATIAL_FEATURE_CHANNELS if c in fused_df.columns]
    n_channels = len(available_channels)

    if n_channels == 0:
        logger.error("No spatial feature channels found in fused DataFrame.")
        raise ValueError("No spatial feature channels available for export.")

    # Build 3D array: (H, W, C), fill with NaN for missing cells
    grid = np.full((n_rows, n_cols, n_channels), np.nan, dtype=np.float32)

    for ch_idx, col_name in enumerate(available_channels):
        values = fused_df[col_name].values.astype(np.float32)
        grid[row_idx, col_idx, ch_idx] = values

    # Export
    fname = f"spatial_grid_{date_str}.npz" if date_str else "spatial_grid.npz"
    file_path = out_path / fname

    np.savez_compressed(
        file_path,
        grid=grid,
        channel_names=np.array(available_channels),
        lat_min=fused_df["latitude"].min(),
        lat_max=fused_df["latitude"].max(),
        lon_min=fused_df["longitude"].min(),
        lon_max=fused_df["longitude"].max(),
        resolution_km=resolution_km,
        n_rows=n_rows,
        n_cols=n_cols,
        grid_ids=np.array(fused_df["grid_id"].tolist() if "grid_id" in fused_df.columns else []),
    )

    logger.info(
        f"Spatial grid exported: {file_path} "
        f"(shape={n_rows}×{n_cols}×{n_channels}, "
        f"channels={available_channels})"
    )
    return file_path


def export_adjacency_matrix(
    fused_df: pd.DataFrame,
    output_dir: str,
    resolution_km: float = 22.0,
    date_str: Optional[str] = None,
) -> Path:
    """Export adjacency matrix in sparse COO format for GCN models.

    Connectivity: each cell is connected to its 8 immediate neighbors
    (queen contiguity). Edge weight = 1.0.

    Args:
        fused_df: Fused features DataFrame.
        output_dir: Output directory.
        resolution_km: Pipeline grid resolution.
        date_str: Optional date string.

    Returns:
        Path to the written .npz file.
    """
    if fused_df.empty:
        logger.warning("Empty DataFrame — skipping adjacency export.")
        return Path(output_dir) / "adjacency_empty.npz"

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    row_idx, col_idx, cell_map, n_rows, n_cols = _build_grid_indices(
        fused_df, resolution_km
    )

    n_cells = len(fused_df)

    # Build pixel-to-index lookup
    pixel_to_idx: dict[tuple[int, int], int] = {}
    for i in range(n_cells):
        pixel_to_idx[(row_idx[i], col_idx[i])] = i

    # 8-connected neighbors (queen contiguity)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    rows_coo = []
    cols_coo = []
    weights = []

    for i in range(n_cells):
        r, c = row_idx[i], col_idx[i]
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if (nr, nc) in pixel_to_idx:
                j = pixel_to_idx[(nr, nc)]
                rows_coo.append(i)
                cols_coo.append(j)
                weights.append(1.0)

    fname = f"adjacency_{date_str}.npz" if date_str else "adjacency.npz"
    file_path = out_path / fname

    np.savez_compressed(
        file_path,
        row=np.array(rows_coo, dtype=np.int32),
        col=np.array(cols_coo, dtype=np.int32),
        weight=np.array(weights, dtype=np.float32),
        n_nodes=n_cells,
        n_edges=len(rows_coo),
    )

    logger.info(
        f"Adjacency matrix exported: {file_path} "
        f"(nodes={n_cells}, edges={len(rows_coo)})"
    )
    return file_path
