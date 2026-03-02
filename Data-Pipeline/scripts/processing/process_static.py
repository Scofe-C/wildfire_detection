from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.utils.grid_utils import generate_full_grid

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
    # 1) paths — 一定要先定義
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"static_features_{resolution_km}km.parquet"

    # 2) cache check
    if out_path.exists() and not force_rebuild:
        return out_path

    # 3) build grid
    grid = generate_full_grid(resolution_km)
    df = grid[["grid_id", "latitude", "longitude"]].copy()
    df["grid_id"] = df["grid_id"].astype(str)

    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()

    # 4) DEM Route A (synthetic but deterministic)
    df["elevation_m"] = (1000 * np.abs(np.sin(np.radians(lat)))).round(2)
    df["slope_degrees"] = (30 * np.abs(np.cos(np.radians(lon)))).round(2)
    df["aspect_degrees"] = (np.mod(lon + 360, 360)).round(2)

    # 5) other static placeholders
    df["fuel_model_fbfm40"] = "UNKNOWN"
    df["canopy_cover_pct"] = 0.0
    df["vegetation_type"] = "UNKNOWN"
    df["ndvi"] = 0.0
    df["dominant_fuel_fraction"] = 0.0

    df = df[STATIC_COLUMNS]

    # 6) write
    df.to_parquet(out_path, index=False)
    return out_path
