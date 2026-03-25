"""
Tests for static feature processing (process_static.py).
"""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.processing.process_static import load_and_process_static


EXPECTED_COLUMNS = [
    "grid_id",
    "latitude",
    "longitude",
    "fuel_model_fbfm40",
    "canopy_cover_pct",
    "vegetation_type",
    "dominant_fuel_fraction",
    "elevation_m",
    "slope_degrees",
    "aspect_degrees",
    "ndvi",
]


@pytest.fixture
def temp_output_dir():
    """Temporary directory for output files."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


class TestLoadAndProcessStatic:
    """Tests for load_and_process_static()."""

    def test_returns_path(self, temp_output_dir):
        """Function should return a Path to the output Parquet."""
        result = load_and_process_static(resolution_km=22, output_dir=temp_output_dir)
        assert isinstance(result, Path)
        assert result.exists()
        assert result.suffix == ".parquet"

    def test_cache_is_reused_on_second_call(self, temp_output_dir):
        """The static cache should be reused on a second call."""
        path1 = load_and_process_static(resolution_km=22, output_dir=temp_output_dir)
        mtime1 = os.path.getmtime(path1)

        path2 = load_and_process_static(resolution_km=22, output_dir=temp_output_dir)
        mtime2 = os.path.getmtime(path2)

        assert path1 == path2
        assert mtime1 == mtime2, "Cache should not be regenerated"

    def test_all_static_columns_present(self, temp_output_dir):
        """Output should contain all expected static columns."""
        result_path = load_and_process_static(resolution_km=22, output_dir=temp_output_dir)
        df = pd.read_parquet(result_path)

        for col in EXPECTED_COLUMNS:
            assert col in df.columns, f"Missing expected column: {col}"

    def test_no_nan_in_identifier_columns(self, temp_output_dir):
        """Grid ID, latitude, longitude must never contain NaN.
        Static feature columns may be NaN if source caches are absent
        (no LANDFIRE/SRTM parquet in temp_output_dir), or non-NaN if real
        caches were pre-computed and placed there.
        """
        result_path = load_and_process_static(resolution_km=22, output_dir=temp_output_dir)
        df = pd.read_parquet(result_path)

        # Identifiers must always be complete
        assert not df["grid_id"].isna().any(),   "grid_id must not contain NaN"
        assert not df["latitude"].isna().any(),  "latitude must not contain NaN"
        assert not df["longitude"].isna().any(), "longitude must not contain NaN"

    def test_static_cols_are_nan_when_no_source_caches(self, temp_output_dir):
        """When no LANDFIRE/SRTM/NDVI parquets exist in the output dir,
        all static feature columns must fall back to NaN stubs gracefully.
        """
        result_path = load_and_process_static(resolution_km=22, output_dir=temp_output_dir)
        df = pd.read_parquet(result_path)

        stub_cols = [
            "fuel_model_fbfm40", "canopy_cover_pct", "vegetation_type",
            "dominant_fuel_fraction", "elevation_m", "slope_degrees",
            "aspect_degrees", "ndvi",
        ]
        for col in stub_cols:
            if col in df.columns:
                assert df[col].isna().all(), (
                    f"Column '{col}' should be NaN stub when source cache is absent"
                )

    def test_static_cols_populated_when_cache_present(self, temp_output_dir):
        """When a pre-computed LANDFIRE parquet is placed in the output dir,
        the corresponding columns must be non-NaN for matched grid IDs.
        """
        from scripts.utils.grid_utils import generate_full_grid
        grid = generate_full_grid(22)

        # Build a minimal fake LANDFIRE parquet
        fake_lf = pd.DataFrame({
            "grid_id":              grid["grid_id"].astype(str).values,
            "fuel_model_fbfm40":    [101.0] * len(grid),
            "canopy_cover_pct":     [30.0]  * len(grid),
            "vegetation_type":      [3105.0] * len(grid),
            "dominant_fuel_fraction": [0.7] * len(grid),
        })
        lf_path = Path(temp_output_dir) / "landfire_features_22km.parquet"
        fake_lf.to_parquet(lf_path, index=False)

        result_path = load_and_process_static(
            resolution_km=22, output_dir=temp_output_dir, force_rebuild=True
        )
        df = pd.read_parquet(result_path)

        assert df["fuel_model_fbfm40"].notna().any(), (
            "fuel_model_fbfm40 should be populated when LANDFIRE cache exists"
        )
        assert df["canopy_cover_pct"].notna().any(), (
            "canopy_cover_pct should be populated when LANDFIRE cache exists"
        )

    def test_force_rebuild_regenerates(self, temp_output_dir):
        """force_rebuild=True should regenerate even when cache exists."""
        path1 = load_and_process_static(resolution_km=22, output_dir=temp_output_dir)
        mtime1 = os.path.getmtime(path1)

        import time
        time.sleep(0.1)  # ensure different mtime

        path2 = load_and_process_static(
            resolution_km=22, output_dir=temp_output_dir, force_rebuild=True
        )
        mtime2 = os.path.getmtime(path2)

        assert mtime2 > mtime1, "Cache file should be regenerated"

    def test_output_is_valid_parquet(self, temp_output_dir):
        """Output file should be a valid Parquet with rows."""
        result_path = load_and_process_static(resolution_km=22, output_dir=temp_output_dir)
        df = pd.read_parquet(result_path)
        assert len(df) > 0, "Parquet file should not be empty"
