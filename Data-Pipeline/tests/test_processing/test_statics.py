"""
Tests for static feature processing (process_static.py).
"""

import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from scripts.processing.process_static import load_and_process_static


EXPECTED_COLUMNS = [
    "grid_id",
    "latitude",
    "longitude",
    "elevation_m",
    "slope_degrees",
    "aspect_degrees",
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

    def test_no_nan_in_key_columns(self, temp_output_dir):
        """Grid ID and terrain columns should not contain NaN."""
        result_path = load_and_process_static(resolution_km=22, output_dir=temp_output_dir)
        df = pd.read_parquet(result_path)

        for col in ["grid_id", "elevation_m", "slope_degrees", "aspect_degrees"]:
            if col in df.columns:
                assert not df[col].isna().any(), f"Column '{col}' contains NaN values"

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
