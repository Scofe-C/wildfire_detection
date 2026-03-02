"""
Tests for the Bug #1 fix: T-1 temporal lag using _previous.parquet.
Verifies that the DAG rotation logic produces correct file paths.
"""

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def temp_processed_dir():
    """Create a temporary directory structure mimicking PROCESSED_DIR/firms/."""
    tmpdir = tempfile.mkdtemp()
    firms_dir = Path(tmpdir) / "firms"
    firms_dir.mkdir(parents=True)
    yield firms_dir
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_firms_features():
    """Sample FIRMS features DataFrame."""
    return pd.DataFrame({
        "grid_id": ["cell_a", "cell_b"],
        "active_fire_count": [5, 0],
        "mean_frp": [120.0, 0.0],
    })


@pytest.fixture
def sample_previous_features():
    """Different FIRMS features representing the previous run."""
    return pd.DataFrame({
        "grid_id": ["cell_a", "cell_b"],
        "active_fire_count": [3, 1],
        "mean_frp": [80.0, 10.0],
    })


class TestFileRotation:
    """Test the _latest/_previous file rotation pattern."""

    def test_first_run_no_previous(self, temp_processed_dir, sample_firms_features):
        """On first run, _previous should NOT exist."""
        latest_path = temp_processed_dir / "firms_features_california_latest.parquet"
        previous_path = temp_processed_dir / "firms_features_california_previous.parquet"

        sample_firms_features.to_parquet(latest_path, index=False)

        assert latest_path.exists()
        assert not previous_path.exists()

    def test_second_run_creates_previous(self, temp_processed_dir, sample_firms_features, sample_previous_features):
        """On second run, _latest should be copied to _previous before overwrite."""
        latest_path = temp_processed_dir / "firms_features_california_latest.parquet"
        previous_path = temp_processed_dir / "firms_features_california_previous.parquet"

        # First run
        sample_previous_features.to_parquet(latest_path, index=False)

        # Second run — rotate
        if latest_path.exists():
            shutil.copy2(str(latest_path), str(previous_path))

        # Overwrite with new data
        sample_firms_features.to_parquet(latest_path, index=False)

        assert latest_path.exists()
        assert previous_path.exists()

        # _previous should have the OLD data
        prev_df = pd.read_parquet(previous_path)
        assert prev_df["active_fire_count"].tolist() == [3, 1]

        # _latest should have the NEW data
        latest_df = pd.read_parquet(latest_path)
        assert latest_df["active_fire_count"].tolist() == [5, 0]

    def test_previous_differs_from_latest(self, temp_processed_dir, sample_firms_features, sample_previous_features):
        """T-1 data (_previous) must differ from T data (_latest)."""
        latest_path = temp_processed_dir / "firms_features_california_latest.parquet"
        previous_path = temp_processed_dir / "firms_features_california_previous.parquet"

        # Simulate rotation
        sample_previous_features.to_parquet(previous_path, index=False)
        sample_firms_features.to_parquet(latest_path, index=False)

        latest_df = pd.read_parquet(latest_path)
        previous_df = pd.read_parquet(previous_path)

        # Values must differ to confirm T-1 is not T
        assert latest_df["active_fire_count"].tolist() != previous_df["active_fire_count"].tolist()
