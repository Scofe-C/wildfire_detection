"""
Tests for dual-track ML evaluation pipeline invariants.
Verifies that Track A (tabular) and Track B (spatial) maintain consistency.
"""

import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.export.export_spatial import (
    export_spatial_grid,
    export_adjacency_matrix,
    SPATIAL_FEATURE_CHANNELS,
)


@pytest.fixture
def fused_ml_df():
    """ML-ready fused DataFrame for testing both tracks."""
    return pd.DataFrame({
        "grid_id": ["cell_a", "cell_b", "cell_c", "cell_d"],
        "latitude": [34.0, 34.2, 34.4, 34.6],
        "longitude": [-118.0, -118.25, -118.5, -118.75],
        "timestamp": pd.Timestamp("2025-01-15 06:00"),
        "active_fire_count": [5, 0, 2, 8],
        "mean_frp": [120.0, 0.0, 45.0, 200.0],
        "max_confidence": [90, 0, 75, 95],
        "nearest_fire_distance_km": [0.0, 50.0, 10.0, 0.0],
        "fire_detected_binary": [1, 0, 1, 1],
        "temperature_2m": [30.0, 28.0, 32.0, 35.0],
        "relative_humidity_2m": [15.0, 20.0, 10.0, 8.0],
        "wind_speed_10m": [15.0, 10.0, 20.0, 25.0],
        "wind_direction_10m": [180.0, 90.0, 270.0, 0.0],
        "elevation_m": [500.0, 600.0, 700.0, 800.0],
        "slope_degrees": [10.0, 15.0, 20.0, 25.0],
        "aspect_degrees": [180.0, 90.0, 270.0, 0.0],
        "fuel_model": [1, 2, 3, 1],
        "canopy_cover_pct": [50.0, 60.0, 70.0, 40.0],
        "data_source_priority": [2, 2, 2, 2],
    })


@pytest.fixture
def temp_output_dir():
    """Temporary output directory."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


class TestDualTrackConsistency:
    """Critical invariants between Track A (tabular) and Track B (spatial)."""

    def test_same_cell_count(self, fused_ml_df, temp_output_dir):
        """Track A row count == Track B node count."""
        track_a_count = len(fused_ml_df)

        grid_path = export_spatial_grid(fused_ml_df, temp_output_dir, resolution_km=22.0, date_str="2025-01-15")
        data = np.load(grid_path, allow_pickle=True)
        track_b_count = len(data["grid_ids"])
        data.close()

        assert track_a_count == track_b_count

    def test_same_grid_ids(self, fused_ml_df, temp_output_dir):
        """Both tracks should cover identical grid cell IDs."""
        track_a_ids = set(fused_ml_df["grid_id"].tolist())

        grid_path = export_spatial_grid(fused_ml_df, temp_output_dir, resolution_km=22.0, date_str="2025-01-15")
        data = np.load(grid_path, allow_pickle=True)
        track_b_ids = set(data["grid_ids"].tolist())
        data.close()

        assert track_a_ids == track_b_ids

    def test_same_label_values(self, fused_ml_df, temp_output_dir):
        """Both tracks must have identical fire_detected_binary labels."""
        track_a_labels = sorted(fused_ml_df["fire_detected_binary"].tolist())

        grid_path = export_spatial_grid(fused_ml_df, temp_output_dir, resolution_km=22.0, date_str="2025-01-15")
        data = np.load(grid_path, allow_pickle=True)
        grid = data["grid"]
        channels = data["channel_names"].tolist()
        data.close()

        if "fire_detected_binary" in channels:
            label_idx = channels.index("fire_detected_binary")
            track_b_grid_values = grid[:, :, label_idx]
            # Convert to Python floats for consistent comparison
            track_b_labels = sorted([float(v) for v in track_b_grid_values.flatten() if not np.isnan(v)])
            track_a_labels_f = sorted([float(v) for v in track_a_labels])
            assert track_a_labels_f == track_b_labels

    def test_feature_values_match(self, fused_ml_df, temp_output_dir):
        """Feature values should match between tracks for non-NaN cells."""
        grid_path = export_spatial_grid(fused_ml_df, temp_output_dir, resolution_km=22.0, date_str="2025-01-15")
        data = np.load(grid_path, allow_pickle=True)
        grid = data["grid"]
        channels = data["channel_names"].tolist()
        data.close()

        for col in ["active_fire_count", "mean_frp", "elevation_m"]:
            if col in channels and col in fused_ml_df.columns:
                ch_idx = channels.index(col)
                # Use np.float32 for consistent comparison
                grid_vals = sorted([float(v) for v in grid[:, :, ch_idx].flatten() if not np.isnan(v)])
                df_vals = sorted([float(v) for v in fused_ml_df[col].tolist()])
                assert grid_vals == pytest.approx(df_vals, rel=1e-5), f"Mismatch in {col}"


class TestSpatialGridExport:
    """Tests for spatial grid export."""

    def test_grid_dimensions(self, fused_ml_df, temp_output_dir):
        """Grid H×W should cover the lat/lon extent."""
        grid_path = export_spatial_grid(fused_ml_df, temp_output_dir, resolution_km=22.0, date_str="2025-01-15")
        data = np.load(grid_path, allow_pickle=True)
        assert data["grid"].ndim == 3
        assert data["n_rows"] > 0
        assert data["n_cols"] > 0
        data.close()

    def test_channel_count_matches_schema(self, fused_ml_df, temp_output_dir):
        """Number of channels should match available features."""
        grid_path = export_spatial_grid(fused_ml_df, temp_output_dir, resolution_km=22.0, date_str="2025-01-15")
        data = np.load(grid_path, allow_pickle=True)
        n_channels = data["grid"].shape[2]
        expected = len([c for c in SPATIAL_FEATURE_CHANNELS if c in fused_ml_df.columns])
        data.close()
        assert n_channels == expected

    def test_missing_cells_are_nan(self, fused_ml_df, temp_output_dir):
        """Cells without data should have NaN sentinel."""
        grid_path = export_spatial_grid(fused_ml_df, temp_output_dir, resolution_km=22.0, date_str="2025-01-15")
        data = np.load(grid_path, allow_pickle=True)
        grid = data["grid"]
        n_rows, n_cols = int(data["n_rows"]), int(data["n_cols"])
        data.close()
        total_cells = n_rows * n_cols
        data_cells = len(fused_ml_df)
        if total_cells > data_cells:
            assert np.any(np.isnan(grid))


class TestAdjacencyExport:
    """Tests for adjacency matrix export."""

    def test_adjacency_is_symmetric(self, fused_ml_df, temp_output_dir):
        """Adjacency matrix should be symmetric (if i→j then j→i)."""
        adj_path = export_adjacency_matrix(fused_ml_df, temp_output_dir, resolution_km=22.0, date_str="2025-01-15")
        data = np.load(adj_path)
        edges = set(zip(data["row"].tolist(), data["col"].tolist()))
        data.close()
        for i, j in list(edges):
            assert (j, i) in edges, f"Edge ({i},{j}) exists but ({j},{i}) does not"

    def test_node_count(self, fused_ml_df, temp_output_dir):
        """Number of nodes should equal number of data points."""
        adj_path = export_adjacency_matrix(fused_ml_df, temp_output_dir, resolution_km=22.0, date_str="2025-01-15")
        data = np.load(adj_path)
        n_nodes = int(data["n_nodes"])
        data.close()
        assert n_nodes == len(fused_ml_df)
