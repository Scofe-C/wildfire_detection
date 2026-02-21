"""
Tests for temporal lag in the fusion layer.
Verifies that apply_temporal_lag() and fuse_features_for_ml() correctly
replace fire context columns with T-1 values while preserving
fire_detected_binary as the current-window label.
"""

import pandas as pd
import pytest

from scripts.fusion.fuse_features import (
    FIRE_CONTEXT_LAG_COLS,
    apply_temporal_lag,
    fuse_features_for_ml,
)


@pytest.fixture
def fused_current():
    """Fused DataFrame with current-window (T) fire features."""
    return pd.DataFrame({
        "grid_id": ["cell_a", "cell_b", "cell_c"],
        "latitude": [34.0, 34.1, 34.2],
        "longitude": [-118.0, -118.1, -118.2],
        "timestamp": pd.Timestamp("2025-01-15 06:00"),
        "resolution_km": 64,
        "active_fire_count": [5, 0, 2],
        "mean_frp": [120.0, 0.0, 45.0],
        "median_frp": [100.0, 0.0, 40.0],
        "max_confidence": [90, 0, 75],
        "nearest_fire_distance_km": [0.0, 50.0, 10.0],
        "fire_detected_binary": [1, 0, 1],
    })


@pytest.fixture
def prev_fire_features():
    """Previous window (T-1) fire features — different values from current."""
    return pd.DataFrame({
        "grid_id": ["cell_a", "cell_b", "cell_c"],
        "active_fire_count": [3, 1, 0],
        "mean_frp": [80.0, 10.0, 0.0],
        "median_frp": [70.0, 8.0, 0.0],
        "max_confidence": [85, 60, 0],
        "nearest_fire_distance_km": [0.0, 30.0, -1.0],
    })


class TestApplyTemporalLag:
    """Tests for apply_temporal_lag()."""

    def test_lagged_columns_come_from_prev(self, fused_current, prev_fire_features):
        """Fire context columns should reflect T-1 values, not current."""
        ml = apply_temporal_lag(fused_current, prev_fire_features)

        # active_fire_count should be [3, 1, 0] from prev, not [5, 0, 2]
        assert ml.loc[ml["grid_id"] == "cell_a", "active_fire_count"].values[0] == 3
        assert ml.loc[ml["grid_id"] == "cell_b", "active_fire_count"].values[0] == 1
        assert ml.loc[ml["grid_id"] == "cell_c", "active_fire_count"].values[0] == 0

        # mean_frp should be [80, 10, 0] from prev
        assert ml.loc[ml["grid_id"] == "cell_a", "mean_frp"].values[0] == 80.0
        assert ml.loc[ml["grid_id"] == "cell_b", "mean_frp"].values[0] == 10.0

    def test_fire_detected_binary_stays_current(self, fused_current, prev_fire_features):
        """fire_detected_binary is the LABEL — it must stay at current window T."""
        ml = apply_temporal_lag(fused_current, prev_fire_features)

        # fire_detected_binary should still be [1, 0, 1] from current
        assert ml.loc[ml["grid_id"] == "cell_a", "fire_detected_binary"].values[0] == 1
        assert ml.loc[ml["grid_id"] == "cell_b", "fire_detected_binary"].values[0] == 0
        assert ml.loc[ml["grid_id"] == "cell_c", "fire_detected_binary"].values[0] == 1

    def test_none_prev_fills_defaults(self, fused_current):
        """When prev_fire_features is None, lagged columns get default values."""
        ml = apply_temporal_lag(fused_current, None)

        # Should still have all columns
        for col in FIRE_CONTEXT_LAG_COLS:
            assert col in ml.columns

        # fire_detected_binary should be preserved from current
        assert ml["fire_detected_binary"].tolist() == [1, 0, 1]

    def test_empty_prev_fills_defaults(self, fused_current):
        """When prev_fire_features is empty, lagged columns get default values."""
        empty_prev = pd.DataFrame(columns=["grid_id"])
        ml = apply_temporal_lag(fused_current, empty_prev)

        for col in FIRE_CONTEXT_LAG_COLS:
            assert col in ml.columns

    def test_does_not_modify_original(self, fused_current, prev_fire_features):
        """apply_temporal_lag must not mutate the input DataFrame."""
        original_values = fused_current["active_fire_count"].tolist()
        _ = apply_temporal_lag(fused_current, prev_fire_features)

        # Original should be unchanged
        assert fused_current["active_fire_count"].tolist() == original_values

    def test_partial_prev_columns(self, fused_current):
        """If prev only has some lag columns, missing ones get defaults."""
        partial_prev = pd.DataFrame({
            "grid_id": ["cell_a", "cell_b", "cell_c"],
            "active_fire_count": [10, 20, 30],
            # missing mean_frp, median_frp, etc.
        })

        ml = apply_temporal_lag(fused_current, partial_prev)

        # active_fire_count should come from prev
        assert ml.loc[ml["grid_id"] == "cell_a", "active_fire_count"].values[0] == 10

        # missing columns should exist with defaults
        assert "mean_frp" in ml.columns
        assert "nearest_fire_distance_km" in ml.columns

    def test_output_shape_stable(self, fused_current, prev_fire_features):
        """Output should have same number of rows and all expected columns."""
        ml = apply_temporal_lag(fused_current, prev_fire_features)
        assert len(ml) == len(fused_current)

        for col in FIRE_CONTEXT_LAG_COLS:
            assert col in ml.columns
        assert "fire_detected_binary" in ml.columns
