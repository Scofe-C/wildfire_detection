"""
Tests for Priority Hierarchy Engine.
"""

import pandas as pd
import pytest

from scripts.fusion.priority_resolver import (
    resolve_priorities,
    _haversine_km,
    OVERRIDABLE_FIRE_COLS,
)


@pytest.fixture
def fused_df():
    """Satellite-based fused DataFrame (Priority 2)."""
    return pd.DataFrame({
        "grid_id": ["cell_a", "cell_b", "cell_c", "cell_d"],
        "latitude": [34.0, 34.01, 34.05, 35.0],
        "longitude": [-118.0, -118.01, -118.05, -119.0],
        "timestamp": pd.Timestamp("2025-01-15 06:00", tz="UTC"),
        "active_fire_count": [5, 3, 0, 10],
        "mean_frp": [120.0, 80.0, 0.0, 200.0],
        "median_frp": [100.0, 70.0, 0.0, 180.0],
        "max_confidence": [90, 75, 0, 95],
        "fire_detected_binary": [1, 1, 0, 1],
        "nearest_fire_distance_km": [0.0, 5.0, 50.0, 0.0],
    })


@pytest.fixture
def ground_truth_df():
    """Ground truth from field telemetry — overrides cell_a and neighbors."""
    return pd.DataFrame({
        "latitude": [34.005],
        "longitude": [-118.005],
        "timestamp": pd.Timestamp("2025-01-15 05:00", tz="UTC"),
        "active_fire_count": [15],
        "mean_frp": [300.0],
        "median_frp": [280.0],
        "max_confidence": [99],
        "fire_detected_binary": [1],
        "confidence": [98],
        "source_type": ["drone"],
    })


class TestNoop:
    """When no ground truth is present, nothing changes."""

    def test_empty_ground_truth(self, fused_df):
        result = resolve_priorities(fused_df, pd.DataFrame())
        assert "data_source_priority" in result.columns
        assert (result["data_source_priority"] == 2).all()
        assert result["active_fire_count"].tolist() == [5, 3, 0, 10]

    def test_none_ground_truth(self, fused_df):
        result = resolve_priorities(fused_df, None)
        assert (result["data_source_priority"] == 2).all()


class TestSpatialOverride:
    """Ground truth overrides nearby cells."""

    def test_nearby_cells_get_priority_1(self, fused_df, ground_truth_df):
        result = resolve_priorities(fused_df, ground_truth_df)
        # cell_a and cell_b are within 5km of ground truth
        nearby = result[result["grid_id"].isin(["cell_a", "cell_b"])]
        assert (nearby["data_source_priority"] == 1).all()

    def test_far_cell_stays_priority_2(self, fused_df, ground_truth_df):
        result = resolve_priorities(fused_df, ground_truth_df)
        # cell_d is 1 degree away (~111km) — well outside 5km radius
        far_cell = result[result["grid_id"] == "cell_d"]
        assert far_cell["data_source_priority"].values[0] == 2

    def test_fire_features_overridden(self, fused_df, ground_truth_df):
        result = resolve_priorities(fused_df, ground_truth_df)
        cell_a = result[result["grid_id"] == "cell_a"]
        # Fire features should reflect ground truth values
        assert cell_a["active_fire_count"].values[0] == 15
        assert cell_a["mean_frp"].values[0] == 300.0


class TestTemporalDecay:
    """Override expires after temporal_decay_hours."""

    def test_expired_ground_truth_ignored(self, fused_df):
        """Ground truth from 24h ago should be ignored (default decay = 6h)."""
        old_gt = pd.DataFrame({
            "latitude": [34.005],
            "longitude": [-118.005],
            "timestamp": pd.Timestamp("2025-01-14 01:00", tz="UTC"),
            "active_fire_count": [99],
            "fire_detected_binary": [1],
        })
        result = resolve_priorities(fused_df, old_gt)
        # Should not override — ground truth is too old
        assert result[result["grid_id"] == "cell_a"]["active_fire_count"].values[0] == 5


class TestMultipleGroundTruth:
    """Multiple ground truth observations."""

    def test_most_recent_wins(self, fused_df):
        gt_multi = pd.DataFrame({
            "latitude": [34.005, 34.005],
            "longitude": [-118.005, -118.005],
            "timestamp": [
                pd.Timestamp("2025-01-15 04:00", tz="UTC"),
                pd.Timestamp("2025-01-15 05:30", tz="UTC"),
            ],
            "active_fire_count": [50, 99],
            "fire_detected_binary": [1, 1],
        })
        result = resolve_priorities(fused_df, gt_multi)
        cell_a = result[result["grid_id"] == "cell_a"]
        # The later observation (99) should be the final value
        assert cell_a["active_fire_count"].values[0] == 99


class TestHaversine:
    """Test haversine distance calculation."""

    def test_same_point_zero_distance(self):
        assert _haversine_km(34.0, -118.0, 34.0, -118.0) == 0.0

    def test_known_distance(self):
        # LA to SF is approximately 559 km
        dist = _haversine_km(34.0522, -118.2437, 37.7749, -122.4194)
        assert 550 < dist < 570
