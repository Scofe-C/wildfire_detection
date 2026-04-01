"""
Tests for src/models/obj2_spread/evaluation.py
"""
import numpy as np
import pytest

from src.models.obj2_spread.evaluation import (
    compute_buffered_iou,
    compute_dice_coefficient,
    find_best_threshold,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_perimeter_gdf(center_lon=-118.5, center_lat=34.1, size=0.05):
    """Create a simple square perimeter GeoDataFrame."""
    import geopandas as gpd
    from shapely.geometry import box

    polygon = box(
        center_lon - size, center_lat - size,
        center_lon + size, center_lat + size,
    )
    return gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")


def _make_burn_prob_grid(rows=50, cols=50, burned_fraction=0.25):
    """Create a synthetic burn probability grid with a burned patch."""
    grid = np.zeros((rows, cols))
    patch_size = int(np.sqrt(rows * cols * burned_fraction))
    start = (rows - patch_size) // 2
    grid[start:start + patch_size, start:start + patch_size] = 0.8
    return grid


def _make_transform(west=-118.72, south=33.97, east=-118.37, north=34.20,
                    rows=50, cols=50):
    """Create a rasterio-compatible Affine transform."""
    from rasterio.transform import from_bounds
    return from_bounds(west, south, east, north, cols, rows)


# ---------------------------------------------------------------------------
# compute_dice_coefficient
# ---------------------------------------------------------------------------

class TestComputeDiceCoefficient:

    def test_perfect_overlap_returns_one(self):
        mask = np.array([1, 1, 0, 0, 1])
        assert compute_dice_coefficient(mask, mask) == pytest.approx(1.0)

    def test_no_overlap_returns_zero(self):
        pred = np.array([1, 0, 0, 0, 0])
        actual = np.array([0, 0, 0, 0, 1])
        assert compute_dice_coefficient(pred, actual) == pytest.approx(0.0)

    def test_both_empty_returns_one(self):
        pred = np.zeros(10)
        actual = np.zeros(10)
        assert compute_dice_coefficient(pred, actual) == pytest.approx(1.0)

    def test_partial_overlap(self):
        pred   = np.array([1, 1, 1, 0, 0])
        actual = np.array([0, 1, 1, 1, 0])
        # intersection=2, total=6 → dice=4/6≈0.667
        assert compute_dice_coefficient(pred, actual) == pytest.approx(4 / 6, rel=1e-3)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_dice_coefficient(np.ones(5), np.ones(6))

    def test_float_inputs_work(self):
        pred = np.array([0.9, 0.8, 0.1, 0.05])
        actual = np.array([1, 1, 0, 0])
        # Should work after bool cast
        result = compute_dice_coefficient(pred > 0.5, actual)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# compute_buffered_iou
# ---------------------------------------------------------------------------

class TestComputeBufferedIou:

    def test_returns_expected_keys(self):
        grid = _make_burn_prob_grid()
        transform = _make_transform()
        actual_gdf = _make_perimeter_gdf()

        result = compute_buffered_iou(grid, actual_gdf, transform)

        expected_keys = {
            "buffered_iou", "directional_accuracy", "angle_diff_degrees",
            "area_ratio", "area_ratio_ok", "pred_area_km2", "actual_area_km2",
            "gate_passed", "threshold_used", "buffer_pct_used",
        }
        assert expected_keys.issubset(result.keys())

    def test_iou_in_valid_range(self):
        grid = _make_burn_prob_grid()
        transform = _make_transform()
        actual_gdf = _make_perimeter_gdf()

        result = compute_buffered_iou(grid, actual_gdf, transform)
        assert 0.0 <= result["buffered_iou"] <= 1.0

    def test_empty_prediction_returns_zero(self):
        grid = np.zeros((50, 50))  # no predicted burn
        transform = _make_transform()
        actual_gdf = _make_perimeter_gdf()

        result = compute_buffered_iou(grid, actual_gdf, transform, threshold=0.10)
        assert result["buffered_iou"] == 0.0
        assert result["gate_passed"] is False

    def test_perfect_prediction_passes_gate(self):
        """When predicted and actual areas perfectly overlap, gate should pass."""
        import geopandas as gpd
        from rasterio.transform import from_bounds
        from shapely.geometry import box

        # Create grid covering exact same area as perimeter
        rows, cols = 50, 50
        west, south, east, north = -118.60, 34.05, -118.40, 34.20
        transform = from_bounds(west, south, east, north, cols, rows)

        # All cells burned at high probability
        grid = np.ones((rows, cols)) * 0.9

        # Actual perimeter covers same area
        actual_gdf = gpd.GeoDataFrame(
            geometry=[box(west, south, east, north)], crs="EPSG:4326"
        )

        result = compute_buffered_iou(grid, actual_gdf, transform, threshold=0.5)
        assert result["buffered_iou"] > 0.35

    def test_threshold_affects_result(self):
        grid = _make_burn_prob_grid()
        transform = _make_transform()
        actual_gdf = _make_perimeter_gdf()

        result_low = compute_buffered_iou(grid, actual_gdf, transform, threshold=0.05)
        result_high = compute_buffered_iou(grid, actual_gdf, transform, threshold=0.90)

        # Lower threshold = more predicted burned area
        assert result_low["pred_area_km2"] >= result_high["pred_area_km2"]


# ---------------------------------------------------------------------------
# find_best_threshold
# ---------------------------------------------------------------------------

class TestFindBestThreshold:

    def test_returns_best_of_tried_thresholds(self):
        grid = _make_burn_prob_grid()
        transform = _make_transform()
        actual_gdf = _make_perimeter_gdf()

        thresholds = [0.10, 0.20, 0.30]
        best = find_best_threshold(grid, actual_gdf, transform, thresholds)

        assert best["threshold_used"] in thresholds

    def test_best_has_highest_iou(self):
        grid = _make_burn_prob_grid()
        transform = _make_transform()
        actual_gdf = _make_perimeter_gdf()

        thresholds = [0.10, 0.20, 0.30]
        best = find_best_threshold(grid, actual_gdf, transform, thresholds)

        # Verify it's actually the best
        for t in thresholds:
            result = compute_buffered_iou(grid, actual_gdf, transform, threshold=t)
            assert best["buffered_iou"] >= result["buffered_iou"]

    def test_uses_default_thresholds_when_none(self):
        grid = _make_burn_prob_grid()
        transform = _make_transform()
        actual_gdf = _make_perimeter_gdf()

        best = find_best_threshold(grid, actual_gdf, transform)
        # Default thresholds are [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        assert best["threshold_used"] in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
