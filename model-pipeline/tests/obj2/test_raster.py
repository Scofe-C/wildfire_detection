"""
Tests for src/models/obj2_spread/raster.py
"""
import numpy as np
import pytest

from src.models.obj2_spread.exceptions import Cell2FireError
from src.models.obj2_spread.raster import clip_raster_to_aoi, parse_burn_probability

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_geotiff(path, width=100, height=100, crs="EPSG:4326"):
    """Create a small synthetic GeoTIFF for testing."""
    import rasterio
    from rasterio.transform import from_bounds

    transform = from_bounds(-119.0, 33.5, -118.0, 34.5, width, height)
    data = np.random.randint(100, 165, (height, width), dtype=np.int16)

    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=height, width=width,
        count=1, dtype=data.dtype,
        crs=crs, transform=transform,
    ) as dst:
        dst.write(data, 1)
    return path


def _make_grid_files(output_dir, n_sims=5, rows=10, cols=10):
    """Create fake Cell2Fire output grid CSV files."""
    grids_dir = output_dir / "Grids" / "Grids1"
    grids_dir.mkdir(parents=True)
    for i in range(1, n_sims + 1):
        grid = np.zeros((rows, cols), dtype=int)
        # Burn a small patch
        grid[3:6, 3:6] = 1
        header = ",".join([str(j) for j in range(cols)])
        np.savetxt(
            grids_dir / f"ForestGrid{i:02d}.csv",
            grid, delimiter=",", header=header, comments="",
        )
    return grids_dir


# ---------------------------------------------------------------------------
# clip_raster_to_aoi
# ---------------------------------------------------------------------------

class TestClipRasterToAoi:

    def test_produces_smaller_raster(self, tmp_path):
        src = tmp_path / "source.tif"
        _make_test_geotiff(src)
        out = tmp_path / "clipped.tif"

        # Clip to half the extent
        clip_raster_to_aoi(src, (-118.75, 33.5, -118.0, 34.5), out)

        import rasterio
        with rasterio.open(src) as s, rasterio.open(out) as c:
            assert c.width < s.width

    def test_returns_path(self, tmp_path):
        src = tmp_path / "source.tif"
        _make_test_geotiff(src)
        out = tmp_path / "clipped.tif"
        result = clip_raster_to_aoi(src, (-118.75, 33.5, -118.0, 34.5), out)
        assert result == out
        assert out.exists()

    def test_raises_on_empty_clip(self, tmp_path):
        src = tmp_path / "source.tif"
        _make_test_geotiff(src)
        out = tmp_path / "clipped.tif"

        # Bounds completely outside raster extent
        with pytest.raises(Cell2FireError, match="empty raster"):
            clip_raster_to_aoi(src, (10.0, 10.0, 11.0, 11.0), out)

    def test_creates_output_directory(self, tmp_path):
        src = tmp_path / "source.tif"
        _make_test_geotiff(src)
        out = tmp_path / "subdir" / "nested" / "clipped.tif"

        clip_raster_to_aoi(src, (-118.75, 33.5, -118.0, 34.5), out)
        assert out.exists()


# ---------------------------------------------------------------------------
# parse_burn_probability
# ---------------------------------------------------------------------------

class TestParseBurnProbability:

    def test_probability_range(self, tmp_path):
        _make_grid_files(tmp_path, n_sims=5)
        prob = parse_burn_probability(tmp_path, n_simulations=5)
        assert prob.min() >= 0.0
        assert prob.max() <= 1.0

    def test_correct_shape(self, tmp_path):
        _make_grid_files(tmp_path, n_sims=5, rows=10, cols=10)
        prob = parse_burn_probability(tmp_path, n_simulations=5)
        assert prob.shape == (10, 10)

    def test_burned_patch_has_high_probability(self, tmp_path):
        _make_grid_files(tmp_path, n_sims=5)
        prob = parse_burn_probability(tmp_path, n_simulations=5)
        # The patch at rows 3-5, cols 3-5 should have prob = 1.0
        assert prob[3:6, 3:6].mean() == pytest.approx(1.0)

    def test_unburned_cells_have_zero_probability(self, tmp_path):
        _make_grid_files(tmp_path, n_sims=5)
        prob = parse_burn_probability(tmp_path, n_simulations=5)
        # Corners should be unburned
        assert prob[0, 0] == 0.0

    def test_raises_if_no_grid_files(self, tmp_path):
        # Create empty Grids directory with no CSV files
        (tmp_path / "Grids" / "Grids1").mkdir(parents=True)
        with pytest.raises(Cell2FireError, match="No output grids"):
            parse_burn_probability(tmp_path, n_simulations=5)

    def test_handles_zero_simulations_gracefully(self, tmp_path):
        _make_grid_files(tmp_path, n_sims=3)
        # n_simulations=0 should not divide by zero
        prob = parse_burn_probability(tmp_path, n_simulations=0)
        assert np.all(np.isfinite(prob))
