"""
Tests for src/models/obj2_spread/raster.py
"""
import numpy as np
import pytest

from src.models.obj2_spread.raster import clip_raster_to_aoi

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
        with pytest.raises(ValueError, match="empty raster"):
            clip_raster_to_aoi(src, (10.0, 10.0, 11.0, 11.0), out)

    def test_creates_output_directory(self, tmp_path):
        src = tmp_path / "source.tif"
        _make_test_geotiff(src)
        out = tmp_path / "subdir" / "nested" / "clipped.tif"

        clip_raster_to_aoi(src, (-118.75, 33.5, -118.0, 34.5), out)
        assert out.exists()

