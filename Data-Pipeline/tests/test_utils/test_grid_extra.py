"""
Tests for additional grid_utils functions
==========================================
Covers the four untested public functions:
  - points_to_grid_ids()    — vectorised lat/lon → H3 cell ID array
  - get_cell_area_km2()     — approximate area of a single H3 cell
  - get_cell_neighbors()    — ring-1 (and ring-N) neighbors of a cell
  - get_parent_cell()       — coarser-resolution parent of a cell
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_h3_cell(cell_id: str) -> bool:
    """Return True if cell_id looks like a valid H3 hexagon string."""
    try:
        import h3
        # h3 v3: h3.h3_is_valid, h3 v4: h3.is_valid_cell
        if hasattr(h3, "is_valid_cell"):
            return h3.is_valid_cell(cell_id)
        return h3.h3_is_valid(cell_id)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# points_to_grid_ids
# ---------------------------------------------------------------------------

class TestPointsToGridIds:

    def test_returns_numpy_array(self):
        """Return type must be a numpy array."""
        from scripts.utils.grid_utils import points_to_grid_ids
        lats = np.array([37.5, 36.0])
        lons = np.array([-120.5, -119.0])
        result = points_to_grid_ids(lats, lons, resolution_km=64)
        assert isinstance(result, np.ndarray)

    def test_output_length_matches_input(self):
        """Output array length must equal the number of input points."""
        from scripts.utils.grid_utils import points_to_grid_ids
        n = 10
        lats = np.linspace(32.5, 42.0, n)
        lons = np.linspace(-124.0, -114.0, n)
        result = points_to_grid_ids(lats, lons, resolution_km=64)
        assert len(result) == n

    def test_all_output_cells_are_valid_h3(self):
        """Every cell ID in the output must be a valid H3 hexagon."""
        from scripts.utils.grid_utils import points_to_grid_ids
        lats = np.array([37.5, 36.0, 34.0])
        lons = np.array([-120.5, -119.0, -118.0])
        result = points_to_grid_ids(lats, lons, resolution_km=64)
        for cell_id in result:
            assert _valid_h3_cell(cell_id), f"Invalid H3 cell: {cell_id}"

    def test_same_point_produces_same_cell(self):
        """The same lat/lon must always map to the same cell ID."""
        from scripts.utils.grid_utils import points_to_grid_ids
        lats = np.array([37.774929, 37.774929])
        lons = np.array([-122.419418, -122.419418])
        result = points_to_grid_ids(lats, lons, resolution_km=22)
        assert result[0] == result[1]

    def test_different_resolutions_produce_different_cells(self):
        """Points at 64 km and 22 km resolution must produce different cell IDs."""
        from scripts.utils.grid_utils import points_to_grid_ids
        lats = np.array([37.5])
        lons = np.array([-120.5])
        cell_64 = points_to_grid_ids(lats, lons, resolution_km=64)[0]
        cell_22 = points_to_grid_ids(lats, lons, resolution_km=22)[0]
        assert cell_64 != cell_22

    def test_empty_input_returns_empty_array(self):
        """Empty input arrays must produce an empty output array."""
        from scripts.utils.grid_utils import points_to_grid_ids
        result = points_to_grid_ids(np.array([]), np.array([]), resolution_km=64)
        assert len(result) == 0

    def test_single_point_returns_single_cell(self):
        """A single-point input must produce a single-element array."""
        from scripts.utils.grid_utils import points_to_grid_ids
        result = points_to_grid_ids(np.array([37.5]), np.array([-120.5]), resolution_km=22)
        assert len(result) == 1
        assert _valid_h3_cell(result[0])


# ---------------------------------------------------------------------------
# get_cell_area_km2
# ---------------------------------------------------------------------------

class TestGetCellAreaKm2:

    @pytest.fixture
    def cell_64km(self):
        """A valid H3 cell at res 2 (~64 km)."""
        from scripts.utils.grid_utils import point_to_grid_id
        return point_to_grid_id(37.5, -120.5, resolution_km=64)

    @pytest.fixture
    def cell_22km(self):
        """A valid H3 cell at res 5 (~22 km)."""
        from scripts.utils.grid_utils import point_to_grid_id
        return point_to_grid_id(37.5, -120.5, resolution_km=22)

    def test_returns_positive_float(self, cell_64km):
        """Cell area must be a positive float."""
        from scripts.utils.grid_utils import get_cell_area_km2
        area = get_cell_area_km2(cell_64km)
        assert isinstance(area, float)
        assert area > 0.0

    def test_64km_cell_larger_than_22km_cell(self, cell_64km, cell_22km):
        """A coarser-resolution cell must have a larger area than a finer one."""
        from scripts.utils.grid_utils import get_cell_area_km2
        area_64 = get_cell_area_km2(cell_64km)
        area_22 = get_cell_area_km2(cell_22km)
        assert area_64 > area_22

    def test_area_is_in_plausible_range_for_res2(self, cell_64km):
        """H3 res-2 cells have an average area of ~86,000 km² — should be in range."""
        from scripts.utils.grid_utils import get_cell_area_km2
        area = get_cell_area_km2(cell_64km)
        # H3 res-2 average: ~86,000 km². Allow ±50% for edge hexagons.
        assert 40_000 < area < 150_000, f"Area {area:.0f} km² is outside expected range"

    def test_same_cell_id_produces_same_area(self, cell_22km):
        """get_cell_area_km2 must be deterministic for the same cell."""
        from scripts.utils.grid_utils import get_cell_area_km2
        assert get_cell_area_km2(cell_22km) == get_cell_area_km2(cell_22km)


# ---------------------------------------------------------------------------
# get_cell_neighbors
# ---------------------------------------------------------------------------

class TestGetCellNeighbors:

    @pytest.fixture
    def center_cell(self):
        from scripts.utils.grid_utils import point_to_grid_id
        return point_to_grid_id(37.5, -120.5, resolution_km=22)

    def test_returns_list(self, center_cell):
        """get_cell_neighbors must return a list."""
        from scripts.utils.grid_utils import get_cell_neighbors
        result = get_cell_neighbors(center_cell, ring_size=1)
        assert isinstance(result, list)

    def test_ring_1_returns_six_neighbors(self, center_cell):
        """A ring-1 disk of a non-pentagon H3 cell has exactly 6 neighbors."""
        from scripts.utils.grid_utils import get_cell_neighbors
        neighbors = get_cell_neighbors(center_cell, ring_size=1)
        # Standard hexagonal ring-1 has 6 neighbors (center excluded)
        assert len(neighbors) == 6

    def test_center_cell_not_in_neighbors(self, center_cell):
        """The center cell must not appear in its own neighbor list."""
        from scripts.utils.grid_utils import get_cell_neighbors
        neighbors = get_cell_neighbors(center_cell, ring_size=1)
        assert center_cell not in neighbors

    def test_all_neighbors_are_valid_h3_cells(self, center_cell):
        """Every neighbor cell ID must be a valid H3 hexagon."""
        from scripts.utils.grid_utils import get_cell_neighbors
        for cell in get_cell_neighbors(center_cell, ring_size=1):
            assert _valid_h3_cell(cell), f"Invalid neighbor cell: {cell}"

    def test_ring_2_returns_more_neighbors_than_ring_1(self, center_cell):
        """ring_size=2 must return more neighbors than ring_size=1."""
        from scripts.utils.grid_utils import get_cell_neighbors
        ring1 = get_cell_neighbors(center_cell, ring_size=1)
        ring2 = get_cell_neighbors(center_cell, ring_size=2)
        assert len(ring2) > len(ring1)

    def test_neighbors_are_adjacent_to_center(self, center_cell):
        """Each ring-1 neighbor must itself list the center cell as a neighbor."""
        from scripts.utils.grid_utils import get_cell_neighbors
        for neighbor in get_cell_neighbors(center_cell, ring_size=1):
            neighbor_of_neighbor = get_cell_neighbors(neighbor, ring_size=1)
            assert center_cell in neighbor_of_neighbor, (
                f"Center {center_cell} not found in neighbors of {neighbor}"
            )


# ---------------------------------------------------------------------------
# get_parent_cell
# ---------------------------------------------------------------------------

class TestGetParentCell:

    @pytest.fixture
    def fine_cell(self):
        """A cell at 22 km resolution (H3 res 5)."""
        from scripts.utils.grid_utils import point_to_grid_id
        return point_to_grid_id(37.5, -120.5, resolution_km=22)

    def test_returns_valid_h3_cell(self, fine_cell):
        """get_parent_cell must return a valid H3 cell ID."""
        from scripts.utils.grid_utils import get_parent_cell
        parent = get_parent_cell(fine_cell, parent_resolution_km=64)
        assert _valid_h3_cell(parent)

    def test_parent_is_different_from_child(self, fine_cell):
        """Parent cell must differ from the input cell."""
        from scripts.utils.grid_utils import get_parent_cell
        parent = get_parent_cell(fine_cell, parent_resolution_km=64)
        assert parent != fine_cell

    def test_parent_resolution_is_coarser(self, fine_cell):
        """The parent cell must be at a coarser H3 resolution."""
        import h3
        from scripts.utils.grid_utils import get_parent_cell, _h3_get_resolution_compat
        parent = get_parent_cell(fine_cell, parent_resolution_km=64)
        child_res = _h3_get_resolution_compat(fine_cell)
        parent_res = _h3_get_resolution_compat(parent)
        assert parent_res < child_res, (
            f"Parent res {parent_res} should be smaller (coarser) than child res {child_res}"
        )

    def test_multiple_children_share_same_parent(self, fine_cell):
        """Neighboring fine cells that lie within the same coarse cell share a parent."""
        from scripts.utils.grid_utils import get_cell_neighbors, get_parent_cell
        parent = get_parent_cell(fine_cell, parent_resolution_km=64)
        # At least some neighbors must have the same parent (they're nearby cells)
        neighbors = get_cell_neighbors(fine_cell, ring_size=1)
        neighbor_parents = [get_parent_cell(n, parent_resolution_km=64) for n in neighbors]
        assert parent in neighbor_parents, (
            "At least one immediate neighbor should share the same 64km parent cell"
        )

    def test_deterministic_for_same_input(self, fine_cell):
        """get_parent_cell must return the same result for the same input."""
        from scripts.utils.grid_utils import get_parent_cell
        p1 = get_parent_cell(fine_cell, parent_resolution_km=64)
        p2 = get_parent_cell(fine_cell, parent_resolution_km=64)
        assert p1 == p2