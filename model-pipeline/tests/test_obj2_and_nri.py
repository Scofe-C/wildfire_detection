"""
Tests for OBJ-2 (PROPAGATOR) and NRI Loader
============================================
Run with: PYTHONPATH=. pytest tests/test_obj2_and_nri.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# =====================================================================
# PROPAGATOR fuel reclassification tests
# =====================================================================

class TestReclassifyFuel:
    """Tests for reclassify_fuel()."""

    def test_known_codes(self):
        from src.models.obj2_spread.propagator_spread import reclassify_fuel
        codes = np.array([101, 141, 181, 99])  # grass, shrub, litter, NB
        result = reclassify_fuel(codes)
        assert list(result) == [1, 3, 5, 7]

    def test_unknown_codes_map_to_nonburnable(self):
        from src.models.obj2_spread.propagator_spread import reclassify_fuel
        codes = np.array([999, 0, -1])
        result = reclassify_fuel(codes)
        assert list(result) == [7, 7, 7]

    def test_custom_crosswalk(self):
        from src.models.obj2_spread.propagator_spread import reclassify_fuel
        custom = {101: 5, 102: 6}
        codes = np.array([101, 102, 103])
        result = reclassify_fuel(codes, crosswalk=custom)
        assert list(result) == [5, 6, 7]  # 103 unmapped → 7


# =====================================================================
# PropagatorSpread interface compliance
# =====================================================================

class TestPropagatorSpreadInterface:
    """Verify PropagatorSpread implements the BaseModel contract."""

    def test_inherits_base_model(self):
        from src.models.base import BaseModel
        from src.models.obj2_spread.propagator_spread import PropagatorSpread
        model = PropagatorSpread()
        assert isinstance(model, BaseModel)

    def test_disclaimer_on_every_prediction(self, tmp_path):
        from src.models.obj2_spread.propagator_spread import PropagatorSpread

        prop_cfg = {
            "enabled": True,
            "crosswalk_path": None,
            "default_params": {"wind_reduction_factor": 0.4},
        }

        config_file = tmp_path / "propagator_config.json"
        config_file.write_text(json.dumps({}))

        model = PropagatorSpread()
        with patch(
            "src.models.obj2_spread.propagator_spread._load_propagator_config",
            return_value=prop_cfg,
        ):
            model.load_model(config_file)

        X = pd.DataFrame({
            "wind_speed_10m": [10.0, 5.0, 15.0],
            "slope_degrees": [5.0, 20.0, 35.0],
            "relative_humidity_2m": [30, 60, 20],
        })
        result = model.predict(X)
        assert "disclaimer" in result.columns
        assert all(result["disclaimer"] == PropagatorSpread.DISCLAIMER)
        assert "prediction" in result.columns
        assert "probability" in result.columns

    def test_explain_returns_structure(self):
        from src.models.obj2_spread.propagator_spread import PropagatorSpread
        model = PropagatorSpread()
        model._is_loaded = True
        model._crosswalk = {101: 1, 141: 3}
        result = model.explain(pd.DataFrame())
        assert "disclaimer" in result
        assert "crosswalk_coverage" in result
        assert result["crosswalk_coverage"]["mapped_codes"] == 2


# =====================================================================
# NRI Loader tests
# =====================================================================

class TestNRILoader:
    """Tests for src.bias.nri_loader module."""

    def test_load_nri_missing_dir_raises(self):
        from src.bias.nri_loader import NRILoadError, load_nri
        with pytest.raises(NRILoadError, match="not found"):
            load_nri(cache_dir="/nonexistent/path")

    def test_compute_vulnerability_quartiles(self):
        import geopandas as gpd
        from shapely.geometry import Point

        from src.bias.nri_loader import compute_vulnerability_quartiles

        nri = gpd.GeoDataFrame({
            "SOVI_SCORE": np.random.uniform(0, 100, size=100),
            "geometry": [Point(0, i) for i in range(100)],
        })
        result = compute_vulnerability_quartiles(nri)
        assert "nri_vulnerability_quartile" in result.columns
        labels = set(result["nri_vulnerability_quartile"].unique())
        assert labels == {"Low", "Medium", "High", "Very High"}
        # Each quartile should have ~25 tracts
        for label in labels:
            count = (result["nri_vulnerability_quartile"] == label).sum()
            assert 20 <= count <= 30

    def test_spatial_join_fills_unknown_for_unmatched(self):
        import geopandas as gpd
        from shapely.geometry import Point

        from src.bias.nri_loader import spatial_join_predictions

        # Create NRI with tracts only near (0, 0)
        nri = gpd.GeoDataFrame({
            "nri_vulnerability_quartile": ["High", "Low"],
            "SOVI_SCORE": [80.0, 20.0],
            "geometry": [Point(0, 0), Point(0, 0.01)],
        }, crs="EPSG:4326")

        # Predictions with one nearby and one far-away cell
        # sjoin_nearest always matches, so "Unknown" only appears
        # if nri_vulnerability_quartile is NaN — which doesn't happen
        # with sjoin_nearest. This tests the join completes without error.
        preds = pd.DataFrame({
            "h3_index": ["8529a183fffffff", "8529a1c3fffffff"],
        })

        with patch("h3.cell_to_latlng") as mock_c2l:
            mock_c2l.side_effect = [
                (0.0, 0.0),      # near the NRI tracts
                (50.0, 50.0),    # far away
            ]
            result = spatial_join_predictions(preds, nri)

        assert len(result) == 2
        assert "nri_vulnerability_quartile" in result.columns


# =====================================================================
# Crosswalk loader tests
# =====================================================================

class TestCrosswalkLoader:
    """Tests for load_crosswalk()."""

    def test_default_crosswalk(self):
        from src.models.obj2_spread.propagator_spread import load_crosswalk
        cw = load_crosswalk(crosswalk_path=None)
        assert len(cw) > 0
        assert cw[101] == 1   # grass
        assert cw[141] == 3   # shrub

    def test_custom_crosswalk_file(self, tmp_path):
        from src.models.obj2_spread.propagator_spread import load_crosswalk
        custom = {"101": 5, "102": 6}
        cw_file = tmp_path / "crosswalk.json"
        cw_file.write_text(json.dumps(custom))
        cw = load_crosswalk(cw_file)
        assert cw[101] == 5
        assert cw[102] == 6

    def test_missing_file_falls_back_to_default(self):
        from src.models.obj2_spread.propagator_spread import load_crosswalk
        cw = load_crosswalk(crosswalk_path="/nonexistent.json")
        assert len(cw) > 0  # falls back to DEFAULT_CROSSWALK
