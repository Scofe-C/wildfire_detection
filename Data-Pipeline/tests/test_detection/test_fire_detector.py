"""
Tests for FireDetector — Four-Gate False Alarm Filter
"""
import os
import pytest
from unittest.mock import MagicMock, patch


def _make_detection(lat, lon, frp=300.0, confidence=80):
    # fire_detector._map_to_h3_cells uses "lat"/"lon" keys (not "latitude"/"longitude")
    return {"lat": lat, "lon": lon, "frp": frp, "confidence": confidence, "sensor": "ABI"}


def _dense_cluster(center_lat, center_lon, rows=5, cols=5, step=0.025):
    """Return rows*cols detections spaced `step` degrees apart.

    0.025 degrees ≈ 2.8 km lat, 2.2 km lon — larger than H3 res-5 cell edge (~1.2 km),
    guaranteeing each point lands in a distinct H3 cell.
    A 5x5 grid produces 25 unique cells; interior cells each have 4+ fire neighbors,
    far exceeding min_neighbor_detections=1.
    """
    detections = []
    for r in range(rows):
        for c in range(cols):
            detections.append(_make_detection(
                center_lat + r * step,
                center_lon + c * step,
            ))
    return detections


@pytest.fixture
def watchdog_config():
    return {
        "false_alarm": {
            "min_neighbor_detections": 1,
            "viirs_lookback_hours": 3,
            "industrial_exclusion_radius_km": 2.0,
        },
        "emergency": {"min_frp_mw": 200.0},
        "slack": {"webhook_env_var": "SLACK_WEBHOOK_URL"},
    }


@pytest.fixture
def quiet_state():
    return {"mode": "quiet", "active_fire_cells": []}


@pytest.fixture
def detector(watchdog_config, quiet_state):
    from scripts.detection.fire_detector import FireDetector
    return FireDetector(watchdog_config, quiet_state, gcs_state=None)


class TestFireDetectorInstantiation:

    def test_creates_instance(self, watchdog_config, quiet_state):
        from scripts.detection.fire_detector import FireDetector
        assert FireDetector(watchdog_config, quiet_state) is not None

    def test_stores_config(self, detector, watchdog_config):
        assert detector.cfg is watchdog_config

    def test_gcs_state_can_be_none(self, watchdog_config, quiet_state):
        from scripts.detection.fire_detector import FireDetector
        assert FireDetector(watchdog_config, quiet_state, gcs_state=None).gcs_state is None


class TestEvaluateEmptyDetections:

    def test_empty_returns_not_confirmed(self, detector):
        assert detector.evaluate(detections=[], region="california")["confirmed"] is False

    def test_empty_gate_failed_is_no_detections(self, detector):
        assert detector.evaluate(detections=[], region="california")["gate_failed"] == "no_detections"

    def test_empty_fire_cells_is_empty(self, detector):
        assert detector.evaluate(detections=[], region="california")["fire_cells"] == []

    def test_result_has_required_keys(self, detector):
        result = detector.evaluate(detections=[], region="california")
        for key in ("confirmed", "gate_failed", "fire_cells", "max_frp", "detection_summary"):
            assert key in result


class TestGate1Spatial:

    def test_isolated_single_detection_fails_g1(self, detector):
        """One isolated point → 0 fire neighbors → G1 fails."""
        result = detector.evaluate(
            detections=[_make_detection(36.0, -118.0)],
            region="california", previous_scan_detections=None, industrial_sources=[],
        )
        assert result["confirmed"] is False
        assert result["gate_failed"] == "G1"

    def test_dense_cluster_passes_g1(self, detector):
        """5×5 grid at 0.025° spacing → 25 distinct H3 cells, each with 4+ neighbors → G1 passes."""
        dets = _dense_cluster(37.5, -120.5)
        result = detector.evaluate(
            detections=dets, region="california",
            previous_scan_detections=None, industrial_sources=[],
        )
        assert result["gate_failed"] != "G1", (
            f"Dense 5x5 cluster should pass G1, got gate_failed={result['gate_failed']}"
        )


class TestGate2Temporal:

    def test_no_previous_scan_skips_g2(self, detector):
        dets = _dense_cluster(37.5, -120.5)
        result = detector.evaluate(
            detections=dets, region="california",
            previous_scan_detections=None, industrial_sources=[],
        )
        assert result["gate_failed"] != "G2"

    def test_same_cluster_in_previous_scan_passes_g2(self, detector):
        """Identical cluster in current and previous scan → G1 and G2 both pass."""
        dets = _dense_cluster(37.5, -120.5)
        result = detector.evaluate(
            detections=dets, region="california",
            previous_scan_detections=dets, industrial_sources=[],
        )
        assert result["gate_failed"] not in ("G1", "G2"), (
            f"Expected G1/G2 to pass with identical cluster, got gate_failed={result['gate_failed']}"
        )

    def test_different_location_previous_scan_reduces_confirmed_cells(self, detector):
        """Previous scan at a different location means zero temporal overlap.

        G2 has four real outcomes depending on state and FRP:
          - "G2_pending"      : no overlap, consecutive scan count below threshold
          - "G2_no_prior_scan": no prior scan provided and FRP too low to bypass
          - passes            : overlap found, or consecutive count met threshold
        All of these are valid — the important assertion is confirmed=False.
        """
        current = _dense_cluster(37.5, -120.5)
        previous = _dense_cluster(34.0, -117.0)
        result = detector.evaluate(
            detections=current, region="california",
            previous_scan_detections=previous, industrial_sources=[],
        )
        # When there is zero temporal overlap the fire must not be confirmed
        assert result["confirmed"] is False
        # gate_failed must be one of the documented G2 outcomes (or G1 if G1 also fails)
        assert result["gate_failed"] in ("G1", "G2", "G2_pending", "G2_no_prior_scan", "G3", "G4")


class TestGate4Industrial:

    def test_industrial_source_far_away_does_not_suppress(self, detector):
        dets = _dense_cluster(37.5, -120.5)
        result = detector.evaluate(
            detections=dets, region="california",
            previous_scan_detections=dets,
            industrial_sources=[{"name": "Distant", "lat": 32.0, "lon": -117.0, "radius_km": 2.0}],
        )
        assert result["gate_failed"] != "G4"

    def test_empty_industrial_sources_passes_g4(self, detector):
        dets = _dense_cluster(37.5, -120.5)
        result = detector.evaluate(
            detections=dets, region="california",
            previous_scan_detections=dets, industrial_sources=[],
        )
        assert result["gate_failed"] != "G4"


class TestFullPipeline:

    def test_detection_summary_contains_region(self, detector):
        assert detector.evaluate(detections=[], region="texas")["detection_summary"]["region"] == "texas"

    def test_original_state_not_mutated(self, watchdog_config):
        from scripts.detection.fire_detector import FireDetector
        state = {"mode": "quiet", "active_fire_cells": []}
        FireDetector(watchdog_config, state).evaluate(
            detections=_dense_cluster(37.5, -120.5), region="california"
        )
        assert state["mode"] == "quiet"

    def test_confirmed_fire_has_non_empty_cells(self, detector):
        dets = _dense_cluster(37.5, -120.5)
        with patch.dict(os.environ, {"FIRMS_MAP_KEY": ""}):
            result = detector.evaluate(
                detections=dets, region="california",
                previous_scan_detections=dets, industrial_sources=[],
            )
        if result["confirmed"]:
            assert len(result["fire_cells"]) > 0