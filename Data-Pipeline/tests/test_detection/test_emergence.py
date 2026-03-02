"""
Tests for detection/emergency.py
==================================
"""
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def watchdog_config():
    return {
        "emergency": {
            "min_frp_mw": 200.0,
            "min_expanding_scans": 2,
            "deactivate_no_expansion_scans": 3,
            "deactivate_frp_mw": 50.0,
            "deactivate_low_frp_scans": 2,
        },
        "modes": {
            "quiet":     {"resolution_km": 64},
            "active":    {"resolution_km": 22},
            "emergency": {"resolution_km": 22},
        },
        "detection": {"max_range_km": 25, "h3_ring_max": 5},
        "slack": {"webhook_env_var": "SLACK_WEBHOOK_URL"},
    }


@pytest.fixture
def quiet_state():
    return {"mode": "quiet", "active_fire_cells": []}


@pytest.fixture
def active_state():
    return {
        "mode": "active",
        "active_fire_cells": ["cell_a", "cell_b"],
        "consecutive_expanding_scans": 0,
        "consecutive_no_expansion_scans": 0,
        "consecutive_low_frp_scans": 0,
    }


@pytest.fixture
def active_state_primed():
    """Already has 1 expanding scan — one more will reach min_expanding_scans=2."""
    return {
        "mode": "active",
        "active_fire_cells": ["cell_a", "cell_b"],
        "consecutive_expanding_scans": 1,
        "consecutive_no_expansion_scans": 0,
        "consecutive_low_frp_scans": 0,
    }


@pytest.fixture
def emergency_state():
    return {
        "mode": "emergency",
        "active_fire_cells": ["cell_a", "cell_b"],
        "consecutive_expanding_scans": 0,
        "consecutive_no_expansion_scans": 0,
        "consecutive_low_frp_scans": 0,
        "emergency_activated_at": "2026-08-15T18:00:00+00:00",
        "prior_mode": "active",
    }


class TestEvaluateEmergency:

    def test_quiet_state_stays_quiet_below_frp_threshold(self, quiet_state, watchdog_config):
        """In quiet mode with FRP below threshold, mode must remain quiet."""
        from scripts.detection.emergency import evaluate_emergency
        result = evaluate_emergency(
            state=quiet_state, confirmed_cells=["cell_a"],
            max_frp=100.0, watchdog_config=watchdog_config,
        )
        assert result["mode"] == "quiet"

    def test_active_state_does_not_activate_on_first_expanding_scan(
        self, active_state, watchdog_config
    ):
        """Emergency requires min_expanding_scans=2 consecutive calls.
        The first scan with new cells only sets consecutive_expanding_scans=1."""
        from scripts.detection.emergency import evaluate_emergency
        result = evaluate_emergency(
            state=active_state,
            confirmed_cells=["cell_a", "cell_b", "cell_c"],  # cell_c is new
            max_frp=650.0,
            watchdog_config=watchdog_config,
        )
        # Should NOT activate on first scan — counter only reaches 1
        assert result["mode"] == "active"
        assert result.get("consecutive_expanding_scans", 0) == 1

    def test_active_state_activates_emergency_on_second_consecutive_expanding_scan(
        self, active_state_primed, watchdog_config
    ):
        """Emergency activates when consecutive_expanding_scans reaches min_expanding_scans=2."""
        from scripts.detection.emergency import evaluate_emergency
        result = evaluate_emergency(
            state=active_state_primed,
            confirmed_cells=["cell_a", "cell_b", "cell_c"],  # cell_c is new → scans=2
            max_frp=650.0,
            watchdog_config=watchdog_config,
        )
        assert result["mode"] == "emergency"

    def test_emergency_state_stays_emergency_while_expanding(
        self, emergency_state, watchdog_config
    ):
        from scripts.detection.emergency import evaluate_emergency
        result = evaluate_emergency(
            state=emergency_state,
            confirmed_cells=["cell_a", "cell_b", "cell_c", "cell_d"],
            max_frp=800.0,
            watchdog_config=watchdog_config,
        )
        assert result["mode"] == "emergency"

    def test_emergency_deactivates_after_no_expansion_scans(
        self, emergency_state, watchdog_config
    ):
        from scripts.detection.emergency import evaluate_emergency
        state = dict(emergency_state)
        state["consecutive_no_expansion_scans"] = 2
        result = evaluate_emergency(
            state=state, confirmed_cells=["cell_a", "cell_b"],
            max_frp=300.0, watchdog_config=watchdog_config,
        )
        assert result["mode"] != "emergency"

    def test_emergency_deactivates_after_low_frp_scans(
        self, emergency_state, watchdog_config
    ):
        from scripts.detection.emergency import evaluate_emergency
        state = dict(emergency_state)
        state["consecutive_low_frp_scans"] = 1
        result = evaluate_emergency(
            state=state, confirmed_cells=["cell_a", "cell_b"],
            max_frp=20.0, watchdog_config=watchdog_config,
        )
        assert result["mode"] != "emergency"

    def test_input_state_is_not_mutated(self, emergency_state, watchdog_config):
        from scripts.detection.emergency import evaluate_emergency
        original_mode = emergency_state["mode"]
        evaluate_emergency(
            state=emergency_state,
            confirmed_cells=["cell_a", "cell_b", "cell_c"],
            max_frp=500.0, watchdog_config=watchdog_config,
        )
        assert emergency_state["mode"] == original_mode

    def test_no_expansion_counter_resets_on_new_cells(self, emergency_state, watchdog_config):
        from scripts.detection.emergency import evaluate_emergency
        state = dict(emergency_state)
        state["consecutive_no_expansion_scans"] = 2
        result = evaluate_emergency(
            state=state,
            confirmed_cells=["cell_a", "cell_b", "cell_c"],
            max_frp=400.0, watchdog_config=watchdog_config,
        )
        assert result.get("consecutive_no_expansion_scans", 0) == 0

    def test_gcs_state_called_on_deactivation(self, emergency_state, watchdog_config):
        from scripts.detection.emergency import evaluate_emergency
        state = dict(emergency_state)
        state["consecutive_no_expansion_scans"] = 2
        mock_gcs = MagicMock()
        result = evaluate_emergency(
            state=state, confirmed_cells=["cell_a", "cell_b"],
            max_frp=300.0, watchdog_config=watchdog_config, gcs_state=mock_gcs,
        )
        if result["mode"] != "emergency":
            mock_gcs.write_emergency_log.assert_called_once()

    def test_gcs_state_none_does_not_raise(self, emergency_state, watchdog_config):
        from scripts.detection.emergency import evaluate_emergency
        evaluate_emergency(
            state=emergency_state,
            confirmed_cells=["cell_a", "cell_b", "cell_c"],
            max_frp=500.0, watchdog_config=watchdog_config, gcs_state=None,
        )

    def test_returned_state_contains_active_fire_cells(self, active_state, watchdog_config):
        from scripts.detection.emergency import evaluate_emergency
        result = evaluate_emergency(
            state=active_state, confirmed_cells=["cell_x"],
            max_frp=50.0, watchdog_config=watchdog_config,
        )
        assert "active_fire_cells" in result


class TestGetPipelineParamsForMode:

    def test_returns_dict(self, watchdog_config):
        from scripts.detection.emergency import get_pipeline_params_for_mode
        result = get_pipeline_params_for_mode("quiet", [], watchdog_config, "california")
        assert isinstance(result, dict)

    def test_quiet_mode_uses_64km_resolution(self, watchdog_config):
        from scripts.detection.emergency import get_pipeline_params_for_mode
        result = get_pipeline_params_for_mode("quiet", [], watchdog_config, "california")
        assert result["resolution_km"] == 64

    def test_active_mode_uses_22km_resolution(self, watchdog_config):
        from scripts.detection.emergency import get_pipeline_params_for_mode
        result = get_pipeline_params_for_mode("active", ["cell_a"], watchdog_config, "california")
        assert result["resolution_km"] == 22

    def test_emergency_mode_uses_22km_resolution(self, watchdog_config):
        from scripts.detection.emergency import get_pipeline_params_for_mode
        result = get_pipeline_params_for_mode("emergency", [], watchdog_config, "texas")
        assert result["resolution_km"] == 22

    def test_fire_cells_passed_through(self, watchdog_config):
        from scripts.detection.emergency import get_pipeline_params_for_mode
        cells = ["abc123", "def456"]
        result = get_pipeline_params_for_mode("active", cells, watchdog_config, "california")
        assert result["fire_cells"] == cells

    def test_region_passed_through(self, watchdog_config):
        from scripts.detection.emergency import get_pipeline_params_for_mode
        result = get_pipeline_params_for_mode("active", [], watchdog_config, "texas")
        assert "texas" in result.get("regions", [])

    def test_trigger_source_includes_mode_name(self, watchdog_config):
        from scripts.detection.emergency import get_pipeline_params_for_mode
        result = get_pipeline_params_for_mode("active", [], watchdog_config, "california")
        assert "active" in result["trigger_source"]

    def test_unknown_mode_falls_back_gracefully(self, watchdog_config):
        from scripts.detection.emergency import get_pipeline_params_for_mode
        result = get_pipeline_params_for_mode("nonexistent", [], watchdog_config, "california")
        assert "resolution_km" in result

    def test_required_keys_present(self, watchdog_config):
        from scripts.detection.emergency import get_pipeline_params_for_mode
        required = {"trigger_source", "resolution_km", "regions", "fire_cells", "mode"}
        result = get_pipeline_params_for_mode("emergency", [], watchdog_config, "california")
        assert not (required - set(result.keys()))