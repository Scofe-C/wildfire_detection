"""Unit tests for state_machine.py — §5.1 test_state_machine."""

from __future__ import annotations

import pytest

from src.models.obj3_gemini.state_machine import (
    AdminToggle,
    EmergencySubState,
    OperationalMode,
    mode_to_report_type,
    resolve_mode,
)

# ---------------------------------------------------------------------------
# resolve_mode tests
# ---------------------------------------------------------------------------

class TestResolveMode:
    def test_quiet_mode_low_risk(self, mock_pipeline_result):
        mode, sub = resolve_mode(mock_pipeline_result)
        assert mode == OperationalMode.QUIET
        assert sub is None

    def test_active_mode_high_risk(self, active_pipeline_result):
        mode, sub = resolve_mode(active_pipeline_result)
        assert mode == OperationalMode.ACTIVE
        assert sub is None

    def test_emergency_mode_critical(self):
        result = {"risk_level": "CRITICAL", "firms_hotspot_count": 0}
        mode, sub = resolve_mode(result)
        assert mode == OperationalMode.EMERGENCY
        assert sub == EmergencySubState.ACTIVE_FIRE

    def test_emergency_mode_firms_hotspot(self):
        result = {"risk_level": "LOW", "firms_hotspot_count": 5}
        mode, sub = resolve_mode(result)
        assert mode == OperationalMode.EMERGENCY
        assert sub == EmergencySubState.ACTIVE_FIRE

    def test_moderate_risk_no_hotspot(self):
        result = {"risk_level": "MODERATE", "firms_hotspot_count": 0}
        mode, sub = resolve_mode(result)
        assert mode == OperationalMode.ACTIVE
        assert sub is None

    def test_missing_risk_level_raises(self):
        with pytest.raises(ValueError, match="risk_level"):
            resolve_mode({"firms_hotspot_count": 0})

    def test_missing_firms_count_raises(self):
        with pytest.raises(ValueError, match="firms_hotspot_count"):
            resolve_mode({"risk_level": "LOW"})


# ---------------------------------------------------------------------------
# mode_to_report_type tests
# ---------------------------------------------------------------------------

class TestModeToReportType:
    def test_mode_to_report_type_quiet(self):
        assert mode_to_report_type(OperationalMode.QUIET, None) == "daily"

    def test_mode_to_report_type_active(self):
        assert mode_to_report_type(OperationalMode.ACTIVE, None) == "high_risk"

    def test_mode_to_report_type_emergency_active_fire(self):
        assert mode_to_report_type(
            OperationalMode.EMERGENCY, EmergencySubState.ACTIVE_FIRE
        ) == "incident"

    def test_mode_to_report_type_emergency_interim(self):
        assert mode_to_report_type(
            OperationalMode.EMERGENCY, EmergencySubState.INTERIM
        ) == "incident"

    def test_mode_to_report_type_emergency_post_fire(self):
        assert mode_to_report_type(
            OperationalMode.EMERGENCY, EmergencySubState.POST_FIRE
        ) == "incident"

    def test_mode_to_report_type_final(self):
        assert mode_to_report_type(
            OperationalMode.EMERGENCY, EmergencySubState.FINAL
        ) == "final"


# ---------------------------------------------------------------------------
# AdminToggle tests
# ---------------------------------------------------------------------------

class TestAdminToggle:
    def test_admin_toggle_default_on(self, toggle_on):
        assert toggle_on.is_on is True

    def test_admin_toggle_default_off(self, toggle_off):
        assert toggle_off.is_on is False

    def test_admin_toggle_disable(self, toggle_on):
        toggle_on.disable("admin_1")
        assert toggle_on.is_on is False

    def test_admin_toggle_enable(self, toggle_off):
        toggle_off.enable("admin_1")
        assert toggle_off.is_on is True

    def test_admin_toggle_local_persistence(self, tmp_path):
        """Disable, reinstantiate from same config → new instance reads False."""
        import yaml

        cfg_file = tmp_path / "reporting_config.yaml"
        cfg_data = {"admin_toggle": {"default": True, "current_state": True, "persistence": "local"}}
        cfg_file.write_text(yaml.safe_dump(cfg_data))

        toggle = AdminToggle({
            "default": True, "current_state": True,
            "persistence": "local", "_config_path": cfg_file,
        })
        assert toggle.is_on is True

        toggle.disable("test_admin")
        assert toggle.is_on is False

        # Re-read from file
        with open(cfg_file) as fh:
            reloaded = yaml.safe_load(fh)
        assert reloaded["admin_toggle"]["current_state"] is False

        # New instance reads the updated value
        toggle2 = AdminToggle({
            "default": True,
            "current_state": reloaded["admin_toggle"]["current_state"],
            "persistence": "local",
            "_config_path": cfg_file,
        })
        assert toggle2.is_on is False
