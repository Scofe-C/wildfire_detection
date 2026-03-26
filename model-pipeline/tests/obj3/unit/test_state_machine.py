"""Unit tests for state_machine.py — 9-cell routing matrix + is_deployable gate."""

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
# resolve_mode tests — 9-cell matrix + is_deployable gate
# ---------------------------------------------------------------------------

class TestResolveMode:
    """Tests for the 9-cell routing matrix + is_deployable gate."""

    # --- Row 1: LOW/MODERATE + firms=0 → QUIET ---
    def test_low_risk_no_firms_quiet(self):
        r = {"risk_level": "LOW", "firms_hotspot_count": 0, "is_deployable": True}
        mode, sub, flag = resolve_mode(r)
        assert mode == OperationalMode.QUIET
        assert sub is None
        assert flag is False

    def test_moderate_risk_no_firms_quiet(self):
        r = {"risk_level": "MODERATE", "firms_hotspot_count": 0, "is_deployable": True}
        mode, sub, flag = resolve_mode(r)
        assert mode == OperationalMode.QUIET
        assert sub is None
        assert flag is False

    # --- Row 2: HIGH/CRITICAL + firms=0 → ACTIVE ---
    def test_high_risk_no_firms_active(self):
        r = {"risk_level": "HIGH", "firms_hotspot_count": 0, "is_deployable": True}
        mode, sub, flag = resolve_mode(r)
        assert mode == OperationalMode.ACTIVE
        assert sub is None
        assert flag is False

    def test_critical_risk_no_firms_active(self):
        r = {"risk_level": "CRITICAL", "firms_hotspot_count": 0, "is_deployable": True}
        mode, sub, flag = resolve_mode(r)
        assert mode == OperationalMode.ACTIVE
        assert sub is None
        assert flag is False

    # --- Row 3: LOW/MODERATE + firms>0 → ACTIVE + disagreement ---
    def test_low_risk_with_firms_active_disagreement(self):
        r = {"risk_level": "LOW", "firms_hotspot_count": 7, "is_deployable": True}
        mode, sub, flag = resolve_mode(r)
        assert mode == OperationalMode.ACTIVE
        assert sub is None
        assert flag is True   # ← MODEL DISAGREEMENT

    def test_moderate_risk_with_firms_active_disagreement(self):
        r = {"risk_level": "MODERATE", "firms_hotspot_count": 3, "is_deployable": True}
        mode, sub, flag = resolve_mode(r)
        assert mode == OperationalMode.ACTIVE
        assert sub is None
        assert flag is True

    # --- Row 4: HIGH/CRITICAL + firms>0 → EMERGENCY ---
    def test_high_risk_with_firms_emergency(self):
        r = {"risk_level": "HIGH", "firms_hotspot_count": 12, "is_deployable": True}
        mode, sub, flag = resolve_mode(r)
        assert mode == OperationalMode.EMERGENCY
        assert sub == EmergencySubState.ACTIVE_FIRE
        assert flag is False

    def test_critical_risk_with_firms_emergency(self):
        r = {"risk_level": "CRITICAL", "firms_hotspot_count": 5, "is_deployable": True}
        mode, sub, flag = resolve_mode(r)
        assert mode == OperationalMode.EMERGENCY
        assert sub == EmergencySubState.ACTIVE_FIRE
        assert flag is False

    # --- Row 5: is_deployable=False → always QUIET ---
    def test_non_deployable_low_no_firms_quiet(self):
        r = {"risk_level": "LOW", "firms_hotspot_count": 0, "is_deployable": False}
        mode, sub, flag = resolve_mode(r)
        assert mode == OperationalMode.QUIET
        assert flag is False

    def test_non_deployable_high_with_firms_still_quiet(self):
        """Even HIGH + firms>0 → QUIET when model is not deployable."""
        r = {"risk_level": "HIGH", "firms_hotspot_count": 12, "is_deployable": False}
        mode, sub, flag = resolve_mode(r)
        assert mode == OperationalMode.QUIET
        assert flag is False

    def test_non_deployable_critical_with_firms_still_quiet(self):
        r = {"risk_level": "CRITICAL", "firms_hotspot_count": 5, "is_deployable": False}
        mode, sub, flag = resolve_mode(r)
        assert mode == OperationalMode.QUIET
        assert flag is False

    # --- Disagreement safety test (Fixture E) ---
    def test_fixture_e_disagreement_guaranteed_pending(self):
        """Fixture E: LOW + 7 firms + deployable → ACTIVE + disagreement.
        review_status is guaranteed PENDING_REVIEW regardless of LLM output
        because disagreement_flag fires trigger #3."""
        r = {"risk_level": "LOW", "firms_hotspot_count": 7, "is_deployable": True}
        mode, sub, flag = resolve_mode(r)
        assert mode == OperationalMode.ACTIVE
        assert flag is True  # This alone guarantees PENDING_REVIEW in reporter.py

    # --- Error cases ---
    def test_missing_risk_level_raises(self):
        with pytest.raises(ValueError, match="risk_level"):
            resolve_mode({"firms_hotspot_count": 0})

    def test_missing_firms_count_raises(self):
        with pytest.raises(ValueError, match="firms_hotspot_count"):
            resolve_mode({"risk_level": "LOW"})

    def test_unknown_risk_level_raises(self):
        r = {"risk_level": "EXTREME", "firms_hotspot_count": 0}
        with pytest.raises(ValueError, match="Unknown risk_level"):
            resolve_mode(r)


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
        with open(cfg_file, encoding="utf-8") as fh:
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
