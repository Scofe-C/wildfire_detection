"""Shared fixtures for OBJ-3 tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models.obj3_gemini.state_machine import AdminToggle

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # tests/obj3/ → model-pipeline/
TEMPLATE_DIR = PROJECT_ROOT / "templates"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


# ---------------------------------------------------------------------------
# Pipeline result fixtures (A-E) — match mock JSON files
# ---------------------------------------------------------------------------

@pytest.fixture()
def fixture_a_quiet() -> dict:
    """Fixture A: LOW + 0 firms + deployable → QUIET → daily."""
    return {
        "run_id": "test-fixture-a", "is_deployable": True,
        "risk_level": "LOW", "firms_hotspot_count": 0,
        "metrics": {"auc_pr": 0.82, "f1": 0.74, "fnr": 0.12},
        "xgboost_top_cells": [], "cell2fire_geojson": None,
        "propagator_summary": None,
        "telemetry": {"temperature_max": 28.5, "wind_speed_mph": 12.0,
                      "relative_humidity": 58.0, "soil_moisture": 0.22},
        "fema_nri_tracts": [],
        "bias_report": {"gate_result": "PASS", "observed_disparity": 0.02},
    }


@pytest.fixture()
def fixture_b_active() -> dict:
    """Fixture B: HIGH + 0 firms + deployable → ACTIVE → high_risk."""
    return {
        "run_id": "test-fixture-b", "is_deployable": True,
        "risk_level": "HIGH", "firms_hotspot_count": 0,
        "metrics": {"auc_pr": 0.78, "f1": 0.70, "fnr": 0.15},
        "xgboost_top_cells": [
            {"h3_index": "8928308280fffff", "probability": 0.81, "lat": 34.1, "lon": -118.3},
        ],
        "cell2fire_geojson": None, "propagator_summary": None,
        "telemetry": {"temperature_max": 38.0, "wind_speed_mph": 25.0,
                      "relative_humidity": 15.0, "soil_moisture": 0.08},
        "fema_nri_tracts": [{"tract_id": "06037701000", "nri_score": 42.5}],
        "bias_report": {"gate_result": "PASS", "observed_disparity": 0.04},
    }


@pytest.fixture()
def fixture_c_emergency() -> dict:
    """Fixture C: HIGH + 12 firms + deployable → EMERGENCY → incident."""
    return {
        "run_id": "test-fixture-c", "is_deployable": True,
        "risk_level": "HIGH", "firms_hotspot_count": 12,
        "metrics": {"auc_pr": 0.85, "f1": 0.76, "fnr": 0.10},
        "xgboost_top_cells": [
            {"h3_index": "8928308280fffff", "probability": 0.92, "lat": 34.1, "lon": -118.3},
            {"h3_index": "8928308281fffff", "probability": 0.88, "lat": 34.2, "lon": -118.4},
        ],
        "cell2fire_geojson": '{"type": "FeatureCollection", "features": []}',
        "propagator_summary": "Fire spreading NE at 2.3 mph, 150 acres estimated.",
        "telemetry": {"temperature_max": 41.0, "wind_speed_mph": 35.0,
                      "relative_humidity": 8.0, "soil_moisture": 0.04},
        "fema_nri_tracts": [{"tract_id": "06037701000", "nri_score": 78.2}],
        "bias_report": {"gate_result": "PASS", "observed_disparity": 0.03},
    }


@pytest.fixture()
def fixture_d_non_deployable() -> dict:
    """Fixture D: LOW + 0 firms + NOT deployable → QUIET → daily."""
    return {
        "run_id": "test-fixture-d", "is_deployable": False,
        "risk_level": "LOW", "firms_hotspot_count": 0,
        "metrics": {"auc_pr": 0.55, "f1": 0.48, "fnr": 0.35},
        "xgboost_top_cells": [],
        "cell2fire_geojson": None, "propagator_summary": None,
        "telemetry": {"temperature_max": 22.0, "wind_speed_mph": 5.0,
                      "relative_humidity": 65.0, "soil_moisture": 0.30},
        "fema_nri_tracts": [],
        "bias_report": {"gate_result": "FAIL", "observed_disparity": 0.18},
    }


@pytest.fixture()
def fixture_e_disagreement() -> dict:
    """Fixture E: LOW + 7 firms + deployable → ACTIVE + disagreement → high_risk.
    MOST CRITICAL SAFETY TEST. review_status is guaranteed PENDING_REVIEW
    because disagreement_flag fires before the LLM is called."""
    return {
        "run_id": "test-fixture-e", "is_deployable": True,
        "risk_level": "LOW", "firms_hotspot_count": 7,
        "metrics": {"auc_pr": 0.80, "f1": 0.72, "fnr": 0.13},
        "xgboost_top_cells": [
            {"h3_index": "8928308280fffff", "probability": 0.15, "lat": 37.4, "lon": -119.5},
        ],
        "cell2fire_geojson": None,
        "propagator_summary": None,
        "telemetry": {"temperature_max": 26.0, "wind_speed_mph": 8.0,
                      "relative_humidity": 55.0, "soil_moisture": 0.20},
        "fema_nri_tracts": [],
        "bias_report": {"gate_result": "PASS", "observed_disparity": 0.02},
    }


# Backward-compat aliases for existing tests that reference old fixture names
@pytest.fixture()
def mock_pipeline_result(fixture_a_quiet) -> dict:
    """Alias for fixture_a_quiet (backward compat)."""
    return fixture_a_quiet


@pytest.fixture()
def active_pipeline_result(fixture_b_active) -> dict:
    """Alias for fixture_b_active (backward compat)."""
    return fixture_b_active


@pytest.fixture()
def emergency_pipeline_result(fixture_c_emergency) -> dict:
    """Alias for fixture_c_emergency (backward compat)."""
    return fixture_c_emergency


# ---------------------------------------------------------------------------
# Report JSON fixtures (pre-built valid report data)
# ---------------------------------------------------------------------------

@pytest.fixture()
def valid_daily_json() -> dict:
    """Minimal valid DailyReport JSON dict."""
    return {
        "incident_id": "550e8400-e29b-41d4-a716-446655440000",
        "report_type": "daily",
        "report_confidence": 0.92,
        "generated_at": "2026-03-20T06:00:00Z",
        "operating_mode": "QUIET",
        "risk_level": "LOW",
        "human_review_required": False,
        "human_input_included": False,
        "disclaimer": "AI-generated. Not for operational use without human review.",
        "data_sources_used": ["xgboost_scores", "owm_telemetry", "firms_hotspots"],
        "summary": "All monitored areas remain at low risk.",
        "monitored_area_count": 847,
        "highest_risk_cell": None,
        "weather_summary": "Relative humidity 65%, wind 8 mph NW.",
        "notable_changes": [],
        "next_check_recommendation": "Routine check in 24 hours.",
    }


@pytest.fixture()
def valid_incident_json() -> dict:
    """Minimal valid IncidentReport JSON dict."""
    return {
        "incident_id": "660f9500-f30c-52e5-b827-557766550001",
        "report_type": "incident",
        "report_confidence": 0.85,
        "generated_at": "2026-03-20T16:00:00Z",
        "operating_mode": "EMERGENCY",
        "risk_level": "CRITICAL",
        "human_review_required": False,
        "human_input_included": True,
        "disclaimer": "AI-generated. Not for operational use without human review.",
        "data_sources_used": ["xgboost_scores", "cell2fire", "firms_hotspots", "fema_nri"],
        "incident_name": "Oak Ridge Fire",
        "incident_status": "ACTIVE",
        "affected_communities": ["Cedar Valley", "Pine Ridge"],
        "spread_summary": "Fire spreading northeast driven by 25 mph winds.",
        "estimated_area_acres": 120.5,
        "resource_requirements": [
            {"resource_type": "Type 1 Engine", "quantity": 3, "ics_type": "E-1", "notes": None}
        ],
        "projected_losses": {
            "structures_at_risk": 45,
            "population_at_risk": 1200,
            "infrastructure_notes": "Highway 49 may need closure.",
            "estimated_cost_usd": 5000000.0,
        },
        "vulnerable_populations": [
            {
                "group_name": "Elderly residents",
                "location": "Cedar Valley Senior Center",
                "vulnerability_score": 0.78,
                "notes": "Limited mobility, requires evacuation assistance.",
            }
        ],
        "immediate_actions": [
            "Evacuate Zone A within 2 hours",
            "Deploy Type 1 engines to northern perimeter",
            "Activate emergency shelters at Pine Ridge Community Center",
        ],
        "operator_notes_incorporated": "Wind shift noted by operator at 14:00.",
        "ics_form_references": ["ICS-209", "ICS-204"],
    }


@pytest.fixture()
def valid_final_json() -> dict:
    """Minimal valid FinalReport JSON dict."""
    return {
        "incident_id": "770fa600-a41d-63f6-c938-668877660002",
        "report_type": "final",
        "report_confidence": 0.88,
        "generated_at": "2026-03-21T09:00:00Z",
        "operating_mode": "EMERGENCY",
        "risk_level": "HIGH",
        "human_review_required": True,
        "human_input_included": True,
        "disclaimer": "AI-generated. Not for operational use without human review.",
        "data_sources_used": ["xgboost_scores", "cell2fire", "firms_hotspots"],
        "incident_name": "Oak Ridge Fire",
        "linked_incident_id": "660f9500-f30c-52e5-b827-557766550001",
        "incident_timeline": [
            {"timestamp": "2026-03-20T14:00:00Z", "event": "Fire detected by FIRMS", "source": "satellite"},
            {"timestamp": "2026-03-20T16:00:00Z", "event": "Evacuation ordered for Zone A", "source": "IC"},
            {"timestamp": "2026-03-21T06:00:00Z", "event": "Fire contained", "source": "ground crews"},
        ],
        "total_area_burned_acres": 200.0,
        "resources_deployed": [
            {"resource_type": "Type 1 Engine", "quantity": 3, "deployed_at": "2026-03-20T15:00:00Z", "notes": None}
        ],
        "losses_summary": {
            "structures_at_risk": 2,
            "population_at_risk": 0,
            "infrastructure_notes": "Highway 49 reopened.",
            "estimated_cost_usd": 1500000.0,
        },
        "response_effectiveness": "Rapid deployment limited damage. Evacuation was orderly.",
        "lessons_learned": [
            "Pre-positioning of engines significantly reduced response time.",
            "Community alert system worked effectively.",
        ],
        "recommendations_for_future": [
            "Install additional weather stations in Cedar Valley.",
            "Update evacuation routes for Pine Ridge.",
        ],
        "attachments_referenced": None,
    }


@pytest.fixture()
def valid_high_risk_json() -> dict:
    """Minimal valid HighRiskReport JSON dict."""
    return {
        "incident_id": "880fb700-b52e-74g7-da49-779988770003",
        "report_type": "high_risk",
        "report_confidence": 0.81,
        "generated_at": "2026-03-20T14:30:00Z",
        "operating_mode": "ACTIVE",
        "risk_level": "HIGH",
        "human_review_required": False,
        "human_input_included": False,
        "disclaimer": "AI-generated. Not for operational use without human review.",
        "data_sources_used": ["xgboost_scores", "owm_telemetry"],
        "risk_summary": "High risk due to low soil moisture and strong winds.",
        "top_risk_cells": [
            {"h3_index": "8928308280fffff", "risk_score": 0.87, "lat": 37.4, "lon": -119.5}
        ],
        "contributing_factors": ["Low soil moisture", "Wind > 30 mph", "High temperature"],
        "preventive_recommendations": [
            {
                "title": "Pre-position crews",
                "description": "Deploy Type 2 crews to high-risk sectors.",
                "priority": "HIGH",
            },
            {
                "title": "Issue public advisory",
                "description": "Alert residents in Cedar Valley of elevated risk.",
                "priority": "MEDIUM",
            },
        ],
        "vulnerable_populations": None,
        "escalation_trigger": "FIRMS hotspot detected OR wind exceeds 40 mph.",
    }


# ---------------------------------------------------------------------------
# Toggle fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def toggle_on() -> AdminToggle:
    """AdminToggle in ON state (local persistence, no file writes)."""
    return AdminToggle({"default": True, "current_state": True, "persistence": "local"})


@pytest.fixture()
def toggle_off() -> AdminToggle:
    """AdminToggle in OFF state."""
    return AdminToggle({"default": False, "current_state": False, "persistence": "local"})
