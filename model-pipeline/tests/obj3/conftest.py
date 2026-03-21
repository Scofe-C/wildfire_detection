"""Shared fixtures for OBJ-3 tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "templates"
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


# ---------------------------------------------------------------------------
# Pipeline result fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_pipeline_result() -> dict:
    """Load the standard QUIET-mode pipeline result fixture."""
    with open(FIXTURE_DIR / "mock_pipeline_result.json") as fh:
        return json.load(fh)


@pytest.fixture()
def emergency_pipeline_result(mock_pipeline_result: dict) -> dict:
    """Pipeline result that triggers EMERGENCY mode."""
    result = dict(mock_pipeline_result)
    result["risk_level"] = "CRITICAL"
    result["firms_hotspot_count"] = 5
    result["xgboost_top_cells"] = [
        {"h3_index": "8928308280fffff", "probability": 0.91, "lat": 37.4, "lon": -119.5},
        {"h3_index": "8928308281fffff", "probability": 0.87, "lat": 37.5, "lon": -119.6},
    ]
    return result


@pytest.fixture()
def active_pipeline_result(mock_pipeline_result: dict) -> dict:
    """Pipeline result that triggers ACTIVE mode."""
    result = dict(mock_pipeline_result)
    result["risk_level"] = "HIGH"
    result["firms_hotspot_count"] = 0
    return result


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
def toggle_on() -> "AdminToggle":
    """AdminToggle in ON state (local persistence, no file writes)."""
    from src.models.obj3_gemini.state_machine import AdminToggle
    return AdminToggle({"default": True, "current_state": True, "persistence": "local"})


@pytest.fixture()
def toggle_off() -> "AdminToggle":
    """AdminToggle in OFF state."""
    from src.models.obj3_gemini.state_machine import AdminToggle
    return AdminToggle({"default": False, "current_state": False, "persistence": "local"})
