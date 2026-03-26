"""Unit tests for Pydantic schemas — §5.1 test_schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.obj3_gemini.schemas import SCHEMA_MAP
from src.models.obj3_gemini.schemas.daily_schema import DailyReport
from src.models.obj3_gemini.schemas.final_schema import FinalReport
from src.models.obj3_gemini.schemas.high_risk_schema import HighRiskReport
from src.models.obj3_gemini.schemas.incident_schema import IncidentReport


class TestDailyReport:
    def test_daily_report_valid(self, valid_daily_json):
        report = DailyReport(**valid_daily_json)
        assert report.report_type == "daily"
        assert report.incident_id is not None

    def test_daily_report_missing_required(self, valid_daily_json):
        del valid_daily_json["incident_id"]
        with pytest.raises(ValidationError):
            DailyReport(**valid_daily_json)

    def test_daily_report_missing_summary(self, valid_daily_json):
        del valid_daily_json["summary"]
        with pytest.raises(ValidationError):
            DailyReport(**valid_daily_json)


class TestIncidentReport:
    def test_incident_report_valid(self, valid_incident_json):
        report = IncidentReport(**valid_incident_json)
        assert report.report_type == "incident"
        assert report.incident_name == "Oak Ridge Fire"

    def test_incident_report_disclaimer_enforced(self, valid_incident_json):
        valid_incident_json["disclaimer"] = "Wrong disclaimer"
        with pytest.raises(ValidationError, match="Disclaimer must be exactly"):
            IncidentReport(**valid_incident_json)

    def test_incident_min_actions(self, valid_incident_json):
        valid_incident_json["immediate_actions"] = ["only one"]
        with pytest.raises(ValidationError):
            IncidentReport(**valid_incident_json)


class TestFinalReport:
    def test_final_report_valid(self, valid_final_json):
        report = FinalReport(**valid_final_json)
        assert report.report_type == "final"
        assert report.linked_incident_id is not None

    def test_final_min_timeline(self, valid_final_json):
        valid_final_json["incident_timeline"] = [
            {"timestamp": "t1", "event": "e1"},
        ]
        with pytest.raises(ValidationError):
            FinalReport(**valid_final_json)

    def test_final_min_lessons(self, valid_final_json):
        valid_final_json["lessons_learned"] = ["only one"]
        with pytest.raises(ValidationError):
            FinalReport(**valid_final_json)


class TestHighRiskReport:
    def test_high_risk_valid(self, valid_high_risk_json):
        report = HighRiskReport(**valid_high_risk_json)
        assert report.report_type == "high_risk"

    def test_high_risk_min_recommendations(self, valid_high_risk_json):
        valid_high_risk_json["preventive_recommendations"] = [
            {"title": "One", "description": "Only one", "priority": "HIGH"}
        ]
        with pytest.raises(ValidationError):
            HighRiskReport(**valid_high_risk_json)


class TestBaseFieldValidation:
    def test_confidence_bounds(self, valid_daily_json):
        valid_daily_json["report_confidence"] = 1.5
        with pytest.raises(ValidationError):
            DailyReport(**valid_daily_json)

    def test_confidence_negative(self, valid_daily_json):
        valid_daily_json["report_confidence"] = -0.1
        with pytest.raises(ValidationError):
            DailyReport(**valid_daily_json)

    def test_risk_level_enum(self, valid_daily_json):
        valid_daily_json["risk_level"] = "EXTREME"
        with pytest.raises(ValidationError):
            DailyReport(**valid_daily_json)

    def test_nested_resource_requirement(self, valid_incident_json):
        report = IncidentReport(**valid_incident_json)
        req = report.resource_requirements[0]
        assert req.resource_type == "Type 1 Engine"
        assert req.quantity == 3

    def test_all_schema_to_json_schema(self):
        """All 4 schema classes must produce valid JSON schema dicts."""
        for name, cls in SCHEMA_MAP.items():
            schema = cls.model_json_schema()
            assert isinstance(schema, dict), f"{name} schema is not a dict"
            assert "properties" in schema, f"{name} schema missing 'properties'"
