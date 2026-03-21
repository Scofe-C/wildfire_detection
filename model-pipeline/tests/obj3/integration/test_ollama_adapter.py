"""Integration tests for Ollama adapter — requires local Ollama server.

Run with: pytest tests/obj3/integration/test_ollama_adapter.py -m integration
Prerequisite: `ollama serve` running, `qwen2.5:14b` pulled.
"""

from __future__ import annotations

import json

import pytest

from src.models.obj3_gemini.context_builder import ContextBundle

pytestmark = pytest.mark.integration


def _ollama_available() -> bool:
    try:
        from src.models.obj3_gemini.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter({"ollama": {"model": "qwen2.5:14b"}})
        return adapter.is_available()
    except Exception:
        return False


@pytest.fixture()
def adapter():
    from src.models.obj3_gemini.adapters.ollama_adapter import OllamaAdapter
    return OllamaAdapter({"ollama": {
        "model": "qwen2.5:14b",
        "base_url": "http://localhost:11434",
        "temperature": 0.0,
        "max_retries": 2,
    }})


@pytest.fixture()
def daily_context_bundle(mock_pipeline_result, toggle_on):
    from src.models.obj3_gemini.context_builder import assemble
    from src.models.obj3_gemini.state_machine import OperationalMode
    return assemble(
        mode=OperationalMode.QUIET,
        sub_state=None,
        pipeline_result=mock_pipeline_result,
        human_inputs=[],
        corpus_ref=None,
        corpus_text=None,
        toggle=toggle_on,
        config={"reporting": {}},
    )


@pytest.fixture()
def incident_context_bundle(emergency_pipeline_result, toggle_on):
    from src.models.obj3_gemini.context_builder import assemble
    from src.models.obj3_gemini.state_machine import EmergencySubState, OperationalMode
    return assemble(
        mode=OperationalMode.EMERGENCY,
        sub_state=EmergencySubState.ACTIVE_FIRE,
        pipeline_result=emergency_pipeline_result,
        human_inputs=[],
        corpus_ref=None,
        corpus_text=None,
        toggle=toggle_on,
        config={"reporting": {}},
    )


@pytest.fixture()
def daily_schema():
    from src.models.obj3_gemini.schemas.daily_schema import DailyReport
    return DailyReport.model_json_schema()


@pytest.fixture()
def incident_schema():
    from src.models.obj3_gemini.schemas.incident_schema import IncidentReport
    return IncidentReport.model_json_schema()


@pytest.mark.integration
class TestOllamaAdapter:
    """All tests require Ollama running locally with qwen2.5:14b pulled."""

    @pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
    def test_is_available_returns_true(self, adapter):
        assert adapter.is_available() is True

    @pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
    def test_generate_daily_report_schema(self, adapter, daily_context_bundle, daily_schema):
        raw = adapter.generate(daily_context_bundle, daily_schema)
        parsed = json.loads(raw)
        assert "report_type" in parsed
        assert "incident_id" in parsed
        assert "summary" in parsed

    @pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
    def test_generate_incident_report_schema(self, adapter, incident_context_bundle, incident_schema):
        raw = adapter.generate(incident_context_bundle, incident_schema)
        parsed = json.loads(raw)
        assert "incident_name" in parsed
        assert "immediate_actions" in parsed

    @pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
    def test_schema_compliance_confidence_field(self, adapter, daily_context_bundle, daily_schema):
        raw = adapter.generate(daily_context_bundle, daily_schema)
        parsed = json.loads(raw)
        assert "report_confidence" in parsed
        conf = parsed["report_confidence"]
        assert isinstance(conf, (int, float))
        assert 0.0 <= conf <= 1.0

    @pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
    def test_disclaimer_field_exact_match(self, adapter, daily_context_bundle, daily_schema):
        raw = adapter.generate(daily_context_bundle, daily_schema)
        parsed = json.loads(raw)
        expected = "AI-generated. Not for operational use without human review."
        assert parsed.get("disclaimer") == expected

    @pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
    def test_human_block_incorporated(self, adapter, mock_pipeline_result, toggle_on, daily_schema):
        from src.models.obj3_gemini.context_builder import HumanInput, assemble
        from src.models.obj3_gemini.state_machine import OperationalMode

        inputs = [HumanInput(
            text_notes="Wind shift expected from the west at 14:00 UTC",
            source="operator",
            submitted_at="2026-03-20T12:30:00Z",
        )]
        bundle = assemble(
            mode=OperationalMode.QUIET,
            sub_state=None,
            pipeline_result=mock_pipeline_result,
            human_inputs=inputs,
            corpus_ref=None,
            corpus_text=None,
            toggle=toggle_on,
            config={"reporting": {}},
        )
        raw = adapter.generate(bundle, daily_schema)
        parsed = json.loads(raw)
        assert parsed.get("human_input_included") is True

    @pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
    def test_temperature_zero_determinism(self, adapter, daily_context_bundle, daily_schema):
        """Same input twice → same risk_level output both times."""
        raw1 = adapter.generate(daily_context_bundle, daily_schema)
        raw2 = adapter.generate(daily_context_bundle, daily_schema)
        p1 = json.loads(raw1)
        p2 = json.loads(raw2)
        assert p1.get("risk_level") == p2.get("risk_level")
