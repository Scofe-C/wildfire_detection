"""Integration tests for Gemini Developer API adapter — requires GEMINI_API_KEY.

Run with: pytest tests/obj3/integration/test_gemini_dev_adapter.py -m integration

Prerequisites:
  - pip install google-generativeai
  - export GEMINI_API_KEY=<your-key-from-ai-studio>
"""

from __future__ import annotations

import json
import os

import pytest

from src.models.obj3_gemini.context_builder import ContextBundle

pytestmark = pytest.mark.integration

_HAS_API_KEY = bool(os.environ.get("GEMINI_API_KEY"))


@pytest.fixture()
def adapter():
    from src.models.obj3_gemini.adapters.gemini_dev_adapter import GeminiDevAdapter
    return GeminiDevAdapter({"gemini_dev": {"model": "gemini-2.5-flash"}})


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
def daily_schema():
    from src.models.obj3_gemini.schemas.daily_schema import DailyReport
    return DailyReport.model_json_schema()


@pytest.mark.integration
class TestGeminiDevAdapter:
    @pytest.mark.skipif(not _HAS_API_KEY, reason="GEMINI_API_KEY not set")
    def test_is_available_returns_true(self, adapter):
        assert adapter.is_available() is True

    @pytest.mark.skipif(not _HAS_API_KEY, reason="GEMINI_API_KEY not set")
    def test_generate_daily_report_schema(self, adapter, daily_context_bundle, daily_schema):
        raw = adapter.generate(daily_context_bundle, daily_schema)
        parsed = json.loads(raw)
        assert "report_type" in parsed
        assert "incident_id" in parsed

    @pytest.mark.skipif(not _HAS_API_KEY, reason="GEMINI_API_KEY not set")
    def test_schema_compliance_confidence_field(self, adapter, daily_context_bundle, daily_schema):
        raw = adapter.generate(daily_context_bundle, daily_schema)
        parsed = json.loads(raw)
        assert "report_confidence" in parsed
        conf = parsed["report_confidence"]
        assert isinstance(conf, (int, float))
        assert 0.0 <= conf <= 1.0

    @pytest.mark.skipif(not _HAS_API_KEY, reason="GEMINI_API_KEY not set")
    def test_disclaimer_field_exact_match(self, adapter, daily_context_bundle, daily_schema):
        raw = adapter.generate(daily_context_bundle, daily_schema)
        parsed = json.loads(raw)
        expected = "AI-generated. Not for operational use without human review."
        assert parsed.get("disclaimer") == expected

    def test_no_api_key_is_not_available(self):
        """Without API key, is_available() should return False."""
        from src.models.obj3_gemini.adapters.gemini_dev_adapter import GeminiDevAdapter
        adapter = GeminiDevAdapter({"gemini_dev": {"model": "gemini-2.5-flash"}})
        # Temporarily unset key
        original = os.environ.pop("GEMINI_API_KEY", None)
        try:
            adapter._api_key = ""
            assert adapter.is_available() is False
        finally:
            if original:
                os.environ["GEMINI_API_KEY"] = original
