"""Unit tests for context_builder.py — §5.1 test_context_builder."""

from __future__ import annotations

from src.models.obj3_gemini.context_builder import (
    ContextBundle,
    HumanInput,
    assemble,
    build_data_block,
    build_human_block,
    build_instruction,
    build_ml_block,
    build_system_prompt,
)
from src.models.obj3_gemini.state_machine import OperationalMode


class TestBuildHumanBlock:
    def test_human_block_empty_when_toggle_off(self, toggle_off):
        inputs = [HumanInput(text_notes="Important note", source="operator", submitted_at="now")]
        result = build_human_block(inputs, toggle_off)
        assert result == ""

    def test_human_block_present_when_toggle_on(self, toggle_on):
        inputs = [HumanInput(text_notes="Wind shift expected", source="operator", submitted_at="2026-03-20T12:00Z")]
        result = build_human_block(inputs, toggle_on)
        assert "Wind shift expected" in result
        assert "OPERATOR" in result

    def test_human_block_no_inputs(self, toggle_on):
        result = build_human_block([], toggle_on)
        assert "No operator input provided" in result


class TestBuildMlBlock:
    def test_ml_block_contains_cells(self, mock_pipeline_result):
        block = build_ml_block(mock_pipeline_result)
        assert "8928308280fffff" in block

    def test_ml_block_truncation(self, mock_pipeline_result):
        block = build_ml_block(mock_pipeline_result, max_chars=50)
        assert len(block) <= 50


class TestBuildDataBlock:
    def test_data_block_contains_telemetry(self, mock_pipeline_result):
        block = build_data_block(mock_pipeline_result)
        assert "temperature_max" in block
        assert "28.5" in block

    def test_data_block_firms_count(self, mock_pipeline_result):
        block = build_data_block(mock_pipeline_result)
        assert "FIRMS Hotspots" in block


class TestBuildInstruction:
    def test_instruction_contains_incident_id(self):
        result = build_instruction("incident", "test-123", "2026-03-20T16:00Z")
        assert "test-123" in result
        assert "incident" in result

    def test_instruction_contains_datetime(self):
        result = build_instruction("daily", "id-1", "2026-03-20T06:00Z")
        assert "2026-03-20T06:00Z" in result


class TestBuildSystemPrompt:
    def test_system_prompt_contains_schema(self):
        schema = {"properties": {"incident_name": {"type": "string"}}}
        prompt = build_system_prompt("incident", schema)
        assert "incident_name" in prompt
        assert "incident" in prompt
        assert "RULES" in prompt

    def test_system_prompt_contains_disclaimer(self):
        prompt = build_system_prompt("daily", {})
        assert "AI-generated" in prompt


class TestAssemble:
    def test_assemble_returns_context_bundle(self, mock_pipeline_result, toggle_on):
        config = {"reporting": {"max_ml_block_chars": 20000, "max_data_block_chars": 20000}}
        bundle = assemble(
            mode=OperationalMode.QUIET,
            sub_state=None,
            pipeline_result=mock_pipeline_result,
            human_inputs=[],
            corpus_ref=None,
            corpus_text="test corpus",
            toggle=toggle_on,
            config=config,
        )
        assert isinstance(bundle, ContextBundle)
        assert bundle.report_type == "daily"
        assert bundle.corpus_text == "test corpus"
        assert bundle.system_prompt  # non-empty
        assert bundle.ml_block  # non-empty
        assert bundle.instruction  # non-empty
