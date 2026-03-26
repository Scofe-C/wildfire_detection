"""Unit tests for src/models/obj3_gemini/reporter.py.

Tests focus on:
  - ReportResult / ValidationResult / GeneratedReport dataclasses
  - _compute_human_review_required (3-trigger OR logic)
  - GeminiDisasterReporter._check_sections
  - GeminiDisasterReporter._create_adapter
  - GeminiDisasterReporter._append_review_manifest
  - GeminiDisasterReporter.predict (mocked adapter)

No real LLM calls made — all adapters are mocked.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.models.obj3_gemini.reporter import (
    GeminiDisasterReporter,
    GeneratedReport,
    ReportResult,
    ValidationResult,
    _compute_human_review_required,
)
from src.models.obj3_gemini.schemas.base_schema import REQUIRED_DISCLAIMER
from src.models.obj3_gemini.schemas.daily_schema import DailyReport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daily_report(**kwargs: Any) -> DailyReport:
    defaults = dict(
        # BaseReport required fields
        incident_id="INC-001",
        report_type="daily",
        generated_at=datetime.now(UTC).isoformat(),
        operating_mode="QUIET",
        human_input_included=False,
        data_sources_used=["firms", "weather"],
        # DailyReport fields
        summary="Fire risk summary.",
        risk_level="LOW",
        monitored_area_count=5,
        notable_changes=[],
        report_confidence=0.85,
        grounding_search_count=4,
        human_review_required=False,
        disclaimer=REQUIRED_DISCLAIMER,
    )
    defaults.update(kwargs)
    return DailyReport(**defaults)


def _make_report_result(**kwargs: Any) -> ReportResult:
    defaults = dict(
        raw_json='{"summary": "test"}',
        parsed_report=None,
        report_type="daily",
        incident_id="INC-001",
    )
    defaults.update(kwargs)
    return ReportResult(**defaults)


# ===========================================================================
# Dataclass smoke tests
# ===========================================================================

class TestReportResult:
    def test_defaults(self) -> None:
        r = _make_report_result()
        assert r.error is None
        assert r.latency_ms == 0.0

    def test_with_error(self) -> None:
        r = _make_report_result(error="LLM timeout", latency_ms=1500.0)
        assert r.error == "LLM timeout"
        assert r.latency_ms == 1500.0


class TestValidationResult:
    def test_passed_all_true(self) -> None:
        v = ValidationResult(
            schema_valid=True,
            sections_complete=True,
            confidence_ok=True,
            review_flag_correct=True,
        )
        assert v.passed is True

    def test_passed_any_false(self) -> None:
        v = ValidationResult(
            schema_valid=True,
            sections_complete=False,
            confidence_ok=True,
            review_flag_correct=True,
        )
        assert v.passed is False

    def test_defaults_all_false(self) -> None:
        assert ValidationResult().passed is False


class TestGeneratedReport:
    def test_defaults(self) -> None:
        result = _make_report_result()
        validation = ValidationResult()
        g = GeneratedReport(report_result=result, validation=validation)
        assert g.markdown_path is None
        assert g.html_path is None
        assert g.json_path is None
        assert g.gcs_paths == []
        assert g.explain_output == {}


# ===========================================================================
# _compute_human_review_required
# ===========================================================================

class TestComputeHumanReviewRequired:
    def _report(self, confidence: float = 0.85, grounding: int = 4) -> DailyReport:
        return _make_daily_report(
            report_confidence=confidence,
            grounding_search_count=grounding,
        )

    def test_no_triggers_returns_false(self) -> None:
        report = self._report(confidence=0.85, grounding=4)
        assert _compute_human_review_required(report, False, {}) is False

    def test_low_confidence_triggers(self) -> None:
        report = self._report(confidence=0.50)
        assert _compute_human_review_required(report, False, {}) is True

    def test_low_grounding_triggers(self) -> None:
        report = self._report(grounding=1)
        assert _compute_human_review_required(report, False, {}) is True

    def test_disagreement_flag_triggers(self) -> None:
        report = self._report(confidence=0.85, grounding=4)
        assert _compute_human_review_required(report, True, {}) is True

    def test_all_three_triggers(self) -> None:
        report = self._report(confidence=0.3, grounding=0)
        assert _compute_human_review_required(report, True, {}) is True

    def test_custom_threshold_from_config(self) -> None:
        config = {"reporting": {"confidence_threshold": 0.95, "min_grounding_sources": 10}}
        report = self._report(confidence=0.80, grounding=8)
        # Both below custom thresholds
        assert _compute_human_review_required(report, False, config) is True

    def test_exactly_at_threshold_passes(self) -> None:
        # confidence == threshold → NOT strictly less than, so should pass
        config = {"reporting": {"confidence_threshold": 0.70}}
        report = self._report(confidence=0.70, grounding=5)
        assert _compute_human_review_required(report, False, config) is False


# ===========================================================================
# GeminiDisasterReporter._check_sections
# ===========================================================================

class TestCheckSections:
    def test_passes_for_valid_report(self) -> None:
        report = _make_daily_report()
        assert GeminiDisasterReporter._check_sections(report) is True

    def test_fails_when_required_str_is_empty(self) -> None:
        report = _make_daily_report(summary="")
        assert GeminiDisasterReporter._check_sections(report) is False


# ===========================================================================
# GeminiDisasterReporter._create_adapter
# ===========================================================================

class TestCreateAdapter:
    def _reporter(self) -> GeminiDisasterReporter:
        r = GeminiDisasterReporter()
        r._config = {}
        return r

    def test_creates_ollama_adapter(self) -> None:
        from src.models.obj3_gemini.adapters.ollama_adapter import OllamaAdapter
        reporter = self._reporter()
        adapter = reporter._create_adapter("ollama")
        assert isinstance(adapter, OllamaAdapter)

    def test_creates_gemini_dev_adapter(self) -> None:
        from src.models.obj3_gemini.adapters.gemini_dev_adapter import GeminiDevAdapter
        reporter = self._reporter()
        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake"}):
            adapter = reporter._create_adapter("gemini_dev")
        assert isinstance(adapter, GeminiDevAdapter)

    def test_raises_on_unknown_backend(self) -> None:
        reporter = self._reporter()
        with pytest.raises(ValueError, match="Unknown LLM backend"):
            reporter._create_adapter("nonexistent_backend")


# ===========================================================================
# GeminiDisasterReporter._append_review_manifest
# ===========================================================================

class TestAppendReviewManifest:
    def test_creates_manifest_on_first_call(self, tmp_path: Path) -> None:
        reporter = GeminiDisasterReporter()
        reporter._output_dir = tmp_path

        reporter._append_review_manifest(
            incident_id="INC-001",
            report_type="daily",
            json_path=tmp_path / "report.json",
            rendered_path=None,
            disagreement_flag=False,
            confidence=0.85,
            grounding_count=4,
            generated_at="2026-01-01T00:00:00Z",
        )

        manifest_path = tmp_path / "review_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert len(manifest) == 1
        assert manifest[0]["incident_id"] == "INC-001"

    def test_appends_to_existing_manifest(self, tmp_path: Path) -> None:
        reporter = GeminiDisasterReporter()
        reporter._output_dir = tmp_path

        # Pre-populate
        manifest_path = tmp_path / "review_manifest.json"
        manifest_path.write_text(json.dumps([{"incident_id": "INC-000"}]))

        reporter._append_review_manifest(
            incident_id="INC-001",
            report_type="daily",
            json_path=None,
            rendered_path=None,
            disagreement_flag=False,
            confidence=0.9,
            grounding_count=5,
            generated_at="2026-01-01T00:00:00Z",
        )

        manifest = json.loads(manifest_path.read_text())
        assert len(manifest) == 2
        assert manifest[1]["incident_id"] == "INC-001"

    def test_paths_serialized_as_strings(self, tmp_path: Path) -> None:
        reporter = GeminiDisasterReporter()
        reporter._output_dir = tmp_path

        reporter._append_review_manifest(
            incident_id="INC-001",
            report_type="daily",
            json_path=tmp_path / "report.json",
            rendered_path=tmp_path / "report.md",
            disagreement_flag=False,
            confidence=0.9,
            grounding_count=5,
            generated_at="2026-01-01T00:00:00Z",
        )

        manifest = json.loads((tmp_path / "review_manifest.json").read_text())
        # Paths must be strings, not Path objects
        assert isinstance(manifest[0]["json_path"], str)
        assert isinstance(manifest[0]["rendered_path"], str)

    def test_does_not_crash_on_write_error(self, tmp_path: Path) -> None:
        reporter = GeminiDisasterReporter()
        reporter._output_dir = tmp_path / "nonexistent_deep_dir" / "sub"

        # Should not raise — best-effort, silently logs
        reporter._append_review_manifest(
            incident_id="INC-001",
            report_type="daily",
            json_path=None,
            rendered_path=None,
            disagreement_flag=False,
            confidence=0.9,
            grounding_count=5,
            generated_at="2026-01-01T00:00:00Z",
        )


# ===========================================================================
# GeminiDisasterReporter.predict — mocked adapter
# ===========================================================================

class TestReporterPredict:
    def _reporter_with_mock_adapter(self, raw_json: str) -> GeminiDisasterReporter:
        reporter = GeminiDisasterReporter()
        mock_adapter = MagicMock()
        mock_adapter.generate.return_value = raw_json
        reporter._adapter = mock_adapter
        reporter._config = {}
        return reporter

    def _valid_daily_json(self) -> str:
        from datetime import UTC, datetime
        return json.dumps({
            "incident_id": "INC-001",
            "report_type": "daily",
            "generated_at": datetime.now(UTC).isoformat(),
            "operating_mode": "QUIET",
            "human_input_included": False,
            "data_sources_used": ["firms", "weather"],
            "summary": "Low risk conditions across monitored area.",
            "risk_level": "LOW",
            "monitored_area_count": 3,
            "notable_changes": [],
            "report_confidence": 0.9,
            "grounding_search_count": 5,
            "human_review_required": False,
            "disclaimer": REQUIRED_DISCLAIMER,
        })

    def _make_bundle(self) -> Any:
        from src.models.obj3_gemini.context_builder import ContextBundle
        return ContextBundle(
            system_prompt="sys",
            corpus_text="corpus",
            corpus_ref=None,
            ml_block="ml",
            data_block="data",
            human_block="",
            instruction="generate",
            report_type="daily",
            incident_id="INC-001",
        )

    def test_returns_report_result_on_success(self) -> None:
        reporter = self._reporter_with_mock_adapter(self._valid_daily_json())
        bundle = self._make_bundle()
        result = reporter.predict(bundle)
        assert result.error is None
        assert result.parsed_report is not None
        assert result.report_type == "daily"
        assert result.incident_id == "INC-001"

    def test_returns_error_result_on_llm_failure(self) -> None:
        from src.models.obj3_gemini.adapters.base_adapter import LLMGenerationError
        reporter = GeminiDisasterReporter()
        mock_adapter = MagicMock()
        mock_adapter.generate.side_effect = LLMGenerationError("timeout")
        reporter._adapter = mock_adapter
        reporter._config = {}

        bundle = self._make_bundle()
        result = reporter.predict(bundle)
        assert result.error is not None
        assert "timeout" in result.error
        assert result.parsed_report is None
        assert result.raw_json == ""

    def test_raises_when_adapter_not_loaded(self) -> None:
        reporter = GeminiDisasterReporter()
        bundle = self._make_bundle()
        with pytest.raises(RuntimeError, match="load_model"):
            reporter.predict(bundle)

    def test_latency_is_positive(self) -> None:
        reporter = self._reporter_with_mock_adapter(self._valid_daily_json())
        bundle = self._make_bundle()
        result = reporter.predict(bundle)
        assert result.latency_ms >= 0
