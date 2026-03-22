"""Integration tests for full OBJ-3 pipeline — requires LLM backend running.

Run with: pytest tests/obj3/integration/test_full_pipeline.py -m integration

These tests exercise the complete generate_report() flow.
By default they use Ollama. Set llm_backend in the config to test other adapters.
"""

from __future__ import annotations

import json
import re

import pytest

pytestmark = pytest.mark.integration


def _ollama_available() -> bool:
    try:
        from src.models.obj3_gemini.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter({"ollama": {"model": "qwen2.5:14b"}})
        return adapter.is_available()
    except Exception:
        return False


@pytest.fixture()
def reporter(tmp_path):
    """Create a reporter with isolated output dir (avoid polluting real reports/)."""
    import yaml

    from src.models.obj3_gemini.reporter import GeminiDisasterReporter

    # Write a temporary config pointing at tmp_path for output
    config = {
        "llm_backend": "ollama",
        "ollama": {"model": "qwen2.5:14b", "base_url": "http://localhost:11434", "temperature": 0.0, "max_retries": 2},
        "corpus": {"version": "v1", "local_dir": "corpus/", "max_corpus_chars": 500000},
        "reporting": {
            "confidence_threshold": 0.70,
            "output_dir": str(tmp_path / "reports"),
            "gcs_bucket": "",
            "max_ml_block_chars": 20000,
            "max_data_block_chars": 20000,
        },
        "admin_toggle": {"default": True, "current_state": True, "persistence": "local"},
    }
    config_file = tmp_path / "reporting_config.yaml"
    config_file.write_text(yaml.safe_dump(config))

    # Create templates dir
    import shutil

    from tests.obj3.conftest import TEMPLATE_DIR
    tmp_templates = tmp_path / "templates"
    shutil.copytree(TEMPLATE_DIR, tmp_templates)

    r = GeminiDisasterReporter()
    r.load_model(config_file)
    return r


@pytest.mark.integration
class TestFullPipeline:
    @pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
    def test_generate_report_quiet_mode(self, reporter, mock_pipeline_result):
        result = reporter.generate_report(pipeline_result=mock_pipeline_result)
        assert result.report_result.report_type == "daily"
        assert result.json_path is not None
        assert result.json_path.exists()
        assert result.markdown_path is not None
        assert result.markdown_path.exists()

    @pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
    def test_generate_report_emergency_mode(self, reporter, emergency_pipeline_result):
        result = reporter.generate_report(pipeline_result=emergency_pipeline_result)
        assert result.report_result.report_type == "incident"
        assert result.html_path is not None or result.report_result.error is not None

    @pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
    def test_validation_passes_on_valid_output(self, reporter, mock_pipeline_result):
        result = reporter.generate_report(pipeline_result=mock_pipeline_result)
        if result.report_result.parsed_report is not None:
            assert result.validation.schema_valid is True

    @pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
    def test_no_human_input_when_toggle_off(self, reporter, mock_pipeline_result):
        reporter._toggle.disable("test")
        result = reporter.generate_report(pipeline_result=mock_pipeline_result)
        if result.report_result.parsed_report:
            assert result.report_result.parsed_report.human_input_included is False

    @pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
    def test_report_naming_convention(self, reporter, mock_pipeline_result):
        result = reporter.generate_report(pipeline_result=mock_pipeline_result)
        if result.json_path:
            # Filename should match ReportType_YYYYMMDD_HHMM.json
            assert re.match(r"DailyReport_\d{8}_\d{4}\.json", result.json_path.name)

    @pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
    def test_report_files_exist_after_generate(self, reporter, mock_pipeline_result):
        result = reporter.generate_report(pipeline_result=mock_pipeline_result)
        if result.json_path:
            assert result.json_path.exists()
            content = json.loads(result.json_path.read_text())
            assert "report_type" in content
        if result.markdown_path:
            assert result.markdown_path.exists()
            assert len(result.markdown_path.read_text()) > 0
