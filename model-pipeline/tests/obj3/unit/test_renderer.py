"""Unit tests for renderer.py — §5.1 test_renderer."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models.obj3_gemini.renderer import (
    get_template,
    markdown_to_html,
    render_html,
    render_markdown,
)
from src.models.obj3_gemini.schemas.daily_schema import DailyReport
from src.models.obj3_gemini.schemas.incident_schema import IncidentReport
from tests.obj3.conftest import TEMPLATE_DIR


def _weasyprint_available() -> bool:
    try:
        import weasyprint  # noqa: F401
        return True
    except ImportError:
        return False


class TestRenderMarkdownDaily:
    def test_render_markdown_daily(self, valid_daily_json):
        report = DailyReport(**valid_daily_json)
        md = render_markdown(report, TEMPLATE_DIR)
        assert isinstance(md, str)
        assert len(md) > 0
        assert report.summary in md

    def test_render_markdown_contains_risk_level(self, valid_daily_json):
        report = DailyReport(**valid_daily_json)
        md = render_markdown(report, TEMPLATE_DIR)
        assert "LOW" in md


class TestRenderHtmlIncident:
    def test_render_html_incident(self, valid_incident_json):
        report = IncidentReport(**valid_incident_json)
        html = render_html(report, TEMPLATE_DIR)
        assert isinstance(html, str)
        assert "<html" in html
        assert report.incident_name in html

    def test_render_html_contains_actions(self, valid_incident_json):
        report = IncidentReport(**valid_incident_json)
        html = render_html(report, TEMPLATE_DIR)
        assert "Evacuate Zone A" in html


class TestRenderMissingField:
    def test_render_missing_required_field(self, valid_daily_json):
        """If a template references a missing field, Jinja2 StrictUndefined raises."""
        import tempfile

        report = DailyReport(**valid_daily_json)
        with tempfile.TemporaryDirectory() as td:
            bad_template = Path(td) / "daily.md.j2"
            bad_template.write_text("{{ nonexistent_field }}")
            with pytest.raises(Exception):  # jinja2.UndefinedError
                render_markdown(report, Path(td))


class TestMarkdownToHtml:
    def test_markdown_to_html(self):
        md = "# Hello\n\nThis is a test."
        html = markdown_to_html(md)
        assert "<h1>" in html
        assert "Hello" in html


class TestGetTemplate:
    def test_get_template_daily(self):
        assert get_template("daily", "md") == "daily.md.j2"

    def test_get_template_incident(self):
        assert get_template("incident", "html") == "incident.html.j2"

    def test_get_template_invalid(self):
        with pytest.raises(ValueError):
            get_template("nonexistent", "md")


class TestRenderPdf:
    """PDF tests require WeasyPrint. Skipped if not installed."""

    @pytest.mark.skipif(
        not _weasyprint_available(),
        reason="WeasyPrint not installed",
    )
    def test_render_pdf_returns_bytes(self, valid_incident_json):
        from src.models.obj3_gemini.renderer import render_pdf

        report = IncidentReport(**valid_incident_json)
        html = render_html(report, TEMPLATE_DIR)
        pdf = render_pdf(html)
        assert isinstance(pdf, bytes)
        assert len(pdf) > 0

    @pytest.mark.skipif(
        not _weasyprint_available(),
        reason="WeasyPrint not installed",
    )
    def test_pdf_content_type(self, valid_incident_json):
        from src.models.obj3_gemini.renderer import render_pdf

        report = IncidentReport(**valid_incident_json)
        html = render_html(report, TEMPLATE_DIR)
        pdf = render_pdf(html)
        assert pdf[:5] == b"%PDF-"
