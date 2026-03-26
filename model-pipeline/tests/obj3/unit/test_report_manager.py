"""Unit tests for report_manager.py — §5.1 test_report_manager."""

from __future__ import annotations

from datetime import date, datetime

from src.reports.report_manager import (
    ReportIndex,
    list_reports,
    make_filename,
    save_report,
)


class TestMakeFilename:
    def test_filename_format_daily(self):
        dt = datetime(2026, 3, 20, 6, 0)
        assert make_filename("daily", dt) == "DailyReport_20260320_0600"

    def test_filename_format_incident(self):
        dt = datetime(2026, 3, 20, 16, 30)
        assert make_filename("incident", dt) == "IncidentReport_20260320_1630"

    def test_filename_format_high_risk(self):
        dt = datetime(2026, 3, 20, 14, 30)
        assert make_filename("high_risk", dt) == "HighRiskReport_20260320_1430"

    def test_filename_format_final(self):
        dt = datetime(2026, 3, 21, 9, 0)
        assert make_filename("final", dt) == "FinalReport_20260321_0900"


class TestSaveReport:
    def test_save_creates_files(self, tmp_path):
        dt = datetime(2026, 3, 20, 6, 0)
        json_path, rendered_path = save_report(
            report_json='{"test": true}',
            rendered_content="# Test Report",
            report_type="daily",
            incident_id="test-id",
            dt=dt,
            fmt="md",
            output_dir=tmp_path,
        )
        assert json_path.exists()
        assert rendered_path.exists()
        assert json_path.suffix == ".json"
        assert rendered_path.suffix == ".md"

    def test_save_creates_subdirectory(self, tmp_path):
        dt = datetime(2026, 3, 20, 16, 0)
        subdir = tmp_path / "daily"
        assert not subdir.exists()

        save_report(
            report_json="{}",
            rendered_content="",
            report_type="daily",
            incident_id="test-id",
            dt=dt,
            fmt="md",
            output_dir=tmp_path,
        )
        assert subdir.exists()
        assert subdir.is_dir()

    def test_save_html_format(self, tmp_path):
        dt = datetime(2026, 3, 20, 16, 0)
        json_path, rendered_path = save_report(
            report_json='{"test": true}',
            rendered_content="<html></html>",
            report_type="incident",
            incident_id="test-id",
            dt=dt,
            fmt="html",
            output_dir=tmp_path,
        )
        assert rendered_path.suffix == ".html"


class TestListReports:
    def test_list_reports_empty(self, tmp_path):
        result = list_reports("daily", tmp_path)
        assert result == []

    def test_list_reports_with_files(self, tmp_path):
        # Create 3 saved reports
        for hour in [6, 12, 18]:
            dt = datetime(2026, 3, 20, hour, 0)
            save_report(
                report_json=f'{{"hour": {hour}}}',
                rendered_content=f"# Report at {hour}",
                report_type="daily",
                incident_id="test-id",
                dt=dt,
                fmt="md",
                output_dir=tmp_path,
            )

        result = list_reports("daily", tmp_path)
        assert len(result) == 3
        assert all(isinstance(r, ReportIndex) for r in result)

    def test_list_reports_date_filter(self, tmp_path):
        for day in [19, 20, 21]:
            dt = datetime(2026, 3, day, 6, 0)
            save_report(
                report_json="{}",
                rendered_content="",
                report_type="daily",
                incident_id="test-id",
                dt=dt,
                fmt="md",
                output_dir=tmp_path,
            )

        result = list_reports(
            "daily", tmp_path,
            date_range=(date(2026, 3, 20), date(2026, 3, 20)),
        )
        assert len(result) == 1
