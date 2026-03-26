"""DailyReport schema — generated during QUIET mode."""

from __future__ import annotations

from src.models.obj3_gemini.schemas.base_schema import BaseReport


class DailyReport(BaseReport):
    """Routine daily situation report. Optional sections may be null."""

    summary: str
    monitored_area_count: int
    highest_risk_cell: str | None = None
    weather_summary: str | None = None
    notable_changes: list[str]
    next_check_recommendation: str | None = None
