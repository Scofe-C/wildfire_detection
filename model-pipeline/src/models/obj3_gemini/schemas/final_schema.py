"""FinalReport schema — post-incident formal close-out (EMERGENCY/FINAL sub-state)."""

from __future__ import annotations

from pydantic import Field

from src.models.obj3_gemini.schemas.base_schema import (
    BaseReport,
    ProjectedLoss,
    ResourceDeployed,
    TimelineEvent,
)


class FinalReport(BaseReport):
    """Post-incident formal report. human_review_required is always True."""

    incident_name: str
    linked_incident_id: str
    incident_timeline: list[TimelineEvent] = Field(min_length=3)
    total_area_burned_acres: float | None = None
    resources_deployed: list[ResourceDeployed]
    losses_summary: ProjectedLoss
    response_effectiveness: str
    lessons_learned: list[str] = Field(min_length=2)
    recommendations_for_future: list[str] = Field(min_length=2)
    attachments_referenced: list[str] | None = None
