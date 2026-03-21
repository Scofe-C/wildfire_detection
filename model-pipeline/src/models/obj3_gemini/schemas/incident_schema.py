"""IncidentReport schema — generated during EMERGENCY mode (ACTIVE_FIRE / INTERIM / POST_FIRE)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from src.models.obj3_gemini.schemas.base_schema import (
    BaseReport,
    ProjectedLoss,
    ResourceRequirement,
    VulnerableGroup,
)


class IncidentReport(BaseReport):
    """Full formal incident report for active fire events."""

    incident_name: str
    incident_status: Literal["ACTIVE", "CONTAINED", "CONTROLLED", "OUT"]
    affected_communities: list[str] = Field(min_length=1)
    spread_summary: str
    estimated_area_acres: float | None = None
    resource_requirements: list[ResourceRequirement] = Field(min_length=1)
    projected_losses: ProjectedLoss
    vulnerable_populations: list[VulnerableGroup] = Field(min_length=1)
    immediate_actions: list[str] = Field(min_length=3)
    operator_notes_incorporated: str | None = None
    ics_form_references: list[str] | None = None
