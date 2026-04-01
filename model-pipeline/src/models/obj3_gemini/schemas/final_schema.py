"""FinalReport schema — post-incident formal close-out (EMERGENCY/FINAL sub-state).

Comprehensive post-incident report aligned with ICS-209 final submission
and FEMA After Action Report (AAR) structure. Includes actual costs,
casualty summary, structures summary, evacuation history, and containment
timeline alongside the original AAR-style lessons learned.
"""

from __future__ import annotations

from pydantic import Field

from src.models.obj3_gemini.schemas.base_schema import (
    BaseReport,
    Casualties,
    EvacuationEvent,
    ProjectedLoss,
    ResourceDeployed,
    StructureSummary,
    TimelineEvent,
)


class FinalReport(BaseReport):
    """Post-incident formal report. human_review_required is always True.

    Structure follows ICS-209 final report + FEMA AAR:
      - Incident identity & final status
      - Timeline & containment progression
      - Resources deployed & costs
      - Damage assessment (actual, not projected)
      - Casualties & evacuation history
      - Response effectiveness & lessons learned
    """

    # --- Incident identity ---
    incident_name: str
    linked_incident_id: str

    # --- Timeline ---
    incident_timeline: list[TimelineEvent] = Field(min_length=3)
    containment_timeline: list[TimelineEvent] | None = None

    # --- Final size & containment ---
    total_area_burned_acres: float | None = None
    percent_contained_final: float | None = Field(default=None, ge=0.0, le=100.0)

    # --- Resources & costs ---
    resources_deployed: list[ResourceDeployed]
    total_cost: float | None = Field(default=None, ge=0.0)

    # --- Damage assessment (actual counts, not projections) ---
    losses_summary: ProjectedLoss
    structures_summary: StructureSummary | None = None

    # --- Casualties (ICS-209 Block 30-33 final) ---
    casualties_summary: Casualties | None = None

    # --- Evacuation history ---
    evacuation_history: list[EvacuationEvent] | None = None

    # --- Weather progression ---
    weather_progression: str | None = None

    # --- AAR: effectiveness & lessons ---
    response_effectiveness: str
    lessons_learned: list[str] = Field(min_length=2)
    recommendations_for_future: list[str] = Field(min_length=2)
    attachments_referenced: list[str] | None = None
