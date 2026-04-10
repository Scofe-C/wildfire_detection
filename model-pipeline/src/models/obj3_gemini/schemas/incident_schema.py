"""IncidentReport schema — generated during EMERGENCY mode (ACTIVE_FIRE / INTERIM / POST_FIRE).

Aligned with ICS-209 Incident Status Summary structure. Includes tiered
time-horizon projections, structured weather/fire behavior observations,
evacuation tracking, and casualty reporting.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from src.models.obj3_gemini.schemas.base_schema import (
    BaseReport,
    Casualties,
    EvacuationZone,
    FireBehavior,
    ProjectedActivity,
    ProjectedLoss,
    ResourceRequirement,
    VulnerableGroup,
    WeatherObservation,
)


class IncidentReport(BaseReport):
    """Full formal incident report for active fire events.

    Structure follows ICS-209 block groups:
      - Identity & status (Blocks 1-15)
      - Weather & fire behavior (Blocks 28-29, 35)
      - Damage & casualties (Blocks 30-33)
      - Projections & threats (Blocks 36-38)
      - Resources & actions (Blocks 39-41)
    """

    # --- Incident identity & status ---
    incident_name: str
    incident_status: Literal["ACTIVE", "CONTAINED", "CONTROLLED", "OUT"]
    incident_complexity: Literal["TYPE_1", "TYPE_2", "TYPE_3", "TYPE_4", "TYPE_5"] | None = None
    percent_contained: float | None = Field(default=None, ge=0.0, le=100.0)
    affected_communities: list[str] = Field(min_length=1)

    # --- Fire behavior & weather (ICS-209 Blocks 28-29, 35) ---
    spread_summary: str
    estimated_area_acres: float | None = None
    fire_behavior: FireBehavior | None = None
    weather_observations: WeatherObservation | None = None

    # --- Damage & casualties (ICS-209 Blocks 30-33) ---
    resource_requirements: list[ResourceRequirement] = Field(min_length=1)
    projected_losses: ProjectedLoss
    casualties: Casualties | None = None
    vulnerable_populations: list[VulnerableGroup] = Field(min_length=1)

    # --- Evacuation (CAL FIRE 4-tier + ICS-209 Block 36) ---
    evacuation_status: list[EvacuationZone] | None = None
    closures: list[str] | None = None

    # --- Projections (ICS-209 Blocks 36-38) ---
    projected_activity: ProjectedActivity | None = None

    # --- Actions & strategy (ICS-209 Blocks 37, 41) ---
    immediate_actions: list[str] = Field(min_length=3)
    strategic_objectives: list[str] | None = None
    planned_next_actions: list[str] | None = None

    # --- Cost (ICS-209 Blocks 45-46) ---
    cost_to_date: float | None = Field(default=None, ge=0.0)

    # --- Operator & ICS references ---
    operator_notes_incorporated: str | None = None
    ics_form_references: list[str] | None = None
