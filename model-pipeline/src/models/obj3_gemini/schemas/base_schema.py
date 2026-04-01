"""Shared base report model and nested types used across all OBJ-3 report schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Disclaimer constant — must appear verbatim in every generated report.
# ---------------------------------------------------------------------------
REQUIRED_DISCLAIMER = "AI-generated. Not for operational use without human review."


# ---------------------------------------------------------------------------
# Nested supporting types
# ---------------------------------------------------------------------------

class RiskCell(BaseModel):
    """A single H3 cell with its risk score and coordinates."""

    h3_index: str
    risk_score: float = Field(ge=0.0, le=1.0)
    lat: float
    lon: float


class Recommendation(BaseModel):
    """An actionable recommendation for fire prevention or response."""

    title: str
    description: str
    priority: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class VulnerableGroup(BaseModel):
    """A community or population identified as vulnerable."""

    group_name: str
    location: str
    vulnerability_score: float = Field(ge=0.0, le=1.0)
    notes: str | None = None


class ResourceRequirement(BaseModel):
    """ICS-typed resource needed for incident response."""

    resource_type: str
    quantity: int = Field(ge=1)
    ics_type: str | None = None
    notes: str | None = None


class ProjectedLoss(BaseModel):
    """Projected or actual losses from an incident."""

    structures_at_risk: int = Field(ge=0)
    population_at_risk: int = Field(ge=0)
    infrastructure_notes: str | None = None
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)


class TimelineEvent(BaseModel):
    """A single event in an incident timeline."""

    timestamp: str
    event: str
    source: str | None = None


class ResourceDeployed(BaseModel):
    """A resource that was deployed during an incident."""

    resource_type: str
    quantity: int = Field(ge=1)
    deployed_at: str
    notes: str | None = None


# ---------------------------------------------------------------------------
# ICS-209 aligned nested types
# ---------------------------------------------------------------------------

class WeatherObservation(BaseModel):
    """Structured weather conditions at the incident — ICS-209 Block 35."""

    temperature_f: float | None = None
    relative_humidity_pct: float | None = None
    wind_speed_mph: float | None = None
    wind_direction: str | None = None
    fuel_moisture_1hr: float | None = None


class FireBehavior(BaseModel):
    """Observed fire behavior — ICS-209 Block 28 fire-specific fields."""

    rate_of_spread: str | None = None
    flame_length_ft: float | None = None
    spotting_distance: str | None = None
    fire_type: Literal["SURFACE", "CROWN", "GROUND", "MIXED"] | None = None


class EvacuationZone(BaseModel):
    """Evacuation zone with status — aligned with CAL FIRE 4-tier system."""

    zone_name: str
    status: Literal["ORDER", "WARNING", "LIFTED", "NORMAL"]
    population_affected: int | None = None


class ProjectedActivity(BaseModel):
    """Tiered time-horizon projections — ICS-209 Block 36/38."""

    hours_12: str | None = None
    hours_24: str | None = None
    hours_48: str | None = None
    hours_72: str | None = None


class Casualties(BaseModel):
    """Civilian and responder casualties — ICS-209 Block 30-33."""

    civilian_fatalities: int = Field(default=0, ge=0)
    civilian_injuries: int = Field(default=0, ge=0)
    responder_fatalities: int = Field(default=0, ge=0)
    responder_injuries: int = Field(default=0, ge=0)


class StructureSummary(BaseModel):
    """Structures impacted by type — ICS-209 Block 30."""

    residential_destroyed: int = Field(default=0, ge=0)
    residential_damaged: int = Field(default=0, ge=0)
    commercial_destroyed: int = Field(default=0, ge=0)
    commercial_damaged: int = Field(default=0, ge=0)
    infrastructure_notes: str | None = None


class EvacuationEvent(BaseModel):
    """A single evacuation event in the incident history."""

    timestamp: str
    zone_name: str
    action: Literal["ORDER_ISSUED", "WARNING_ISSUED", "LIFTED", "EXPANDED"]
    population_affected: int | None = None


# ---------------------------------------------------------------------------
# Base report — all report types inherit from this.
# ---------------------------------------------------------------------------

class BaseReport(BaseModel):
    """Fields present in ALL OBJ-3 report types."""

    incident_id: str
    report_type: Literal["daily", "high_risk", "incident", "final"]
    report_confidence: float = Field(ge=0.0, le=1.0)
    generated_at: str
    operating_mode: Literal["QUIET", "ACTIVE", "EMERGENCY"]
    risk_level: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]
    human_review_required: bool
    human_input_included: bool
    # B1: review_status — stamped by reporter.py AFTER the LLM call, not by
    # the LLM itself.  Default is PENDING_REVIEW (fail-safe).
    review_status: Literal["PENDING_REVIEW", "AUTO_APPROVED"] = "PENDING_REVIEW"
    # B2: grounding fields — populated by the LLM; read by reporter.py for
    # the deterministic human_review_required calculation.
    grounding_sources: list[str] = Field(default_factory=list)
    grounding_search_count: int = Field(default=0, ge=0)
    # B3: disagreement_flag — set by state machine before the LLM is called.
    disagreement_flag: bool = False
    disclaimer: str
    data_sources_used: list[str]

    @field_validator("disclaimer")
    @classmethod
    def disclaimer_must_match(cls, v: str) -> str:
        if v != REQUIRED_DISCLAIMER:
            raise ValueError(
                f"Disclaimer must be exactly: {REQUIRED_DISCLAIMER!r}"
            )
        return v
