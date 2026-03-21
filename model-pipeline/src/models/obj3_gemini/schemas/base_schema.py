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
