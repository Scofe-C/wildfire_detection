"""HighRiskReport schema — generated during ACTIVE mode."""

from __future__ import annotations

from pydantic import Field

from src.models.obj3_gemini.schemas.base_schema import (
    BaseReport,
    Recommendation,
    RiskCell,
    VulnerableGroup,
)


class HighRiskReport(BaseReport):
    """Elevated-risk report when fire risk is high but no confirmed ignition."""

    risk_summary: str
    top_risk_cells: list[RiskCell] = Field(min_length=1, max_length=5)
    contributing_factors: list[str]
    preventive_recommendations: list[Recommendation] = Field(min_length=2)
    vulnerable_populations: list[VulnerableGroup] | None = None
    escalation_trigger: str
