"""OBJ-3 Pydantic report schemas — re-exports for convenience."""

from src.models.obj3_gemini.schemas.base_schema import (
    BaseReport,
    ProjectedLoss,
    Recommendation,
    ResourceDeployed,
    ResourceRequirement,
    RiskCell,
    TimelineEvent,
    VulnerableGroup,
)
from src.models.obj3_gemini.schemas.daily_schema import DailyReport
from src.models.obj3_gemini.schemas.final_schema import FinalReport
from src.models.obj3_gemini.schemas.high_risk_schema import HighRiskReport
from src.models.obj3_gemini.schemas.incident_schema import IncidentReport

SCHEMA_MAP: dict[str, type[BaseReport]] = {
    "daily": DailyReport,
    "high_risk": HighRiskReport,
    "incident": IncidentReport,
    "final": FinalReport,
}

__all__ = [
    "BaseReport",
    "DailyReport",
    "FinalReport",
    "HighRiskReport",
    "IncidentReport",
    "ProjectedLoss",
    "Recommendation",
    "ResourceDeployed",
    "ResourceRequirement",
    "RiskCell",
    "SCHEMA_MAP",
    "TimelineEvent",
    "VulnerableGroup",
]
