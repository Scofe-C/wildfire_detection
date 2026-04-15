"""OBJ-3 Pydantic report schemas — re-exports for convenience."""

from src.models.obj3_gemini.schemas.base_schema import (
    BaseReport,
    Casualties,
    EvacuationEvent,
    EvacuationZone,
    FireBehavior,
    ProjectedActivity,
    ProjectedLoss,
    Recommendation,
    ReasoningStep,
    ResourceDeployed,
    ResourceRequirement,
    RiskCell,
    StructureSummary,
    TimelineEvent,
    VulnerableGroup,
    WeatherObservation,
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
    "Casualties",
    "DailyReport",
    "EvacuationEvent",
    "EvacuationZone",
    "FinalReport",
    "FireBehavior",
    "HighRiskReport",
    "IncidentReport",
    "ProjectedActivity",
    "ProjectedLoss",
    "Recommendation",
    "ReasoningStep",
    "ResourceDeployed",
    "ResourceRequirement",
    "RiskCell",
    "SCHEMA_MAP",
    "StructureSummary",
    "TimelineEvent",
    "VulnerableGroup",
    "WeatherObservation",
]
