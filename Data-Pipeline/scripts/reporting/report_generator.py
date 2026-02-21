"""
LLM Report Generator
=====================
Assembles pipeline data + ML model predictions into structured context,
then generates ICS-209 style wildfire incident reports using Gemini 2.0 Flash.

Reference: ml_llm_readiness_and_plan.md §Part 3

Usage:
    # As a module
    from scripts.reporting.report_generator import generate_report, ReportContext

    # As a FastAPI server
    uvicorn scripts.reporting.report_generator:app --host 0.0.0.0 --port 8000
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

from fastapi import FastAPI, HTTPException

from scripts.reporting.prompts import (
    SYSTEM_PROMPT,
    SECTION_TEMPLATES,
    RESOURCE_GUIDELINES,
    COST_REFERENCE,
    format_context_for_llm,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Wildfire Report Generator",
    description="Generates ICS-209 style wildfire reports using Gemini 2.0 Flash",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Report Context dataclass
# ---------------------------------------------------------------------------
@dataclass
class RiskAssessment:
    region: str = ""
    high_risk_cells: list = field(default_factory=list)
    risk_scores_by_cell: dict = field(default_factory=dict)
    top_risk_factors: list = field(default_factory=list)


@dataclass
class SpreadPrediction:
    current_fire_perimeter: list = field(default_factory=list)
    predicted_24h_perimeter: list = field(default_factory=list)
    spread_direction: str = "unknown"
    spread_rate_km_per_hour: float = 0.0
    confidence: float = 0.0


@dataclass
class WeatherConditions:
    temperature_c: float = 0.0
    relative_humidity_pct: float = 0.0
    wind_speed_kmh: float = 0.0
    wind_direction: str = ""
    vpd_kpa: float = 0.0
    fire_weather_index: float = 0.0
    precipitation_last_7d_mm: float = 0.0


@dataclass
class FireData:
    active_fire_cells: int = 0
    max_frp_mw: float = 0.0
    nearest_populated_area_km: float = 0.0
    nearest_populated_area_name: str = ""


@dataclass
class TerrainData:
    dominant_fuel_model: str = ""
    average_slope_degrees: float = 0.0
    elevation_range_m: str = ""
    canopy_cover_pct: float = 0.0


@dataclass
class ReferenceData:
    historical_fires_nearby: list = field(default_factory=list)
    fire_station_locations: list = field(default_factory=list)
    hospital_locations: list = field(default_factory=list)
    road_network_summary: str = ""
    population_in_threat_zone: int = 0
    structures_in_threat_zone: int = 0


@dataclass
class ReportContext:
    """Full context package assembled from pipeline + ML predictions."""
    risk_assessment: RiskAssessment = field(default_factory=RiskAssessment)
    spread_prediction: SpreadPrediction = field(default_factory=SpreadPrediction)
    weather: WeatherConditions = field(default_factory=WeatherConditions)
    fire_data: FireData = field(default_factory=FireData)
    terrain: TerrainData = field(default_factory=TerrainData)
    reference_data: ReferenceData = field(default_factory=ReferenceData)

    def to_dict(self) -> dict:
        return {
            "risk_assessment": asdict(self.risk_assessment),
            "spread_prediction": asdict(self.spread_prediction),
            "current_conditions": {
                "weather": asdict(self.weather),
                "fire_data": asdict(self.fire_data),
                "terrain": asdict(self.terrain),
            },
            "reference_data": asdict(self.reference_data),
        }


# ---------------------------------------------------------------------------
# Gemini 2.0 Flash client
# ---------------------------------------------------------------------------
def _get_gemini_client():
    """Initialize Gemini client from google-genai SDK."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable not set. "
            "Get an API key from https://aistudio.google.com/apikey"
        )

    from google import genai
    client = genai.Client(api_key=api_key)
    return client


def generate_report(
    context: dict,
    sections: Optional[list[str]] = None,
    model: str = "gemini-2.0-flash",
) -> dict:
    """Generate a full ICS-209 style wildfire incident report.

    Args:
        context: Structured context dict (from ReportContext.to_dict() or pipeline).
        sections: Optional list of section names to generate. If None, generates all.
        model: Gemini model name.

    Returns:
        Dict with 'report' (full text) and 'sections' (individual section texts).
    """
    client = _get_gemini_client()
    data_block = format_context_for_llm(context)

    if sections is None:
        sections = list(SECTION_TEMPLATES.keys())

    section_results = {}

    for section_name in sections:
        if section_name not in SECTION_TEMPLATES:
            logger.warning(f"Unknown section '{section_name}' — skipping")
            continue

        template = SECTION_TEMPLATES[section_name]

        # Format template with available data
        format_kwargs = {
            "data_block": data_block,
            "resource_guidelines": json.dumps(RESOURCE_GUIDELINES, indent=2),
            "cost_reference": json.dumps(COST_REFERENCE, indent=2),
        }

        # Extract spread-specific values for firebreak template
        spread = context.get("spread_prediction", {})
        terrain = context.get("current_conditions", {}).get("terrain", {})
        format_kwargs.update({
            "spread_direction": spread.get("spread_direction", "unknown"),
            "spread_rate": spread.get("spread_rate_km_per_hour", "N/A"),
            "slope": terrain.get("average_slope_degrees", "N/A"),
            "fuel_model": terrain.get("dominant_fuel_model", "N/A"),
        })

        try:
            prompt = template.format(**format_kwargs)
        except KeyError as e:
            logger.warning(f"Template format error for {section_name}: {e}")
            prompt = template.replace("{data_block}", data_block)

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "max_output_tokens": 4000,
                    "temperature": 0.3,  # Low temp for factual reporting
                },
            )
            section_results[section_name] = response.text
        except Exception as e:
            logger.error(f"Gemini API error for section '{section_name}': {e}")
            section_results[section_name] = f"[Error generating {section_name}: {e}]"

    # Combine all sections into full report
    full_report = "\n\n---\n\n".join(
        f"## {name.replace('_', ' ').title()}\n\n{text}"
        for name, text in section_results.items()
    )

    # Add limitations section
    full_report += (
        "\n\n---\n\n## Limitations\n\n"
        "This report is generated by AI models and requires human verification "
        "before operational use. Model limitations include: weather data age, "
        "spread model trained on historical patterns that may not reflect "
        "current conditions, cost estimates based on regional averages."
    )

    return {"report": full_report, "sections": section_results}


def assemble_context(
    fused_df=None,
    ml_predictions: Optional[dict] = None,
    spread_predictions: Optional[dict] = None,
    reference_data: Optional[dict] = None,
) -> dict:
    """Assemble pipeline output + ML predictions into report context.

    Args:
        fused_df: The fused features DataFrame from the pipeline.
        ml_predictions: Dict with ignition model predictions (risk scores, SHAP).
        spread_predictions: Dict with spread model output (perimeter, direction).
        reference_data: Dict with static reference data (population, roads, etc).

    Returns:
        Context dict ready for generate_report().
    """
    context = ReportContext()

    if ml_predictions:
        context.risk_assessment = RiskAssessment(
            region=ml_predictions.get("region", ""),
            high_risk_cells=ml_predictions.get("high_risk_cells", []),
            risk_scores_by_cell=ml_predictions.get("risk_scores_by_cell", {}),
            top_risk_factors=ml_predictions.get("top_risk_factors", []),
        )

    if spread_predictions:
        context.spread_prediction = SpreadPrediction(
            current_fire_perimeter=spread_predictions.get("current_fire_perimeter", []),
            predicted_24h_perimeter=spread_predictions.get("predicted_24h_perimeter", []),
            spread_direction=spread_predictions.get("spread_direction", "unknown"),
            spread_rate_km_per_hour=spread_predictions.get("spread_rate_km_per_hour", 0.0),
            confidence=spread_predictions.get("confidence", 0.0),
        )

    if fused_df is not None and len(fused_df) > 0:
        # Extract current conditions from latest fused features
        latest = fused_df.iloc[-1] if len(fused_df) > 0 else {}

        context.weather = WeatherConditions(
            temperature_c=float(latest.get("temperature_2m", 0)),
            relative_humidity_pct=float(latest.get("relative_humidity_2m", 0)),
            wind_speed_kmh=float(latest.get("wind_speed_10m", 0)),
            wind_direction=str(latest.get("wind_direction_10m", "")),
            vpd_kpa=float(latest.get("vpd", 0)),
            fire_weather_index=float(latest.get("fire_weather_index", 0)),
        )

        context.fire_data = FireData(
            active_fire_cells=int(fused_df["active_fire_count"].sum()) if "active_fire_count" in fused_df.columns else 0,
            max_frp_mw=float(fused_df["mean_frp"].max()) if "mean_frp" in fused_df.columns else 0.0,
        )

        context.terrain = TerrainData(
            average_slope_degrees=float(fused_df["slope_degrees"].mean()) if "slope_degrees" in fused_df.columns else 0.0,
            canopy_cover_pct=float(fused_df["canopy_cover_pct"].mean()) if "canopy_cover_pct" in fused_df.columns else 0.0,
        )

    if reference_data:
        context.reference_data = ReferenceData(**reference_data)

    return context.to_dict()


# ---------------------------------------------------------------------------
# FastAPI endpoint
# ---------------------------------------------------------------------------
@app.post("/generate_report")
async def api_generate_report(context: dict):
    """Generate a wildfire incident report from pipeline context data."""
    try:
        result = generate_report(context)
        return result
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "model": "gemini-2.0-flash"}
