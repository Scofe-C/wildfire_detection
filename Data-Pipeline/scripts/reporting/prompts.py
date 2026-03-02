"""
LLM Prompt Templates
=====================
Prompt templates for the Gemini 2.0 Flash-powered wildfire report generator.
Each template injects structured pipeline data into a focused prompt for one
report section.

Reference: ml_llm_readiness_and_plan.md §Part 3
"""

# ---------------------------------------------------------------------------
# System Prompt — ICS-209 analyst persona
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a senior wildfire incident analyst generating an
ICS-209 style situation report. You MUST:
1. Base ALL claims on the data provided — never fabricate numbers
2. Express uncertainty where model confidence is below 0.7
3. Use fire service terminology (ICS, NIMS standards)
4. Flag any data quality issues (stale weather, missing VIIRS confirmation)
5. Include specific, actionable recommendations with estimated timelines
6. All cost estimates must show assumptions and range (low/high)
7. Evacuation recommendations must specify Zone, Timeline, and Route
8. End every report with a Limitations section acknowledging AI generation
"""


# ---------------------------------------------------------------------------
# Resource guidelines (injected into prompts for personnel/cost sections)
# ---------------------------------------------------------------------------
RESOURCE_GUIDELINES = {
    "firefighters_per_mile_of_line": {
        "light_fuels_flat": 20,       # Grass, 0-15° slope
        "medium_fuels_moderate": 35,  # Brush, 15-30° slope
        "heavy_fuels_steep": 50,      # Timber, 30°+ slope
    },
    "engine_to_firefighter_ratio": "1:5",
    "dozer_effectiveness_terrain": "only below 35% slope",
    "helicopter_bucket_capacity_gallons": 300,
    "air_tanker_retardant_gallons": 3000,
    "crew_rotation_hours": 14,  # Max shift length per OSHA
}


COST_REFERENCE = {
    "suppression_cost_per_acre": {
        "grass_flat": 500,
        "brush_moderate": 2000,
        "timber_steep": 5000,
        "wui_interface": 10000,   # Wildland-urban interface
    },
    "median_home_value": {
        "california": 750000,
        "texas": 310000,
    },
    "firefighter_cost_per_day": 500,
    "engine_cost_per_day": 2500,
    "helicopter_cost_per_hour": 5000,
    "air_tanker_cost_per_hour": 15000,
}


# ---------------------------------------------------------------------------
# Section prompt templates
# ---------------------------------------------------------------------------
SECTION_TEMPLATES = {
    "situation_assessment": """Generate a SITUATION ASSESSMENT section based on the data below.
Include: fire location, current size, behavior, weather conditions and trend.

INCIDENT DATA:
{data_block}
""",

    "spread_prediction": """Generate a SPREAD PREDICTION ANALYSIS section.
Include: predicted fire movement in 6h / 24h / 48h, confidence level,
key uncertainty factors.

INCIDENT DATA:
{data_block}
""",

    "firebreak_recommendations": """Generate FIREBREAK LAYING RECOMMENDATIONS.
Given the predicted fire spread toward the {spread_direction} at {spread_rate} km/h,
and the terrain profile showing {slope}° slope with {fuel_model} fuel:

1. Identify natural firebreak opportunities (roads, rivers, ridgelines)
   within the predicted 24-hour perimeter
2. Recommend constructed firebreak locations where natural barriers
   are insufficient
3. Specify firebreak type for each segment based on fuel and terrain:
   - Bulldozer line: flat terrain, accessible, fuel height < 2m
   - Hand line: steep terrain (>30°), sensitive areas
   - Burnout operations: where containment line exists downwind
4. Estimate construction time and crew requirements per segment

INCIDENT DATA:
{data_block}
""",

    "personnel_resources": """Generate PERSONNEL & RESOURCE MOBILIZATION estimates.
Include: estimated firefighters needed, equipment recommendations,
mutual aid requirements.

RESOURCE GUIDELINES:
{resource_guidelines}

INCIDENT DATA:
{data_block}
""",

    "evacuation": """Generate EVACUATION RECOMMENDATIONS.
Include: zones to evacuate, evacuation routes, timeline
(warning vs mandatory), special populations.

INCIDENT DATA:
{data_block}
""",

    "damage_cost_estimation": """Generate DAMAGE ASSESSMENT & COST ESTIMATION.
Include: structures in fire path, estimated property damage,
suppression cost estimation, economic impact.
All cost estimates must show assumptions and range (low/high).

COST REFERENCE:
{cost_reference}

INCIDENT DATA:
{data_block}
""",
}


def format_context_for_llm(context: dict) -> str:
    """Format a structured context dict into a readable text block for LLM.

    Args:
        context: The REPORT_CONTEXT dict from the pipeline.

    Returns:
        Formatted string suitable for injection into prompt templates.
    """
    import json

    sections = []

    if "risk_assessment" in context:
        ra = context["risk_assessment"]
        sections.append(f"=== RISK ASSESSMENT ===\n"
                        f"Region: {ra.get('region', 'unknown')}\n"
                        f"High-risk cells: {len(ra.get('high_risk_cells', []))}\n"
                        f"Top risk factors: {ra.get('top_risk_factors', [])}")

    if "spread_prediction" in context:
        sp = context["spread_prediction"]
        sections.append(f"=== SPREAD PREDICTION ===\n"
                        f"Spread direction: {sp.get('spread_direction', 'unknown')}\n"
                        f"Spread rate: {sp.get('spread_rate_km_per_hour', 'N/A')} km/h\n"
                        f"Confidence: {sp.get('confidence', 'N/A')}\n"
                        f"Current perimeter cells: {len(sp.get('current_fire_perimeter', []))}\n"
                        f"Predicted 24h perimeter cells: {len(sp.get('predicted_24h_perimeter', []))}")

    if "current_conditions" in context:
        cc = context["current_conditions"]
        weather = cc.get("weather", {})
        fire = cc.get("fire_data", {})
        terrain = cc.get("terrain", {})
        sections.append(
            f"=== CURRENT CONDITIONS ===\n"
            f"Temperature: {weather.get('temperature_c', 'N/A')}°C\n"
            f"Humidity: {weather.get('relative_humidity_pct', 'N/A')}%\n"
            f"Wind: {weather.get('wind_speed_kmh', 'N/A')} km/h from {weather.get('wind_direction', 'N/A')}\n"
            f"VPD: {weather.get('vpd_kpa', 'N/A')} kPa\n"
            f"FWI: {weather.get('fire_weather_index', 'N/A')}\n"
            f"Precipitation (7d): {weather.get('precipitation_last_7d_mm', 'N/A')} mm\n"
            f"Active fire cells: {fire.get('active_fire_cells', 'N/A')}\n"
            f"Max FRP: {fire.get('max_frp_mw', 'N/A')} MW\n"
            f"Nearest populated area: {fire.get('nearest_populated_area_name', 'N/A')} "
            f"({fire.get('nearest_populated_area_km', 'N/A')} km)\n"
            f"Dominant fuel: {terrain.get('dominant_fuel_model', 'N/A')}\n"
            f"Slope: {terrain.get('average_slope_degrees', 'N/A')}°\n"
            f"Elevation: {terrain.get('elevation_range_m', 'N/A')} m\n"
            f"Canopy cover: {terrain.get('canopy_cover_pct', 'N/A')}%"
        )

    if "reference_data" in context:
        ref = context["reference_data"]
        sections.append(
            f"=== REFERENCE DATA ===\n"
            f"Population in threat zone: {ref.get('population_in_threat_zone', 'N/A')}\n"
            f"Structures in threat zone: {ref.get('structures_in_threat_zone', 'N/A')}\n"
            f"Road network: {ref.get('road_network_summary', 'N/A')}"
        )

    return "\n\n".join(sections) if sections else json.dumps(context, indent=2, default=str)
