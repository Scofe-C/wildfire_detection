"""Context builder — assembles multi-source context for each LLM report call.

This is the *only* place where input sources are combined before being sent
to the adapter.  No LLM calls happen here.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

logger = logging.getLogger(__name__)

from src.models.obj3_gemini.state_machine import (
    AdminToggle,
    EmergencySubState,
    OperationalMode,
    mode_to_report_type,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class UploadedFile:
    """A file uploaded by an operator or manager."""

    filename: str
    content_bytes: bytes
    mime_type: str


@dataclass
class HumanInput:
    """Operator / management input for context injection."""

    text_notes: str | None = None
    uploaded_files: list[UploadedFile] = field(default_factory=list)
    source: Literal["operator", "management"] = "operator"
    submitted_at: str = ""


@dataclass
class ContextBundle:
    """Complete context payload sent to the LLM adapter."""

    system_prompt: str
    corpus_ref: str | None          # cache_name (Phase 3) or None
    corpus_text: str | None         # inline text (Phase 1/2) or None
    ml_block: str                   # serialised ML pipeline outputs
    data_block: str                 # serialised data pipeline snapshot
    human_block: str                # operator input (empty if toggle OFF)
    instruction: str                # final generation directive
    report_type: str
    incident_id: str
    # Processed uploaded files — adapter decides how to inject (text vs vision)
    # Import is local to avoid circular deps at module level
    uploaded_files: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------

def build_system_prompt(report_type: str, schema: dict[str, Any]) -> str:
    """Construct the system prompt string for this report type.

    Includes role definition, output schema as JSON string,
    hallucination rules, and disclaimer injection instruction.
    """
    schema_str = json.dumps(schema, indent=2)
    return (
        "You are a professional disaster reporting assistant specialised in "
        "wildfire analysis. You generate structured reports aligned with "
        "ICS-209 Incident Status Summary conventions, based on real-time "
        "ML pipeline outputs, environmental data, and official emergency "
        "management doctrine.\n\n"
        "RULES:\n"
        "1. Output ONLY valid JSON matching the schema below.\n"
        "2. Do NOT hallucinate data — if a value is unknown, use null for "
        "optional fields. Say 'insufficient data' rather than inventing figures.\n"
        "3. Every report MUST include the disclaimer: "
        '"AI-generated. Not for operational use without human review."\n'
        "4. Set human_review_required=true if report_confidence < 0.70.\n"
        "5. Do NOT add markdown code fences or text outside the JSON object.\n"
        "6. GROUNDING: You MUST list the corpus documents you referenced in "
        "the 'grounding_sources' field. Use the exact filenames from the "
        "REFERENCE CORPUS section. Set 'grounding_search_count' to the number "
        "of distinct sources you actually consulted.\n"
        "7. Be GEOGRAPHICALLY SPECIFIC — use H3 cell indices, specific "
        "neighborhoods, and precise coordinates rather than vague county-level "
        "references.\n"
        "8. For incident reports: populate weather_observations and "
        "fire_behavior from the telemetry and ML data provided. Use ICS-209 "
        "tiered projections (12/24/48/72h) in projected_activity when data "
        "supports forecasting.\n"
        "9. For resource_requirements: reference ICS resource typing standards "
        "(Type 1-7) from the IRPG corpus when available.\n\n"
        f"REPORT TYPE: {report_type}\n\n"
        f"RESPONSE SCHEMA:\n{schema_str}"
    )


def build_ml_block(
    pipeline_result: dict[str, Any],
    max_chars: int = 20_000,
) -> str:
    """Serialise ML pipeline outputs into a structured text block.

    Uses priority-based assembly: when truncating (Ollama fallback),
    always keeps top 5 XGBoost cells and OBJ-2 simulation data first,
    then adds lower-priority sections if space remains.
    """
    # Build sections in priority order (highest first)
    sections: list[str] = []

    # Priority 1: XGBoost top cells (always keep at least 5)
    top_cells = pipeline_result.get("xgboost_top_cells") or []
    if top_cells:
        xgb_lines = ["## XGBoost Top Risk Cells"]
        # Under tight limits, show fewer cells
        cell_limit = 5 if max_chars < 10_000 else 20
        for cell in top_cells[:cell_limit]:
            xgb_lines.append(
                f"- H3: {cell.get('h3_index')}  "
                f"P={cell.get('probability', 'N/A')}  "
                f"({cell.get('lat', '?')}, {cell.get('lon', '?')})"
            )
        sections.append("\n".join(xgb_lines))

    # Priority 2: OBJ-2 Rothermel Simulation (critical for emergency reports)
    sim = pipeline_result.get("obj2_simulation")
    if sim:
        sim_lines = ["\n## OBJ-2 Fire Spread Simulation (Rothermel)"]
        sim_lines.append(f"- Ignition cell: {sim.get('ignition_cell', 'N/A')}")
        sim_lines.append(f"- Ignition probability: {sim.get('ignition_probability', 'N/A')}")
        sim_lines.append(f"- Spread direction: {sim.get('spread_direction_deg', 'N/A')} degrees")
        sim_lines.append(f"- Spread speed: {sim.get('spread_speed_kmh', 'N/A')} km/h")
        sim_lines.append(f"- Crown fire status: {sim.get('crown_fire_status', 'N/A')}")
        sim_lines.append(f"- Byram fire intensity: {sim.get('byram_intensity_kwm', 'N/A')} kW/m")
        sim_lines.append(f"- Dead fuel moisture: {sim.get('dead_fuel_moisture_pct', 'N/A')}%")
        sim_lines.append(f"- Foliar moisture: {sim.get('foliar_moisture_content_pct', 'N/A')}%")
        sim_lines.append(f"- Dominant spread factor: {sim.get('dominant_factor', 'N/A')}")
        inputs_used = sim.get("inputs_used") or {}
        if inputs_used:
            sim_lines.append(f"- Wind speed (10m): {inputs_used.get('wind_speed_10m_ms', 'N/A')} m/s")
            sim_lines.append(f"- Midflame wind: {inputs_used.get('midflame_wind_mph', 'N/A')} mph")
            sim_lines.append(f"- Slope: {inputs_used.get('ignition_cell_slope_deg', 'N/A')} degrees")
            sim_lines.append(f"- FBFM40 fuel model: {inputs_used.get('ignition_cell_fbfm40', 'N/A')}")
        warnings = sim.get("warnings") or []
        if warnings:
            sim_lines.append(f"- Warnings: {', '.join(str(w) for w in warnings)}")
        sections.append("\n".join(sim_lines))

    # Priority 3: Propagator summary
    prop = pipeline_result.get("propagator_summary")
    if prop:
        sections.append(f"\n## Propagator Summary (secondary comparison)\n{str(prop)[:2000]}")

    # Priority 4: Bias gate
    bias = pipeline_result.get("bias_report")
    if bias:
        sections.append(
            f"\n## Bias Gate Result\n"
            f"- Gate: {bias.get('gate_result', 'N/A')}\n"
            f"- Observed disparity: {bias.get('observed_disparity', 'N/A')}"
        )
    else:
        sections.append("\n## Bias Gate Result: not run (no bias evaluation available)")

    # Priority 5: Cell2Fire (lowest priority, often large)
    c2f = pipeline_result.get("cell2fire_geojson")
    if c2f:
        c2f_lines = ["\n## Cell2Fire Spread Model (top-10)"]
        if isinstance(c2f, list):
            for feat in c2f[:10]:
                c2f_lines.append(f"- {json.dumps(feat)}")
        else:
            c2f_lines.append(str(c2f)[:2000])
        sections.append("\n".join(c2f_lines))

    # Assemble with priority-aware truncation
    block = ""
    for section in sections:
        if len(block) + len(section) + 1 > max_chars:
            break
        block = block + ("\n" if block else "") + section

    # Log warning if context was truncated
    total_len = sum(len(s) for s in sections)
    if total_len > max_chars:
        logger.warning(
            "ML block truncated: %d chars available, %d chars total content. "
            "Lower-priority sections may be dropped.",
            max_chars, total_len,
        )

    return block


def build_data_block(
    pipeline_result: dict[str, Any],
    max_chars: int = 20_000,
) -> str:
    """Serialise data pipeline snapshot into a structured text block.

    Sections: source staleness, Open-Meteo/SMAP telemetry, FIRMS hotspots
    (with spatial detail), FEMA NRI tracts.
    """
    parts: list[str] = []

    # Data completeness summary (from bridge)
    completeness = pipeline_result.get("data_completeness")
    if completeness:
        parts.append("## Data Completeness")
        for key, available in completeness.items():
            status_str = "available" if available else "NOT AVAILABLE"
            parts.append(f"- {key}: {status_str}")

    # Source staleness warnings (from orchestrator resilience)
    source_status = pipeline_result.get("source_status")
    if source_status:
        parts.append("\n## Data Source Status")
        for source_name, status in source_status.items():
            if isinstance(status, dict):
                state = status.get("status", "UNKNOWN")
                detail = status.get("detail", "")
                if state in ("STALE", "UNAVAILABLE"):
                    parts.append(f"- [{state}] {source_name}: {detail}")
                else:
                    parts.append(f"- [OK] {source_name}")
            elif status in ("STALE", "UNAVAILABLE"):
                parts.append(f"- [{status}] {source_name}")
            else:
                parts.append(f"- [OK] {source_name}")
    else:
        parts.append("\n## Data Source Status: unavailable (no freshness info provided)")

    # Telemetry — structured for WeatherObservation fields
    telem = pipeline_result.get("telemetry")
    if telem:
        parts.append("\n## Environmental Telemetry (use for weather_observations field)")
        for k, v in telem.items():
            parts.append(f"- {k}: {v}")
        parts.append(
            "Map these to weather_observations: temperature_f, "
            "relative_humidity_pct, wind_speed_mph, wind_direction, fuel_moisture_1hr"
        )

    # FIRMS — count + spatial detail
    firms_count = pipeline_result.get("firms_hotspot_count", 0)
    parts.append(f"\n## FIRMS Hotspots (last 6 hours): {firms_count}")

    firms_hotspots = pipeline_result.get("firms_hotspots") or []
    if firms_hotspots:
        # Sort by FRP descending, take top 20
        sorted_hs = sorted(firms_hotspots, key=lambda h: float(h.get("frp", 0)), reverse=True)
        for h in sorted_hs[:20]:
            parts.append(
                f"- lat={h.get('lat', '?')}, lon={h.get('lon', '?')}, "
                f"FRP={h.get('frp', 'N/A')} MW, "
                f"confidence={h.get('confidence', 'N/A')}, "
                f"time={h.get('acq_datetime', h.get('acq_date', 'N/A'))}"
            )

    # FEMA NRI
    nri = pipeline_result.get("fema_nri_tracts") or []
    if nri:
        parts.append("\n## FEMA NRI Vulnerability Data")
        for tract in nri[:20]:
            parts.append(f"- {json.dumps(tract)}")
    else:
        parts.append("\n## FEMA NRI Vulnerability Data: none available")

    block = "\n".join(parts)
    return block[:max_chars] if len(block) > max_chars else block


def build_human_block(
    human_inputs: list[HumanInput],
    toggle: AdminToggle,
) -> str:
    """Format human/operator input for context injection.

    Returns empty string if toggle is OFF.
    """
    if not toggle.is_on:
        return ""

    if not human_inputs:
        return "No operator input provided for this report period."

    parts: list[str] = []
    for inp in human_inputs:
        header = f"[{inp.source.upper()} — {inp.submitted_at}]"
        parts.append(header)
        if inp.text_notes:
            parts.append(inp.text_notes)
        for uf in inp.uploaded_files:
            parts.append(f"[File: {uf.filename}]")
            if uf.mime_type.startswith("text/"):
                try:
                    parts.append(uf.content_bytes.decode("utf-8", errors="replace"))
                except Exception:
                    parts.append("[Could not decode file]")
        parts.append("")  # blank line separator

    return "\n".join(parts).strip()


def build_instruction(
    report_type: str,
    incident_id: str,
    datetime_str: str,
) -> str:
    """Return the final directive message for the LLM."""
    return (
        f"Generate a {report_type} report. "
        f"Current datetime: {datetime_str}. "
        f"Incident ID: {incident_id}. "
        "Return ONLY valid JSON matching the provided schema. "
        "Do not add markdown code fences or any text outside the JSON object."
    )


# ---------------------------------------------------------------------------
# Main assembler
# ---------------------------------------------------------------------------

def assemble(
    mode: OperationalMode,
    sub_state: EmergencySubState | None,
    pipeline_result: dict[str, Any],
    human_inputs: list[HumanInput],
    corpus_ref: str | None,
    corpus_text: str | None,
    toggle: AdminToggle,
    config: dict[str, Any],
    incident_id: str | None = None,
    uploaded_files: list[Any] | None = None,
) -> ContextBundle:
    """Orchestrate all builder functions and return a complete ContextBundle.

    Parameters
    ----------
    incident_id:
        If provided (e.g. from IncidentTracker), reuse this ID for continuity
        across sequential reports about the same fire. If None, falls back to
        ``pipeline_result["run_id"]`` or generates a new UUID.
    """
    from src.models.obj3_gemini.schemas import SCHEMA_MAP

    report_type = mode_to_report_type(mode, sub_state)
    schema_cls = SCHEMA_MAP[report_type]
    schema_dict = schema_cls.model_json_schema()

    if incident_id is None:
        incident_id = pipeline_result.get("run_id") or str(uuid.uuid4())
    dt_str = datetime.now(tz=UTC).isoformat()

    reporting_cfg = config.get("reporting", {})
    backend = config.get("llm_backend", "")
    if backend == "ollama":
        max_ml = reporting_cfg.get("ollama_max_ml_block_chars", 6_000)
        max_data = reporting_cfg.get("ollama_max_data_block_chars", 6_000)
    else:
        max_ml = reporting_cfg.get("max_ml_block_chars", 20_000)
        max_data = reporting_cfg.get("max_data_block_chars", 20_000)

    system_prompt = build_system_prompt(report_type, schema_dict)
    ml_block = build_ml_block(pipeline_result, max_chars=max_ml)
    data_block = build_data_block(pipeline_result, max_chars=max_data)
    human_block = build_human_block(human_inputs, toggle)
    instruction = build_instruction(report_type, incident_id, dt_str)

    # Token estimation — warn when approaching context limits (especially Ollama 32K)
    total_chars = (
        len(system_prompt) + len(ml_block) + len(data_block)
        + len(human_block) + len(instruction) + len(corpus_text or "")
    )
    estimated_tokens = total_chars // 4
    warn_threshold = reporting_cfg.get("max_context_tokens_warn", 28_000)
    if estimated_tokens > warn_threshold:
        logger.warning(
            "Estimated context tokens (%d) exceeds threshold (%d). "
            "LLM may truncate or degrade quality. Consider reducing corpus "
            "or context block sizes.",
            estimated_tokens, warn_threshold,
        )

    return ContextBundle(
        system_prompt=system_prompt,
        corpus_ref=corpus_ref,
        corpus_text=corpus_text,
        ml_block=ml_block,
        data_block=data_block,
        human_block=human_block,
        instruction=instruction,
        report_type=report_type,
        incident_id=incident_id,
        uploaded_files=uploaded_files or [],
    )
