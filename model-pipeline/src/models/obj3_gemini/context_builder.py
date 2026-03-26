"""Context builder — assembles multi-source context for each LLM report call.

This is the *only* place where input sources are combined before being sent
to the adapter.  No LLM calls happen here.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from src.models.obj3_gemini.state_machine import (
    AdminToggle,
    EmergencySubState,
    OperationalMode,
    mode_to_report_type,
)

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
        "wildfire analysis. You generate structured reports based on real-time "
        "ML pipeline outputs, environmental data, and official emergency "
        "management doctrine.\n\n"
        "RULES:\n"
        "1. Output ONLY valid JSON matching the schema below.\n"
        "2. Do NOT hallucinate data — if a value is unknown, use null for "
        "optional fields.\n"
        "3. Every report MUST include the disclaimer: "
        '"AI-generated. Not for operational use without human review."\n'
        "4. Set human_review_required=true if report_confidence < 0.70.\n"
        "5. Do NOT add markdown code fences or text outside the JSON object.\n\n"
        f"REPORT TYPE: {report_type}\n\n"
        f"RESPONSE SCHEMA:\n{schema_str}"
    )


def build_ml_block(
    pipeline_result: dict[str, Any],
    max_chars: int = 20_000,
) -> str:
    """Serialise ML pipeline outputs into a structured text block.

    Sections: XGBoost top cells, Cell2Fire GeoJSON, Propagator summary,
    bias gate result.
    """
    parts: list[str] = []

    # XGBoost scores
    top_cells = pipeline_result.get("xgboost_top_cells") or []
    if top_cells:
        parts.append("## XGBoost Top Risk Cells")
        for cell in top_cells[:20]:
            parts.append(
                f"- H3: {cell.get('h3_index')}  "
                f"P={cell.get('probability', 'N/A')}  "
                f"({cell.get('lat', '?')}, {cell.get('lon', '?')})"
            )

    # Cell2Fire
    c2f = pipeline_result.get("cell2fire_geojson")
    if c2f:
        parts.append("\n## Cell2Fire Spread Model (top-10)")
        if isinstance(c2f, list):
            for feat in c2f[:10]:
                parts.append(f"- {json.dumps(feat)}")
        else:
            parts.append(str(c2f)[:2000])

    # Propagator
    prop = pipeline_result.get("propagator_summary")
    if prop:
        parts.append("\n## Propagator Summary (secondary comparison)")
        parts.append(str(prop)[:2000])

    # Bias gate
    bias = pipeline_result.get("bias_report")
    if bias:
        parts.append("\n## Bias Gate Result")
        parts.append(f"- Gate: {bias.get('gate_result', 'N/A')}")
        parts.append(f"- Observed disparity: {bias.get('observed_disparity', 'N/A')}")

    block = "\n".join(parts)
    return block[:max_chars] if len(block) > max_chars else block


def build_data_block(
    pipeline_result: dict[str, Any],
    max_chars: int = 20_000,
) -> str:
    """Serialise data pipeline snapshot into a structured text block.

    Sections: OWM/SMAP telemetry, FIRMS hotspots, FEMA NRI tracts.
    """
    parts: list[str] = []

    # Telemetry
    telem = pipeline_result.get("telemetry")
    if telem:
        parts.append("## Environmental Telemetry")
        for k, v in telem.items():
            parts.append(f"- {k}: {v}")

    # FIRMS
    firms = pipeline_result.get("firms_hotspot_count", 0)
    parts.append(f"\n## FIRMS Hotspots (last 6 hours): {firms}")

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
) -> ContextBundle:
    """Orchestrate all builder functions and return a complete ContextBundle."""
    from src.models.obj3_gemini.schemas import SCHEMA_MAP

    report_type = mode_to_report_type(mode, sub_state)
    schema_cls = SCHEMA_MAP[report_type]
    schema_dict = schema_cls.model_json_schema()

    incident_id = pipeline_result.get("run_id") or str(uuid.uuid4())
    dt_str = datetime.now(tz=UTC).isoformat()

    reporting_cfg = config.get("reporting", {})
    max_ml = reporting_cfg.get("max_ml_block_chars", 20_000)
    max_data = reporting_cfg.get("max_data_block_chars", 20_000)

    return ContextBundle(
        system_prompt=build_system_prompt(report_type, schema_dict),
        corpus_ref=corpus_ref,
        corpus_text=corpus_text,
        ml_block=build_ml_block(pipeline_result, max_chars=max_ml),
        data_block=build_data_block(pipeline_result, max_chars=max_data),
        human_block=build_human_block(human_inputs, toggle),
        instruction=build_instruction(report_type, incident_id, dt_str),
        report_type=report_type,
        incident_id=incident_id,
    )
