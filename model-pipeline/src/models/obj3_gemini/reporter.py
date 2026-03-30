"""GeminiDisasterReporter — main OBJ-3 orchestrator class.

Inherits from ``src.models.base.BaseModel`` but overrides the method
signatures to accept ``ContextBundle`` / ``ReportResult`` rather than
raw DataFrames (see implementation plan for rationale).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from src.models.base import BaseModel
from src.models.obj3_gemini.adapters.base_adapter import LLMAdapter, LLMGenerationError
from src.models.obj3_gemini.context_builder import (
    ContextBundle,
    HumanInput,
    assemble,
)
from src.models.obj3_gemini.corpus_loader import (
    get_corpus_as_text,
    load_corpus_texts,
)
from src.models.obj3_gemini.renderer import render_html, render_markdown
from src.models.obj3_gemini.schemas import SCHEMA_MAP
from src.models.obj3_gemini.schemas.base_schema import REQUIRED_DISCLAIMER, BaseReport
from src.models.obj3_gemini.state_machine import (
    AdminToggle,
    EmergencySubState,
    IncidentTracker,
    OperationalMode,
    mode_to_report_type,
    resolve_mode,
)
from src.reports.report_manager import save_report, sync_to_gcs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class ReportResult:
    """Output of a single predict() call."""

    raw_json: str
    parsed_report: BaseReport | None  # None if parsing failed
    report_type: str
    incident_id: str
    error: str | None = None          # None if successful
    latency_ms: float = 0.0


@dataclass
class ValidationResult:
    """Result of validate() checks."""

    schema_valid: bool = False
    sections_complete: bool = False
    confidence_ok: bool = False
    review_flag_correct: bool = False

    @property
    def passed(self) -> bool:
        return all([
            self.schema_valid,
            self.sections_complete,
            self.confidence_ok,
            self.review_flag_correct,
        ])


@dataclass
class GeneratedReport:
    """Full output of generate_report() — all artefacts."""

    report_result: ReportResult
    validation: ValidationResult
    markdown_path: Path | None = None
    html_path: Path | None = None
    json_path: Path | None = None
    gcs_paths: list[str] = field(default_factory=list)
    explain_output: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GeminiDisasterReporter(BaseModel):
    """OBJ-3 disaster reporting engine. Orchestrates state machine,
    context building, LLM generation, validation, rendering, and saving.
    """

    DISCLAIMER = REQUIRED_DISCLAIMER

    def __init__(self) -> None:
        super().__init__(model_name="gemini_disaster_reporter", version="1.0.0")
        self._config: dict[str, Any] = {}
        self._adapter: LLMAdapter | None = None
        self._toggle: AdminToggle | None = None
        self._incident_tracker: IncidentTracker | None = None
        self._corpus_text: str | None = None
        self._corpus_ref: str | None = None
        self._template_dir: Path | None = None
        self._output_dir: Path | None = None

    # -- BaseModel overrides ---------------------------------------------------

    def load_model(self, model_path: str | Path) -> None:  # type: ignore[override]
        """Load reporting config and initialise adapter + toggle.

        Parameters
        ----------
        model_path:
            Path to ``reporting_config.yaml``.
        """
        config_path = Path(model_path)
        with open(config_path, encoding="utf-8") as fh:
            self._config = yaml.safe_load(fh) or {}

        # Resolve paths relative to config file location
        base_dir = config_path.resolve().parent.parent  # model-pipeline/
        self._template_dir = base_dir / "templates"
        self._output_dir = base_dir / self._config.get(
            "reporting", {}
        ).get("output_dir", "reports/disaster_reports")

        # Instantiate adapter
        backend = self._config.get("llm_backend", "ollama")
        self._adapter = self._create_adapter(backend)

        # Health check
        if not self._adapter.is_available():
            raise RuntimeError(
                f"LLM backend '{backend}' is not available. "
                "Check that the service is running."
            )

        # Load corpus (best-effort — absence is not fatal for Phase 1)
        try:
            corpus_dir = base_dir / self._config.get("corpus", {}).get("local_dir", "corpus/")
            version = self._config.get("corpus", {}).get("version", "v1")
            corpus_docs = load_corpus_texts(corpus_dir, version)
            corpus_cfg = self._config.get("corpus", {})
            # Local models have limited effective context — use a much smaller cap
            if backend == "ollama":
                max_chars = corpus_cfg.get("ollama_max_corpus_chars", 20_000)
            else:
                max_chars = corpus_cfg.get("max_corpus_chars", 500_000)
            self._corpus_text = get_corpus_as_text(corpus_docs, max_chars)

            # Phase 3: load corpus into Vertex AI context cache
            if backend == "vertex_ai":
                try:
                    from src.models.obj3_gemini.adapters.vertex_adapter import VertexAdapter
                    if isinstance(self._adapter, VertexAdapter):
                        cache_name = self._adapter.load_corpus_cache(
                            corpus_docs=corpus_docs,
                            system_prompt="",  # System prompt injected per-call
                            ttl=self._config.get("vertex_ai", {}).get(
                                "corpus_cache_ttl_seconds", 3600
                            ),
                        )
                        if cache_name:
                            self._corpus_ref = cache_name
                            self._corpus_text = None  # Use cache instead of inline
                            logger.info("Corpus cached in Vertex AI: %s", cache_name)
                except Exception as exc:
                    logger.warning(
                        "Vertex AI corpus caching failed, falling back to inline: %s", exc
                    )
        except Exception as exc:
            logger.warning("Corpus loading skipped: %s", exc)
            self._corpus_text = None

        # Admin toggle
        toggle_cfg = dict(self._config.get("admin_toggle", {}))
        toggle_cfg["_config_path"] = config_path
        self._toggle = AdminToggle(toggle_cfg)

        # Incident tracker — persists EMERGENCY sub-state transitions
        incident_state_file = self._output_dir / ".incident_state.yaml"
        self._incident_tracker = IncidentTracker(
            state_file=incident_state_file, config=self._config,
        )

        self._is_loaded = True
        logger.info("GeminiDisasterReporter loaded (backend=%s)", backend)

    def predict(self, context_bundle: ContextBundle) -> ReportResult:  # type: ignore[override]
        """Send context bundle to the LLM and parse the response.

        Parameters
        ----------
        context_bundle:
            Assembled context from :func:`context_builder.assemble`.

        Returns
        -------
        ReportResult
        """
        if self._adapter is None:
            raise RuntimeError("Call load_model() before predict().")

        schema_cls = SCHEMA_MAP[context_bundle.report_type]
        schema_dict = schema_cls.model_json_schema()

        t0 = time.perf_counter()
        try:
            raw_json = self._adapter.generate(context_bundle, schema_dict)
        except LLMGenerationError as exc:
            latency = (time.perf_counter() - t0) * 1000
            logger.error("LLM generation failed: %s", exc)
            return ReportResult(
                raw_json="",
                parsed_report=None,
                report_type=context_bundle.report_type,
                incident_id=context_bundle.incident_id,
                error=str(exc),
                latency_ms=latency,
            )

        latency = (time.perf_counter() - t0) * 1000

        # Parse JSON → Pydantic
        try:
            raw_json = _preprocess_report_json(raw_json, context_bundle)
            parsed = schema_cls.model_validate_json(raw_json)
        except Exception as exc:
            logger.warning("JSON parse failed, retrying: %s", exc)
            # Retry once
            try:
                raw_json = self._adapter.generate(context_bundle, schema_dict)
                raw_json = _preprocess_report_json(raw_json, context_bundle)
                parsed = schema_cls.model_validate_json(raw_json)
            except Exception as retry_exc:
                return ReportResult(
                    raw_json=raw_json,
                    parsed_report=None,
                    report_type=context_bundle.report_type,
                    incident_id=context_bundle.incident_id,
                    error=f"Parse failed after retry: {retry_exc}",
                    latency_ms=latency,
                )

        return ReportResult(
            raw_json=raw_json,
            parsed_report=parsed,
            report_type=context_bundle.report_type,
            incident_id=context_bundle.incident_id,
            latency_ms=latency,
        )

    def validate(self, report_result: ReportResult, *, disagreement_flag: bool = False) -> ValidationResult:  # type: ignore[override]
        """Run all 4 validation criteria on a ReportResult.

        Does NOT raise — caller decides how to handle failures.
        """
        vr = ValidationResult()
        parsed = report_result.parsed_report

        # 1. Schema valid
        vr.schema_valid = parsed is not None and report_result.error is None

        if parsed is None:
            return vr

        # 2. Sections complete — check all required fields are non-null
        vr.sections_complete = self._check_sections(parsed)

        # 3. Confidence threshold
        threshold = self._config.get("reporting", {}).get("confidence_threshold", 0.70)
        vr.confidence_ok = parsed.report_confidence >= threshold

        # 4. Review flag correct (Option A — verify hrr matches deterministic computation)
        expected_hrr = _compute_human_review_required(parsed, disagreement_flag, self._config)
        vr.review_flag_correct = parsed.human_review_required == expected_hrr

        # Special rule: Final reports always require human review
        if report_result.report_type == "final":
            vr.review_flag_correct = parsed.human_review_required is True

        # Consistency: review_status must match human_review_required
        if parsed.review_status == "PENDING_REVIEW" and not parsed.human_review_required:
            vr.review_flag_correct = False
        if parsed.review_status == "AUTO_APPROVED" and parsed.human_review_required:
            vr.review_flag_correct = False

        return vr

    def explain(self, report_result: ReportResult) -> dict[str, Any]:  # type: ignore[override]
        """Return explanation metadata derived from the report — no LLM call."""
        parsed = report_result.parsed_report
        if parsed is None:
            return {"error": report_result.error}
        return {
            "confidence": parsed.report_confidence,
            "human_input_included": parsed.human_input_included,
            "data_sources_used": parsed.data_sources_used,
            "human_review_required": parsed.human_review_required,
            "review_status": parsed.review_status,
            "disagreement_flag": parsed.disagreement_flag,
            "grounding_search_count": parsed.grounding_search_count,
            "report_type": parsed.report_type,
            "latency_ms": report_result.latency_ms,
        }

    # -- High-level convenience method -----------------------------------------

    def generate_report(
        self,
        pipeline_result: dict[str, Any],
        human_inputs: list[HumanInput] | None = None,
        uploaded_files: list[Any] | None = None,
        mode: OperationalMode | None = None,
        sub_state: EmergencySubState | None = None,
    ) -> GeneratedReport:
        """Full report generation pipeline: resolve → build → generate →
        validate → render → save.

        Parameters
        ----------
        pipeline_result:
            Raw dict of ML pipeline outputs.
        human_inputs:
            Operator/management inputs (empty list if none).
        mode, sub_state:
            Override auto-resolution if provided.
        """
        if not self._is_loaded:
            raise RuntimeError("Call load_model() before generate_report().")

        human_inputs = human_inputs or []

        # 1–2: Resolve mode
        disagreement_flag = False
        if mode is None:
            mode, sub_state, disagreement_flag = resolve_mode(pipeline_result)

        # 2.5: For EMERGENCY mode, use IncidentTracker for sub-state + incident_id
        tracked_incident_id: str | None = None
        if mode == OperationalMode.EMERGENCY and self._incident_tracker is not None:
            tracked_incident_id, sub_state = self._incident_tracker.update(pipeline_result)

        report_type = mode_to_report_type(mode, sub_state)
        logger.info("Mode: %s/%s → report_type: %s", mode.value, sub_state, report_type)

        # 3–4: Build context
        context = assemble(
            mode=mode,
            sub_state=sub_state,
            pipeline_result=pipeline_result,
            human_inputs=human_inputs,
            corpus_ref=self._corpus_ref,
            corpus_text=self._corpus_text,
            toggle=self._toggle,
            config=self._config,
            incident_id=tracked_incident_id,
            uploaded_files=uploaded_files or [],
        )

        # 5–6: Generate + parse
        result = self.predict(context)

        # 6.5: Deterministic stamping (post-LLM, pre-validate)
        if result.parsed_report is not None:
            # Stamp disagreement_flag from state machine
            result.parsed_report.disagreement_flag = disagreement_flag

            # Compute human_review_required (OR logic across 3 independent triggers)
            hrr = _compute_human_review_required(
                result.parsed_report, disagreement_flag, self._config,
            )
            result.parsed_report.human_review_required = hrr

            # Stamp review_status
            result.parsed_report.review_status = (
                "PENDING_REVIEW" if hrr else "AUTO_APPROVED"
            )
            # Re-serialize so the saved JSON reflects deterministic stamps
            result.raw_json = result.parsed_report.model_dump_json(indent=2)

        # 7: Validate
        validation = self.validate(result, disagreement_flag=disagreement_flag)

        # 8: Render
        rendered_content = ""
        fmt = "md"
        if result.parsed_report is not None:
            if report_type in ("incident", "final"):
                rendered_content = render_html(result.parsed_report, self._template_dir)
                fmt = "html"
            else:
                rendered_content = render_markdown(result.parsed_report, self._template_dir)
                fmt = "md"

        # 9: Save
        now = datetime.now(tz=UTC)
        json_path: Path | None = None
        rendered_path: Path | None = None
        if result.raw_json and self._output_dir:
            json_path, rendered_path = save_report(
                report_json=result.raw_json,
                rendered_content=rendered_content,
                report_type=report_type,
                incident_id=result.incident_id,
                dt=now,
                fmt=fmt,
                output_dir=self._output_dir,
            )

        # 9.5: Append to review manifest if PENDING_REVIEW
        if (
            result.parsed_report is not None
            and result.parsed_report.review_status == "PENDING_REVIEW"
            and self._output_dir
        ):
            self._append_review_manifest(
                incident_id=result.incident_id,
                report_type=report_type,
                json_path=json_path,
                rendered_path=rendered_path,
                disagreement_flag=disagreement_flag,
                confidence=result.parsed_report.report_confidence,
                grounding_count=result.parsed_report.grounding_search_count,
                generated_at=now.isoformat(),
            )

        # 9.7: Update incident tracker (report count + archive on FINAL)
        if (
            tracked_incident_id
            and self._incident_tracker is not None
            and result.parsed_report is not None
        ):
            self._incident_tracker.increment_report_count(tracked_incident_id)
            if report_type == "final":
                self._incident_tracker.archive_incident(tracked_incident_id)

        # 10: GCS sync
        gcs_bucket = self._config.get("reporting", {}).get("gcs_bucket", "")
        gcs_paths: list[str] = []
        if gcs_bucket and json_path and rendered_path:
            gcs_paths = sync_to_gcs(
                [json_path, rendered_path],
                gcs_bucket,
                gcs_prefix=report_type + "/",
            )

        # 11: Return
        explain_output = self.explain(result)

        gen = GeneratedReport(
            report_result=result,
            validation=validation,
            json_path=json_path,
            explain_output=explain_output,
            gcs_paths=gcs_paths,
        )
        if fmt == "html":
            gen.html_path = rendered_path
        else:
            gen.markdown_path = rendered_path

        return gen

    # -- Private helpers -------------------------------------------------------

    def _create_adapter(self, backend: str) -> LLMAdapter:
        """Instantiate the correct adapter based on config."""
        if backend == "ollama":
            from src.models.obj3_gemini.adapters.ollama_adapter import OllamaAdapter
            return OllamaAdapter(self._config)
        elif backend == "gemini_dev":
            from src.models.obj3_gemini.adapters.gemini_dev_adapter import GeminiDevAdapter
            return GeminiDevAdapter(self._config)
        elif backend == "vertex_ai":
            from src.models.obj3_gemini.adapters.vertex_adapter import VertexAdapter
            return VertexAdapter(self._config)
        else:
            raise ValueError(f"Unknown LLM backend: {backend!r}")

    @staticmethod
    def _check_sections(report: BaseReport) -> bool:
        """Check that all required fields are non-empty and list minimums are met.

        Enforces Pydantic ``min_length`` constraints that are defined in the
        schema (e.g. ``immediate_actions >= 3``, ``preventive_recommendations >= 2``).
        """
        data = report.model_dump()
        for field_name, field_info in type(report).model_fields.items():
            if field_info.is_required():
                val = data.get(field_name)
                if val is None:
                    return False
                if isinstance(val, str) and not val.strip():
                    return False
                if isinstance(val, list):
                    # Enforce min_length from Pydantic Field metadata
                    min_len = _get_field_min_length(field_info)
                    if min_len is not None and len(val) < min_len:
                        return False
        return True

    def _append_review_manifest(self, **entry_kwargs: Any) -> None:
        """Append an entry to review_manifest.json (create if missing).

        Best-effort: a manifest write failure must NOT crash report generation.
        """
        import json as _json

        manifest_path = self._output_dir / "review_manifest.json"  # type: ignore[operator]
        try:
            if manifest_path.exists():
                with open(manifest_path,encoding="utf-8") as f:
                    manifest = _json.load(f)
            else:
                manifest = []

            # Convert Path objects to strings for JSON serialization
            entry = {
                k: str(v) if isinstance(v, Path) else v
                for k, v in entry_kwargs.items()
            }
            manifest.append(entry)

            with open(manifest_path, "w",encoding="utf-8") as f:
                _json.dump(manifest, f, indent=2)

            logger.info("Review manifest updated: %s entries", len(manifest))
        except Exception:
            logger.exception("Failed to update review manifest at %s", manifest_path)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _get_field_min_length(field_info: Any) -> int | None:
    """Extract ``min_length`` from a Pydantic FieldInfo, if set."""
    # Pydantic v2: min_length is stored in field_info.metadata as
    # annotated_types.MinLen, or directly on the FieldInfo object.
    min_len = getattr(field_info, "min_length", None)
    if min_len is not None:
        return int(min_len)
    # Check metadata list (Pydantic v2 annotated style)
    for meta in getattr(field_info, "metadata", []):
        if hasattr(meta, "min_length"):
            return int(meta.min_length)
    return None


def _preprocess_report_json(raw_json: str, context_bundle: ContextBundle) -> str:
    """Normalise and stamp deterministic fields before Pydantic validation.

    Local models (Qwen3:8b) commonly:
      - Skip metadata fields when context is large
      - Lowercase Literal enum values  (``"active"`` instead of ``"ACTIVE"``)
      - Paraphrase the disclaimer instead of copying it verbatim

    All three are fixed here — deterministically — so validation succeeds
    without requiring a retry.

    Fields stamped unconditionally (values are known before the LLM is called):
      ``disclaimer``, ``report_type``, ``incident_id``, ``generated_at``,
      ``operating_mode``, ``human_input_included``, ``data_sources_used``

    Fields normalised (uppercase):
      ``incident_status``, ``risk_level``, ``operating_mode``,
      ``review_status`` and ``priority`` in nested recommendation lists.
    """
    import json as _json  # noqa: PLC0415
    from datetime import UTC, datetime as _dt  # noqa: PLC0415

    _UPPERCASE_FIELDS = {
        "incident_status", "risk_level", "operating_mode", "review_status",
    }
    _RECOMMENDATION_LIST_FIELDS = (
        "recommendations", "preventive_recommendations", "resource_requirements",
    )

    try:
        data = _json.loads(raw_json)
        if not isinstance(data, dict):
            return raw_json
    except Exception:
        return raw_json

    # --- Stamp: disclaimer (fixed constant, never trust LLM) ---
    data["disclaimer"] = REQUIRED_DISCLAIMER

    # --- Stamp: metadata known at call time (always override — never trust LLM) ---
    data["report_type"] = context_bundle.report_type
    data["incident_id"] = context_bundle.incident_id
    # Always use real current time — model often hallucinates past dates
    data["generated_at"] = _dt.now(tz=UTC).isoformat()
    # Derive operating_mode from report_type — deterministic, no guessing
    _rt = context_bundle.report_type
    data["operating_mode"] = (
        "EMERGENCY" if _rt in ("incident", "final")
        else "ACTIVE" if _rt == "high_risk"
        else "QUIET"
    )
    data["human_input_included"] = bool(context_bundle.human_block.strip())
    data.setdefault("human_review_required", True)   # fail-safe; stamped again later

    # Sanitize data_sources_used — remove garbled/non-printable characters
    raw_sources = data.get("data_sources_used")
    if isinstance(raw_sources, list):
        cleaned_sources = []
        for s in raw_sources:
            if isinstance(s, str):
                # Keep only printable ASCII + common symbols; collapse whitespace
                sanitized = "".join(c for c in s if c.isprintable()).strip()
                if sanitized:
                    cleaned_sources.append(sanitized)
        data["data_sources_used"] = cleaned_sources or ["XGBoost", "FIRMS", "OWM", "FEMA NRI"]
    else:
        data["data_sources_used"] = ["XGBoost", "FIRMS", "OWM", "FEMA NRI"]

    # --- Normalise: uppercase all known enum fields ---
    for field in _UPPERCASE_FIELDS:
        if field in data and isinstance(data[field], str):
            data[field] = data[field].upper()

    # --- Normalise: uppercase priority in nested lists ---
    for list_field in _RECOMMENDATION_LIST_FIELDS:
        items = data.get(list_field) or []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and isinstance(item.get("priority"), str):
                    item["priority"] = item["priority"].upper()

    # --- Normalise: scores that must be 0.0–1.0 but model outputs 0–100 ---
    # vulnerability_score in vulnerable_populations
    for pop in data.get("vulnerable_populations") or []:
        if isinstance(pop, dict):
            vs = pop.get("vulnerability_score")
            if isinstance(vs, (int, float)) and vs > 1.0:
                pop["vulnerability_score"] = round(vs / 100.0, 4)
    # risk_score in top_risk_cells
    for cell in data.get("top_risk_cells") or []:
        if isinstance(cell, dict):
            rs = cell.get("risk_score")
            if isinstance(rs, (int, float)) and rs > 1.0:
                cell["risk_score"] = round(rs / 100.0, 4)
    # report_confidence itself
    rc = data.get("report_confidence")
    if isinstance(rc, (int, float)) and rc > 1.0:
        data["report_confidence"] = round(rc / 100.0, 4)

    return _json.dumps(data)


def _compute_human_review_required(
    report: BaseReport,
    disagreement_flag: bool,
    config: dict[str, Any],
) -> bool:
    """Deterministic OR logic — 3 independent triggers.

    1. report_confidence < threshold (default 0.7)
    2. grounding_search_count < min_grounding (default 3)
    3. disagreement_flag is True

    Returns True if ANY trigger fires.
    """
    threshold = config.get("reporting", {}).get("confidence_threshold", 0.70)
    min_grounding = config.get("reporting", {}).get("min_grounding_sources", 3)

    trigger_low_confidence = report.report_confidence < threshold
    trigger_low_grounding = report.grounding_search_count < min_grounding
    trigger_disagreement = disagreement_flag

    return trigger_low_confidence or trigger_low_grounding or trigger_disagreement
