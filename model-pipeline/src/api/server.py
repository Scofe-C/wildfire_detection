"""FastAPI server — OBJ-3 operator dashboard backend.

Endpoints
---------
GET  /                    → index.html (report list)
GET  /generate            → generate.html (input form)
GET  /static/{path}       → static assets
POST /api/generate        → generate a report (multipart form + files)
GET  /api/reports         → list all saved reports (JSON)
GET  /api/reports/{id}    → read one report JSON
GET  /api/report-file     → serve a report HTML/MD file from disk
GET  /api/status          → system status (backends, corpus)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OBJ-3 Wildfire Dashboard",
    description="Operator console for AI-powered wildfire disaster reports",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
_DASHBOARD = _ROOT / "dashboard"
_STATIC = _DASHBOARD / "static"
_STATIC.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

# ---------------------------------------------------------------------------
# Reporter singleton — loaded once at startup
# ---------------------------------------------------------------------------

_reporter: Any = None
_config_path = _ROOT / "configs" / "reporting_config.yaml"


@app.on_event("startup")
async def startup_event() -> None:
    global _reporter
    try:
        from src.models.obj3_gemini.reporter import GeminiDisasterReporter  # noqa
        _reporter = GeminiDisasterReporter()
        _reporter.load_model(_config_path)
        logger.info("Reporter loaded successfully")
    except Exception as exc:
        logger.error("Reporter failed to load: %s", exc)
        _reporter = None


# ---------------------------------------------------------------------------
# HTML routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    path = _DASHBOARD / "index.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/generate", response_class=HTMLResponse)
async def generate_page() -> HTMLResponse:
    path = _DASHBOARD / "generate.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="generate.html not found")
    return HTMLResponse(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# API — status
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def status() -> JSONResponse:
    import yaml  # noqa

    cfg: dict[str, Any] = {}
    try:
        with open(_config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        pass

    backend = cfg.get("llm_backend", "unknown")

    # Check Ollama
    ollama_ok = False
    ollama_model = cfg.get("ollama", {}).get("model", "?")
    if backend == "ollama":
        try:
            import ollama as ollama_lib  # noqa
            client = ollama_lib.Client(host=cfg.get("ollama", {}).get("base_url", "http://localhost:11434"))
            available = [m.model for m in client.list().models]
            ollama_ok = any(
                name == ollama_model or name == f"{ollama_model}:latest"
                for name in available
            )
        except Exception:
            ollama_ok = False

    # Check Gemini key
    gemini_ok = bool(os.getenv("GEMINI_API_KEY"))

    # Corpus
    corpus_dir = _ROOT / cfg.get("corpus", {}).get("local_dir", "corpus/")
    corpus_count = len(list(corpus_dir.glob("processed/*.json"))) if corpus_dir.exists() else 0

    return JSONResponse({
        "backend": backend,
        "reporter_loaded": _reporter is not None,
        "ollama": {"available": ollama_ok, "model": ollama_model} if backend == "ollama" else None,
        "gemini": {"api_key_set": gemini_ok} if backend != "ollama" else None,
        "corpus_chunks": corpus_count,
        "timestamp": datetime.now(tz=UTC).isoformat(),
    })


# ---------------------------------------------------------------------------
# API — list reports
# ---------------------------------------------------------------------------

@app.get("/api/reports")
async def list_reports(limit: int = 50) -> JSONResponse:
    reports_dir = _ROOT / "reports" / "disaster_reports"
    if not reports_dir.exists():
        return JSONResponse([])

    entries = []
    for json_file in sorted(reports_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if "review_manifest" in json_file.name or "incident_state" in json_file.name:
            continue
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            # Find companion HTML/MD
            stem = json_file.stem
            parent = json_file.parent
            rendered = None
            for ext in (".html", ".md"):
                candidate = parent / (stem + ext)
                if candidate.exists():
                    # Use forward slashes — avoids URL encoding issues on Windows
                    rendered = candidate.relative_to(_ROOT).as_posix()
                    break
            entries.append({
                "id": stem,
                "report_type": data.get("report_type", "unknown"),
                "risk_level": data.get("risk_level", "?"),
                "incident_id": data.get("incident_id", "?"),
                "generated_at": data.get("generated_at", ""),
                "confidence": data.get("report_confidence"),
                "human_review_required": data.get("human_review_required", False),
                "review_status": data.get("review_status", "?"),
                "json_path": str(json_file.relative_to(_ROOT)),
                "rendered_path": rendered,
            })
        except Exception:
            continue
        if len(entries) >= limit:
            break

    return JSONResponse(entries)


@app.get("/api/reports/{report_id}")
async def get_report(report_id: str) -> JSONResponse:
    reports_dir = _ROOT / "reports" / "disaster_reports"
    matches = list(reports_dir.rglob(f"{report_id}.json"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    return JSONResponse(json.loads(matches[0].read_text(encoding="utf-8")))


@app.delete("/api/reports/{report_id}")
async def delete_report_endpoint(report_id: str) -> JSONResponse:
    """Delete a report and its companion files by ID."""
    from src.reports.report_manager import delete_report  # noqa

    reports_dir = _ROOT / "reports" / "disaster_reports"
    deleted = delete_report(report_id, reports_dir)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    return JSONResponse({
        "deleted": [str(p.relative_to(_ROOT)) for p in deleted],
        "count": len(deleted),
    })


@app.get("/api/report-file")
async def serve_report_file(path: str) -> FileResponse:
    """Serve a report HTML or MD file from disk.

    ``path`` is a relative path from the project root, e.g.
    ``reports/disaster_reports/incident/IncidentReport_20260330_0338.html``.
    Both forward and back slashes are accepted (Windows compat).
    """
    # Normalise separators so Windows backslashes don't break pathlib
    safe_path = path.replace("\\", "/")

    # Resolve to absolute and verify the file is inside reports/
    full = (_ROOT / safe_path).resolve()
    reports_root = (_ROOT / "reports").resolve()

    if not str(full).startswith(str(reports_root)):
        raise HTTPException(status_code=403, detail="Access denied")
    if not full.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {full}")

    media = "text/html" if full.suffix == ".html" else "text/plain"
    return FileResponse(str(full), media_type=media)


@app.get("/api/reports/{report_id}/render")
async def render_report_on_demand(report_id: str, format: str = "auto") -> HTMLResponse:
    """Render a saved JSON report to HTML or Markdown on demand.

    This enables JSON-only storage while still allowing readable report views.
    The rendered output is NOT saved to disk — it is computed on the fly.

    Parameters
    ----------
    report_id:
        Report filename stem (e.g. ``IncidentReport_20260330_0338``).
    format:
        ``"html"``, ``"md"``, or ``"auto"`` (picks based on report type).
    """
    from src.models.obj3_gemini.renderer import render_html, render_markdown, markdown_to_html  # noqa
    from src.models.obj3_gemini.schemas import SCHEMA_MAP  # noqa

    reports_dir = _ROOT / "reports" / "disaster_reports"
    matches = list(reports_dir.rglob(f"{report_id}.json"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")

    data = json.loads(matches[0].read_text(encoding="utf-8"))
    report_type = data.get("report_type", "daily")

    schema_cls = SCHEMA_MAP.get(report_type)
    if schema_cls is None:
        raise HTTPException(status_code=400, detail=f"Unknown report_type: {report_type}")

    try:
        parsed = schema_cls.model_validate(data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Schema validation failed: {exc}") from exc

    template_dir = _ROOT / "templates"

    # Determine output format
    if format == "auto":
        format = "html" if report_type in ("incident", "final") else "md"

    if format == "html":
        if report_type in ("incident", "final"):
            content = render_html(parsed, template_dir)
        else:
            md_content = render_markdown(parsed, template_dir)
            content = markdown_to_html(md_content)
        return HTMLResponse(content)
    else:
        md_content = render_markdown(parsed, template_dir)
        return HTMLResponse(f"<pre style='font-family:monospace;white-space:pre-wrap;padding:24px'>{md_content}</pre>")


# ---------------------------------------------------------------------------
# API — generate report
# ---------------------------------------------------------------------------

@app.post("/api/generate")
async def generate_report(
    # Situation data
    risk_level: str = Form("HIGH"),
    firms_hotspot_count: int = Form(0),
    temperature_max: float | None = Form(None),
    wind_speed_mph: float | None = Form(None),
    relative_humidity: float | None = Form(None),
    soil_moisture: float | None = Form(None),
    # ML outputs
    propagator_summary: str | None = Form(None),
    xgboost_cells_json: str | None = Form(None),
    cell2fire_summary: str | None = Form(None),
    obj2_simulation_json: str | None = Form(None),
    # Operator input
    operator_notes: str | None = Form(None),
    # Settings
    report_type_override: str = Form("auto"),
    backend_override: str | None = Form(None),
    # Files
    files: list[UploadFile] = File(default=[]),  # noqa: B008
) -> JSONResponse:
    """Generate a disaster report from operator-supplied data and files."""

    if _reporter is None:
        raise HTTPException(status_code=503, detail="Reporter not loaded — check server logs")

    # --- Build pipeline_result ---
    telemetry: dict[str, Any] = {}
    if temperature_max is not None:
        telemetry["temperature_max"] = temperature_max
    if wind_speed_mph is not None:
        telemetry["wind_speed_mph"] = wind_speed_mph
    if relative_humidity is not None:
        telemetry["relative_humidity"] = relative_humidity
    if soil_moisture is not None:
        telemetry["soil_moisture"] = soil_moisture

    xgboost_top_cells: list[dict] = []
    if xgboost_cells_json:
        try:
            parsed = json.loads(xgboost_cells_json)
            if isinstance(parsed, list):
                xgboost_top_cells = parsed
        except json.JSONDecodeError:
            pass

    obj2_sim: dict[str, Any] | None = None
    if obj2_simulation_json:
        import contextlib
        with contextlib.suppress(json.JSONDecodeError):
            obj2_sim = json.loads(obj2_simulation_json)

    pipeline_result: dict[str, Any] = {
        "run_id": f"dashboard-{datetime.now(tz=UTC).strftime('%Y%m%d-%H%M%S')}",
        "is_deployable": True,
        "risk_level": risk_level.upper(),
        "firms_hotspot_count": firms_hotspot_count,
        "firms_hotspots": [],
        "xgboost_top_cells": xgboost_top_cells,
        "cell2fire_geojson": cell2fire_summary,
        "obj2_simulation": obj2_sim,
        "propagator_summary": propagator_summary,
        "telemetry": telemetry or None,
        "fema_nri_tracts": [],
        "bias_report": None,          # Dashboard-generated reports have no bias evaluation
        "metrics": {},
        "source_status": None,         # Operator-supplied data has no staleness tracking
        "data_completeness": {
            "xgboost_predictions": bool(xgboost_top_cells),
            "obj2_simulation": obj2_sim is not None,
            "firms_hotspots": firms_hotspot_count > 0,
            "telemetry": bool(telemetry),
            "fema_nri": False,
            "bias_report": False,
            "source_status": False,
        },
    }

    # --- Process uploaded files ---
    from src.api.file_processor import process_files  # noqa

    # Determine active backend
    import yaml  # noqa
    try:
        with open(_config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        active_backend = backend_override or cfg.get("llm_backend", "ollama")
    except Exception:
        active_backend = "ollama"

    raw_files: list[tuple[str, bytes, str]] = []
    for uf in files:
        if uf.filename and uf.size and uf.size > 0:
            content = await uf.read()
            mime = uf.content_type or "application/octet-stream"
            raw_files.append((uf.filename, content, mime))

    processed_files = process_files(raw_files, active_backend)

    # --- Build HumanInput ---
    from src.models.obj3_gemini.context_builder import HumanInput, UploadedFile  # noqa

    human_inputs = []
    if operator_notes or raw_files:
        # Convert ProcessedFile → UploadedFile for HumanInput text injection
        uf_list: list[UploadedFile] = []
        for pf in processed_files:
            uf_list.append(UploadedFile(
                filename=pf.filename,
                content_bytes=pf.text_content.encode("utf-8"),
                mime_type="text/plain",  # already extracted
            ))
        human_inputs.append(HumanInput(
            text_notes=operator_notes or "",
            uploaded_files=uf_list,
            source="operator",
            submitted_at=datetime.now(tz=UTC).isoformat(),
        ))

    # --- Mode override ---
    from src.models.obj3_gemini.state_machine import (  # noqa
        EmergencySubState, OperationalMode, mode_to_report_type,
    )

    mode_arg = None
    sub_state_arg = None
    if report_type_override != "auto":
        _type_to_mode = {
            "daily": OperationalMode.QUIET,
            "high_risk": OperationalMode.ACTIVE,
            "incident": OperationalMode.EMERGENCY,
            "final": OperationalMode.EMERGENCY,
        }
        mode_arg = _type_to_mode.get(report_type_override)
        if report_type_override == "final":
            sub_state_arg = EmergencySubState.FINAL

    # --- Run in thread (blocking call) ---
    def _run() -> Any:
        return _reporter.generate_report(
            pipeline_result=pipeline_result,
            human_inputs=human_inputs,
            uploaded_files=processed_files,
            mode=mode_arg,
            sub_state=sub_state_arg,
        )

    try:
        result = await asyncio.to_thread(_run)
    except Exception as exc:
        logger.exception("generate_report failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    rr = result.report_result
    val = result.validation

    return JSONResponse({
        "success": rr.error is None,
        "error": rr.error,
        "report_type": rr.report_type,
        "incident_id": rr.incident_id,
        "latency_ms": rr.latency_ms,
        "validation": {
            "passed": val.passed,
            "schema_valid": val.schema_valid,
            "sections_ok": val.sections_complete,
            "confidence_ok": val.confidence_ok,
        },
        "confidence": rr.parsed_report.report_confidence if rr.parsed_report else None,
        "human_review_required": rr.parsed_report.human_review_required if rr.parsed_report else True,
        "json_path": str(result.json_path.relative_to(_ROOT)) if result.json_path else None,
        "rendered_path": str(
            (result.html_path or result.markdown_path).relative_to(_ROOT)
        ) if (result.html_path or result.markdown_path) else None,
        "files_processed": len(processed_files),
        "backend_used": active_backend,
    })


# ---------------------------------------------------------------------------
# API — operator re-run with local observations
# ---------------------------------------------------------------------------

@app.post("/api/rerun")
async def rerun_with_local_data(
    grid_id: str = Form(...),
    region: str = Form("california"),
    temperature_f: float | None = Form(None),
    wind_speed_mph: float | None = Form(None),
    relative_humidity: float | None = Form(None),
    soil_moisture: float | None = Form(None),
    fire_weather_index: float | None = Form(None),
    operator_notes: str | None = Form(None),
    backend_override: str | None = Form(None),
) -> JSONResponse:
    """Re-run OBJ-1 + OBJ-2 with operator-supplied local observations.

    Loads the latest pipeline data for the region from disk/GCS, replaces
    operator-overridden columns in the target grid cell, re-scores with the
    production model, then generates an OBJ-3 report with real predictions.
    """
    if _reporter is None:
        raise HTTPException(status_code=503, detail="Reporter not loaded — check server logs")

    import json as _json

    # Build override dict from non-None form fields
    overrides: dict[str, float] = {}
    for field_name, value in [
        ("temperature_f", temperature_f),
        ("wind_speed_mph", wind_speed_mph),
        ("relative_humidity", relative_humidity),
        ("soil_moisture", soil_moisture),
        ("fire_weather_index", fire_weather_index),
    ]:
        if value is not None:
            overrides[field_name] = value

    # Load latest pipeline data for the region
    import pandas as pd
    pipeline_data_path = _ROOT / "historical_data" / f"{region}_merged.parquet"
    if not pipeline_data_path.exists():
        # Fallback: look for any parquet with region name
        candidates = list(_ROOT.rglob(f"*{region}*.parquet"))
        if not candidates:
            raise HTTPException(
                status_code=404,
                detail=f"No pipeline data found for region '{region}'. Run inference first.",
            )
        pipeline_data_path = candidates[0]

    try:
        df = pd.read_parquet(pipeline_data_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load pipeline data: {e}") from e

    # Load production model metadata
    local_model_dir = _ROOT / "models" / "ignition"
    pointer = local_model_dir / f"latest_{region}.txt"
    if not pointer.exists():
        raise HTTPException(
            status_code=503,
            detail=f"No local model pointer for '{region}'. Run training first.",
        )
    model_dir = pointer.read_text().strip()
    try:
        meta = _json.loads((Path(model_dir) / "model_metadata.json").read_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model metadata: {e}") from e

    # Re-run with overrides
    def _run_rerun() -> Any:
        from src.pipeline.rerun_engine import RerunEngine

        engine = RerunEngine(model_path=model_dir, config=meta)
        df_overridden = engine.apply_overrides(df, grid_id=grid_id, overrides=overrides)
        predictions, input_df = engine.run_obj1(df_overridden)
        obj2_sim = engine.run_obj2(df_overridden, predictions)
        pipeline_result = engine.build_result(predictions, input_df, obj2_sim, firms=None)

        # Wire operator notes into HumanInput
        from src.models.obj3_gemini.context_builder import HumanInput

        human_inputs = []
        if operator_notes:
            override_summary = ", ".join(f"{k}={v}" for k, v in overrides.items())
            human_inputs.append(HumanInput(
                text_notes=(
                    f"Operator local observations applied to grid_id={grid_id}: "
                    f"{override_summary}. Notes: {operator_notes}"
                ),
                uploaded_files=[],
                source="operator",
                submitted_at=datetime.now(tz=UTC).isoformat(),
            ))

        return _reporter.generate_report(
            pipeline_result=pipeline_result,
            human_inputs=human_inputs,
            uploaded_files=[],
        )

    try:
        result = await asyncio.to_thread(_run_rerun)
    except Exception as exc:
        logger.exception("rerun failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    rr = result.report_result
    return JSONResponse({
        "success": rr.error is None,
        "error": rr.error,
        "report_type": rr.report_type,
        "grid_id": grid_id,
        "region": region,
        "overrides_applied": overrides,
        "json_path": str(result.json_path.relative_to(_ROOT)) if result.json_path else None,
        "rendered_path": str(
            (result.html_path or result.markdown_path).relative_to(_ROOT)
        ) if (result.html_path or result.markdown_path) else None,
    })
