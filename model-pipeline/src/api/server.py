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
    # Operator input
    operator_notes: str | None = Form(None),
    # Settings
    report_type_override: str = Form("auto"),
    backend_override: str | None = Form(None),
    # Files
    files: list[UploadFile] = File(default=[]),
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

    pipeline_result: dict[str, Any] = {
        "run_id": f"dashboard-{datetime.now(tz=UTC).strftime('%Y%m%d-%H%M%S')}",
        "is_deployable": True,
        "risk_level": risk_level.upper(),
        "firms_hotspot_count": firms_hotspot_count,
        "firms_hotspots": [],
        "xgboost_top_cells": xgboost_top_cells,
        "cell2fire_geojson": cell2fire_summary,
        "propagator_summary": propagator_summary,
        "telemetry": telemetry or None,
        "fema_nri_tracts": [],
        "bias_report": {"gate_result": "PASS", "observed_disparity": 0.0},
        "metrics": {},
        "source_status": {
            "FIRMS": {"status": "OK", "detail": "Operator-supplied"},
            "OWM": {"status": "OK", "detail": "Operator-supplied"},
            "SMAP": {"status": "UNAVAILABLE", "detail": "Not provided"},
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
        raise HTTPException(status_code=500, detail=str(exc))

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
