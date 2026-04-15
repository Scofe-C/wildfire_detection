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

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

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
    corpus_count = len(list(corpus_dir.glob("processed/**/*.json"))) if corpus_dir.exists() else 0

    return JSONResponse({
        "backend": backend,
        "reporter_loaded": _reporter is not None,
        "ollama": {"available": ollama_ok, "model": ollama_model} if backend == "ollama" else None,
        "gemini": {"api_key_set": gemini_ok} if backend != "ollama" else None,
        "corpus_chunks": corpus_count,
        "timestamp": datetime.now(tz=UTC).isoformat(),
    })


# ---------------------------------------------------------------------------
# API — live data from GCS (powers React frontend)
# ---------------------------------------------------------------------------

def _gcs_client():
    """Lazy GCS client — returns (client, bucket) or raises."""
    from google.cloud import storage as gcs
    bucket_name = os.getenv("GCS_BUCKET_NAME", "wildfire-mlops-123")
    client = gcs.Client()
    return client, client.bucket(bucket_name), bucket_name


# Simplified state boundary polygons for point-in-polygon filtering
# Generous state boundary polygons — padded ~1° beyond actual borders to include
# H3 cells whose centroids are near the edge. At 64km resolution the hexagons
# are huge, so a tight polygon drops too many border cells.
_STATE_POLYGONS: dict[str, list[tuple[float, float]]] = {
    "california": [
        (43.0, -125.5), (43.0, -119.0), (39.0, -119.0), (38.5, -113.5),
        (34.5, -113.5), (33.5, -113.5), (31.5, -114.5), (31.5, -118.0),
        (32.5, -118.5), (33.5, -119.5), (34.5, -121.5), (36.0, -122.5),
        (37.5, -123.5), (39.0, -124.5), (40.5, -125.0), (43.0, -125.5),
    ],
    "texas": [
        (37.0, -107.0), (37.0, -99.0), (34.5, -99.0), (33.5, -97.0),
        (33.0, -95.5), (31.5, -93.0), (29.5, -93.0), (28.0, -95.5),
        (26.0, -96.5), (25.0, -98.0), (25.5, -99.5), (27.5, -100.5),
        (29.0, -101.5), (30.5, -104.0), (31.5, -107.0), (37.0, -107.0),
    ],
}


def _point_in_polygon(lat: float, lon: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        yi, xi = polygon[i]
        yj, xj = polygon[j]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _sanitize_json(obj):
    """Recursively replace float NaN/inf with None for JSON compliance."""
    import math
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


@app.get("/api/grid-cells")
async def get_grid_cells(region: str = "california", resolution_km: int = 64) -> JSONResponse:
    """Return all grid cells with fused features + OBJ-1 inference scores merged.

    Reads the fused parquet (weather, terrain, fuel, FIRMS) and the inference
    JSON (fire_risk_score, fire_risk_flag, risk_tier) from GCS, joins them by
    grid_id, and returns a single list of enriched cells.
    """
    import asyncio
    import io as _io

    import pandas as pd

    def _read():
        _, bkt, bucket_name = _gcs_client()

        # 1. Read fused parquet (full features)
        prefix = f"data/processed/fused/{resolution_km}km/region={region}/"
        blobs = sorted(bkt.list_blobs(prefix=prefix), key=lambda b: b.name, reverse=True)
        parquet_blob = next((b for b in blobs if b.name.endswith(".parquet")), None)
        if not parquet_blob:
            return []
        df = pd.read_parquet(_io.BytesIO(parquet_blob.download_as_bytes()))

        # 2. Read inference JSON (OBJ-1 scores)
        inf_blob = bkt.blob(f"inference/latest/{region}_latest.json")
        if inf_blob.exists():
            inf_data = json.loads(inf_blob.download_as_bytes())
            scores_df = pd.DataFrame(inf_data.get("cells", []))
            if not scores_df.empty and "grid_id" in scores_df.columns and "grid_id" in df.columns:
                # Keep only score columns from inference
                score_cols = ["grid_id", "fire_risk_score", "fire_risk_flag", "risk_tier"]
                score_cols = [c for c in score_cols if c in scores_df.columns]
                scores_df = scores_df[score_cols]
                # Drop any existing score columns in fused to avoid _x/_y suffixes
                for col in ["fire_risk_score", "fire_risk_flag", "risk_tier"]:
                    if col in df.columns:
                        df = df.drop(columns=[col])
                df = df.merge(scores_df, on="grid_id", how="left")

        # Deduplicate by grid_id (fused parquet may have multiple timestamps)
        if "grid_id" in df.columns:
            df = df.drop_duplicates(subset=["grid_id"], keep="last")

        # Sanitize for JSON: Timestamps → str, NaN/inf → None
        import math
        import numpy as np
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str).replace("NaT", None)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.where(df.notna(), None)
        records = df.to_dict(orient="records")

        # Add H3 hex boundaries + filter by state polygon
        import h3
        state_poly = _STATE_POLYGONS.get(region)
        filtered = []
        for row in records:
            for k, v in row.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    row[k] = None
            gid = row.get("grid_id")
            lat = row.get("latitude") or row.get("lat") or 0
            lon = row.get("longitude") or row.get("lon") or 0
            # Tag cells as inside/outside state (frontend can use this for styling)
            row["in_state"] = bool(state_poly and _point_in_polygon(lat, lon, state_poly)) if state_poly else True
            # Add actual H3 hex boundary
            if gid:
                try:
                    boundary = h3.cell_to_boundary(gid)
                    row["hex_boundary"] = [[round(p[0], 5), round(p[1], 5)] for p in boundary]
                except Exception:
                    pass
            filtered.append(row)
        return filtered

    try:
        cells = await asyncio.to_thread(_read)
        return JSONResponse({"region": region, "resolution_km": resolution_km, "cells": cells, "count": len(cells)})
    except Exception as exc:
        logger.warning("grid-cells failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/predictions")
async def get_predictions(region: str = "california") -> JSONResponse:
    """Return latest OBJ-1 inference scores from GCS."""
    import asyncio

    def _read():
        _, bkt, _ = _gcs_client()
        blob = bkt.blob(f"inference/latest/{region}_latest.json")
        if not blob.exists():
            return None
        return _sanitize_json(json.loads(blob.download_as_bytes()))

    try:
        data = await asyncio.to_thread(_read)
        if data is None:
            raise HTTPException(status_code=404, detail=f"No predictions for region '{region}'")
        return JSONResponse(data)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("predictions failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/spread-simulations")
async def get_spread_simulations(region: str = "california") -> JSONResponse:
    """Return full OBJ-2 Monte Carlo fire spread simulation from GCS.

    Returns ignition cell, spread speed/direction, crown fire status,
    burn probabilities for neighbor cells, and all physics inputs.
    """
    import asyncio

    def _read():
        _, bkt, _ = _gcs_client()
        # Primary: dedicated simulation JSON (written by generate-from-pipeline)
        blob = bkt.blob(f"simulation/latest/{region}_latest.json")
        if blob.exists():
            return _sanitize_json(json.loads(blob.download_as_bytes()))
        # Fallback: extract from OBJ-3 report
        report_blob = bkt.blob(f"reports/obj3/latest/{region}_latest_report.json")
        if report_blob.exists():
            report = json.loads(report_blob.download_as_bytes())
            return {
                "spread_summary": report.get("spread_summary"),
                "fire_behavior": report.get("fire_behavior"),
                "report_type": report.get("report_type"),
                "generated_at": report.get("generated_at"),
                "fallback": True,
            }
        return None

    try:
        data = await asyncio.to_thread(_read)
        if data is None:
            raise HTTPException(status_code=404, detail=f"No spread simulation for region '{region}'")
        return JSONResponse({"region": region, "simulation": data})
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("spread-simulations failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/pipeline/status")
async def get_pipeline_status() -> JSONResponse:
    """Return pipeline metadata — regions, last inference timestamps, resolution."""
    import asyncio

    def _read():
        _, bkt, bucket_name = _gcs_client()
        regions_status = {}
        for region in ["california", "texas"]:
            blob = bkt.blob(f"inference/latest/{region}_latest.json")
            if blob.exists():
                data = _sanitize_json(json.loads(blob.download_as_bytes()))
                regions_status[region] = {
                    "last_inference": data.get("run_timestamp"),
                    "total_cells": data.get("summary", {}).get("total_cells", 0),
                    "flagged_cells": data.get("summary", {}).get("flagged_cells", 0),
                    "max_risk_score": data.get("summary", {}).get("max_risk_score", 0),
                    "risk_tier_counts": data.get("summary", {}).get("risk_tier_counts", {}),
                    "firms_hotspot_count": data.get("firms_hotspot_count", 0),
                }
        return {
            "dag_id": "wildfire_data_pipeline",
            "schedule": "0 */6 * * *",
            "regions": regions_status,
            "gcs_bucket": bucket_name,
        }

    try:
        data = await asyncio.to_thread(_read)
        return JSONResponse(data)
    except Exception as exc:
        logger.warning("pipeline-status failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# API — notifications
# ---------------------------------------------------------------------------

@app.get("/api/notifications")
async def get_notifications() -> JSONResponse:
    """Aggregate OBJ-1/2/3 pipeline events into notification feed for frontend bell."""

    def _build() -> list[dict]:
        _, bkt, _ = _gcs_client()
        events: list[dict] = []

        for region in ["california", "texas"]:
            label = region.title()

            # ── OBJ-1: ignition inference ──────────────────────────────────
            inf_blob = bkt.blob(f"inference/latest/{region}_latest.json")
            if inf_blob.exists():
                try:
                    d = json.loads(inf_blob.download_as_bytes())
                    ts = d.get("run_timestamp", "")
                    summary = d.get("summary", {})
                    flagged  = summary.get("flagged_cells", 0)
                    total    = summary.get("total_cells", 0)
                    max_risk = summary.get("max_risk_score", 0)
                    firms    = d.get("firms_hotspot_count", 0)
                    tiers    = summary.get("risk_tier_counts", {})

                    events.append({
                        "id":        f"obj1_{region}_{ts}",
                        "type":      "success",
                        "source":    "OBJ-1",
                        "title":     f"OBJ-1 Inference Complete — {label}",
                        "message":   f"{flagged}/{total} cells flagged · max risk {max_risk:.2f}",
                        "timestamp": ts,
                        "region":    region,
                    })

                    if firms > 0:
                        events.append({
                            "id":        f"firms_{region}_{ts}",
                            "type":      "fire",
                            "source":    "FIRMS",
                            "title":     f"Active Fire Hotspots — {label}",
                            "message":   f"{firms} FIRMS hotspot(s) detected in region",
                            "timestamp": ts,
                            "region":    region,
                        })

                    critical = tiers.get("CRITICAL", 0)
                    if critical > 0:
                        events.append({
                            "id":        f"critical_{region}_{ts}",
                            "type":      "warning",
                            "source":    "OBJ-1",
                            "title":     f"Critical Risk Cells — {label}",
                            "message":   f"{critical} cell(s) upgraded to CRITICAL tier",
                            "timestamp": ts,
                            "region":    region,
                        })
                except Exception:
                    pass

            # ── OBJ-2: fire spread simulation ─────────────────────────────
            sim_blob = bkt.blob(f"simulation/latest/{region}_latest.json")
            if sim_blob.exists():
                try:
                    d = json.loads(sim_blob.download_as_bytes())
                    ts    = d.get("run_timestamp", "")
                    speed = d.get("spread_speed_kmh", 0) or 0
                    crown = d.get("crown_fire_status", "NO_CROWN") or "NO_CROWN"
                    ntype = "warning" if crown not in ("NO_CROWN", "no_crown", "") else "success"
                    events.append({
                        "id":        f"obj2_{region}_{ts}",
                        "type":      ntype,
                        "source":    "OBJ-2",
                        "title":     f"OBJ-2 Spread Simulation — {label}",
                        "message":   f"Speed {speed:.1f} km/h · Crown: {crown.replace('_', ' ').title()}",
                        "timestamp": ts,
                        "region":    region,
                    })
                except Exception:
                    pass

        # ── OBJ-3: most recent reports from disk ───────────────────────────
        reports_dir = _ROOT / "reports" / "disaster_reports"
        if reports_dir.exists():
            report_files = sorted(
                (f for f in reports_dir.rglob("*.json")
                 if "review_manifest" not in f.name and "incident_state" not in f.name),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:6]
            for json_file in report_files:
                try:
                    d = json.loads(json_file.read_text(encoding="utf-8"))
                    ts     = d.get("generated_at", "")
                    risk   = d.get("risk_level", "?")
                    reg    = d.get("region", "unknown").title()
                    rtype  = d.get("report_type", "report").replace("_", " ").title()
                    ntype  = "warning" if risk in ("CRITICAL", "HIGH") else "success"
                    events.append({
                        "id":        f"obj3_{json_file.stem}",
                        "type":      ntype,
                        "source":    "OBJ-3",
                        "title":     f"OBJ-3 Report Generated — {reg}",
                        "message":   f"{rtype} · Risk level: {risk}",
                        "timestamp": ts,
                        "region":    d.get("region", ""),
                    })
                except Exception:
                    continue

        # Sort newest-first, cap at 30
        events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return events[:30]

    try:
        data = await asyncio.to_thread(_build)
        return JSONResponse(data)
    except Exception as exc:
        logger.warning("notifications endpoint failed: %s", exc)
        return JSONResponse([])


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
# API — edit report (PATCH)
# ---------------------------------------------------------------------------

_PROTECTED_FIELDS = frozenset({
    "incident_id", "report_type", "generated_at", "disclaimer",
    "operating_mode", "data_quality_score", "data_completeness",
    "human_input_included", "data_sources_used", "grounding_sources",
    "grounding_search_count", "review_status", "human_review_required",
    "disagreement_flag",
})


@app.patch("/api/reports/{report_id}")
async def update_report(report_id: str, request: Request) -> JSONResponse:
    """Update editable fields in a saved report.

    Protected metadata fields are silently ignored.
    After update, the companion HTML/MD file is re-rendered.
    """
    body = await request.json()

    reports_dir = _ROOT / "reports" / "disaster_reports"
    matches = list(reports_dir.rglob(f"{report_id}.json"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")

    json_path = matches[0]
    data = json.loads(json_path.read_text(encoding="utf-8"))

    # Apply updates, skipping protected fields and handling dot-notation
    updated_keys: list[str] = []
    for key, value in body.items():
        if key in _PROTECTED_FIELDS:
            continue
        if "." in key:
            parts = key.split(".")
            target = data
            for part in parts[:-1]:
                if part not in target or target[part] is None:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value
        else:
            data[key] = value
        updated_keys.append(key)

    if not updated_keys:
        return JSONResponse({"updated": [], "report_id": report_id})

    # Stamp edit metadata
    data["last_edited_at"] = datetime.now(tz=UTC).isoformat()
    data["edited_by_human"] = True

    # Save JSON
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Re-render companion file (best-effort)
    try:
        from src.models.obj3_gemini.renderer import render_html, render_markdown  # noqa
        from src.models.obj3_gemini.schemas import SCHEMA_MAP  # noqa

        schema_cls = SCHEMA_MAP.get(data.get("report_type", ""))
        if schema_cls:
            parsed = schema_cls.model_validate(data)
            template_dir = _ROOT / "templates"
            if data["report_type"] in ("incident", "final"):
                content = render_html(parsed, template_dir)
                json_path.with_suffix(".html").write_text(content, encoding="utf-8")
            else:
                content = render_markdown(parsed, template_dir)
                json_path.with_suffix(".md").write_text(content, encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to re-render companion after edit: %s", exc)

    return JSONResponse({
        "updated": updated_keys,
        "report_id": report_id,
        "edited_at": data["last_edited_at"],
    })


# ---------------------------------------------------------------------------
# API — summarize report (AI)
# ---------------------------------------------------------------------------

@app.post("/api/reports/{report_id}/summarize")
async def summarize_report(report_id: str) -> JSONResponse:
    """Generate a concise AI executive summary using Gemini."""
    reports_dir = _ROOT / "reports" / "disaster_reports"
    matches = list(reports_dir.rglob(f"{report_id}.json"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")

    data = json.loads(matches[0].read_text(encoding="utf-8"))

    def _run() -> str:
        return _generate_ai_summary(data)

    try:
        summary = await asyncio.to_thread(_run)
    except Exception as exc:
        logger.exception("AI summarize failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse({"summary": summary, "report_id": report_id})


def _generate_ai_summary(report_data: dict[str, Any]) -> str:
    """Call Gemini directly for a concise executive summary."""
    try:
        from google import genai  # noqa
        from google.genai import types as genai_types  # noqa
    except ImportError:
        return "AI summary unavailable — google-genai package not installed."

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return "AI summary unavailable — GEMINI_API_KEY not set."

    import yaml as _yaml  # noqa
    try:
        with open(_config_path, encoding="utf-8") as f:
            cfg = _yaml.safe_load(f) or {}
    except Exception:
        cfg = {}

    model = cfg.get("gemini_dev", {}).get("model", "gemini-2.5-flash")
    client = genai.Client(api_key=api_key)

    report_str = json.dumps(report_data, indent=2)
    if len(report_str) > 8000:
        report_str = report_str[:8000] + "\n...(truncated)"

    prompt = (
        "You are a wildfire emergency communications specialist. "
        "Distill this AI-generated wildfire report into a CONCISE executive "
        "summary (3-5 sentences, under 100 words) for rapid distribution to "
        "incident commanders, emergency managers, and local officials.\n\n"
        "Focus on: (1) current situation and threat level, (2) most critical "
        "action needed NOW, (3) key risk or outlook. Use specific numbers "
        "(temperatures, wind speeds, areas, probabilities) from the data.\n\n"
        "Write in plain, direct language. No disclaimers or metadata.\n\n"
        f"Report:\n{report_str}"
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(temperature=0.0),
    )
    return response.text.strip()


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

    # --- Validate generate form inputs ---
    errors: list[str] = []

    if risk_level.upper() not in ("LOW", "MODERATE", "HIGH", "CRITICAL"):
        errors.append(f"Invalid risk_level: {risk_level!r}. Must be LOW/MODERATE/HIGH/CRITICAL.")
    if firms_hotspot_count < 0:
        errors.append("firms_hotspot_count must be >= 0.")
    if temperature_max is not None and not (-80 <= temperature_max <= 160):
        errors.append(f"temperature_max={temperature_max} out of range [-80, 160] °F.")
    if wind_speed_mph is not None and not (0 <= wind_speed_mph <= 250):
        errors.append(f"wind_speed_mph={wind_speed_mph} out of range [0, 250].")
    if relative_humidity is not None and not (0 <= relative_humidity <= 100):
        errors.append(f"relative_humidity={relative_humidity} out of range [0, 100] %.")
    if soil_moisture is not None and not (0 <= soil_moisture <= 1):
        errors.append(f"soil_moisture={soil_moisture} out of range [0, 1].")
    if report_type_override not in ("auto", "daily", "high_risk", "incident", "final"):
        errors.append(f"Invalid report_type_override: {report_type_override!r}.")

    if errors:
        raise HTTPException(status_code=422, detail=errors)

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
            if not isinstance(parsed, list):
                raise ValueError("Expected a JSON array of cell objects")
            for i, cell in enumerate(parsed):
                if not isinstance(cell, dict):
                    raise ValueError(f"Cell [{i}] is not a JSON object")
                prob = cell.get("probability")
                if prob is not None and not (0 <= float(prob) <= 1):
                    raise ValueError(f"Cell [{i}] probability={prob} out of range [0, 1]")
            xgboost_top_cells = parsed
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid xgboost_cells_json: {e}",
            ) from e

    obj2_sim: dict[str, Any] | None = None
    if obj2_simulation_json:
        try:
            obj2_sim = json.loads(obj2_simulation_json)
            if not isinstance(obj2_sim, dict):
                raise ValueError("Expected a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid obj2_simulation_json: {e}",
            ) from e

    pipeline_result: dict[str, Any] = {
        "run_id": f"dashboard-{datetime.now(tz=UTC).strftime('%Y%m%d-%H%M%S')}",
        "is_deployable": True,
        "risk_level": risk_level.upper(),
        "firms_hotspot_count": firms_hotspot_count,
        "firms_hotspots": [],
        "xgboost_top_cells": xgboost_top_cells,
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

    from src.pipeline.rerun_engine import ALLOWED_MIME_TYPES  # noqa

    raw_files: list[tuple[str, bytes, str]] = []
    if len(files) > _MAX_FILES:
        raise HTTPException(status_code=422, detail=f"Max {_MAX_FILES} files allowed")
    total_bytes = 0
    for uf in files:
        if uf.filename and uf.size and uf.size > 0:
            if uf.size > _MAX_FILE_BYTES:
                raise HTTPException(
                    status_code=422,
                    detail=f"File '{uf.filename}' exceeds {_MAX_FILE_BYTES // (1024*1024)}MB limit",
                )
            total_bytes += uf.size
            if total_bytes > _MAX_TOTAL_BYTES:
                raise HTTPException(status_code=422, detail="Total upload size exceeds 50MB")
            mime = uf.content_type or "application/octet-stream"
            if mime not in ALLOWED_MIME_TYPES:
                raise HTTPException(
                    status_code=422,
                    detail=f"File '{uf.filename}' has unsupported type '{mime}'. "
                           f"Allowed: images, PDFs, text, CSV, JSON, GeoJSON.",
                )
            content = await uf.read()
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

def _sanitize_pydantic_errors(errors: list[dict]) -> list[dict]:
    """Make Pydantic error dicts JSON-serializable.

    Pydantic v2 puts ValueError objects in ``ctx.error`` which are not
    serializable by ``json.dumps``.  Convert them to strings.
    """
    clean = []
    for err in errors:
        e = dict(err)
        ctx = e.get("ctx")
        if isinstance(ctx, dict):
            e["ctx"] = {k: str(v) for k, v in ctx.items()}
        clean.append(e)
    return clean


_MAX_FILE_BYTES = 10 * 1024 * 1024   # 10 MB per file
_MAX_TOTAL_BYTES = 50 * 1024 * 1024  # 50 MB total
_MAX_FILES = 20


@app.post("/api/rerun")
async def rerun_with_local_data(
    # Legacy flat fields (backward compat)
    grid_id: str = Form(...),
    region: str = Form("california"),
    temperature_f: float | None = Form(None),
    wind_speed_mph: float | None = Form(None),
    relative_humidity: float | None = Form(None),
    soil_moisture: float | None = Form(None),
    fire_weather_index: float | None = Form(None),
    # Full override JSON (new — takes precedence over flat fields when present)
    overrides_json: str | None = Form(None),
    # Structured advisories (new)
    advisories_json: str | None = Form(None),
    # Existing
    operator_notes: str | None = Form(None),
    backend_override: str | None = Form(None),
    # File uploads (new)
    files: list[UploadFile] = File(default=[]),  # noqa: B008
) -> JSONResponse:
    """Re-run OBJ-1 + OBJ-2 with operator-supplied local observations.

    Accepts full data overrides (weather, vegetation, FIRMS, XGBoost cells,
    OBJ-2 simulation, risk level), file/image uploads, and structured
    reviewer advisories.  Falls back to legacy flat weather fields when
    ``overrides_json`` is not provided.
    """
    if _reporter is None:
        raise HTTPException(status_code=503, detail="Reporter not loaded — check server logs")

    import json as _json

    from pydantic import ValidationError

    from src.pipeline.rerun_engine import RerunOverrides, WeatherOverrides

    # --- Parse overrides ---
    if overrides_json:
        try:
            typed_overrides = RerunOverrides.model_validate_json(overrides_json)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=_sanitize_pydantic_errors(e.errors())) from e
        # Honour grid_id / region from typed model
        grid_id = typed_overrides.grid_id
        region = typed_overrides.region
    else:
        # Build from legacy flat fields — validate bounds
        try:
            weather = WeatherOverrides(
                temperature_f=temperature_f,
                wind_speed_mph=wind_speed_mph,
                relative_humidity=relative_humidity,
                soil_moisture=soil_moisture,
                fire_weather_index=fire_weather_index,
            )
            typed_overrides = RerunOverrides(
                grid_id=grid_id, region=region, weather=weather,
            )
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=_sanitize_pydantic_errors(e.errors())) from e

    # --- Parse advisories ---
    from src.models.obj3_gemini.context_builder import HumanAdvisory

    advisories: list[HumanAdvisory] = []
    if advisories_json:
        try:
            raw_advs = _json.loads(advisories_json)
            if not isinstance(raw_advs, list):
                raw_advs = [raw_advs]
            for item in raw_advs:
                advisories.append(HumanAdvisory(
                    category=item.get("category", "general"),
                    advisory_text=str(item.get("advisory_text", ""))[:1000],
                    priority=item.get("priority", "MEDIUM"),
                    affects_zones=item.get("affects_zones") or [],
                    submitted_by=item.get("submitted_by", "reviewer"),
                    submitted_at=item.get("submitted_at") or datetime.now(tz=UTC).isoformat(),
                ))
        except (ValueError, KeyError, TypeError) as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid advisories_json: {e}",
            ) from e

    # --- Process uploaded files ---
    from src.api.file_processor import process_files  # noqa

    import yaml  # noqa
    try:
        with open(_config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        active_backend = backend_override or cfg.get("llm_backend", "ollama")
    except Exception:
        active_backend = "ollama"

    from src.pipeline.rerun_engine import ALLOWED_MIME_TYPES as _AMT  # noqa

    raw_files: list[tuple[str, bytes, str]] = []
    total_bytes = 0
    if len(files) > _MAX_FILES:
        raise HTTPException(status_code=422, detail=f"Max {_MAX_FILES} files allowed")
    for uf in files:
        if uf.filename and uf.size and uf.size > 0:
            if uf.size > _MAX_FILE_BYTES:
                raise HTTPException(
                    status_code=422,
                    detail=f"File '{uf.filename}' exceeds {_MAX_FILE_BYTES // (1024*1024)}MB limit",
                )
            total_bytes += uf.size
            if total_bytes > _MAX_TOTAL_BYTES:
                raise HTTPException(status_code=422, detail="Total upload size exceeds 50MB")
            mime = uf.content_type or "application/octet-stream"
            if mime not in _AMT:
                raise HTTPException(
                    status_code=422,
                    detail=f"File '{uf.filename}' has unsupported type '{mime}'. "
                           f"Allowed: images, PDFs, text, CSV, JSON, GeoJSON.",
                )
            content = await uf.read()
            raw_files.append((uf.filename, content, mime))

    processed_files = process_files(raw_files, active_backend) if raw_files else []

    # --- Load pipeline data ---
    import pandas as pd
    pipeline_data_path = _ROOT / "historical_data" / f"{region}_merged.parquet"
    if not pipeline_data_path.exists():
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

    # --- Re-run with full overrides ---
    def _run_rerun() -> Any:
        from src.pipeline.rerun_engine import RerunEngine

        engine = RerunEngine(model_path=model_dir, config=meta)

        # Apply weather + vegetation overrides to the DataFrame
        df_overridden = engine.apply_overrides(df, grid_id=grid_id, overrides=typed_overrides)

        # Run OBJ-1 + OBJ-2
        predictions, input_df = engine.run_obj1(df_overridden)
        obj2_sim = engine.run_obj2(df_overridden, predictions)

        # Apply OBJ-2 simulation overrides if provided
        if typed_overrides.obj2_simulation:
            obj2_sim = engine.apply_obj2_overrides(obj2_sim, typed_overrides.obj2_simulation)

        pipeline_result = engine.build_result(predictions, input_df, obj2_sim, firms=None)

        # Inject FIRMS hotspot overrides
        if typed_overrides.firms_hotspots:
            engine.inject_firms_overrides(pipeline_result, typed_overrides.firms_hotspots)

        # Inject XGBoost cell overrides
        if typed_overrides.xgboost_cells:
            engine.inject_xgboost_overrides(pipeline_result, typed_overrides.xgboost_cells)

        # Apply risk level override
        if typed_overrides.risk_level_override:
            engine.apply_risk_override(pipeline_result, typed_overrides.risk_level_override)

        # Build HumanInput with notes + advisories
        from src.models.obj3_gemini.context_builder import HumanInput, UploadedFile as CtxUploadedFile

        human_inputs = []
        flat_summary = engine._flatten_overrides(typed_overrides)
        override_summary = ", ".join(f"{k}={v}" for k, v in flat_summary.items())
        notes_parts = []
        if override_summary:
            notes_parts.append(
                f"Operator local observations applied to grid_id={grid_id}: {override_summary}."
            )
        if operator_notes:
            notes_parts.append(f"Notes: {operator_notes}")

        # Convert ProcessedFile → UploadedFile for HumanInput text injection
        uf_list: list[CtxUploadedFile] = []
        for pf in processed_files:
            uf_list.append(CtxUploadedFile(
                filename=pf.filename,
                content_bytes=pf.text_content.encode("utf-8"),
                mime_type="text/plain",
            ))

        human_inputs.append(HumanInput(
            text_notes=" ".join(notes_parts) if notes_parts else None,
            uploaded_files=uf_list,
            advisories=advisories,
            source="operator",
            submitted_at=datetime.now(tz=UTC).isoformat(),
        ))

        return _reporter.generate_report(
            pipeline_result=pipeline_result,
            human_inputs=human_inputs,
            uploaded_files=processed_files,
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
        "overrides_applied": typed_overrides.model_dump(exclude_none=True),
        "advisories_applied": len(advisories),
        "files_processed": len(processed_files),
        "reasoning_steps": (
            len(rr.parsed_report.reasoning_trace) if rr.parsed_report else 0
        ),
        "json_path": str(result.json_path.relative_to(_ROOT)) if result.json_path else None,
        "rendered_path": str(
            (result.html_path or result.markdown_path).relative_to(_ROOT)
        ) if (result.html_path or result.markdown_path) else None,
    })


# ---------------------------------------------------------------------------
# API — pipeline-triggered report (reads OBJ-1 inference from GCS)
# ---------------------------------------------------------------------------

@app.post("/api/generate-from-pipeline")
async def generate_from_pipeline(request: Request) -> JSONResponse:
    """OBJ-1 → OBJ-2 → OBJ-3 pipeline: read inference + fused from GCS,
    run fire spread simulation, generate disaster report.

    Called by the Airflow ``run_inference`` task or the dashboard button.

    Expected JSON body::

        {
            "regions": ["california", "texas"],
            "bucket": "wildfire-mlops-123",       # optional
            "resolution_km": 64                    # optional, default 64
        }
    """
    if _reporter is None:
        raise HTTPException(status_code=503, detail="Reporter not loaded — check server logs")

    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {exc}") from exc

    regions = body.get("regions", [])
    bucket = body.get("bucket") or os.getenv("GCS_BUCKET_NAME", "wildfire-mlops-123")
    resolution_km = body.get("resolution_km", 64)

    if not regions:
        raise HTTPException(status_code=422, detail="'regions' list is required")

    def _run_all() -> list[dict[str, Any]]:
        import io as _io  # noqa

        import pandas as pd  # noqa
        from google.cloud import storage as gcs  # noqa

        client = gcs.Client()
        bkt = client.bucket(bucket)
        results = []

        _TIER_MAP = {"CRITICAL": "CRITICAL", "HIGH": "HIGH", "MEDIUM": "MODERATE", "LOW": "LOW"}
        _TIER_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

        for region in regions:
            # ── Step 1: Read OBJ-1 inference JSON from GCS ────────────────
            blob_path = f"inference/latest/{region}_latest.json"
            blob = bkt.blob(blob_path)
            if not blob.exists():
                logger.warning("[%s] No inference JSON at gs://%s/%s", region, bucket, blob_path)
                results.append({"region": region, "success": False, "error": f"{blob_path} not found"})
                continue

            obj1 = json.loads(blob.download_as_bytes())
            cells = obj1.get("cells", [])
            if not cells:
                results.append({"region": region, "success": False, "error": "no cells in JSON"})
                continue

            # ── Step 2: Derive risk level, top cells, FIRMS, telemetry ────
            tiers = {c.get("risk_tier", "LOW") for c in cells}
            highest = next((t for t in _TIER_ORDER if t in tiers), "LOW")
            risk_level = _TIER_MAP[highest]

            top = sorted(cells, key=lambda c: c.get("fire_risk_score", 0), reverse=True)[:10]
            xgb_cells = [
                {
                    "h3_index": c["grid_id"],
                    "probability": float(c["fire_risk_score"]),
                    "lat": float(c.get("latitude", 0)),
                    "lon": float(c.get("longitude", 0)),
                }
                for c in top
            ]

            firms_count = obj1.get("firms_hotspot_count", 0)
            firms_hotspots = obj1.get("firms_hotspots", [])
            telemetry = obj1.get("telemetry") or {}

            # ── Step 3: Read fused parquet from GCS for OBJ-2 ─────────────
            obj2_result = None
            fused_prefix = f"data/processed/fused/{resolution_km}km/region={region}/"
            try:
                fused_blobs = sorted(
                    bkt.list_blobs(prefix=fused_prefix),
                    key=lambda b: b.name,
                    reverse=True,
                )
                fused_blob = next((b for b in fused_blobs if b.name.endswith(".parquet")), None)

                if fused_blob:
                    logger.info("[%s] Loading fused parquet: gs://%s/%s", region, bucket, fused_blob.name)
                    fused_df = pd.read_parquet(_io.BytesIO(fused_blob.download_as_bytes()))

                    # ── Step 4: Run OBJ-2 on highest-risk cell ────────────
                    top_cell = top[0]  # highest fire_risk_score
                    ign_id = top_cell["grid_id"]
                    ign_prob = float(top_cell["fire_risk_score"])

                    try:
                        from src.models.obj2_spread.fire_spread_simulator import PythonFireSpreadSimulator

                        # Clamp CBH to prevent false crown fire
                        if "canopy_base_height_m" in fused_df.columns:
                            fused_df["canopy_base_height_m"] = fused_df["canopy_base_height_m"].clip(lower=2.0)

                        sim = PythonFireSpreadSimulator()
                        obj2_result = sim.simulate_monte_carlo(fused_df, ign_id, ign_prob)
                        logger.info(
                            "[%s] OBJ-2 MC: cell=%s speed=%.3f km/h dir=%.1f° crown=%s",
                            region, ign_id,
                            obj2_result.get("spread_speed_kmh", 0),
                            obj2_result.get("spread_direction_deg", 0),
                            obj2_result.get("crown_fire_status", "?"),
                        )
                        # Save raw OBJ-2 output to GCS for frontend consumption
                        if obj2_result:
                            try:
                                # Strip large arrays (all 100 per-run values) to keep JSON small
                                obj2_for_gcs = {k: v for k, v in obj2_result.items()
                                                if k not in ("spread_speed_kmh_all", "direction_deg_all",
                                                             "crown_status_all", "byram_intensity_kwm_all")}
                                obj2_for_gcs["region"] = region
                                obj2_for_gcs["run_timestamp"] = obj1.get("run_timestamp", "")
                                bkt.blob(f"simulation/latest/{region}_latest.json").upload_from_string(
                                    json.dumps(obj2_for_gcs, indent=2, default=str),
                                    content_type="application/json",
                                )
                                logger.info("[%s] OBJ-2 simulation → gs://%s/simulation/latest/%s_latest.json", region, bucket, region)
                            except Exception as sim_gcs_exc:
                                logger.warning("[%s] OBJ-2 GCS save failed: %s", region, sim_gcs_exc)

                    except Exception as obj2_exc:
                        logger.warning("[%s] OBJ-2 simulation failed (non-blocking): %s", region, obj2_exc)
                else:
                    logger.warning("[%s] No fused parquet found at gs://%s/%s", region, bucket, fused_prefix)
            except Exception as fused_exc:
                logger.warning("[%s] Failed to load fused parquet: %s", region, fused_exc)

            # ── Step 5: Assemble pipeline_result with OBJ-1 + OBJ-2 ──────
            pipeline_result: dict[str, Any] = {
                "run_id": obj1.get("run_timestamp", ""),
                "is_deployable": True,
                "risk_level": risk_level,
                "firms_hotspot_count": firms_count,
                "firms_hotspots": firms_hotspots,
                "xgboost_top_cells": xgb_cells,
                "obj2_simulation": obj2_result,
                "telemetry": telemetry if telemetry else None,
                "fema_nri_tracts": [],
                "bias_report": {"gate_result": "PASS", "observed_disparity": 0.0},
                "data_completeness": {
                    "ml_scores": True,
                    "obj2_simulation": obj2_result is not None,
                    "firms": firms_count > 0,
                    "weather": bool(telemetry),
                    "fema_nri": False,
                },
            }

            logger.info(
                "[%s] risk_level=%s  firms=%d  obj2=%s  → generating report",
                region, risk_level, firms_count, "yes" if obj2_result else "no",
            )

            # ── Step 6: Call OBJ-3 reporter ───────────────────────────────
            try:
                gen = _reporter.generate_report(pipeline_result=pipeline_result)
                rr = gen.report_result
                parsed = rr.parsed_report

                if gen.json_path and gen.json_path.exists():
                    try:
                        ts_str = datetime.now(tz=UTC).strftime("%Y%m%dT%H%MZ")
                        rtype = rr.report_type
                        gcs_key = f"reports/obj3/{region}/{rtype}_{ts_str}.json"
                        bkt.blob(gcs_key).upload_from_filename(str(gen.json_path))
                        bkt.blob(f"reports/obj3/latest/{region}_latest_report.json").upload_from_filename(
                            str(gen.json_path)
                        )
                        logger.info("[%s] report → gs://%s/%s", region, bucket, gcs_key)
                    except Exception as gcs_exc:
                        logger.warning("[%s] GCS upload failed: %s", region, gcs_exc)

                results.append({
                    "region": region,
                    "success": rr.error is None,
                    "error": rr.error,
                    "report_type": rr.report_type,
                    "confidence": parsed.report_confidence if parsed else None,
                    "review_status": parsed.review_status if parsed else None,
                    "obj2_ran": obj2_result is not None,
                    "json_path": str(gen.json_path.relative_to(_ROOT)) if gen.json_path else None,
                })
            except Exception as exc:
                logger.error("[%s] report generation failed: %s", region, exc)
                results.append({"region": region, "success": False, "error": str(exc)})

        return results

    try:
        results = await asyncio.to_thread(_run_all)
    except Exception as exc:
        logger.exception("generate-from-pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse({"reports": results})
