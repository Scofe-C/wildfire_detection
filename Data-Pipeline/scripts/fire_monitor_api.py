"""Fire monitor control API — mode override, field telemetry submission, status.

Runs as a background thread inside fire_monitor.py (--with-api flag)
or standalone: ``uvicorn scripts.fire_monitor_api:app --port 8001``
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIELD_TELEMETRY_DIR = PROJECT_ROOT / "data" / "raw" / "field_telemetry"
DASHBOARD_PATH = Path(__file__).resolve().parent / "monitor_dashboard.html"

# Shared state reference — set by start_api_background()
_state: dict[str, Any] = {}

app = FastAPI(title="Fire Monitor Control API", version="1.0.0")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ModeOverride(BaseModel):
    mode: str  # quiet | active | emergency
    reason: str = ""


class FalseAlarm(BaseModel):
    reason: str = "false alarm confirmed"


class FieldTelemetryPayload(BaseModel):
    source_type: str = "drone"
    latitude: float
    longitude: float
    confidence: int = 80
    frp: float | None = None
    report_text: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    if DASHBOARD_PATH.exists():
        return HTMLResponse(DASHBOARD_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Fire Monitor</h1><p>Dashboard HTML not found.</p>")


@app.get("/status")
async def status() -> JSONResponse:
    return JSONResponse({
        "mode": _state.get("mode", "unknown"),
        "cycle_count": _state.get("cycle_count", 0),
        "last_cycle_at": _state.get("last_cycle_at"),
        "next_cycle_at": _state.get("next_cycle_at"),
        "fire_cells_detected": _state.get("fire_cells_detected", []),
        "no_fire_streak": _state.get("no_fire_streak", 0),
        "running": _state.get("running", False),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.post("/mode")
async def set_mode(override: ModeOverride) -> JSONResponse:
    if override.mode not in ("quiet", "active", "emergency"):
        raise HTTPException(400, f"Invalid mode: {override.mode}")

    old_mode = _state.get("mode", "unknown")
    _state["user_override"] = {
        "mode": override.mode,
        "reason": override.reason,
        "set_by": "api",
        "set_at": datetime.now(timezone.utc).isoformat(),
    }
    logger.info("Mode override queued: %s → %s (reason: %s)", old_mode, override.mode, override.reason)
    return JSONResponse({"previous_mode": old_mode, "new_mode": override.mode, "applied": "next_cycle"})


@app.post("/false-alarm")
async def false_alarm(body: FalseAlarm) -> JSONResponse:
    _state["user_override"] = {
        "mode": "quiet",
        "reason": f"False alarm: {body.reason}",
        "set_by": "api",
        "set_at": datetime.now(timezone.utc).isoformat(),
    }
    logger.info("False alarm declared: %s", body.reason)
    return JSONResponse({"mode": "quiet", "reason": body.reason, "applied": "next_cycle"})


@app.post("/field-telemetry")
async def submit_field_telemetry(payload: FieldTelemetryPayload) -> JSONResponse:
    FIELD_TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)

    observation = {
        "source_type": payload.source_type,
        "priority": 1,
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "confidence": payload.confidence,
        "frp": payload.frp,
        "report_text": payload.report_text,
        "spatial_trust_radius_km": 5.0,
    }

    filename = f"field_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{payload.source_type}.json"
    filepath = FIELD_TELEMETRY_DIR / filename
    filepath.write_text(json.dumps(observation, indent=2), encoding="utf-8")

    # Track in state
    _state.setdefault("field_telemetry_log", []).append({
        **observation,
        "filename": filename,
    })

    logger.info("Field telemetry saved: %s (%s, conf=%d)", filename, payload.source_type, payload.confidence)
    return JSONResponse({"saved": filename, "observation": observation})


@app.get("/field-telemetry")
async def list_field_telemetry() -> JSONResponse:
    return JSONResponse(_state.get("field_telemetry_log", []))


@app.get("/cycles")
async def list_cycles() -> JSONResponse:
    return JSONResponse(_state.get("cycle_history", []))


# ---------------------------------------------------------------------------
# Background startup
# ---------------------------------------------------------------------------

def start_api_background(port: int = 8001, state: dict | None = None) -> None:
    """Start the API server in a background daemon thread."""
    global _state
    if state is not None:
        _state = state

    import uvicorn

    def _run() -> None:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

    thread = threading.Thread(target=_run, daemon=True, name="fire-monitor-api")
    thread.start()
