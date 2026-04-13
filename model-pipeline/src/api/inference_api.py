"""Cloud Run inference API — health check, monitoring trigger, and prediction stub.

Endpoints
---------
GET  /health         → readiness probe (Cloud Run)
POST /predict        → STUB: returns mock predictions (see wiring instructions below)
POST /monitor        → real: triggers drift detection via monitor_runner

Wiring /predict to a real model
-------------------------------
The /predict endpoint currently returns mock MODERATE predictions for all cells.
To wire it to a real model:

1. **Model loading strategy** — decide one of:
   a. Load from GCS blob at startup (``model-artifacts/{run_id}/model.bst``)
   b. Load from Vertex AI Model Registry (``VertexRegistry.load_production()``)
   c. Load from local disk pointer (``models/ignition/latest_{region}.txt``)

2. **Startup handler** — add an ``@app.on_event("startup")`` that:
   - Loads the model for each active region (california, texas)
   - Loads the preprocessing medians from ``model_metadata.json``
   - Stores them in module-level dicts (``_models``, ``_medians``, ``_thresholds``)

3. **Predict handler** — replace the mock logic with:
   - Build a DataFrame from ``request.features`` + ``request.grid_ids``
   - Run ``full_pipeline()`` with stored medians
   - Call ``model.predict()`` to get probabilities
   - Apply decision threshold for binary flags and risk tiers

4. **Env var ``MODEL_SOURCE``** — set to ``"gcs"``, ``"vertex"``, or ``"local"``
   to select the loading strategy. Default: ``"stub"`` (current behavior).

See ``src/pipeline/rerun_engine.py`` lines 60-75 for the model loading pattern.
"""
from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Check if running in stub mode
_MODEL_SOURCE = os.getenv("MODEL_SOURCE", "stub")
if _MODEL_SOURCE == "stub":
    logger.warning(
        "MODEL_SOURCE=stub — /predict returns mock predictions. "
        "Set MODEL_SOURCE to 'gcs', 'vertex', or 'local' to serve real predictions."
    )

app = FastAPI(
    title="Wildfire Inference API",
    description="Cloud Run endpoint for fire risk scoring and drift monitoring",
    version="1.0.0",
)


# ── Request / response models ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    region: str
    grid_ids: list[str]
    features: dict[str, list[float]]  # feature_name → list of values (one per grid_id)


class PredictResponse(BaseModel):
    region: str
    predictions: list[dict[str, Any]]
    model_version: str
    timestamp: str


class MonitorRequest(BaseModel):
    baseline_run_id: str
    gcs_bucket: str | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> JSONResponse:
    """Cloud Run readiness probe."""
    return JSONResponse({"status": "ok", "timestamp": datetime.now(UTC).isoformat()})


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Fire risk scoring endpoint.

    TODO: Wire to real model loaded from Vertex AI when inference output
    location and Cloud Run serving strategy is decided by the team.
    Currently returns mock predictions for integration testing.
    """
    logger.info("/predict called — region=%s, n_cells=%d", request.region, len(request.grid_ids))

    # Stub: return mock MODERATE predictions for all cells
    mock_predictions = [
        {
            "grid_id": gid,
            "fire_risk_score": 0.35,
            "risk_tier": "MODERATE",
            "fire_risk_flag": 0,
        }
        for gid in request.grid_ids
    ]

    return PredictResponse(
        region=request.region,
        predictions=mock_predictions,
        model_version="stub-v0",
        timestamp=datetime.now(UTC).isoformat(),
    )


@app.post("/monitor")
async def monitor(request: MonitorRequest) -> JSONResponse:
    """Run drift monitoring check — called by Cloud Scheduler every 6 hours."""
    try:
        from src.monitoring.monitor_runner import run_monitoring_check

        run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        result = run_monitoring_check(
            run_id=run_id,
            gcs_bucket=request.gcs_bucket,
            baseline_run_id=request.baseline_run_id,
        )
        status_code = 200
        if result.get("verdict") == "CRITICAL":
            status_code = 200  # still 200 — Cloud Scheduler needs 2xx to not retry
        return JSONResponse(content=result, status_code=status_code)
    except Exception as e:
        logger.exception("Monitoring check failed")
        raise HTTPException(status_code=500, detail=str(e)) from e