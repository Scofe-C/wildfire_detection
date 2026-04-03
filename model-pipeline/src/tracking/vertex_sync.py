from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _load_vertex_config() -> dict[str, Any]:
    cfg = Path(__file__).resolve().parents[2] / "configs" / "model_config.yaml"
    with open(cfg) as f:
        return yaml.safe_load(f)["tracking"]["vertex_ai"]


class VertexAISync:
    def __init__(
        self,
        project_id: str | None = None,
        location: str | None = None,
        experiment_name: str | None = None,
    ):
        config = _load_vertex_config()
        self._project_id = project_id or os.getenv("GCP_PROJECT_ID", config.get("project_id", ""))
        self._location = location or config.get("location", "us-central1")
        self._experiment_name = experiment_name or config.get("experiment_name", "wildfire-model-pipeline")
        self._initialized = False

    def _init_vertex(self, experiment: str | None = None):
        from google.cloud import aiplatform
        # Always re-init when experiment context is needed — init is idempotent
        aiplatform.init(
            project=self._project_id,
            location=self._location,
            experiment=experiment or self._experiment_name,
        )
        self._aiplatform = aiplatform
        self._initialized = True
        logger.info("Vertex AI initialized — project: %s", self._project_id)

    def sync_run(
        self,
        run_id: str,
        metrics: dict[str, float],
        params: dict[str, str],
    ) -> str:
        self._init_vertex()
        # run name must be lowercase alphanumeric + hyphens
        safe_run_id = run_id.lower().replace("_", "-")
        with self._aiplatform.start_run(run=safe_run_id) as run:
            float_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            if float_metrics:
                run.log_metrics(float_metrics)
            str_params = {k: str(v) for k, v in params.items()}
            if str_params:
                run.log_params(str_params)
        logger.info("Synced to Vertex AI Experiment '%s' — run: %s", self._experiment_name, safe_run_id)
        return safe_run_id

    def sync_rollback_event(
        self,
        run_id: str,
        reason_code: str,
        delta_auc_pr: float | None = None,
        delta_fnr_disparity: float | None = None,
    ):
        metrics: dict[str, float] = {"rollback": 1.0}
        if delta_auc_pr is not None:
            metrics["delta_auc_pr"] = delta_auc_pr
        if delta_fnr_disparity is not None:
            metrics["delta_fnr_disparity"] = delta_fnr_disparity
        self.sync_run(
            run_id=f"rollback-{run_id}",
            metrics=metrics,
            params={"reason_code": reason_code, "event_type": "rollback"},
        )
        logger.warning("Rollback recorded — run: %s, reason: %s", run_id, reason_code)
