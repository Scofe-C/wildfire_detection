"""
Vertex AI Model Registry wrapper — artifact storage, versioning, stage promotion.

Responsibilities:
  - Save model artifact + metadata (threshold, medians, feature list) to GCS
  - Register the GCS artifact with Vertex AI Model Registry
  - Promote/demote model versions using resource labels (env=production/staging/archived)
  - Load the production model + metadata at inference time

Storage layout per training run:
  gs://{bucket}/model-artifacts/{run_id}/
    model.bst               (XGBoost booster)  OR
    model.txt               (LightGBM booster)
    model_metadata.json     (threshold, medians, feature list, framework, run_id)
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_XGBOOST_CONTAINER = "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest"
_SKLEARN_CONTAINER  = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"

_MODEL_FILENAME = {
    "xgboost":  "model.bst",
    "lightgbm": "model.txt",
}


class VertexRegistry:
    """Push and load models via Vertex AI Model Registry + GCS artifact store."""

    def __init__(
        self,
        project_id: str | None = None,
        location: str = "us-central1",
        display_name: str = "wildfire-ignition",
        gcs_bucket: str = "wildfire-mlops-123",
        gcs_prefix: str = "model-artifacts",
    ):
        self._project_id  = project_id or os.environ.get("GCP_PROJECT_ID", "")
        self._location    = location
        self._display_name = display_name
        self._gcs_bucket  = gcs_bucket
        self._gcs_prefix  = gcs_prefix
        self._initialized = False

    # ── Vertex AI init (lazy) ─────────────────────────────────────────────────

    def _init(self):
        if self._initialized:
            return
        from google.cloud import aiplatform
        aiplatform.init(project=self._project_id, location=self._location)
        self._aiplatform = aiplatform
        self._initialized = True

    # ── Push ──────────────────────────────────────────────────────────────────

    def push(
        self,
        winner: Any,
        winner_name: str,
        run_id: str,
        threshold: float,
        medians: dict[str, float],
    ) -> str:
        """Save artifact to GCS, register with Vertex AI, promote to Production.

        Returns
        -------
        Vertex AI model resource name (projects/.../models/...)
        """
        self._init()

        artifact_gcs_dir = f"gs://{self._gcs_bucket}/{self._gcs_prefix}/{run_id}"
        self._save_artifacts_to_gcs(winner, winner_name, threshold, medians, artifact_gcs_dir, run_id)

        # Pick serving container (for registry metadata — we don't deploy to endpoint)
        container = _XGBOOST_CONTAINER if winner_name == "xgboost" else _SKLEARN_CONTAINER

        model = self._aiplatform.Model.upload(
            display_name=self._display_name,
            artifact_uri=artifact_gcs_dir,
            serving_container_image_uri=container,
            labels={
                "env": "staging",
                "framework": winner_name,
                "run_id": run_id[:63],   # label values max 63 chars
            },
        )
        logger.info("Registered model in Vertex AI — %s (env=staging)", model.resource_name)

        # Demote any current Production version → archived
        self._demote_current_production()

        # Promote new version → Production
        model.update(labels={
            "env": "production",
            "framework": winner_name,
            "run_id": run_id[:63],
        })
        logger.info("Promoted to Production — %s", model.resource_name)
        return model.resource_name

    # ── Load production ───────────────────────────────────────────────────────

    def load_production(self) -> tuple[Any, dict[str, float], float]:
        """Load the current Production model + medians + threshold from GCS.

        Returns
        -------
        (model_object, medians_dict, threshold_float)
        """
        self._init()

        models = self._aiplatform.Model.list(
            filter=f'display_name="{self._display_name}" AND labels.env="production"',
            order_by="create_time desc",
        )
        if not models:
            raise RuntimeError(
                f"No Production model found for display_name='{self._display_name}'"
            )
        model = models[0]
        framework = model.labels.get("framework", "xgboost")
        run_id = model.labels.get("run_id", "")
        if not run_id:
            raise RuntimeError(
                f"Production model '{self._display_name}' has no run_id label — cannot locate GCS artifacts"
            )
        artifact_uri = f"gs://{self._gcs_bucket}/{self._gcs_prefix}/{run_id}"

        logger.info("Loading Production model from %s (framework=%s)", artifact_uri, framework)
        return self._load_artifacts_from_gcs(artifact_uri, framework)

    # ── Rollback ──────────────────────────────────────────────────────────────

    def rollback(self) -> str:
        """Promote the most recently archived model back to Production.

        Demotes the current Production version to archived first.

        Returns
        -------
        Resource name of the newly promoted model.
        """
        self._init()

        archived = self._aiplatform.Model.list(
            filter=f'display_name="{self._display_name}" AND labels.env="archived"',
            order_by="create_time desc",
        )
        if not archived:
            raise RuntimeError(
                f"No archived model found for '{self._display_name}' — cannot rollback"
            )
        prev = archived[0]

        # Demote current production
        self._demote_current_production()

        # Promote previous archived → production
        labels = dict(prev.labels)
        labels["env"] = "production"
        prev.update(labels=labels)
        logger.info("Rollback complete — promoted %s to Production", prev.resource_name)
        return prev.resource_name

    # ── Private helpers ───────────────────────────────────────────────────────

    def _demote_current_production(self) -> None:
        """Move any current Production models to archived."""
        current_prod = self._aiplatform.Model.list(
            filter=f'display_name="{self._display_name}" AND labels.env="production"',
        )
        for m in current_prod:
            labels = dict(m.labels)
            labels["env"] = "archived"
            m.update(labels=labels)
            logger.info("Archived previous Production model — %s", m.resource_name)

    def _save_artifacts_to_gcs(
        self,
        winner: Any,
        winner_name: str,
        threshold: float,
        medians: dict[str, float],
        artifact_gcs_dir: str,
        run_id: str,
    ) -> None:
        """Save model file + model_metadata.json to the GCS artifact directory."""
        from google.cloud import storage
        from src.preprocessing.feature_engineering import FEATURES

        client = storage.Client()
        bucket_name = self._gcs_bucket
        prefix = artifact_gcs_dir.replace(f"gs://{bucket_name}/", "")
        bkt = client.bucket(bucket_name)

        # ── Model file ────────────────────────────────────────────────────────
        model_filename = _MODEL_FILENAME[winner_name]
        with tempfile.TemporaryDirectory() as tmpdir:
            local_model_path = Path(tmpdir) / model_filename
            if winner_name == "xgboost":
                # Use booster-level save; the sklearn wrapper's save_model()
                # calls _get_type() which is strict in xgboost>=2 and can
                # raise when sklearn metadata isn't fully populated.
                winner._model.get_booster().save_model(str(local_model_path))
            else:
                winner._model.booster_.save_model(str(local_model_path))

            bkt.blob(f"{prefix}/{model_filename}").upload_from_filename(str(local_model_path))
            logger.info("Saved model artifact → gs://%s/%s/%s", bucket_name, prefix, model_filename)

        # ── Metadata JSON ──────────────────────────────────────────────────────
        metadata = {
            "run_id": run_id,
            "framework": winner_name,
            "threshold": threshold,
            "medians": medians,
            "features": FEATURES,
        }
        bkt.blob(f"{prefix}/model_metadata.json").upload_from_string(
            json.dumps(metadata, indent=2),
            content_type="application/json",
        )
        logger.info("Saved model_metadata.json → gs://%s/%s/model_metadata.json", bucket_name, prefix)

    def _load_artifacts_from_gcs(
        self,
        artifact_uri: str,
        framework: str,
    ) -> tuple[Any, dict[str, float], float]:
        """Download model + metadata from GCS, return (model, medians, threshold)."""
        import json as _json

        from google.cloud import storage

        client = storage.Client()
        # artifact_uri = "gs://bucket/prefix/run_id"
        bucket_name = artifact_uri.replace("gs://", "").split("/")[0]
        prefix = "/".join(artifact_uri.replace("gs://", "").split("/")[1:])
        bkt = client.bucket(bucket_name)

        # ── Load metadata ─────────────────────────────────────────────────────
        meta_bytes = bkt.blob(f"{prefix}/model_metadata.json").download_as_bytes()
        metadata   = _json.loads(meta_bytes)
        threshold  = float(metadata["threshold"])
        medians    = metadata["medians"]

        # ── Load model file ───────────────────────────────────────────────────
        model_filename = _MODEL_FILENAME.get(framework, "model.bst")
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / model_filename
            bkt.blob(f"{prefix}/{model_filename}").download_to_filename(str(local_path))

            if framework == "xgboost":
                import xgboost as xgb
                model = xgb.Booster()
                model.load_model(str(local_path))
            else:
                import lightgbm as lgb
                model = lgb.Booster(model_file=str(local_path))

        logger.info("Loaded %s model from %s", framework, artifact_uri)
        return model, medians, threshold
