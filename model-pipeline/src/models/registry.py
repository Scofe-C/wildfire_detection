from __future__ import annotations

import json
import logging
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RegistryError(Exception):
    pass


class ModelRegistry:
    def __init__(
        self,
        local_models_dir: str | Path = "models/ignition",
        gcs_bucket: str = "wildfire-model-pipeline",
        gcs_prefix: str = "wildfire-models",
    ):
        self.local_models_dir = Path(local_models_dir)
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix
        self.local_models_dir.mkdir(parents=True, exist_ok=True)

    def save_local(
        self,
        model_artifact_path: str | Path,
        version: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        src = Path(model_artifact_path)
        version_dir = self.local_models_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        if src.is_file():
            shutil.copy2(src, version_dir / src.name)
        elif src.is_dir():
            dest = version_dir / src.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
        else:
            raise RegistryError(f"Artifact not found: {src}")

        if metadata is not None:
            metadata["saved_at"] = datetime.now(UTC).isoformat()
            metadata["version"] = version
            with open(version_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2, default=str)

        logger.info("Model saved locally: %s", version_dir)
        return version_dir

    def push_to_gcs(self, version: str) -> str:
        version_dir = self.local_models_dir / version
        if not version_dir.exists():
            raise RegistryError(f"Local version not found: {version_dir}")
        gcs_uri = f"gs://{self.gcs_bucket}/{self.gcs_prefix}/{version}/"
        try:
            subprocess.run(
                ["gsutil", "-m", "cp", "-r", str(version_dir) + "/", gcs_uri],
                capture_output=True, text=True, check=True,
            )
        except FileNotFoundError:
            raise RegistryError("gsutil not found — install Google Cloud SDK") from None
        except subprocess.CalledProcessError as e:
            raise RegistryError(f"GCS push failed: {e.stderr}") from e
        logger.info("Pushed to GCS: %s", gcs_uri)
        return gcs_uri

    def tag_previous(self, current_version: str) -> None:
        versions = sorted(
            [d.name for d in self.local_models_dir.iterdir()
             if d.is_dir() and d.name != current_version],
            reverse=True,
        )
        marker = self.local_models_dir / "PREVIOUS_VERSION"
        if versions:
            marker.write_text(versions[0])
            logger.info("Tagged previous version: %s", versions[0])

    def get_previous_version(self) -> str | None:
        marker = self.local_models_dir / "PREVIOUS_VERSION"
        return marker.read_text().strip() if marker.exists() else None

    def list_versions(self) -> list[str]:
        return sorted(
            d.name for d in self.local_models_dir.iterdir()
            if d.is_dir() and (d / "metadata.json").exists()
        )
