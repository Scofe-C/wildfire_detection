"""Unit tests for src/models/registry.py — ModelRegistry.

All tests are offline (no gsutil, no GCS). GCS push is tested via mock.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.models.registry import ModelRegistry, RegistryError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(
        local_models_dir=tmp_path / "models",
        gcs_bucket="test-bucket",
        gcs_prefix="test-prefix",
    )


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    """A fake model artifact file."""
    p = tmp_path / "model.ubj"
    p.write_bytes(b"fake-model-bytes")
    return p


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """A fake model artifact directory."""
    d = tmp_path / "model_dir"
    d.mkdir()
    (d / "weights.ubj").write_bytes(b"weights")
    (d / "config.json").write_text('{"n_estimators": 100}')
    return d


# ---------------------------------------------------------------------------
# save_local — file artifact
# ---------------------------------------------------------------------------

class TestSaveLocalFile:
    def test_saves_file_to_version_dir(self, registry: ModelRegistry, model_file: Path) -> None:
        version_dir = registry.save_local(model_file, version="v1.0")
        saved = version_dir / model_file.name
        assert saved.exists()
        assert saved.read_bytes() == b"fake-model-bytes"

    def test_returns_version_dir_path(self, registry: ModelRegistry, model_file: Path) -> None:
        result = registry.save_local(model_file, version="v1.0")
        assert result.name == "v1.0"

    def test_writes_metadata_json(self, registry: ModelRegistry, model_file: Path) -> None:
        registry.save_local(model_file, version="v1.0", metadata={"auc_pr": 0.82})
        meta_path = registry.local_models_dir / "v1.0" / "metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["auc_pr"] == 0.82
        assert meta["version"] == "v1.0"
        assert "saved_at" in meta

    def test_no_metadata_file_when_none(self, registry: ModelRegistry, model_file: Path) -> None:
        registry.save_local(model_file, version="v2.0")
        meta_path = registry.local_models_dir / "v2.0" / "metadata.json"
        assert not meta_path.exists()

    def test_raises_on_missing_artifact(self, registry: ModelRegistry, tmp_path: Path) -> None:
        with pytest.raises(RegistryError, match="Artifact not found"):
            registry.save_local(tmp_path / "nonexistent.ubj", version="v1.0")


# ---------------------------------------------------------------------------
# save_local — directory artifact
# ---------------------------------------------------------------------------

class TestSaveLocalDir:
    def test_saves_dir_to_version_dir(self, registry: ModelRegistry, model_dir: Path) -> None:
        version_dir = registry.save_local(model_dir, version="v1.0")
        saved_dir = version_dir / model_dir.name
        assert saved_dir.is_dir()
        assert (saved_dir / "weights.ubj").exists()
        assert (saved_dir / "config.json").exists()

    def test_overwrites_existing_dir(self, registry: ModelRegistry, model_dir: Path) -> None:
        # Save twice — second call should replace first without error
        registry.save_local(model_dir, version="v1.0")
        registry.save_local(model_dir, version="v1.0")
        saved_dir = registry.local_models_dir / "v1.0" / model_dir.name
        assert saved_dir.exists()


# ---------------------------------------------------------------------------
# tag_previous / get_previous_version
# ---------------------------------------------------------------------------

class TestVersionTagging:
    def test_get_previous_returns_none_when_no_marker(self, registry: ModelRegistry) -> None:
        assert registry.get_previous_version() is None

    def test_tag_previous_creates_marker(self, registry: ModelRegistry, model_file: Path) -> None:
        registry.save_local(model_file, version="v1.0", metadata={})
        registry.save_local(model_file, version="v2.0", metadata={})
        registry.tag_previous("v2.0")
        assert registry.get_previous_version() == "v1.0"

    def test_tag_previous_no_other_versions(
        self, registry: ModelRegistry, model_file: Path
    ) -> None:
        # Only one version exists — tag_previous should not crash, marker absent
        registry.save_local(model_file, version="v1.0", metadata={})
        registry.tag_previous("v1.0")
        # No other versions to tag — marker file should not exist
        marker = registry.local_models_dir / "PREVIOUS_VERSION"
        assert not marker.exists()

    def test_get_previous_reads_marker(self, registry: ModelRegistry) -> None:
        marker = registry.local_models_dir / "PREVIOUS_VERSION"
        marker.write_text("v0.9")
        assert registry.get_previous_version() == "v0.9"


# ---------------------------------------------------------------------------
# list_versions
# ---------------------------------------------------------------------------

class TestListVersions:
    def test_empty_when_no_versions(self, registry: ModelRegistry) -> None:
        assert registry.list_versions() == []

    def test_lists_only_dirs_with_metadata(
        self, registry: ModelRegistry, model_file: Path
    ) -> None:
        registry.save_local(model_file, version="v1.0", metadata={"x": 1})
        registry.save_local(model_file, version="v2.0", metadata={"x": 2})
        # Add a dir without metadata — should be excluded
        (registry.local_models_dir / "no_meta").mkdir()
        versions = registry.list_versions()
        assert "v1.0" in versions
        assert "v2.0" in versions
        assert "no_meta" not in versions

    def test_returns_sorted(self, registry: ModelRegistry, model_file: Path) -> None:
        registry.save_local(model_file, version="v3.0", metadata={})
        registry.save_local(model_file, version="v1.0", metadata={})
        registry.save_local(model_file, version="v2.0", metadata={})
        assert registry.list_versions() == ["v1.0", "v2.0", "v3.0"]


# ---------------------------------------------------------------------------
# push_to_gcs — mocked
# ---------------------------------------------------------------------------

class TestPushToGcs:
    def test_raises_when_version_not_local(self, registry: ModelRegistry) -> None:
        with pytest.raises(RegistryError, match="Local version not found"):
            registry.push_to_gcs("v99.0")

    def test_raises_when_gsutil_missing(
        self, registry: ModelRegistry, model_file: Path
    ) -> None:
        registry.save_local(model_file, version="v1.0", metadata={})
        with patch("subprocess.run", side_effect=FileNotFoundError), pytest.raises(RegistryError, match="gsutil not found"):
            registry.push_to_gcs("v1.0")

    def test_returns_gcs_uri_on_success(
        self, registry: ModelRegistry, model_file: Path
    ) -> None:
        registry.save_local(model_file, version="v1.0", metadata={})
        mock_result = type("R", (), {"returncode": 0, "stderr": ""})()
        with patch("subprocess.run", return_value=mock_result):
            uri = registry.push_to_gcs("v1.0")
        assert uri == "gs://test-bucket/test-prefix/v1.0/"
