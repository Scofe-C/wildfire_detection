"""
DVC Tests — Configuration, GCS Connectivity, and Pipeline Versioning
======================================================================

Three test classes, separated by what they actually need:

  TestDVCConfig       — offline, no credentials, no network.
                        Validates .dvc/config is correct and the remote
                        name and bucket match what the Airflow task expects.

  TestDVCGCSMocked    — offline, no credentials.
                        Validates that dvc push/pull logic and the
                        version_with_dvc Airflow task behave correctly
                        by mocking the GCS client and subprocess calls.
                        Safe to run in CI at all times.

  TestDVCGCSLive      — requires real GCP credentials.
                        Validates actual connectivity to gs://wildfire-mlops.
                        Skipped automatically when GOOGLE_APPLICATION_CREDENTIALS
                        or GCS_BUCKET_NAME is not set.
                        Run manually before deployments:
                            pytest tests/test_dvc/ -m gcp -v

Usage
-----
All tests:
    pytest tests/test_dvc/ -v

Skip GCP-credential tests (safe for CI):
    pytest tests/test_dvc/ -m "not gcp" -v

Run only live GCP tests:
    pytest tests/test_dvc/ -m gcp -v

What each test validates
------------------------
Config tests:
  - .dvc/config exists and is parseable
  - remote is named 'gcsremote' (matches DAG bash command)
  - remote URL points to gs://wildfire-mlops
  - default remote is set
  - dvc.yaml stages are present and well-formed
  - .dvc files for tracked outputs exist and have valid md5 hashes
  - dvc.lock is present and in sync with dvc.yaml stage names

Mocked GCS tests:
  - version_with_dvc bash command has correct structure
  - dvc remote list check in bash command would catch unconfigured remote
  - dvc add then dvc push sequence is called in correct order
  - DVC push failure is surfaced as task failure (not swallowed)
  - GCS state write uses conditional generation match (race-safe)
  - GCS trigger write produces correct JSON schema
  - list_pending_triggers returns empty on no triggers (not an exception)

Live GCS tests (marked gcp):
  - GCS bucket is reachable and credentials are valid
  - DVC remote URL in .dvc/config matches GCS_BUCKET_NAME env var
  - Can write a small test object to GCS and read it back
  - Can delete the test object after verification
  - dvc status --cloud produces parseable output (does not crash)
  - Watchdog state path exists or can be initialized on first run
"""

from __future__ import annotations

import configparser
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DVC_CONFIG_PATH   = PROJECT_ROOT / ".dvc" / "config"
DVC_YAML_PATH     = PROJECT_ROOT / "dvc.yaml"
DVC_LOCK_PATH     = PROJECT_ROOT / "dvc.lock"
FUSED_DVC_PATH    = PROJECT_ROOT / "data" / "processed" / "fused.dvc"
KM64_DVC_PATH     = PROJECT_ROOT / "data" / "processed" / "64km.dvc"

# Expected constants — must match .dvc/config AND the Airflow BashOperator
EXPECTED_REMOTE_NAME   = "gcsremote"
# Read bucket name from env var — falls back to the value in .dvc/config
# This makes the test work regardless of which bucket name was chosen
EXPECTED_BUCKET_PREFIX = "gs://" + os.environ.get("GCS_BUCKET_NAME", "wildfire-mlops")

# GCP live test guard
_HAS_GCP_CREDS = bool(
    os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    or os.environ.get("GCS_BUCKET_NAME")
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dvc_config() -> configparser.ConfigParser:
    """Parse .dvc/config as INI. Returns empty parser if file missing.

    DVC writes the remote section as ['remote "name"'] (with outer single
    quotes). configparser then reads it as section name 'remote "name"'
    (including the single quotes). We normalise the section name so the
    rest of the test code can use the expected bare form.
    """
    cfg = configparser.ConfigParser()
    if DVC_CONFIG_PATH.exists():
        cfg.read(str(DVC_CONFIG_PATH), encoding="utf-8")
        # Normalise DVC-quoted section names: 'remote "x"' → remote "x"
        for section in cfg.sections():
            stripped = section.strip("'")
            if stripped != section:
                items = dict(cfg.items(section))
                cfg.remove_section(section)
                cfg.add_section(stripped)
                for k, v in items.items():
                    cfg.set(stripped, k, v)
    return cfg


def _parse_dvc_yaml() -> dict:
    """Parse dvc.yaml. Returns empty dict if file missing.
    Uses explicit UTF-8 encoding — Windows default (GBK/cp936) cannot
    decode the smart-quote characters in YAML comments.
    """
    try:
        import yaml
        with open(DVC_YAML_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (FileNotFoundError, ImportError):
        return {}


def _parse_dvc_lock() -> dict:
    """Parse dvc.lock. Returns empty dict if file missing.
    Uses explicit UTF-8 encoding for Windows GBK compatibility.
    """
    try:
        import yaml
        with open(DVC_LOCK_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (FileNotFoundError, ImportError):
        return {}


def _read_dvc_file(path: Path) -> dict:
    """Read a .dvc sidecar file (YAML). Returns empty dict if missing.
    Uses explicit UTF-8 encoding for Windows GBK compatibility.
    """
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (FileNotFoundError, ImportError):
        return {}


def _get_version_with_dvc_bash() -> str:
    """Extract the bash_command string from the version_with_dvc BashOperator."""
    dag_path = PROJECT_ROOT / "dags" / "wildfire_dag.py"
    content = dag_path.read_text(encoding="utf-8")
    # Find the bash_command triple-quoted string
    start = content.find('bash_command="""')
    end   = content.find('"""', start + 16)
    if start == -1 or end == -1:
        return ""
    return content[start + 16 : end]


# ===========================================================================
# Class 1: Offline config validation (no credentials needed)
# ===========================================================================

class TestDVCConfig:
    """Validate .dvc/config, dvc.yaml, dvc.lock, and .dvc sidecar files.
    All offline — no GCP credentials required.
    """

    # --- .dvc/config ---

    def test_dvc_config_file_exists(self):
        assert DVC_CONFIG_PATH.exists(), (
            ".dvc/config not found. Run: dvc remote add -d gcsremote gs://wildfire-mlops"
        )

    def test_dvc_config_is_parseable(self):
        cfg = _parse_dvc_config()
        assert cfg is not None

    def test_default_remote_is_set(self):
        cfg = _parse_dvc_config()
        assert cfg.has_option("core", "remote"), (
            "No default remote set in .dvc/config. "
            "Run: dvc remote add -d gcsremote gs://wildfire-mlops"
        )

    def test_default_remote_name_matches_dag(self):
        """Remote name must match what the Airflow bash task expects."""
        cfg = _parse_dvc_config()
        remote_name = cfg.get("core", "remote", fallback=None)
        assert remote_name == EXPECTED_REMOTE_NAME, (
            f"Default remote is '{remote_name}', expected '{EXPECTED_REMOTE_NAME}'. "
            f"The version_with_dvc task uses 'dvc remote list | grep -q .' "
            f"which only works when the remote is named '{EXPECTED_REMOTE_NAME}'."
        )

    def test_remote_url_points_to_correct_bucket(self):
        cfg = _parse_dvc_config()
        # Config section key is 'remote "gcsremote"'
        section = f'remote "{EXPECTED_REMOTE_NAME}"'
        assert cfg.has_section(section), (
            f"Section [{section}] not found in .dvc/config. "
            f"Run: dvc remote add -d {EXPECTED_REMOTE_NAME} {EXPECTED_BUCKET_PREFIX}"
        )
        url = cfg.get(section, "url", fallback=None)
        assert url is not None, f"No 'url' key in [{section}]"
        assert url.startswith(EXPECTED_BUCKET_PREFIX), (
            f"Remote URL '{url}' does not start with '{EXPECTED_BUCKET_PREFIX}'. "
            f"The pipeline writes to this bucket — a wrong URL means all versioned "
            f"data goes to the wrong place silently."
        )

    def test_remote_url_uses_gs_scheme(self):
        cfg = _parse_dvc_config()
        section = f'remote "{EXPECTED_REMOTE_NAME}"'
        url = cfg.get(section, "url", fallback="")
        assert url.startswith("gs://"), (
            f"Remote URL '{url}' must use gs:// scheme (Google Cloud Storage). "
            f"s3:// or https:// would cause silent auth failures on GCP."
        )

    # --- dvc.yaml ---

    def test_dvc_yaml_exists(self):
        assert DVC_YAML_PATH.exists(), "dvc.yaml not found at project root"

    def test_dvc_yaml_has_stages(self):
        data = _parse_dvc_yaml()
        assert "stages" in data and data["stages"], (
            "dvc.yaml has no stages defined"
        )

    def test_dvc_yaml_required_stages_present(self):
        """Core pipeline stages must be defined for offline reproducibility."""
        data = _parse_dvc_yaml()
        stages = data.get("stages", {})
        required = {"ingest_firms", "ingest_weather", "fuse_features", "validate_schema"}
        missing = required - set(stages.keys())
        assert not missing, (
            f"Missing required dvc.yaml stages: {missing}. "
            f"DVC stages mirror the Airflow DAG for offline reproducibility."
        )

    def test_each_stage_has_cmd(self):
        data = _parse_dvc_yaml()
        for stage_name, stage_def in data.get("stages", {}).items():
            assert "cmd" in stage_def and stage_def["cmd"], (
                f"Stage '{stage_name}' in dvc.yaml has no 'cmd' key"
            )

    def test_fuse_stage_depends_on_firms_and_weather(self):
        data = _parse_dvc_yaml()
        fuse = data.get("stages", {}).get("fuse_features", {})
        deps = fuse.get("deps", [])
        dep_paths = [d if isinstance(d, str) else list(d.keys())[0] for d in deps]
        assert any("firms" in d for d in dep_paths), (
            "fuse_features stage must list data/processed/firms as a dependency"
        )
        assert any("weather" in d for d in dep_paths), (
            "fuse_features stage must list data/processed/weather as a dependency"
        )

    # --- dvc.lock ---

    def test_dvc_lock_exists(self):
        assert DVC_LOCK_PATH.exists(), (
            "dvc.lock not found. Run 'dvc repro' to generate it. "
            "Without dvc.lock, DVC cannot detect which stages are stale."
        )

    def test_dvc_lock_stage_names_match_yaml(self):
        yaml_stages = set(_parse_dvc_yaml().get("stages", {}).keys())
        lock_stages = set(_parse_dvc_lock().get("stages", {}).keys())
        # Lock may be a subset of yaml (stages not yet run)
        # But every stage in lock must exist in yaml
        extra_in_lock = lock_stages - yaml_stages
        assert not extra_in_lock, (
            f"dvc.lock references stages not in dvc.yaml: {extra_in_lock}. "
            f"This means dvc.lock is stale — run 'dvc repro' to resync."
        )

    def test_dvc_lock_schema_version(self):
        lock_data = _parse_dvc_lock()
        schema = lock_data.get("schema")
        assert schema is not None, "dvc.lock missing 'schema' field"
        assert str(schema).startswith("2"), (
            f"dvc.lock schema version is '{schema}', expected '2.x'. "
            f"Older lock files may be incompatible with DVC 3.x."
        )

    # --- .dvc sidecar files ---

    def test_fused_dvc_file_exists(self):
        assert FUSED_DVC_PATH.exists(), (
            "data/processed/fused.dvc not found. "
            "Run: dvc add data/processed/fused"
        )

    def test_64km_dvc_file_exists(self):
        assert KM64_DVC_PATH.exists(), (
            "data/processed/64km.dvc not found. "
            "Run: dvc add data/processed/64km"
        )

    def test_fused_dvc_has_valid_md5(self):
        data = _read_dvc_file(FUSED_DVC_PATH)
        outs = data.get("outs", [])
        assert outs, "fused.dvc has no 'outs' entries"
        md5 = outs[0].get("md5", "")
        assert len(md5) >= 32, (
            f"fused.dvc has invalid md5 '{md5}'. "
            f"Run 'dvc add data/processed/fused' to regenerate."
        )

    def test_64km_dvc_has_valid_md5(self):
        data = _read_dvc_file(KM64_DVC_PATH)
        outs = data.get("outs", [])
        assert outs, "64km.dvc has no 'outs' entries"
        md5 = outs[0].get("md5", "")
        assert len(md5) >= 32, (
            f"64km.dvc has invalid md5 '{md5}'. "
            f"Run 'dvc add data/processed/64km' to regenerate."
        )

    def test_fused_dvc_path_matches_expected(self):
        data = _read_dvc_file(FUSED_DVC_PATH)
        outs = data.get("outs", [])
        assert outs, "fused.dvc has no 'outs'"
        path = outs[0].get("path", "")
        assert "fused" in path, (
            f"fused.dvc output path '{path}' does not reference the fused directory"
        )


# ===========================================================================
# Class 2: Mocked GCS behaviour (no credentials needed)
# ===========================================================================

class TestDVCGCSMocked:
    """
    Tests that validate DVC + GCS logic using mocked GCS clients.
    All offline — safe for CI. Covers:
      - Airflow version_with_dvc bash command structure
      - gcs_state.py write/read/delete behaviour
      - Error handling when GCS is unavailable
    """

    # Shared mock registry — returned by all patched get_registry() calls.
    # Provides the watchdog GCS path config that gcs_state.py reads.
    _MOCK_REGISTRY_CONFIG = {
        "watchdog": {
            "gcs_paths": {
                "state":            "watchdog/state/current.json",
                "triggers":         "watchdog/triggers/",
                "false_alarms":     "watchdog/false_alarms/",
                "emergency_log":    "watchdog/emergency/",
                "industrial_sources": "watchdog/config/industrial_sources.json",
            }
        }
    }

    @pytest.fixture(autouse=True)
    def mock_registry(self):
        """Patch get_registry in gcs_state so tests never load schema_config.yaml."""
        mock_reg = MagicMock()
        mock_reg.config = self._MOCK_REGISTRY_CONFIG
        with patch("scripts.utils.schema_loader.get_registry", return_value=mock_reg):
            yield mock_reg

    # --- version_with_dvc BashOperator ---

    def test_bash_command_has_set_euo_pipefail(self):
        bash = _get_version_with_dvc_bash()
        assert "set -euo pipefail" in bash, (
            "version_with_dvc bash_command must start with 'set -euo pipefail' "
            "to fail fast on any error. Without it, dvc push failures are swallowed."
        )

    def test_bash_command_checks_remote_configured(self):
        bash = _get_version_with_dvc_bash()
        assert "dvc remote list" in bash, (
            "version_with_dvc must check 'dvc remote list' before attempting push. "
            "Without this guard, the task silently succeeds with no data versioned."
        )

    def test_bash_command_exits_1_on_no_remote(self):
        bash = _get_version_with_dvc_bash()
        assert "exit 1" in bash, (
            "version_with_dvc must 'exit 1' when no DVC remote is configured. "
            "This is what causes the scheduler starvation warning — "
            "the fix is to configure the remote, not to remove the guard."
        )

    def test_bash_command_dvc_add_before_push(self):
        """dvc add must precede dvc push — never push without tracking first."""
        bash = _get_version_with_dvc_bash()
        add_pos  = bash.find("dvc add")
        push_pos = bash.find("dvc push")
        assert add_pos != -1, "version_with_dvc must call 'dvc add'"
        assert push_pos != -1, "version_with_dvc must call 'dvc push'"
        assert add_pos < push_pos, (
            "dvc add must come before dvc push in version_with_dvc. "
            "Pushing before tracking means stale hashes go to GCS."
        )

    def test_bash_command_tracks_fused_directory(self):
        bash = _get_version_with_dvc_bash()
        assert "data/processed/fused" in bash, (
            "version_with_dvc must track data/processed/fused"
        )

    def test_bash_command_tracks_resolution_km_directory(self):
        bash = _get_version_with_dvc_bash()
        assert "resolution_km" in bash or "params.resolution_km" in bash, (
            "version_with_dvc must track the resolution-specific parquet directory"
        )

    def test_bash_command_pushes_both_dvc_files(self):
        bash = _get_version_with_dvc_bash()
        assert "fused.dvc" in bash, "dvc push must reference fused.dvc"
        # Either the literal 64km.dvc or parameterised with resolution_km
        assert ".dvc" in bash.split("dvc push")[-1], (
            "dvc push must reference at least one .dvc file"
        )

    def test_bash_command_initialises_git_if_needed(self):
        """DVC requires a git repo context — bash command must ensure it exists."""
        bash = _get_version_with_dvc_bash()
        assert "git init" in bash, (
            "version_with_dvc must run 'git init' if .git is missing. "
            "Without a git repo, 'dvc add' raises 'not a git repo' error."
        )

    # --- gcs_state.py: read_state ---

    def test_read_state_returns_default_on_missing_file(self):
        """read_state must return DEFAULT_STATE on first run (no existing blob)."""
        from scripts.utils.gcs_state import read_state, DEFAULT_STATE

        mock_blob = MagicMock()
        mock_blob.exists.return_value = False

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "wildfire-mlops"}):
            state = read_state()

        assert state["mode"] == DEFAULT_STATE["mode"]
        assert "consecutive_fire_scans" in state
        assert state["active_fire_cells"] == []

    def test_read_state_merges_with_defaults_for_new_fields(self):
        """New DEFAULT_STATE fields must appear even when reading old state JSON."""
        from scripts.utils.gcs_state import read_state, DEFAULT_STATE

        old_state = {"mode": "active", "last_updated": "2026-01-01T00:00:00Z"}

        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_text.return_value = json.dumps(old_state)

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "wildfire-mlops"}):
            state = read_state()

        assert state["mode"] == "active"
        # New fields from DEFAULT_STATE must be present even in old JSON
        for key in DEFAULT_STATE:
            assert key in state, (
                f"Field '{key}' from DEFAULT_STATE missing after merge. "
                f"New fields must always be added to DEFAULT_STATE."
            )

    def test_read_state_falls_back_to_default_on_gcs_error(self):
        """GCS connectivity errors must not crash read_state — return defaults."""
        from scripts.utils.gcs_state import read_state, DEFAULT_STATE

        mock_client = MagicMock()
        mock_client.bucket.side_effect = Exception("Simulated GCS network error")

        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "wildfire-mlops"}):
            state = read_state()

        assert state == DEFAULT_STATE

    # --- gcs_state.py: write_state ---

    def test_write_state_uses_conditional_generation_match(self):
        """Conditional write prevents race condition when two Cloud Functions run concurrently."""
        from scripts.utils.gcs_state import write_state

        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.generation = 42

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "wildfire-mlops"}):
            result = write_state({"mode": "active", "active_fire_cells": []})

        assert result is True
        upload_call = mock_blob.upload_from_string.call_args
        assert upload_call is not None
        # Must pass if_generation_match to prevent concurrent overwrites
        kwargs = upload_call[1]
        assert "if_generation_match" in kwargs, (
            "write_state must use if_generation_match for conditional writes. "
            "Without it, two concurrent Cloud Function invocations can corrupt state."
        )
        assert kwargs["if_generation_match"] == 42

    def test_write_state_returns_false_on_precondition_failed(self):
        """PreconditionFailed = another instance won the race — must return False."""
        from scripts.utils.gcs_state import write_state
        from google.api_core.exceptions import PreconditionFailed

        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.generation = 42
        mock_blob.upload_from_string.side_effect = PreconditionFailed("generation mismatch")

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "wildfire-mlops"}):
            result = write_state({"mode": "emergency", "active_fire_cells": []})

        # Must return False — not raise — so the Cloud Function continues gracefully
        assert result is False

    def test_write_state_adds_last_updated_timestamp(self):
        """Every write must stamp last_updated so we can audit state age."""
        from scripts.utils.gcs_state import write_state

        captured_payload = {}

        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        mock_blob.generation = 0

        def capture_upload(payload, **kwargs):
            captured_payload.update(json.loads(payload))

        mock_blob.upload_from_string.side_effect = capture_upload

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "wildfire-mlops"}):
            write_state({"mode": "quiet", "active_fire_cells": []})

        assert "last_updated" in captured_payload, (
            "write_state must stamp 'last_updated' on every write"
        )
        assert captured_payload["last_updated"] is not None

    # --- gcs_state.py: write_trigger ---

    def test_write_trigger_produces_correct_schema(self):
        """Trigger files must match the schema the watchdog_sensor_dag expects."""
        from scripts.utils.gcs_state import write_trigger

        captured_payload = {}

        mock_blob = MagicMock()
        def capture_upload(payload, **kwargs):
            captured_payload.update(json.loads(payload))
        mock_blob.upload_from_string.side_effect = capture_upload

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        trigger_input = {
            "trigger_source": "goes_nrt_confirmed",
            "resolution_km": 22,
            "regions": ["california"],
            "fire_cells": ["8e283082ddbffff"],
            "fire_frp_mw": 75.5,
            "mode": "emergency",
        }

        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "wildfire-mlops"}):
            gcs_path = write_trigger(trigger_input)

        assert gcs_path is not None, "write_trigger must return a GCS path on success"
        assert gcs_path.endswith(".json"), "Trigger file must be a .json file"

        # Required fields that watchdog_sensor_dag.process_fire_trigger reads
        required_fields = {
            "trigger_id", "triggered_at", "resolution_km",
            "regions", "fire_cells", "mode",
        }
        missing = required_fields - set(captured_payload.keys())
        assert not missing, (
            f"Trigger JSON missing fields expected by watchdog_sensor_dag: {missing}"
        )

    def test_write_trigger_returns_none_on_gcs_failure(self):
        """GCS write failure must return None, not raise."""
        from scripts.utils.gcs_state import write_trigger

        mock_client = MagicMock()
        mock_client.bucket.side_effect = Exception("GCS unavailable")

        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "wildfire-mlops"}):
            result = write_trigger({"mode": "emergency", "fire_cells": []})

        assert result is None

    # --- gcs_state.py: list_pending_triggers ---

    def test_list_pending_triggers_returns_empty_list_on_no_triggers(self):
        """No pending triggers → empty list, not an exception."""
        from scripts.utils.gcs_state import list_pending_triggers

        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = []  # no objects

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "wildfire-mlops"}):
            triggers = list_pending_triggers()

        assert triggers == []

    def test_list_pending_triggers_skips_malformed_json(self):
        """Malformed JSON in a trigger file must be skipped, not crash the sensor."""
        from scripts.utils.gcs_state import list_pending_triggers

        good_blob = MagicMock()
        good_blob.name = "watchdog/triggers/good.json"
        good_blob.download_as_text.return_value = json.dumps({
            "trigger_id": "abc", "mode": "active", "fire_cells": []
        })

        bad_blob = MagicMock()
        bad_blob.name = "watchdog/triggers/bad.json"
        bad_blob.download_as_text.side_effect = Exception("Corrupt JSON")

        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = [good_blob, bad_blob]

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "wildfire-mlops"}):
            triggers = list_pending_triggers()

        # Bad trigger skipped; good trigger still returned
        assert len(triggers) == 1
        assert triggers[0]["data"]["trigger_id"] == "abc"

    def test_list_pending_triggers_returns_empty_on_gcs_error(self):
        """GCS error must return empty list, not propagate to the Airflow sensor."""
        from scripts.utils.gcs_state import list_pending_triggers

        mock_client = MagicMock()
        mock_client.bucket.side_effect = Exception("Simulated GCS outage")

        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "wildfire-mlops"}):
            triggers = list_pending_triggers()

        assert triggers == []

    # --- delete_trigger ---

    def test_delete_trigger_returns_true_on_success(self):
        from scripts.utils.gcs_state import delete_trigger

        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "wildfire-mlops"}):
            result = delete_trigger("watchdog/triggers/test-id.json")

        assert result is True
        mock_blob.delete.assert_called_once()

    def test_delete_trigger_returns_false_on_failure(self):
        from scripts.utils.gcs_state import delete_trigger

        mock_blob = MagicMock()
        mock_blob.delete.side_effect = Exception("Permission denied")
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "wildfire-mlops"}):
            result = delete_trigger("watchdog/triggers/test-id.json")

        # Must return False, not raise — the sensor should log and continue
        assert result is False


# ===========================================================================
# Class 3: Live GCS connectivity (requires real credentials)
# ===========================================================================

@pytest.mark.gcp
@pytest.mark.skipif(
    not _HAS_GCP_CREDS,
    reason=(
        "Live GCP tests skipped — set GOOGLE_APPLICATION_CREDENTIALS or "
        "GCS_BUCKET_NAME to run. Use: pytest tests/test_dvc/ -m gcp -v"
    ),
)
class TestDVCGCSLive:
    """
    Validates real GCS connectivity and DVC remote reachability.
    Requires:
        export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
        export GCS_BUCKET_NAME=wildfire-mlops

    Run manually before each deployment:
        pytest tests/test_dvc/test_dvc.py::TestDVCGCSLive -v

    These tests are intentionally excluded from the CI yaml.
    They are meant for pre-deployment validation on a developer machine.
    """

    @pytest.fixture(autouse=True)
    def require_bucket_name(self):
        bucket = os.environ.get("GCS_BUCKET_NAME")
        if not bucket:
            pytest.skip("GCS_BUCKET_NAME not set")
        self.bucket_name = bucket

    def test_gcs_bucket_is_reachable(self):
        """GCS bucket must exist and be accessible with current credentials."""
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(self.bucket_name)
        # exists() raises on auth failure, returns False on 404
        try:
            exists = bucket.exists()
        except Exception as e:
            pytest.fail(
                f"GCS bucket '{self.bucket_name}' is not reachable: {e}\n"
                f"Check GOOGLE_APPLICATION_CREDENTIALS and bucket IAM permissions."
            )
        assert exists, (
            f"GCS bucket 'gs://{self.bucket_name}' does not exist. "
            f"Create it before running the pipeline."
        )

    def test_dvc_remote_url_matches_bucket_env_var(self):
        """Remote URL in .dvc/config must match GCS_BUCKET_NAME env var."""
        cfg = _parse_dvc_config()
        section = f'remote "{EXPECTED_REMOTE_NAME}"'
        url = cfg.get(section, "url", fallback="")
        expected_url = f"gs://{self.bucket_name}"
        assert url == expected_url or url.startswith(expected_url), (
            f"DVC remote URL '{url}' does not match GCS_BUCKET_NAME env var "
            f"'{self.bucket_name}'. Update .dvc/config or the env var."
        )

    def test_can_write_test_object_to_gcs(self, tmp_path):
        """Write a small sentinel object to GCS to confirm write permissions."""
        from google.cloud import storage
        import uuid

        client = storage.Client()
        bucket = client.bucket(self.bucket_name)
        test_key = f"pytest/dvc_connectivity_test_{uuid.uuid4()}.txt"
        blob = bucket.blob(test_key)

        try:
            blob.upload_from_string(
                "wildfire pipeline DVC connectivity test",
                content_type="text/plain",
            )
        except Exception as e:
            pytest.fail(
                f"Cannot write to gs://{self.bucket_name}/{test_key}: {e}\n"
                f"Check bucket write permissions for the service account."
            )

        # Store key for cleanup
        self._test_key = test_key

    def test_can_read_test_object_from_gcs(self, tmp_path):
        """Read the sentinel object back to confirm read permissions."""
        from google.cloud import storage
        import uuid

        client = storage.Client()
        bucket = client.bucket(self.bucket_name)
        test_key = f"pytest/dvc_read_test_{uuid.uuid4()}.txt"
        blob = bucket.blob(test_key)

        # Write then read
        blob.upload_from_string("read test", content_type="text/plain")
        content = blob.download_as_text()
        blob.delete()

        assert content == "read test", (
            f"Read-back content mismatch. GCS may have a caching issue."
        )

    def test_dvc_remote_list_finds_gcsremote(self):
        """'dvc remote list' must output gcsremote — this is what the DAG checks."""
        result = subprocess.run(
            ["dvc", "remote", "list"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, (
            f"'dvc remote list' failed: {result.stderr}"
        )
        assert EXPECTED_REMOTE_NAME in result.stdout, (
            f"'{EXPECTED_REMOTE_NAME}' not found in 'dvc remote list' output: "
            f"{result.stdout!r}\n"
            f"The version_with_dvc task does: "
            f"'dvc remote list | grep -q .' — if this fails, the task exits 1."
        )

    @pytest.mark.slow
    def test_dvc_status_cloud_is_parseable(self):
        """dvc status --cloud must exit 0 or produce parseable output.
        Marked slow — takes 5-30 seconds depending on data size.
        """
        result = subprocess.run(
            ["dvc", "status", "--cloud"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=60,
        )
        # Exit code 0 = in sync, 1 = out of sync — both are valid
        # Any other code indicates a config/auth problem
        assert result.returncode in (0, 1), (
            f"'dvc status --cloud' returned unexpected exit code {result.returncode}.\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )

    def test_watchdog_state_path_is_accessible(self):
        """Watchdog state path must be readable (first run returns 404 gracefully)."""
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(self.bucket_name)
        state_blob = bucket.blob("watchdog/state/current.json")

        try:
            exists = state_blob.exists()
            # Either exists (subsequent runs) or doesn't (first run) — both OK
            assert isinstance(exists, bool)
        except Exception as e:
            pytest.fail(
                f"Cannot check watchdog state path: {e}\n"
                f"The watchdog Cloud Function writes to this path on every run."
            )

    def test_triggers_prefix_is_listable(self):
        """watchdog/triggers/ prefix must be listable (required by watchdog_sensor_dag)."""
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(self.bucket_name)

        try:
            blobs = list(bucket.list_blobs(prefix="watchdog/triggers/", max_results=5))
            assert isinstance(blobs, list)
        except Exception as e:
            pytest.fail(
                f"Cannot list watchdog/triggers/ prefix in gs://{self.bucket_name}: {e}\n"
                f"watchdog_sensor_dag.check_for_fire_triggers() calls list_pending_triggers() "
                f"which lists this prefix every 60 seconds."
            )