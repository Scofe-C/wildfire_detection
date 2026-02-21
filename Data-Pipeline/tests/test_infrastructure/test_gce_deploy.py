"""
Tests for Sprint 1: Ephemeral GCE Infrastructure
=================================================
Validates deploy_gce_test.sh and gce_startup.sh for:
  - Bash syntax correctness
  - Required safety invariants (resource policy, no hardcoded secrets)
  - Structural completeness (all required steps present)
  - Script consistency (bucket/prefix references match)

These tests run offline — no GCP credentials required.
"""

import re
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEPLOY_SCRIPT = PROJECT_ROOT / "cloud" / "deploy_gce_test.sh"
STARTUP_SCRIPT = PROJECT_ROOT / "cloud" / "gce_startup.sh"


def _bash_works() -> bool:
    """Return True only if bash can actually execute (not a broken WSL shim)."""
    try:
        r = subprocess.run(
            ["bash", "--version"], capture_output=True, timeout=5,
        )
        return r.returncode == 0
    except Exception:
        return False


_SKIP_BASH = not _bash_works()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def deploy_text():
    """Raw text of deploy_gce_test.sh."""
    return DEPLOY_SCRIPT.read_text(encoding="utf-8")


@pytest.fixture
def startup_text():
    """Raw text of gce_startup.sh."""
    return STARTUP_SCRIPT.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Script existence and permissions
# ---------------------------------------------------------------------------

class TestScriptFiles:
    def test_deploy_script_exists(self):
        assert DEPLOY_SCRIPT.exists(), "cloud/deploy_gce_test.sh not found"

    def test_startup_script_exists(self):
        assert STARTUP_SCRIPT.exists(), "cloud/gce_startup.sh not found"

    def test_deploy_script_has_shebang(self, deploy_text):
        assert deploy_text.startswith("#!/"), "deploy script missing shebang"

    def test_startup_script_has_shebang(self, startup_text):
        assert startup_text.startswith("#!/"), "startup script missing shebang"

    @pytest.mark.skipif(_SKIP_BASH, reason="bash not available or broken")
    def test_deploy_script_bash_syntax(self):
        """Validate bash syntax without executing (bash -n)."""
        result = subprocess.run(
            ["bash", "-n", str(DEPLOY_SCRIPT)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"deploy_gce_test.sh has syntax errors:\n{result.stderr}"
        )

    @pytest.mark.skipif(_SKIP_BASH, reason="bash not available or broken")
    def test_startup_script_bash_syntax(self):
        result = subprocess.run(
            ["bash", "-n", str(STARTUP_SCRIPT)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"gce_startup.sh has syntax errors:\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# Safety invariants — deploy_gce_test.sh
# ---------------------------------------------------------------------------

class TestDeploySafety:
    """Verify the deploy script enforces all safety nets."""

    def test_uses_set_euo_pipefail(self, deploy_text):
        assert "set -euo pipefail" in deploy_text, (
            "Deploy script must use 'set -euo pipefail' for strict error handling"
        )

    def test_validates_required_env_vars(self, deploy_text):
        for var in ["GCS_BUCKET_NAME", "FIRMS_MAP_KEY", "GOOGLE_CLOUD_PROJECT"]:
            assert var in deploy_text, (
                f"Deploy script must validate required env var: {var}"
            )

    def test_creates_resource_policy(self, deploy_text):
        assert "resource-policies create" in deploy_text, (
            "Deploy script must create a GCE resource policy for auto-stop"
        )

    def test_resource_policy_uses_vm_stop(self, deploy_text):
        assert "vm-stop-schedule" in deploy_text or "vm-maintenance" in deploy_text, (
            "Resource policy must use vm-stop-schedule (not just instance-schedule)"
        )

    def test_ttl_hours_defined(self, deploy_text):
        match = re.search(r'TTL_HOURS=(\d+)', deploy_text)
        assert match, "TTL_HOURS constant must be defined"
        ttl = int(match.group(1))
        assert 72 <= ttl <= 120, (
            f"TTL_HOURS={ttl} is outside safe range [72, 120]"
        )

    def test_machine_type_is_e2_standard_8(self, deploy_text):
        assert "e2-standard-8" in deploy_text, (
            "Machine type must be e2-standard-8 (32GB RAM) per architecture spec"
        )

    def test_checks_for_existing_vm(self, deploy_text):
        assert "instances describe" in deploy_text, (
            "Deploy script must check for existing VM before creating"
        )

    def test_no_hardcoded_api_keys(self, deploy_text):
        # Must not contain anything that looks like a real API key
        assert "AIza" not in deploy_text, "Hardcoded Google API key detected"
        # Check no literal FIRMS keys (they're typically 32+ hex chars)
        long_hex = re.findall(r'[0-9a-fA-F]{32,}', deploy_text)
        # Filter out SHA256 references which are legitimate
        suspicious = [h for h in long_hex if "sha256" not in deploy_text[max(0, deploy_text.index(h)-30):deploy_text.index(h)].lower()]
        assert len(suspicious) == 0, (
            f"Possible hardcoded secret found: {suspicious[0][:16]}..."
        )

    def test_excludes_gcp_key_from_tar(self, deploy_text):
        assert "gcp-key.json" in deploy_text, (
            "Deploy script must reference gcp-key.json (to exclude it from tar)"
        )
        assert "--exclude" in deploy_text and "gcp-key" in deploy_text, (
            "Deploy script must exclude gcp-key.json from the tar upload"
        )

    def test_uploads_env_to_gcs(self, deploy_text):
        # The .env file must be uploaded so the VM can access secrets
        assert ".env" in deploy_text and "gcloud storage cp" in deploy_text, (
            "Deploy script must upload .env to GCS for the VM"
        )

    def test_prints_cleanup_instructions(self, deploy_text):
        assert "instances delete" in deploy_text, (
            "Deploy script must print cleanup instructions including VM deletion"
        )
        assert "resource-policies delete" in deploy_text, (
            "Deploy script must print cleanup instructions including policy deletion"
        )

    def test_prints_billing_warning(self, deploy_text):
        lower = deploy_text.lower()
        assert "billing" in lower or "budget" in lower, (
            "Deploy script must include a billing/budget warning"
        )

    def test_health_marker_poll(self, deploy_text):
        assert "HEALTH_MARKER" in deploy_text or "health" in deploy_text.lower(), (
            "Deploy script must poll for a health marker from the VM"
        )


# ---------------------------------------------------------------------------
# Safety invariants — gce_startup.sh
# ---------------------------------------------------------------------------

class TestStartupSafety:
    """Verify the startup script handles boot correctly."""

    def test_uses_set_euo_pipefail(self, startup_text):
        assert "set -euo pipefail" in startup_text

    def test_reads_metadata(self, startup_text):
        assert "metadata.google.internal" in startup_text, (
            "Startup script must read GCS config from instance metadata"
        )

    def test_installs_docker(self, startup_text):
        assert "docker-ce" in startup_text, (
            "Startup script must install Docker CE"
        )

    def test_installs_compose_plugin(self, startup_text):
        assert "docker-compose-plugin" in startup_text, (
            "Startup script must install docker-compose-plugin"
        )

    def test_downloads_from_gcs(self, startup_text):
        assert "gcloud storage cp" in startup_text, (
            "Startup script must download pipeline from GCS"
        )

    def test_runs_docker_compose_up(self, startup_text):
        assert "docker compose up" in startup_text, (
            "Startup script must run 'docker compose up'"
        )

    def test_writes_health_marker(self, startup_text):
        # Must write something to GCS to signal readiness
        health_writes = [
            line for line in startup_text.splitlines()
            if "HEALTH_MARKER" in line and "gcloud" in line
        ]
        assert len(health_writes) > 0, (
            "Startup script must write a health marker to GCS"
        )

    def test_handles_reboot_idempotently(self, startup_text):
        # Must detect existing installation and skip reinstall
        assert "docker-compose.yaml" in startup_text or "existing" in startup_text.lower(), (
            "Startup script must handle reboots idempotently"
        )

    def test_waits_for_airflow_health(self, startup_text):
        assert "health" in startup_text.lower() and ("curl" in startup_text or "healthy" in startup_text.lower()), (
            "Startup script must wait for Airflow health check before signaling ready"
        )

    def test_creates_data_directories(self, startup_text):
        for required_dir in ["data/raw", "data/processed", "data/static"]:
            assert required_dir in startup_text, (
                f"Startup script must create {required_dir} directory"
            )

    def test_logs_to_file(self, startup_text):
        assert "LOGFILE" in startup_text or "tee" in startup_text, (
            "Startup script must log to a file for debugging"
        )


# ---------------------------------------------------------------------------
# Consistency between scripts
# ---------------------------------------------------------------------------

class TestCrossScriptConsistency:
    """Verify deploy and startup scripts agree on conventions."""

    def test_gcs_prefix_consistent(self, deploy_text, startup_text):
        # Deploy defines GCS_STAGING_PREFIX, startup reads from metadata.
        # Both must reference pipeline.tar.gz as the artifact name.
        assert "pipeline.tar.gz" in deploy_text, (
            "Deploy must upload pipeline.tar.gz"
        )
        assert "pipeline.tar.gz" in startup_text, (
            "Startup must download pipeline.tar.gz"
        )

    def test_health_marker_key_consistent(self, deploy_text, startup_text):
        # Deploy defines HEALTH_MARKER and polls for it.
        # Startup writes to HEALTH_MARKER.
        # Both must reference the same GCS path pattern.
        assert "HEALTH_MARKER" in deploy_text, "Deploy must define HEALTH_MARKER"
        assert "HEALTH_MARKER" in startup_text, "Startup must write HEALTH_MARKER"

    def test_env_file_transfer(self, deploy_text, startup_text):
        # Deploy uploads .env, startup downloads .env
        assert ".env" in deploy_text, "Deploy must handle .env upload"
        assert ".env" in startup_text, "Startup must handle .env download"
