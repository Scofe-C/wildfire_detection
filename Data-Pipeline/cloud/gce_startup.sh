#!/usr/bin/env bash
# =============================================================================
# Wildfire MLOps — GCE VM Startup Script
# =============================================================================
# This script runs automatically on VM boot via GCE metadata startup-script.
# It installs Docker, downloads the pipeline from GCS, and starts Airflow.
#
# On success, writes a health marker to GCS so deploy_gce_test.sh knows
# the VM is ready.
#
# Logs: sudo journalctl -u google-startup-scripts -f
# =============================================================================

set -euo pipefail

LOGFILE="/var/log/wildfire-startup.log"
exec > >(tee -a "${LOGFILE}") 2>&1

echo "============================================="
echo "Wildfire GCE Startup — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================="

# ---------------------------------------------------------------------------
# Read deployment metadata
# ---------------------------------------------------------------------------
METADATA_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
METADATA_HEADER="Metadata-Flavor: Google"

GCS_BUCKET=$(curl -sf -H "${METADATA_HEADER}" "${METADATA_URL}/gcs-bucket")
GCS_PREFIX=$(curl -sf -H "${METADATA_HEADER}" "${METADATA_URL}/gcs-staging-prefix")
HEALTH_MARKER=$(curl -sf -H "${METADATA_HEADER}" "${METADATA_URL}/health-marker")

echo "GCS Bucket:  ${GCS_BUCKET}"
echo "GCS Prefix:  ${GCS_PREFIX}"
echo "Health Mark: ${HEALTH_MARKER}"

# REPO_ROOT mirrors the local repo root (one level above Data-Pipeline/).
# docker-compose.yaml uses context: .. so Docker needs this parent to exist.
REPO_ROOT="/opt/wildfire"
INSTALL_DIR="${REPO_ROOT}/Data-Pipeline"

# ---------------------------------------------------------------------------
# Guard: skip re-install if Docker is already running and project exists
# (handles VM reboot — startup script re-runs but doesn't need full reinstall)
# ---------------------------------------------------------------------------
if [[ -f "${INSTALL_DIR}/docker-compose.yaml" ]] && docker compose version &>/dev/null; then
    echo "→ Detected existing installation. Restarting containers..."
    cd "${INSTALL_DIR}"
    AIRFLOW_SERVICES="postgres airflow-init airflow-webserver airflow-scheduler"
    docker compose up -d ${AIRFLOW_SERVICES}
    echo "✓ Containers restarted after reboot"

    # Re-write health marker
    echo "restarted $(date -u '+%Y-%m-%dT%H:%M:%SZ')" | \
        gcloud storage cp - "gs://${GCS_BUCKET}/${HEALTH_MARKER}" --quiet
    echo "✓ Health marker updated"
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 1: Install Docker Engine (Debian 12)
# ---------------------------------------------------------------------------
echo "→ Installing Docker..."

apt-get update -qq
apt-get install -y -qq ca-certificates curl gnupg lsb-release

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | \
    gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

ARCH=$(dpkg --print-architecture)
CODENAME=$(. /etc/os-release && echo "${VERSION_CODENAME}")
echo "deb [arch=${ARCH} signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/debian ${CODENAME} stable" > \
    /etc/apt/sources.list.d/docker.list

apt-get update -qq
apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin

systemctl enable docker
systemctl start docker

echo "✓ Docker installed: $(docker --version)"
echo "  Compose: $(docker compose version)"

# ---------------------------------------------------------------------------
# Step 2: Download pipeline from GCS
# ---------------------------------------------------------------------------
echo "→ Downloading pipeline from GCS..."

# Extract into REPO_ROOT preserving the Data-Pipeline/ subdirectory.
# docker-compose.yaml has context: .. — Docker resolves that to REPO_ROOT,
# which must contain Data-Pipeline/ as a child (same layout as local repo).
mkdir -p "${REPO_ROOT}"
gcloud storage cp "gs://${GCS_BUCKET}/${GCS_PREFIX}/pipeline.tar.gz" /tmp/pipeline.tar.gz --quiet

# No --strip-components: tar expands to ${REPO_ROOT}/Data-Pipeline/
tar -xzf /tmp/pipeline.tar.gz -C "${REPO_ROOT}"
rm -f /tmp/pipeline.tar.gz

# .env goes inside Data-Pipeline/ where docker-compose.yaml lives
gcloud storage cp "gs://${GCS_BUCKET}/${GCS_PREFIX}/.env" "${INSTALL_DIR}/.env" --quiet

echo "✓ Pipeline extracted to ${INSTALL_DIR}"

# ---------------------------------------------------------------------------
# Step 3: Create required data directories
# ---------------------------------------------------------------------------
echo "→ Creating data directories..."
cd "${INSTALL_DIR}"
mkdir -p data/raw/firms data/raw/weather
mkdir -p data/processed/firms data/processed/weather data/processed/fused
mkdir -p data/static
mkdir -p logs

echo "✓ Data directories ready"

# ---------------------------------------------------------------------------
# Step 4: Build and start Airflow via Docker Compose
# ---------------------------------------------------------------------------
echo "→ Building and starting Airflow..."

# GCP key stub at REPO_ROOT (docker-compose default: ../gcp-key.json from INSTALL_DIR)
if [[ ! -f "${REPO_ROOT}/gcp-key.json" ]]; then
    echo '{}' > "${REPO_ROOT}/gcp-key.json"
fi

# dvc.lock volume mount will fail if the file doesn't exist
touch "${INSTALL_DIR}/dvc.lock"

# Build only the Airflow services (obj3-dashboard and frontend are deployed
# separately to Cloud Run — their build contexts don't exist on this VM)
AIRFLOW_SERVICES="postgres airflow-init airflow-webserver airflow-scheduler"
docker compose build --quiet ${AIRFLOW_SERVICES} 2>&1 | tail -5

# Start Airflow services (airflow-init runs to completion, then webserver/scheduler stay up)
docker compose up -d ${AIRFLOW_SERVICES}

echo "✓ Docker Compose services started"

# ---------------------------------------------------------------------------
# Step 5: Wait for Airflow webserver health check
# ---------------------------------------------------------------------------
echo "→ Waiting for Airflow webserver to become healthy..."

MAX_WAIT=300
ELAPSED=0
HEALTHY=false

while [[ ${ELAPSED} -lt ${MAX_WAIT} ]]; do
    # Airflow 2.x health endpoint returns {"metadatabase":{"status":"healthy"},...}
    if curl -sf http://localhost:8080/health 2>/dev/null | grep -q '"healthy"'; then
        HEALTHY=true
        break
    fi

    sleep 10
    ELAPSED=$((ELAPSED + 10))
    echo "  ... waiting (${ELAPSED}s / ${MAX_WAIT}s)"
done

if [[ "${HEALTHY}" == "true" ]]; then
    echo "✓ Airflow webserver is healthy"
else
    echo "WARNING: Airflow webserver did not become healthy within ${MAX_WAIT}s"
    echo "  Container status:"
    docker compose -f "${INSTALL_DIR}/docker-compose.yaml" ps
    echo "  Recent logs:"
    docker compose -f "${INSTALL_DIR}/docker-compose.yaml" logs --tail=30 airflow-webserver 2>&1 || true
    # Continue anyway — write a degraded health marker
fi

# ---------------------------------------------------------------------------
# Step 6: Write health marker to GCS
# ---------------------------------------------------------------------------
echo "→ Writing health marker to GCS..."

MARKER_CONTENT="vm=$(hostname),healthy=${HEALTHY},time=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "${MARKER_CONTENT}" | gcloud storage cp - "gs://${GCS_BUCKET}/${HEALTH_MARKER}" --quiet

echo "✓ Health marker written: ${MARKER_CONTENT}"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "============================================="
echo "Startup complete — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Airflow UI:  http://$(curl -sf -H "${METADATA_HEADER}" \
    http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip):8080"
echo "  Credentials: airflow / airflow"
echo "============================================="
