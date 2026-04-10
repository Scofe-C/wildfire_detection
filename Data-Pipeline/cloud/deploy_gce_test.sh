#!/usr/bin/env bash
# =============================================================================
# Wildfire MLOps — Ephemeral GCE Test Deployment
# =============================================================================
# Provisions a time-boxed e2-standard-8 VM for the 3-4 day live pipeline test.
# The VM auto-stops via a GCE Resource Policy — no OS-level timer needed.
#
# Prerequisites:
#   1. gcloud CLI installed and authenticated (gcloud auth login)
#   2. .env file populated (FIRMS_MAP_KEY, GCS_BUCKET_NAME, GOOGLE_CLOUD_PROJECT)
#   3. GCS bucket already created
#   4. Service account with roles/storage.objectAdmin + roles/compute.instanceAdmin
#
# Usage:
#   chmod +x cloud/deploy_gce_test.sh
#   ./cloud/deploy_gce_test.sh
#
# What this creates:
#   - GCS staging:  gs://{BUCKET}/gce-test/pipeline.tar.gz + .env
#   - Resource Policy: wildfire-test-autostop (hard stop at deploy_time + 96h)
#   - GCE VM: wildfire-test-vm (e2-standard-8, 50GB PD-SSD, Debian 12)
#   - Firewall rule: allow TCP 8080 for Airflow UI access
#
# Estimated cost (96h):
#   e2-standard-8: ~$0.268/hr × 96h ≈ $25.73
#   50GB PD-SSD:   ~$0.17/mo prorated  ≈ $0.02
#   Total:         ≈ $25.75 (well under a $30 budget alert)
#
# Cleanup (run after test):
#   gcloud compute instances delete wildfire-test-vm --zone=us-central1-a -q
#   gcloud compute resource-policies delete wildfire-test-autostop --region=us-central1 -q
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VM_NAME="wildfire-test-vm"
MACHINE_TYPE="e2-standard-8"
BOOT_DISK_SIZE="50GB"
BOOT_DISK_TYPE="pd-ssd"
IMAGE_FAMILY="debian-12"
IMAGE_PROJECT="debian-cloud"
TTL_HOURS=96
GCS_STAGING_PREFIX="gce-test"
HEALTH_MARKER="gce-test/health/ready"
POLL_INTERVAL_SEC=60
POLL_TIMEOUT_SEC=900   # 15 minutes

# ---------------------------------------------------------------------------
# Load and validate .env
# ---------------------------------------------------------------------------
ENV_FILE="${PROJECT_ROOT}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: .env not found at ${ENV_FILE}"
    echo "Copy .env.example and fill in values."
    exit 1
fi

# Export variables from .env (skip comments and blank lines)
set -a
# shellcheck disable=SC1090
source <(grep -v '^\s*#' "${ENV_FILE}" | grep -v '^\s*$')
set +a
echo "✓ Loaded .env"

REQUIRED_VARS=("GCS_BUCKET_NAME" "FIRMS_MAP_KEY" "GOOGLE_CLOUD_PROJECT")
for var in "${REQUIRED_VARS[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: Required variable ${var} is not set in .env"
        exit 1
    fi
done

PROJECT_ID="${GOOGLE_CLOUD_PROJECT}"
BUCKET="${GCS_BUCKET_NAME}"
ZONE="${GCE_ZONE:-us-central1-a}"
REGION="${ZONE%-*}"  # Extract region from zone (us-central1-a → us-central1)
POLICY_NAME="wildfire-test-autostop"

echo ""
echo "=== Wildfire Ephemeral GCE Test Deployment ==="
echo "Project:      ${PROJECT_ID}"
echo "Zone:         ${ZONE}"
echo "Machine:      ${MACHINE_TYPE}"
echo "Bucket:       gs://${BUCKET}"
echo "TTL:          ${TTL_HOURS} hours"
echo ""

# ---------------------------------------------------------------------------
# Preflight: verify gcloud auth and project
# ---------------------------------------------------------------------------
echo "→ Verifying gcloud configuration..."
ACTIVE_PROJECT=$(gcloud config get-value project 2>/dev/null || true)
if [[ "${ACTIVE_PROJECT}" != "${PROJECT_ID}" ]]; then
    echo "  Setting active project to ${PROJECT_ID}"
    gcloud config set project "${PROJECT_ID}" --quiet
fi

# Verify authentication
if ! gcloud auth print-access-token &>/dev/null; then
    echo "ERROR: Not authenticated. Run: gcloud auth login"
    exit 1
fi
echo "✓ gcloud authenticated for project ${PROJECT_ID}"

# ---------------------------------------------------------------------------
# Preflight: check for existing VM (prevent double-deploy)
# ---------------------------------------------------------------------------
if gcloud compute instances describe "${VM_NAME}" --zone="${ZONE}" &>/dev/null; then
    echo ""
    echo "WARNING: VM '${VM_NAME}' already exists in ${ZONE}."
    echo "Options:"
    echo "  1. Delete it first:  gcloud compute instances delete ${VM_NAME} --zone=${ZONE} -q"
    echo "  2. SSH into it:      gcloud compute ssh ${VM_NAME} --zone=${ZONE}"
    echo ""
    read -rp "Delete existing VM and redeploy? [y/N] " confirm
    if [[ "${confirm}" =~ ^[Yy]$ ]]; then
        echo "→ Deleting existing VM..."
        gcloud compute instances delete "${VM_NAME}" --zone="${ZONE}" --quiet
        echo "✓ Old VM deleted"
    else
        echo "Aborting."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Step 1: Package and upload project to GCS
# ---------------------------------------------------------------------------
echo "→ Packaging project for upload..."
STAGING_TAR="/tmp/wildfire-pipeline-gce.tar.gz"

# Exclude heavy/unnecessary dirs from the tar
tar -czf "${STAGING_TAR}" \
    -C "${PROJECT_ROOT}/.." \
    --exclude='Data-Pipeline/logs' \
    --exclude='Data-Pipeline/data/raw/*' \
    --exclude='Data-Pipeline/data/processed/*' \
    --exclude='Data-Pipeline/.git' \
    --exclude='Data-Pipeline/__pycache__' \
    --exclude='Data-Pipeline/**/__pycache__' \
    --exclude='Data-Pipeline/gcp-key.json' \
    "Data-Pipeline"

TAR_SHA256=$(sha256sum "${STAGING_TAR}" | awk '{print $1}')
TAR_SIZE=$(du -h "${STAGING_TAR}" | awk '{print $1}')
echo "  Package: ${TAR_SIZE}, SHA256: ${TAR_SHA256:0:16}..."

echo "→ Uploading to gs://${BUCKET}/${GCS_STAGING_PREFIX}/..."
gcloud storage cp "${STAGING_TAR}" "gs://${BUCKET}/${GCS_STAGING_PREFIX}/pipeline.tar.gz" --quiet
gcloud storage cp "${ENV_FILE}" "gs://${BUCKET}/${GCS_STAGING_PREFIX}/.env" --quiet

# Clear any stale health marker from a previous deploy
gcloud storage rm "gs://${BUCKET}/${HEALTH_MARKER}" --quiet 2>/dev/null || true

echo "✓ Project staged to GCS"

# ---------------------------------------------------------------------------
# Step 2: Create Resource Policy (auto-stop after TTL_HOURS)
# ---------------------------------------------------------------------------
echo "→ Creating resource policy for auto-stop..."

# Compute the stop time: now + TTL_HOURS
STOP_TIME_UTC=$(date -u -d "+${TTL_HOURS} hours" "+%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || \
                date -u -v "+${TTL_HOURS}H" "+%Y-%m-%dT%H:%M:%SZ" 2>/dev/null)

if [[ -z "${STOP_TIME_UTC}" ]]; then
    echo "ERROR: Could not compute stop time. Ensure 'date' supports -d or -v."
    exit 1
fi

echo "  VM will auto-stop at: ${STOP_TIME_UTC}"

# Delete old policy if it exists (policies are immutable — must recreate)
gcloud compute resource-policies delete "${POLICY_NAME}" \
    --region="${REGION}" --quiet 2>/dev/null || true

# Resource policy: stop the VM on the computed schedule.
# GCE resource-policies use cron-like schedules. We create one that fires
# once at the target hour on the target day. Since cron doesn't support
# one-shot, we use the exact minute/hour/day-of-month/month and accept
# that it would re-fire next year (the VM will be long deleted).
STOP_MINUTE=$(date -u -d "+${TTL_HOURS} hours" "+%-M" 2>/dev/null || \
              date -u -v "+${TTL_HOURS}H" "+%-M" 2>/dev/null)
STOP_HOUR=$(date -u -d "+${TTL_HOURS} hours" "+%-H" 2>/dev/null || \
            date -u -v "+${TTL_HOURS}H" "+%-H" 2>/dev/null)
STOP_DOM=$(date -u -d "+${TTL_HOURS} hours" "+%-d" 2>/dev/null || \
           date -u -v "+${TTL_HOURS}H" "+%-d" 2>/dev/null)
STOP_MON=$(date -u -d "+${TTL_HOURS} hours" "+%-m" 2>/dev/null || \
           date -u -v "+${TTL_HOURS}H" "+%-m" 2>/dev/null)

STOP_CRON="${STOP_MINUTE} ${STOP_HOUR} ${STOP_DOM} ${STOP_MON} *"

gcloud compute resource-policies create vm-maintenance "${POLICY_NAME}" \
    --region="${REGION}" \
    --vm-stop-schedule="${STOP_CRON}" \
    --timezone="UTC" \
    --description="Auto-stop wildfire test VM at ${STOP_TIME_UTC} (${TTL_HOURS}h TTL)" \
    --quiet

echo "✓ Resource policy '${POLICY_NAME}' created (cron: ${STOP_CRON} UTC)"

# ---------------------------------------------------------------------------
# Step 3: Create firewall rule for Airflow UI (TCP 8080)
# ---------------------------------------------------------------------------
FIREWALL_RULE="allow-airflow-ui"
if ! gcloud compute firewall-rules describe "${FIREWALL_RULE}" &>/dev/null; then
    echo "→ Creating firewall rule for Airflow UI (TCP 8080)..."
    gcloud compute firewall-rules create "${FIREWALL_RULE}" \
        --direction=INGRESS \
        --priority=1000 \
        --network=default \
        --action=ALLOW \
        --rules=tcp:8080 \
        --target-tags=airflow-ui \
        --source-ranges=0.0.0.0/0 \
        --description="Allow Airflow UI access for wildfire test VM" \
        --quiet
    echo "✓ Firewall rule created"
else
    echo "✓ Firewall rule '${FIREWALL_RULE}' already exists"
fi

# ---------------------------------------------------------------------------
# Step 4: Create VM
# ---------------------------------------------------------------------------
echo "→ Creating VM '${VM_NAME}'..."

gcloud compute instances create "${VM_NAME}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --boot-disk-size="${BOOT_DISK_SIZE}" \
    --boot-disk-type="${BOOT_DISK_TYPE}" \
    --image-family="${IMAGE_FAMILY}" \
    --image-project="${IMAGE_PROJECT}" \
    --tags=airflow-ui \
    --resource-policies="${POLICY_NAME}" \
    --scopes=cloud-platform \
    --metadata=gcs-bucket="${BUCKET}",gcs-staging-prefix="${GCS_STAGING_PREFIX}",health-marker="${HEALTH_MARKER}" \
    --metadata-from-file=startup-script="${SCRIPT_DIR}/gce_startup.sh" \
    --quiet

VM_IP=$(gcloud compute instances describe "${VM_NAME}" \
    --zone="${ZONE}" \
    --format="get(networkInterfaces[0].accessConfigs[0].natIP)")

echo "✓ VM created: ${VM_NAME} (${VM_IP})"

# ---------------------------------------------------------------------------
# Step 5: Poll for health marker
# ---------------------------------------------------------------------------
echo ""
echo "→ Waiting for Airflow to come online (polling every ${POLL_INTERVAL_SEC}s, timeout ${POLL_TIMEOUT_SEC}s)..."
echo "  Startup script is installing Docker and launching Airflow..."

ELAPSED=0
while [[ ${ELAPSED} -lt ${POLL_TIMEOUT_SEC} ]]; do
    if gcloud storage ls "gs://${BUCKET}/${HEALTH_MARKER}" &>/dev/null; then
        echo ""
        echo "✓ Health marker found — Airflow is running!"
        break
    fi
    printf "."
    sleep "${POLL_INTERVAL_SEC}"
    ELAPSED=$((ELAPSED + POLL_INTERVAL_SEC))
done

if [[ ${ELAPSED} -ge ${POLL_TIMEOUT_SEC} ]]; then
    echo ""
    echo "WARNING: Health marker not found after ${POLL_TIMEOUT_SEC}s."
    echo "The startup script may still be running. Check with:"
    echo "  gcloud compute ssh ${VM_NAME} --zone=${ZONE} -- 'sudo journalctl -u google-startup-scripts -f'"
fi

# ---------------------------------------------------------------------------
# Deployment Summary
# ---------------------------------------------------------------------------
echo ""
echo "==================================================================="
echo "  DEPLOYMENT COMPLETE"
echo "==================================================================="
echo ""
echo "  VM Name:        ${VM_NAME}"
echo "  External IP:    ${VM_IP}"
echo "  Airflow UI:     http://${VM_IP}:8080  (user: airflow, pass: airflow)"
echo "  Auto-Stop:      ${STOP_TIME_UTC} (${TTL_HOURS}h from now)"
echo "  Package SHA256: ${TAR_SHA256:0:16}..."
echo ""
echo "  SSH:            gcloud compute ssh ${VM_NAME} --zone=${ZONE}"
echo "  Logs:           gcloud compute ssh ${VM_NAME} --zone=${ZONE} -- 'cd /opt/wildfire && docker compose logs -f'"
echo "  Serial console: gcloud compute instances get-serial-port-output ${VM_NAME} --zone=${ZONE}"
echo ""
echo "  ⚠  CLEANUP (run after test is done):"
echo "     gcloud compute instances delete ${VM_NAME} --zone=${ZONE} -q"
echo "     gcloud compute resource-policies delete ${POLICY_NAME} --region=${REGION} -q"
echo "     gcloud storage rm -r gs://${BUCKET}/${GCS_STAGING_PREFIX}/"
echo ""
echo "  ⚠  BILLING: Set a budget alert at \$30 in GCP Console:"
echo "     https://console.cloud.google.com/billing/budgets?project=${PROJECT_ID}"
echo "     (Resource Policy stops the VM, but disk charges continue until deletion.)"
echo ""
