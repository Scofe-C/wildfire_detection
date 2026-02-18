#!/bin/bash
# =============================================================================
# Wildfire MLOps — GCP Cloud Function Deployment
# =============================================================================
# Deploys the fire_watchdog Cloud Function and configures Cloud Scheduler
# for adaptive polling (quiet: 30min, active: 15min, emergency: 5min).
#
# Prerequisites:
#   1. gcloud CLI installed and authenticated (gcloud auth login)
#   2. .env file populated (copy from .env.example and fill in values)
#   3. GCS bucket already created (gcloud storage buckets create gs://{BUCKET})
#   4. Billing enabled on the project
#
# Usage:
#   chmod +x cloud/deploy.sh
#   ./cloud/deploy.sh
#
# What this deploys:
#   - Cloud Function: fire-watchdog (HTTP trigger, 2nd gen)
#   - Cloud Scheduler job: watchdog-quiet (30 min, default)
#   - Initial industrial sources config file to GCS
#   - schema_config.yaml copy to GCS (Cloud Function reads it at runtime)
#
# Cost estimate (free tier):
#   Cloud Functions: 2M invocations/month free → ~1,440 invocations/month at 30min
#   Cloud Scheduler: 3 jobs free → we use 1
#   GCS: < 1 MB state files → negligible cost
#
# Total estimated cost: $0/month on free tier
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
    echo "✓ Loaded .env"
else
    echo "ERROR: .env file not found. Copy .env.example and fill in values."
    exit 1
fi

# Validate required variables
REQUIRED_VARS=("GCS_BUCKET_NAME" "FIRMS_MAP_KEY" "GOOGLE_CLOUD_PROJECT")
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: Required environment variable $var is not set in .env"
        exit 1
    fi
done

PROJECT_ID="${GOOGLE_CLOUD_PROJECT}"
BUCKET="${GCS_BUCKET_NAME}"
REGION="${GCP_REGION:-us-central1}"
FUNCTION_NAME="fire-watchdog"
SCHEDULER_JOB_QUIET="watchdog-quiet"
SCHEDULER_JOB_ACTIVE="watchdog-active"

echo ""
echo "=== Wildfire MLOps GCP Deployment ==="
echo "Project:  ${PROJECT_ID}"
echo "Bucket:   gs://${BUCKET}"
echo "Region:   ${REGION}"
echo "Function: ${FUNCTION_NAME}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Enable required GCP APIs
# ---------------------------------------------------------------------------
echo "→ Enabling GCP APIs..."
gcloud services enable \
    cloudfunctions.googleapis.com \
    cloudscheduler.googleapis.com \
    storage.googleapis.com \
    --project="${PROJECT_ID}" \
    --quiet

echo "✓ APIs enabled"

# ---------------------------------------------------------------------------
# Step 2: Upload schema_config.yaml and initial configs to GCS
# ---------------------------------------------------------------------------
echo "→ Uploading watchdog config to GCS..."

# Upload full schema config (Cloud Function reads watchdog section from it)
gcloud storage cp configs/schema_config.yaml \
    "gs://${BUCKET}/watchdog/config/schema_config.yaml"

# Upload initial industrial sources file (empty list — add sources manually)
if [ ! -f "/tmp/industrial_sources.json" ]; then
    cat > /tmp/industrial_sources.json << 'JSON'
[
  {"name": "Valero Benicia Refinery", "lat": 38.05, "lon": -122.15, "radius_km": 2.5},
  {"name": "Chevron Richmond Refinery", "lat": 37.93, "lon": -122.38, "radius_km": 2.5},
  {"name": "Phillips 66 Rodeo Refinery", "lat": 38.03, "lon": -122.27, "radius_km": 2.0},
  {"name": "PBF Energy Martinez Refinery", "lat": 37.99, "lon": -122.13, "radius_km": 2.0},
  {"name": "Shell Martinez Refinery", "lat": 38.01, "lon": -122.08, "radius_km": 2.5},
  {"name": "Tesoro Golden Eagle Refinery", "lat": 38.05, "lon": -122.13, "radius_km": 2.0},
  {"name": "ExxonMobil Torrance Refinery", "lat": 33.84, "lon": -118.34, "radius_km": 2.0},
  {"name": "Valero Wilmington Refinery", "lat": 33.77, "lon": -118.27, "radius_km": 2.0},
  {"name": "Marathon Carson Refinery", "lat": 33.86, "lon": -118.26, "radius_km": 2.0},
  {"name": "PBF Energy Torrance Refinery", "lat": 33.84, "lon": -118.34, "radius_km": 2.0},
  {"name": "Valero Texas City Refinery", "lat": 29.38, "lon": -94.96, "radius_km": 2.5},
  {"name": "Marathon Galveston Bay Refinery", "lat": 29.74, "lon": -95.01, "radius_km": 2.5},
  {"name": "LyondellBasell Houston Refinery", "lat": 29.72, "lon": -95.17, "radius_km": 2.0},
  {"name": "ExxonMobil Baytown Refinery", "lat": 29.73, "lon": -94.99, "radius_km": 3.0},
  {"name": "Motiva Port Arthur Refinery", "lat": 29.87, "lon": -93.93, "radius_km": 3.0}
]
JSON
fi

gcloud storage cp /tmp/industrial_sources.json \
    "gs://${BUCKET}/watchdog/config/industrial_sources.json"

echo "✓ Config files uploaded to GCS"

# ---------------------------------------------------------------------------
# Step 3: Deploy Cloud Function (2nd gen)
# ---------------------------------------------------------------------------
echo "→ Deploying Cloud Function ${FUNCTION_NAME}..."

# Build environment variables string
ENV_VARS="GCS_BUCKET_NAME=${BUCKET},FIRMS_MAP_KEY=${FIRMS_MAP_KEY}"
if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
    ENV_VARS="${ENV_VARS},SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL}"
fi

gcloud functions deploy "${FUNCTION_NAME}" \
    --gen2 \
    --runtime=python311 \
    --region="${REGION}" \
    --source=cloud/fire_watchdog \
    --entry-point=fire_watchdog \
    --trigger-http \
    --allow-unauthenticated \
    --set-env-vars="${ENV_VARS}" \
    --memory=512MB \
    --timeout=120s \
    --min-instances=0 \
    --max-instances=10 \
    --project="${PROJECT_ID}" \
    --quiet

# Get the function URL
FUNCTION_URL=$(gcloud functions describe "${FUNCTION_NAME}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --format="value(serviceConfig.uri)" 2>/dev/null || \
    gcloud functions describe "${FUNCTION_NAME}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --format="value(httpsTrigger.url)")

echo "✓ Cloud Function deployed: ${FUNCTION_URL}"

# ---------------------------------------------------------------------------
# Step 4: Create Cloud Scheduler jobs
# ---------------------------------------------------------------------------
echo "→ Creating Cloud Scheduler jobs..."

# Quiet mode: every 30 minutes (default, off-season)
gcloud scheduler jobs create http "${SCHEDULER_JOB_QUIET}" \
    --location="${REGION}" \
    --schedule="*/30 * * * *" \
    --uri="${FUNCTION_URL}" \
    --http-method=POST \
    --message-body='{"mode": "scheduled", "source": "cloud_scheduler_quiet"}' \
    --time-zone="UTC" \
    --attempt-deadline=90s \
    --project="${PROJECT_ID}" \
    --quiet \
    2>/dev/null || \
gcloud scheduler jobs update http "${SCHEDULER_JOB_QUIET}" \
    --location="${REGION}" \
    --schedule="*/30 * * * *" \
    --uri="${FUNCTION_URL}" \
    --http-method=POST \
    --message-body='{"mode": "scheduled", "source": "cloud_scheduler_quiet"}' \
    --time-zone="UTC" \
    --project="${PROJECT_ID}" \
    --quiet

echo "✓ Cloud Scheduler job created: ${SCHEDULER_JOB_QUIET} (every 30 min)"

# ---------------------------------------------------------------------------
# Step 5: Write Airflow trigger prefix to .env for local watchdog sensor DAG
# ---------------------------------------------------------------------------
echo "→ Updating .env with watchdog trigger config..."
TRIGGER_PREFIX="watchdog/triggers/"

# Add or update WATCHDOG_TRIGGER_PREFIX in .env
if grep -q "WATCHDOG_TRIGGER_PREFIX" .env; then
    sed -i "s|WATCHDOG_TRIGGER_PREFIX=.*|WATCHDOG_TRIGGER_PREFIX=${TRIGGER_PREFIX}|" .env
else
    echo "WATCHDOG_TRIGGER_PREFIX=${TRIGGER_PREFIX}" >> .env
fi

echo "✓ .env updated with WATCHDOG_TRIGGER_PREFIX=${TRIGGER_PREFIX}"

# ---------------------------------------------------------------------------
# Deployment summary
# ---------------------------------------------------------------------------
echo ""
echo "==================================================="
echo "✅ Deployment complete"
echo "==================================================="
echo ""
echo "Cloud Function URL: ${FUNCTION_URL}"
echo "Cloud Scheduler:    ${SCHEDULER_JOB_QUIET} (*/30 * * * * UTC)"
echo ""
echo "Next steps:"
echo "  1. Start local Airflow: docker compose up -d"
echo "  2. Unpause watchdog_sensor_dag in Airflow UI"
echo "  3. Test: gcloud scheduler jobs run ${SCHEDULER_JOB_QUIET} --location=${REGION}"
echo "  4. Watch trigger files: gcloud storage ls gs://${BUCKET}/watchdog/triggers/"
echo ""
echo "To add industrial exclusion sources:"
echo "  Edit /tmp/industrial_sources.json and re-run:"
echo "  gcloud storage cp /tmp/industrial_sources.json gs://${BUCKET}/watchdog/config/industrial_sources.json"
echo ""
echo "GCS watchdog state: gs://${BUCKET}/watchdog/state/current.json"
