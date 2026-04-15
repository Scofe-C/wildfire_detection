#!/bin/bash
# =============================================================================
# Wildfire MLOps — GCP Cloud Function + Scheduler Deployment
# =============================================================================
# Deploys a lightweight Cloud Function (dag-trigger) and a Cloud Scheduler job
# that fires every 30 minutes to run wildfire_data_pipeline directly via the
# Airflow REST API.
#
# Prerequisites:
#   1. gcloud CLI installed and authenticated (gcloud auth login)
#   2. .env file populated at repo root (copy from .env.example)
#   3. Airflow must be publicly reachable at AIRFLOW_URL
#      — Cloud Composer: use the Composer web server URL
#      — Self-hosted:    expose Airflow behind a load balancer / Cloud Run proxy
#      — Local Docker:   not reachable from GCP; use the built-in 30-min DAG
#                        schedule instead (wildfire_dag SCHEDULE_INTERVAL).
#
# Usage:
#   chmod +x cloud/deploy.sh
#   cd Data-Pipeline && ./cloud/deploy.sh
#
# What this deploys:
#   - Cloud Function: dag-trigger (HTTP, 2nd gen, Python 3.11)
#   - Cloud Scheduler job: wildfire-dag-trigger (every 30 min)
#
# Estimated cost: $0/month (Cloud Functions 2M free, Scheduler 3 jobs free)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------
ENV_FILE="${1:-.env}"
if [ -f "${ENV_FILE}" ]; then
    export $(grep -v '^#' "${ENV_FILE}" | grep -v '^$' | xargs)
    echo "✓ Loaded ${ENV_FILE}"
else
    echo "ERROR: env file '${ENV_FILE}' not found."
    exit 1
fi

REQUIRED_VARS=("GOOGLE_CLOUD_PROJECT" "AIRFLOW_URL")
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: ${var} is not set. Add it to ${ENV_FILE}."
        exit 1
    fi
done

PROJECT_ID="${GOOGLE_CLOUD_PROJECT}"
REGION="${GCP_REGION:-us-central1}"
FUNCTION_NAME="dag-trigger"
SCHEDULER_JOB="wildfire-dag-trigger"
AIRFLOW_USER="${AIRFLOW_USER:-admin}"
AIRFLOW_PASS="${AIRFLOW_PASS:-admin}"

echo ""
echo "=== Wildfire MLOps GCP Deployment ==="
echo "Project:  ${PROJECT_ID}"
echo "Region:   ${REGION}"
echo "Airflow:  ${AIRFLOW_URL}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Enable APIs
# ---------------------------------------------------------------------------
echo "→ Enabling GCP APIs..."
gcloud services enable \
    cloudfunctions.googleapis.com \
    cloudscheduler.googleapis.com \
    --project="${PROJECT_ID}" \
    --quiet
echo "✓ APIs enabled"

# ---------------------------------------------------------------------------
# Step 2: Deploy Cloud Function
# ---------------------------------------------------------------------------
echo "→ Deploying Cloud Function ${FUNCTION_NAME}..."

gcloud functions deploy "${FUNCTION_NAME}" \
    --gen2 \
    --runtime=python311 \
    --region="${REGION}" \
    --source=cloud/dag_trigger \
    --entry-point=trigger_wildfire_dag \
    --trigger-http \
    --no-allow-unauthenticated \
    --set-env-vars="AIRFLOW_URL=${AIRFLOW_URL},AIRFLOW_USER=${AIRFLOW_USER},AIRFLOW_PASS=${AIRFLOW_PASS}" \
    --memory=128MB \
    --timeout=60s \
    --min-instances=0 \
    --max-instances=3 \
    --project="${PROJECT_ID}" \
    --quiet

FUNCTION_URL=$(gcloud functions describe "${FUNCTION_NAME}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --gen2 \
    --format="value(serviceConfig.uri)")

echo "✓ Cloud Function deployed: ${FUNCTION_URL}"

# ---------------------------------------------------------------------------
# Step 3: Service account for Cloud Scheduler to invoke the function
# ---------------------------------------------------------------------------
SA_EMAIL="wildfire-scheduler@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts describe "${SA_EMAIL}" \
    --project="${PROJECT_ID}" > /dev/null 2>&1 \
|| gcloud iam service-accounts create wildfire-scheduler \
    --display-name="Wildfire DAG Scheduler" \
    --project="${PROJECT_ID}"

gcloud functions add-invoker-policy-binding "${FUNCTION_NAME}" \
    --region="${REGION}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --project="${PROJECT_ID}" \
    --quiet 2>/dev/null || true

echo "✓ Service account ${SA_EMAIL} can invoke ${FUNCTION_NAME}"

# ---------------------------------------------------------------------------
# Step 4: Create / update Cloud Scheduler job (every 30 minutes)
# ---------------------------------------------------------------------------
echo "→ Configuring Cloud Scheduler job ${SCHEDULER_JOB}..."

gcloud scheduler jobs describe "${SCHEDULER_JOB}" \
    --location="${REGION}" \
    --project="${PROJECT_ID}" > /dev/null 2>&1 \
&& gcloud scheduler jobs update http "${SCHEDULER_JOB}" \
    --location="${REGION}" \
    --schedule="*/30 * * * *" \
    --uri="${FUNCTION_URL}" \
    --http-method=POST \
    --oidc-service-account-email="${SA_EMAIL}" \
    --time-zone="UTC" \
    --attempt-deadline=55s \
    --project="${PROJECT_ID}" \
    --quiet \
|| gcloud scheduler jobs create http "${SCHEDULER_JOB}" \
    --location="${REGION}" \
    --schedule="*/30 * * * *" \
    --uri="${FUNCTION_URL}" \
    --http-method=POST \
    --oidc-service-account-email="${SA_EMAIL}" \
    --time-zone="UTC" \
    --attempt-deadline=55s \
    --project="${PROJECT_ID}" \
    --quiet

echo "✓ Cloud Scheduler job: ${SCHEDULER_JOB} (every 30 min)"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "==================================================="
echo "✅ Deployment complete"
echo "==================================================="
echo ""
echo "Cloud Function:  ${FUNCTION_URL}"
echo "Cloud Scheduler: ${SCHEDULER_JOB} (*/30 * * * * UTC)"
echo ""
echo "Next steps:"
echo "  1. Test manually:  gcloud scheduler jobs run ${SCHEDULER_JOB} --location=${REGION}"
echo "  2. Check Airflow:  open ${AIRFLOW_URL} → wildfire_data_pipeline runs"
echo "  3. View logs:      gcloud functions logs read ${FUNCTION_NAME} --region=${REGION} --limit=20"
echo ""
