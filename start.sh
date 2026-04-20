#!/usr/bin/env bash
# =============================================================================
# PyroWatch Wildfire MLOps — One-command startup
# Usage: ./start.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DC_DIR="$ROOT/Data-Pipeline"
ENV_FILE="$ROOT/.env"

RED='\033[0;31m'; YELLOW='\033[0;33m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'

info()  { echo -e "  ${CYAN}→${NC}  $*"; }
ok()    { echo -e "  ${GREEN}✓${NC}  $*"; }
warn()  { echo -e "  ${YELLOW}!${NC}  $*"; }
error() { echo -e "  ${RED}✗${NC}  $*"; }

echo ""
echo "  PyroWatch MLOps — starting all services"
echo "  ────────────────────────────────────────"
echo ""

# ── 1. Check Docker ──────────────────────────────────────────────────────────
info "Checking Docker..."
if ! docker info &>/dev/null; then
    error "Docker is not running. Start Docker Desktop and retry."
    exit 1
fi
ok "Docker is running"

# ── 2. Check .env ────────────────────────────────────────────────────────────
if [ ! -f "$ENV_FILE" ]; then
    warn ".env not found — copying from .env.example"
    cp "$ROOT/.env.example" "$ENV_FILE"
    warn "Fill in your API keys in .env, then rerun this script."
    echo ""
    echo "  Required variables:"
    echo "    FIRMS_MAP_KEY         — NASA FIRMS API key"
    echo "    GCS_BUCKET_NAME       — GCS bucket name"
    echo "    GCP_KEY_PATH          — path to gcp-key.json"
    echo "    GOOGLE_CLOUD_PROJECT  — GCP project ID"
    echo "    GEMINI_API_KEY        — Google Gemini key (for OBJ-3 reports)"
    echo ""
    exit 1
fi
ok ".env found"

# ── 3. Check ports ───────────────────────────────────────────────────────────
info "Checking ports..."
for port in 8080 8000 3000; do
    if lsof -ti tcp:"$port" &>/dev/null; then
        warn "Port $port already in use — container may already be running"
    fi
done

# ── 4. Start services ────────────────────────────────────────────────────────
info "Starting all services (this may take 2-3 min on first run)..."
cd "$DC_DIR"
docker compose --env-file "$ENV_FILE" up -d --build
cd "$ROOT"
ok "docker compose started"

# ── 5. Wait for services to be healthy ───────────────────────────────────────
echo ""
info "Waiting for services to become healthy (up to 120s)..."
echo ""

ENDPOINTS=(
    "Airflow|http://localhost:8080/health"
    "OBJ-3 Dashboard|http://localhost:8000/api/status"
    "Frontend|http://localhost:3000"
)

MAX_WAIT=120
INTERVAL=5
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    ALL_UP=true
    for entry in "${ENDPOINTS[@]}"; do
        name="${entry%%|*}"
        url="${entry##*|}"
        if ! curl -sf --max-time 3 "$url" &>/dev/null; then
            ALL_UP=false
        fi
    done
    if $ALL_UP; then break; fi
    printf "  Waiting... %ds\r" "$ELAPSED"
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done
echo ""

# ── 6. Print status table ────────────────────────────────────────────────────
echo "  Service Status"
echo "  ─────────────────────────────────────────────────────"
for entry in "${ENDPOINTS[@]}"; do
    name="${entry%%|*}"
    url="${entry##*|}"
    if curl -sf --max-time 3 "$url" &>/dev/null; then
        printf "  ${GREEN}UP${NC}    %-22s %s\n" "$name" "$url"
    else
        printf "  ${RED}DOWN${NC}  %-22s %s\n" "$name" "$url"
    fi
done
echo ""
echo "  Credentials: Airflow → airflow / airflow"
echo ""
