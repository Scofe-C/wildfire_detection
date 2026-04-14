#!/usr/bin/env bash
# =============================================================================
# Wildfire Detection — Zero-Dependency Startup
# For first-time users who may not have `make` installed.
#
# Usage:
#   ./start.sh              # Start all services (full profile)
#   ./start.sh --airflow    # Start Airflow only (default profile)
# =============================================================================
set -euo pipefail

R='\033[0;31m' G='\033[0;32m' Y='\033[1;33m' B='\033[0;34m' N='\033[0m'
PROFILE="full"

for arg in "$@"; do
  case "$arg" in
    --airflow) PROFILE="default" ;;
    --help|-h)
      echo "Usage: ./start.sh [--airflow]"
      echo ""
      echo "  (no flags)   Start ALL services (Airflow + Dashboard + Monitor + MLflow)"
      echo "  --airflow    Start Airflow only"
      echo ""
      echo "Services:"
      echo "  Airflow      http://localhost:8080  (airflow/airflow)"
      echo "  Dashboard    http://localhost:8000"
      echo "  Monitor      http://localhost:8001"
      echo "  MLflow       http://localhost:5000"
      exit 0 ;;
  esac
done

echo ""
echo -e "${B}╔══════════════════════════════════════════╗${N}"
echo -e "${B}║  Wildfire Detection — Starting Services  ║${N}"
echo -e "${B}╚══════════════════════════════════════════╝${N}"
echo ""

# 1. Check Docker is running
if ! docker info > /dev/null 2>&1; then
  echo -e "${R}ERROR: Docker is not running. Please start Docker Desktop first.${N}"
  exit 1
fi
echo -e "${G}[1/4] Docker is running${N}"

# 2. Check .env exists
if [ ! -f .env ]; then
  if [ -f .env.example ]; then
    echo -e "${Y}[2/4] .env not found — copying .env.example to .env${N}"
    echo -e "${Y}      Fill in your API keys before the pipeline can fetch data.${N}"
    cp .env.example .env
  else
    echo -e "${R}ERROR: .env and .env.example both missing.${N}"
    exit 1
  fi
else
  echo -e "${G}[2/4] .env found${N}"
fi

# 3. Check ports are free
check_port() {
  local port=$1 name=$2
  if command -v ss > /dev/null 2>&1; then
    ss -tlnp 2>/dev/null | grep -q ":${port} " && echo -e "${Y}  Warning: port ${port} (${name}) is already in use${N}"
  elif command -v lsof > /dev/null 2>&1; then
    lsof -i ":${port}" > /dev/null 2>&1 && echo -e "${Y}  Warning: port ${port} (${name}) is already in use${N}"
  fi
}
echo -e "${G}[3/4] Checking ports...${N}"
check_port 8080 "Airflow"
check_port 8000 "Dashboard"
check_port 8001 "Monitor"
check_port 5000 "MLflow"

# 4. Start services
echo -e "${G}[4/4] Starting containers...${N}"
if [ "$PROFILE" = "full" ]; then
  docker compose --profile full up -d --build
else
  docker compose up -d --build
fi

# 5. Poll health (up to 60s)
echo ""
echo -e "${Y}Waiting for services to become healthy (up to 60s)...${N}"
ENDPOINTS=("http://localhost:8080/health|Airflow")
if [ "$PROFILE" = "full" ]; then
  ENDPOINTS+=("http://localhost:8000/api/status|Dashboard")
  ENDPOINTS+=("http://localhost:8001/status|Monitor")
fi

for entry in "${ENDPOINTS[@]}"; do
  url="${entry%%|*}"
  name="${entry##*|}"
  for i in $(seq 1 12); do
    if curl -sf --max-time 3 "$url" > /dev/null 2>&1; then
      echo -e "  ${G}${name} is up${N}"
      break
    fi
    if [ "$i" -eq 12 ]; then
      echo -e "  ${Y}${name} not yet ready (may need more time)${N}"
    fi
    sleep 5
  done
done

# 6. Print status table
echo ""
echo -e "${G}All services started.${N}"
echo -e "  Airflow:    ${B}http://localhost:8080${N}  (airflow / airflow)"
if [ "$PROFILE" = "full" ]; then
  echo -e "  Dashboard:  ${B}http://localhost:8000${N}"
  echo -e "  Monitor:    ${B}http://localhost:8001${N}"
  echo -e "  MLflow:     ${B}http://localhost:5000${N}"
fi
echo ""
echo -e "Stop with: ${Y}docker compose --profile full down${N}"
echo ""