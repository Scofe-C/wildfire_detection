#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Wildfire Detection — Unified Launcher
# Starts: Data Pipeline (Airflow) + Model Pipeline (FastAPI) + Frontend (Vite)
# Usage:  ./start.sh [--skip-airflow] [--skip-api] [--skip-frontend]
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PIDS=()
SKIP_AIRFLOW=false
SKIP_API=false
SKIP_FRONTEND=false

# ── Parse flags ──
for arg in "$@"; do
  case "$arg" in
    --skip-airflow)  SKIP_AIRFLOW=true ;;
    --skip-api)      SKIP_API=true ;;
    --skip-frontend) SKIP_FRONTEND=true ;;
    --help|-h)
      echo "Usage: ./start.sh [--skip-airflow] [--skip-api] [--skip-frontend]"
      echo ""
      echo "Services:"
      echo "  Airflow     http://localhost:8080  (airflow/airflow)"
      echo "  FastAPI     http://localhost:8000"
      echo "  Frontend    http://localhost:5173"
      exit 0 ;;
  esac
done

# ── Colors ──
R='\033[0;31m' G='\033[0;32m' Y='\033[1;33m' B='\033[0;34m' N='\033[0m'

echo ""
echo -e "${B}╔══════════════════════════════════════════╗${N}"
echo -e "${B}║  Wildfire Detection — Starting Services  ║${N}"
echo -e "${B}╚══════════════════════════════════════════╝${N}"
echo ""

# ── Cleanup on exit ──
cleanup() {
  echo ""
  echo -e "${Y}Shutting down...${N}"
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  if [ "$SKIP_AIRFLOW" = false ] && [ -f "$ROOT/Data-Pipeline/docker-compose.yaml" ]; then
    (cd "$ROOT/Data-Pipeline" && docker compose down 2>/dev/null) || true
  fi
  echo -e "${G}All services stopped.${N}"
}
trap cleanup EXIT INT TERM

# ─── 1. Data Pipeline (Airflow via Docker Compose) ───
if [ "$SKIP_AIRFLOW" = false ]; then
  echo -e "${Y}[1/3] Starting Data Pipeline (Airflow)...${N}"
  if [ -f "$ROOT/Data-Pipeline/docker-compose.yaml" ]; then
    (cd "$ROOT/Data-Pipeline" && docker compose up -d --build 2>&1 | tail -5) || {
      echo -e "${R}  Warning: Docker Compose failed. Is Docker running?${N}"
    }
    echo -e "${G}  -> Airflow UI:  http://localhost:8080  (airflow / airflow)${N}"
  else
    echo -e "${R}  Skipped — Data-Pipeline/docker-compose.yaml not found${N}"
  fi
else
  echo -e "${Y}[1/3] Airflow — skipped (--skip-airflow)${N}"
fi

# ─── 2. Model Pipeline (FastAPI) ───
if [ "$SKIP_API" = false ]; then
  echo -e "${Y}[2/3] Starting Model Pipeline (FastAPI)...${N}"
  if [ -f "$ROOT/model-pipeline/scripts/run_dashboard.py" ]; then
    (cd "$ROOT/model-pipeline" && python scripts/run_dashboard.py --no-browser --port 8000 2>&1) &
    PIDS+=($!)
    echo -e "${G}  -> API Server:  http://localhost:8000  (PID: ${PIDS[-1]})${N}"
  else
    echo -e "${R}  Skipped — model-pipeline/scripts/run_dashboard.py not found${N}"
  fi
else
  echo -e "${Y}[2/3] FastAPI — skipped (--skip-api)${N}"
fi

# ─── 3. Frontend (Vite) ───
if [ "$SKIP_FRONTEND" = false ]; then
  echo -e "${Y}[3/3] Starting Frontend (Vite)...${N}"
  if [ -f "$ROOT/Frontend/package.json" ]; then
    (cd "$ROOT/Frontend" && npm run dev 2>&1) &
    PIDS+=($!)
    echo -e "${G}  -> Dashboard:   http://localhost:5173  (PID: ${PIDS[-1]})${N}"
  else
    echo -e "${R}  Skipped — Frontend/package.json not found${N}"
  fi
else
  echo -e "${Y}[3/3] Frontend — skipped (--skip-frontend)${N}"
fi

echo ""
echo -e "${G}All services started.${N}"
echo -e "  Airflow:    ${B}http://localhost:8080${N}"
echo -e "  API:        ${B}http://localhost:8000${N}"
echo -e "  Dashboard:  ${B}http://localhost:5173${N}"
echo ""
echo -e "Press ${Y}Ctrl+C${N} to stop all services."
echo ""

# Keep alive
wait
