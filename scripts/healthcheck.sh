#!/usr/bin/env bash
# =============================================================================
# Wildfire Detection — Service Health Check
# Polls all known endpoints and prints a color-coded status table.
# =============================================================================
set -euo pipefail

R='\033[0;31m' G='\033[0;32m' Y='\033[1;33m' B='\033[0;34m' N='\033[0m'

check() {
  local name="$1" url="$2"
  if curl -sf --max-time 3 "$url" > /dev/null 2>&1; then
    printf "  %-26s ${G}UP${N}    %s\n" "$name" "$url"
  else
    printf "  %-26s ${R}DOWN${N}  %s\n" "$name" "$url"
  fi
}

echo ""
echo -e "${B}Service Health Check${N}"
echo -e "${B}────────────────────────────────────────────${N}"
check "Airflow Webserver"       "http://localhost:8080/health"
check "OBJ-3 Dashboard"        "http://localhost:8000/api/status"
check "Fire Monitor API"       "http://localhost:8001/status"
check "MLflow UI"              "http://localhost:5000"
echo -e "${B}────────────────────────────────────────────${N}"
echo ""