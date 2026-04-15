#!/usr/bin/env bash
# =============================================================================
# PyroWatch — Service health check
# Polls all service endpoints and prints a status table.
# Usage: bash scripts/healthcheck.sh
# =============================================================================

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; NC='\033[0m'

ENDPOINTS=(
    "Airflow Webserver  |http://localhost:8080/health         |airflow / airflow"
    "OBJ-3 Dashboard    |http://localhost:8000/api/status     |—"
    "Frontend SPA       |http://localhost:3000                |—"
)

echo ""
echo "  PyroWatch — Service Status"
echo "  ──────────────────────────────────────────────────────────────"
printf "  %-3s  %-22s %-38s %s\n" "ST" "Service" "URL" "Notes"
echo "  ──────────────────────────────────────────────────────────────"

ALL_UP=true

for entry in "${ENDPOINTS[@]}"; do
    IFS='|' read -r name url notes <<< "$entry"
    # Try curl with a short timeout; suppress output
    if curl -sf --max-time 5 "$url" &>/dev/null; then
        printf "  ${GREEN}UP${NC}   %-22s %-38s %s\n" "$name" "$url" "$notes"
    else
        printf "  ${RED}DOWN${NC} %-22s %-38s %s\n" "$name" "$url" "$notes"
        ALL_UP=false
    fi
done

echo "  ──────────────────────────────────────────────────────────────"
echo ""

if $ALL_UP; then
    echo -e "  ${GREEN}All services healthy.${NC}"
else
    echo -e "  ${YELLOW}Some services are down.${NC}"
    echo "  Run 'make logs' to see container output."
    echo "  If this is a fresh start, wait ~60s for Airflow to initialize."
fi
echo ""
