# Frontend Design: Wildfire Detection Unified Dashboard


**Author perspective:** Senior ML Engineer + UI/UX Designer  
**Date:** 2026-04-13  
**Status:** Design spec — not yet implemented

---

## 1. Problem Statement

The current frontend is a **mock-data prototype** pretending to be a dashboard. It looks like a ChatGPT artifact — every section uses a different badge style, spacing is arbitrary, cards have 5 different padding schemes, and nothing connects to a real backend. It cannot monitor a live pipeline, generate a real report, or show an actual fire on a map.

This document defines what the frontend **must become**: a single, deployable operator console that runs the real data pipeline, model pipeline, deployment, and monitoring — with every button doing something real.

---

## 2. Current State Audit

### What exists

| Component | Lines | Data source | Works? |
|-----------|-------|------------|--------|
| Overview | 511 | 4 mock files | Display only |
| DataPipeline | 356 | mockPipelineData | Display only |
| OBJ1Ignition | 256 | mockModelData | Display only |
| OBJ2Spread | 224 | mockModelData | Display only |
| OBJ3Reporter | 200 | /api/status (live) | Partial — status + generate wired |
| RiskMonitor | 328 | mockGridData | Display only |
| IncidentReports | 266 | /api/reports (live) | Working — list, detail, delete, summarize |
| FireMap | 644 | mockMapData | Display only |

**Verdict:** 2 of 8 pages talk to a real backend. 6 pages are static screenshots.

### Design inconsistencies

- **5 different badge opacity patterns:** `/10`, `/20`, `/25`, `/30`, `/40`, `/50` used randomly
- **4 different card padding schemes:** `p-2.5`, `p-3`, `p-4`, `px-3 py-2.5`
- **6 custom font sizes:** `text-[8px]`, `text-[9px]`, `text-[10px]`, `text-[11px]` mixed with Tailwind `text-xs`, `text-sm`
- **Duplicate components:** `ModeBadge` defined 3 times, `RiskBadge` defined 3 times, `KV` defined 4 times
- **No shared component library** — every page reinvents its own badge/card/layout
- **Hard-coded everything** — dates, thresholds, city coordinates, state outlines, alert counts

### Performance issues

- `Overview.jsx` (511 lines): array spreads + filters on every render, no `useMemo`
- `FireMap.jsx` (644 lines): single monolith, SVG re-renders on any state change
- `RiskMonitor.jsx`: no `React.memo` on CellCard — all cards re-render on selection
- Recharts (~60KB gzipped) imported for 2 simple bar charts
- Zero code splitting, zero lazy loading

### Missing backend APIs

The Data-Pipeline has **no REST API**. The frontend cannot query:
- Airflow DAG status or task logs
- Latest pipeline run results
- FIRMS hotspot data from GCS
- Watchdog state (quiet/active/emergency)
- Feature data for grid cells

Only the OBJ-3 FastAPI server (port 8000) exposes endpoints.

---

## 3. Architecture Target

### Service topology

```
Browser (localhost:5173)
    |
    |--- /api/*  --->  Unified FastAPI (localhost:8000)
    |                    ├── /api/status         (OBJ-3 health)
    |                    ├── /api/reports/*       (report CRUD)
    |                    ├── /api/generate        (report generation)
    |                    ├── /api/rerun           (re-run with overrides)
    |                    ├── /api/pipeline/*      (NEW: Airflow proxy)
    |                    ├── /api/data/*          (NEW: grid cells, features)
    |                    ├── /api/monitor/*       (NEW: drift, PSI, alerts)
    |                    └── /api/watchdog/*      (NEW: fire monitor state)
    |
    +--- Vite proxy (vite.config.js)
```

**Principle:** One API gateway, one port. The frontend never talks to Airflow directly. The FastAPI server wraps everything.

### New backend endpoints needed

| Endpoint | Method | Source | Purpose |
|----------|--------|--------|---------|
| `/api/pipeline/status` | GET | Airflow REST API | DAG run status, task states |
| `/api/pipeline/history` | GET | Airflow REST API | Last N runs with durations |
| `/api/pipeline/trigger` | POST | Airflow REST API | Trigger a pipeline run |
| `/api/data/cells/{region}` | GET | Parquet on disk | Grid cells with latest features |
| `/api/data/firms` | GET | FIRMS CSV/GCS | Active hotspots |
| `/api/monitor/drift` | GET | MLflow / monitoring logs | PSI scores per feature |
| `/api/monitor/components` | GET | Health checks | Component status matrix |
| `/api/watchdog/state` | GET | fire_monitor_api | Current mode, cycle count |
| `/api/watchdog/mode` | POST | fire_monitor_api | Override monitoring mode |

### Frontend routing

Keep client-side view switching (no react-router needed for SPA). Current `VIEWS` map in `App.jsx` is fine.

---

## 4. Design System — One System, Used Everywhere

The current frontend has no design system. It has CSS. Fix this.

### 4.1 Typography scale (strict)

| Token | Size | Use |
|-------|------|-----|
| `text-label` | 9px | Badge text, tiny metadata |
| `text-caption` | 10px | Secondary info, KV labels, timestamps |
| `text-body` | 12px (text-xs) | Body text, table cells |
| `text-subtitle` | 14px (text-sm) | Card titles, section headers |
| `text-heading` | 16px (text-base) | Page section titles |
| `text-display` | 24px (text-2xl) | KPI numbers, hero stats |

**Rule:** No `text-[8px]`, no `text-[11px]`, no ad-hoc pixel sizes. Pick from the scale.

### 4.2 Spacing scale (strict)

| Token | Value | Use |
|-------|-------|-----|
| `space-xs` | 4px (p-1) | Inside badges, tight gaps |
| `space-sm` | 8px (p-2) | Compact cards, icon gaps |
| `space-md` | 12px (p-3) | Standard card padding |
| `space-lg` | 16px (p-4) | Section padding |
| `space-xl` | 24px (p-6) | Page padding |

**Rule:** Cards use `p-3`. Sections use `p-4`. Pages use `p-6`. No `p-2.5`.

### 4.3 Badge system (2 variants only)

```
Outlined (default):  bg-{color}/10  border border-{color}/40  text-{color}
Subtle:              bg-{color}/5   text-{color}               (no border)
```

No `/20`, `/25`, `/30`, `/50`. One opacity for background (`/10`), one for border (`/40`).

### 4.4 Card system (3 variants only)

```
Default:   bg-surface-2  border border-border-subtle  rounded-lg  p-3
Raised:    bg-surface-2  border border-border-default  rounded-lg  p-3
Critical:  bg-surface-2  border border-risk-critical/40  rounded-lg  p-3  glow-critical
```

Every card uses `p-3` and `rounded-lg`. No exceptions.

### 4.5 Shared components to extract

Create `src/components/ui/`:

```
ui/
  Badge.jsx          — <Badge color="green" variant="outlined">QUIET</Badge>
  Card.jsx           — <Card variant="default|raised|critical">...</Card>
  KV.jsx             — <KV label="risk_level" value="HIGH" />
  StatCard.jsx       — <StatCard icon={...} label="..." value="..." />
  StatusDot.jsx      — <StatusDot status="working|partial|broken" />
  Spinner.jsx        — <Spinner size="sm|md" />
  ErrorBanner.jsx    — <ErrorBanner message="..." hint="..." />
  Section.jsx        — <Section title="...">...</Section>
```

**Rule:** If a visual pattern appears in 2+ files, it goes in `ui/`. Components import from `ui/`, never define their own badge/card/KV.

---

## 5. Page-by-Page Wiring Plan

### 5.1 Overview — wire to real APIs

| Section | Current source | Target source | API |
|---------|---------------|---------------|-----|
| Operational banner (mode) | Hard-coded "QUIET" | Live watchdog state | `GET /api/watchdog/state` |
| Data Sources stat | Hard-coded "4/5" | Component health check | `GET /api/monitor/components` |
| Models in Production | mockModelData | MLflow or local model registry | `GET /api/data/models` |
| Grid Cells Monitored | mockGridData | Live cell count | `GET /api/data/cells/summary` |
| Critical Cells | mockGridData | Live risk scores | `GET /api/data/cells/summary` |
| Pipeline Run History | mockPipelineData | Airflow API | `GET /api/pipeline/history` |
| Recent Events | mockPipelineData | Aggregated logs | `GET /api/monitor/events` |
| PSI / Drift | mockPipelineData | Monitoring output | `GET /api/monitor/drift` |
| Component Status | mockPipelineData | Health checks | `GET /api/monitor/components` |
| Top Risk Cells | mockGridData | Live grid data | `GET /api/data/cells/{region}` |
| Model Registry | mockModelData | Model metadata | `GET /api/data/models` |

**Refactor:** Split 511-line monolith into:
- `OperationalBanner.jsx` (~40 lines)
- `StatCardsRow.jsx` (~60 lines)
- `ComponentStatusGrid.jsx` (~80 lines)
- `PipelineHistory.jsx` (~80 lines)
- `DriftMonitor.jsx` (~80 lines)
- `TopRiskTable.jsx` (~80 lines)
- `ModelRegistry.jsx` (~80 lines)

### 5.2 DataPipeline — wire to Airflow

| Section | Target source | API |
|---------|---------------|-----|
| Pipeline metadata | Airflow DAG config | `GET /api/pipeline/status` |
| Stage status (ingest/process/fuse/validate/export) | Task instance states | `GET /api/pipeline/status` |
| Records fetched per stage | Task logs or XCom | `GET /api/pipeline/history` |
| Trigger button | Airflow trigger | `POST /api/pipeline/trigger` |

**New feature:** Add "Trigger Pipeline Run" button that calls Airflow's trigger API. Show live task progress.

### 5.3 OBJ1Ignition — wire to model registry

| Section | Target source | API |
|---------|---------------|-----|
| Run history | MLflow or local JSON | `GET /api/data/models?objective=obj1` |
| Metrics (AUC-PR, FNR) | Model metadata JSON | Same |
| SHAP importance | Saved SHAP output | `GET /api/data/models/{run_id}/shap` |
| Bias analysis | Fairlearn output | `GET /api/data/models/{run_id}/bias` |
| Quality gates | Validation results | Same |

### 5.4 OBJ2Spread — wire to simulation output

| Section | Target source | API |
|---------|---------------|-----|
| Simulation results | Cell2Fire output JSON | `GET /api/data/simulations` |
| Compass / fire behavior | Simulation metadata | Same |

**Note:** OBJ-2 is partially implemented (Cell2Fire binary + DEM missing). This page should show real results when available, with a clear "OBJ-2 not configured" state otherwise.

### 5.5 OBJ3Reporter — already partially wired

**Working:** `/api/status`, `/api/generate`  
**Add:** Show last 3 generated reports inline (fetch from `/api/reports?limit=3`)

### 5.6 RiskMonitor — wire to live grid data

| Section | Target source | API |
|---------|---------------|-----|
| Cell list with risk scores | Latest pipeline output | `GET /api/data/cells/{region}` |
| Cell detail panel | Same | Same |
| Weather features | Same | Same |

**Enhancement:** Add auto-refresh (poll every 30s in QUIET, 5s in EMERGENCY).

### 5.7 IncidentReports — already wired

**Working:** Full CRUD, AI summary, rendered view.  
**Enhance:** Add inline report generation shortcut.

### 5.8 FireMap — wire to real geospatial data

This is the hardest page. Currently 644 lines of SVG with hard-coded polygons.

**Option A (recommended for now):** Keep SVG map but load real data:
- Fetch cell positions + risk scores from `/api/data/cells/{region}`
- Fetch FIRMS hotspots from `/api/data/firms`
- Fetch OBJ-2 spread polygons from `/api/data/simulations`

**Option B (future):** Replace with Leaflet/Mapbox for real tiles, zoom, pan. This is a larger effort and not needed for the academic project.

**Refactor:** Split into:
- `MapContainer.jsx` — layout, region tabs, layer toggle
- `RegionSVG.jsx` — SVG rendering of one region
- `MapLegend.jsx` — color scale
- `CellDetailPanel.jsx` — selected cell info

---

## 6. Memory & Performance Optimization

### 6.1 Eliminate unnecessary re-renders

```jsx
// Wrap all list item components
const CellCard = React.memo(function CellCard({ cell, selected, onSelect }) { ... });
const ReportCard = React.memo(function ReportCard({ report, ... }) { ... });
const StatCard = React.memo(function StatCard({ ... }) { ... });
const ComponentCard = React.memo(function ComponentCard({ ... }) { ... });
```

### 6.2 Memoize derived data

```jsx
// Overview.jsx — currently creates new arrays on every render
const { criticalCount, highCount, allCells } = useMemo(() => {
  const all = [...caCells, ...txCells];
  return {
    allCells: all,
    criticalCount: all.filter(c => getRiskTier(c.score) === 'CRITICAL').length,
    highCount: all.filter(c => getRiskTier(c.score) === 'HIGH').length,
  };
}, [caCells, txCells]);
```

### 6.3 Lazy load heavy pages

```jsx
// App.jsx — don't load FireMap (644 lines + SVG) until user navigates there
const FireMap = lazy(() => import('./components/fire-map/FireMap'));
const OBJ1Ignition = lazy(() => import('./components/model-pipeline/OBJ1Ignition'));
```

### 6.4 Replace Recharts for simple charts

Recharts adds ~60KB gzipped for 2 bar charts and 1 line chart. Options:
- **Custom SVG bars** — 30 lines of code, zero dependency
- **Lightweight alternative** — `react-sparklines` or `unovis` if more chart types needed

### 6.5 Virtualize long lists

If reports or cells exceed 50 items, use `react-window`:
```jsx
import { FixedSizeList } from 'react-window';
// Only render visible cells in RiskMonitor and IncidentReports
```

### 6.6 Data fetching strategy

```
SWR pattern (stale-while-revalidate):
1. Show cached data immediately
2. Fetch fresh data in background
3. Update UI when new data arrives
4. Revalidate on focus / interval

Implementation: lightweight custom hook or `useSWR` from swr package (~4KB)
```

```jsx
function useAPI(url, interval = null) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const res = await fetch(url);
        if (!cancelled) setData(await res.json());
      } catch (e) {
        if (!cancelled) setError(e);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    const id = interval ? setInterval(load, interval) : null;
    return () => { cancelled = true; if (id) clearInterval(id); };
  }, [url, interval]);

  return { data, error, loading };
}
```

---

## 7. Project Structure Optimization

### 7.1 Current structure (problems)

```
wildfire_detection/
  Data-Pipeline/          # Airflow, Docker, 15+ scripts
  model-pipeline/         # ML models, FastAPI, LLM, monitoring
  Frontend/               # React, mock data, no real wiring
  .github/workflows/      # CI/CD
  *.md                    # 10+ scattered docs
```

**Problems:**
- `Data-Pipeline/` and `model-pipeline/` are independent silos with duplicate configs
- Feature lists defined in `schema_config.yaml` AND `feature_schema.yaml` AND inline in DAG code
- Region bounding boxes hard-coded in 3 separate files
- Frontend has no idea what the backend exposes
- No shared config between frontend and backend

### 7.2 Recommended structure

```
wildfire_detection/
  config/                          # SINGLE source of truth
    pipeline.yaml                  # Regions, H3, data sources, FIRMS
    model.yaml                     # OBJ-1/2 thresholds, features, Cell2Fire
    reporting.yaml                 # OBJ-3 LLM, corpus, templates
    monitoring.yaml                # Drift thresholds, alert rules

  data-pipeline/                   # (renamed, lowercase)
    dags/
    scripts/
    docker/
    docker-compose.yaml

  model-pipeline/
    src/
      api/
        server.py                  # Unified API — add pipeline proxy endpoints here
        inference_api.py
      models/
      pipeline/
      monitoring/
    scripts/
    configs/ → symlink to /config  # Or import from shared config

  frontend/                        # (renamed, lowercase)
    src/
      api.js                       # API client
      hooks/                       # useAPI, usePolling, useWatchdog
      components/
        ui/                        # Shared: Badge, Card, KV, Spinner, etc.
        layout/                    # Sidebar, Header
        overview/                  # Split into 7 sub-components
        pipeline/                  # DataPipeline (renamed from data-pipeline)
        models/                    # OBJ1, OBJ2, OBJ3 (renamed from model-pipeline)
        monitor/                   # RiskMonitor, DriftMonitor
        reports/                   # IncidentReports
        map/                       # FireMap (split into 4 files)
      data/                        # DELETE mock files after wiring real APIs

  docs/                            # Consolidate all .md files
    ARCHITECTURE.md
    MODELS.md
    PIPELINE.md
    ISSUES.md
    SETUP.md
    frontend_design.md

  .github/workflows/
  CLAUDE.md
  MEMORY.md
  README.md
```

### 7.3 Config deduplication

**Before:** 3 files define feature lists, 3 files define region bounding boxes, 2 files define H3 resolutions.

**After:** One `config/pipeline.yaml` defines regions + features. Everything else imports or reads from it. The FastAPI server loads config at startup and serves it via `/api/config` so the frontend can also read it.

### 7.4 Delete mock data files (after wiring)

Once all pages fetch from real APIs, delete:
- `src/data/mockPipelineData.js` (309 lines)
- `src/data/mockGridData.js` (63 lines)
- `src/data/mockMapData.js` (152 lines)
- `src/data/mockReports.js` (118 lines)

Keep `mockModelData.js` temporarily — it has `MODE_MATRIX` and `WATCHDOG_CONFIG` which are static config, not mock data. Move those to a `src/config/` file.

---

## 8. Deployment

### 8.1 Local development

```bash
# Terminal 1: Data pipeline (Airflow)
cd data-pipeline && docker compose up -d

# Terminal 2: Backend API
cd model-pipeline && python scripts/run_dashboard.py --port 8000

# Terminal 3: Frontend dev server
cd frontend && npm run dev
# → localhost:5173, proxies /api to :8000
```

### 8.2 Production build

```bash
cd frontend && npm run build
# Output: frontend/dist/

# Serve from FastAPI (add to server.py):
# app.mount("/", StaticFiles(directory="../frontend/dist", html=True), name="spa")
```

One container, one port. FastAPI serves the SPA + API. No separate frontend server needed.

### 8.3 Docker

```dockerfile
# frontend/Dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
COPY --from=build /app/dist ./frontend/dist
COPY model-pipeline/ ./model-pipeline/
RUN pip install -r model-pipeline/requirements.txt
CMD ["uvicorn", "model-pipeline.src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 8.4 Environment variables

```env
GEMINI_API_KEY=...              # Required for OBJ-3
AIRFLOW_API_URL=http://localhost:8080/api/v1   # For pipeline proxy
AIRFLOW_USER=airflow
AIRFLOW_PASS=airflow
GCS_BUCKET=wildfire-mlops-dev   # For data access
```

---

## 9. What "Professional" Means Here

### Do

- Every page loads real data or shows a clear "service unavailable" state
- Every button triggers a real action or is visibly disabled with a reason
- Loading states everywhere — skeleton loaders, not blank screens
- Error states everywhere — "Backend unreachable" with how to fix it
- Consistent visual rhythm — same card, same badge, same spacing on every page
- Keyboard navigable — tab through sidebar, enter to expand, escape to close
- Fast — lazy load heavy pages, memoize derived data, virtualize long lists
- Timestamp everything — "Last updated 3 min ago", not hard-coded dates

### Do not

- Do not show mock data and pretend it is real
- Do not define ModeBadge in 3 different files
- Do not use `text-[8px]` in one place and `text-[10px]` for the same purpose elsewhere
- Do not hard-code "2025-01-15" anywhere
- Do not render 500 SVG circles without virtualization
- Do not import Recharts for a bar chart you can draw with 20 lines of SVG
- Do not add features that do not connect to a backend endpoint

---

## 10. Implementation Priority

### Phase 1: Foundation (before merge to master)

1. Extract shared `ui/` components (Badge, Card, KV, StatCard, Spinner, ErrorBanner)
2. Standardize design tokens (typography scale, spacing scale, badge system)
3. Add `useAPI` hook with loading/error/refresh
4. Build missing backend endpoints: `/api/pipeline/status`, `/api/data/cells/{region}`
5. Wire Overview to real APIs (at least operational banner + top risk cells)
6. Wire RiskMonitor to real cell data

### Phase 2: Full wiring

7. Wire DataPipeline to Airflow API proxy
8. Wire OBJ1/OBJ2 to model registry
9. Wire FireMap to real cell + FIRMS data
10. Delete all mock data files
11. Add production build config (serve dist from FastAPI)

### Phase 3: Polish

12. Lazy loading for heavy pages
13. React.memo on all list items
14. Replace Recharts with custom SVG
15. Add virtualization for 50+ item lists
16. Keyboard navigation for map
17. Auto-refresh intervals based on watchdog mode

---

## 11. Success Criteria

The frontend is done when:

- [ ] `npm run build` produces a deployable bundle under 500KB gzipped
- [ ] Every page shows real data from the backend (zero mock imports)
- [ ] Every button/action triggers a real API call
- [ ] The "Generate Report" flow produces a real Gemini-powered report
- [ ] The "Trigger Pipeline" flow kicks off a real Airflow DAG
- [ ] Loading and error states render on every page
- [ ] A new developer can run the full stack with 3 terminal commands
- [ ] Design audit finds zero duplicate component definitions
- [ ] No custom pixel font sizes outside the typography scale