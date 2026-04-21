# Changes to Apply on Master

Summary of work that's in `dev_ack` (commit `1ac74f7 some changes`) but not in `master`, for re-application directly on `master` after the blocked merge.

**Stats**: 69 files changed, +710 / −2672 lines. Most deletions are removed docs/config; most additions are lint/test fixes + feature work.

---

## 1. Functional changes (session work — must keep)

### 1.1 Cloud Function: fix reserved run_id prefix
`Data-Pipeline/cloud/dag_trigger/main.py` (line 30)
```diff
- run_id = f"scheduled__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
+ run_id = f"cloudscheduler__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
```
**Why**: Airflow reserves the `scheduled__` prefix; using it causes HTTP 400 on every Cloud Scheduler trigger.
**Deploy**: After merging, re-deploy the Cloud Function via `Data-Pipeline/cloud/deploy.sh`.

### 1.2 Airflow DAG: disable internal schedule
`Data-Pipeline/dags/wildfire_dag.py` (line 62)
```diff
- SCHEDULE_INTERVAL = "*/30 * * * *"
+ SCHEDULE_INTERVAL = None  # Triggered externally by Cloud Scheduler
```
**Why**: Prevents dual-triggering when Cloud Scheduler + Airflow internal scheduler both fire every 30 min.

### 1.3 FastAPI server: OBJ-3 notifications from GCS
`model-pipeline/src/api/server.py` — `/api/notifications` endpoint
- OBJ-3 notifications previously only read local disk (ephemeral on Cloud Run → notifications disappeared after restart).
- Now reads `reports/obj3/**/*.json` from GCS first (deduplicated with local disk by filename stem).
- Extracts `incident_name` for titles; falls back to region from GCS path when JSON lacks region field.
- Block starts at `# ── OBJ-3: most recent reports from GCS + disk ──` — adds a `seen_obj3: set[str]` dedup guard.

### 1.4 FastAPI server: report list includes `title` + `operating_mode`
`model-pipeline/src/api/server.py` — `/api/reports` endpoint (both GCS and local-disk branches)
```diff
  _append({
      "id": stem,
+     "title": data.get("incident_name") or data.get("title") or data.get("report_type", "Report"),
      ...
+     "operating_mode": data.get("operating_mode"),
      ...
  })
```
**Why**: Frontend `IncidentReports` cards were showing blank titles because incident reports use `incident_name`, not `title`.

### 1.5 FastAPI server: render endpoint GCS fallback
`model-pipeline/src/api/server.py` — `/api/reports/{id}/render`
- Previously only searched local disk → returned 404 for GCS-only reports → OBJ3Reporter iframe showed blank.
- Now tries local disk first, falls back to GCS `reports/obj3/**/*.json` matching by filename stem.

### 1.6 Frontend: IncidentReports incident-schema rendering
`Frontend/src/components/reports/IncidentReports.jsx` (+273 lines)
Rewrote expanded-detail block to use **incident report schema** fields:
- `detail.incident_name`, `incident_status`, `percent_contained`
- `spread_summary` (fallback to `risk_summary` for non-incident reports)
- `weather_observations` + `fire_behavior` row
- `immediate_actions` (strings, indexed 01, 02, ...) — falls back to `preventive_recommendations` for non-incident
- `affected_communities` + `evacuation_status` row
- `resource_requirements` + `projected_losses` row
- `strategic_objectives`, `projected_activity` (hours_12/24/48/72)
- Falls back to `top_risk_cells` / `escalation_trigger` for daily/high_risk reports
Also removed unused `RiskCountBadge` helper, `topCell` variable, `useEffect` import.

### 1.7 Frontend: Overview + RiskMonitor 60s polling
`Frontend/src/components/overview/Overview.jsx`, `Frontend/src/components/risk-monitor/RiskMonitor.jsx`
```diff
  fetchLive();
- return () => { cancelled = true; };
+ const id = setInterval(fetchLive, 60_000); // refresh every 60s
+ return () => { cancelled = true; clearInterval(id); };
```
**Why**: Critical-cell counts weren't refreshing after DAG runs — required manual page reload.

### 1.8 Frontend: Sidebar logo becomes collapse toggle
`Frontend/src/components/layout/Sidebar.jsx`
- Wrapped logo `<img>` in a `<button onClick={() => setCollapsed(v => !v)}>`.
- Adds `title="Expand sidebar" / "Collapse sidebar"` tooltip.
- Existing bottom-chevron collapse toggle still works.

---

## 2. Cleanup deletions (safe, reduce repo clutter)

- `CICD.md` (−314 lines) — stale docs duplicated in `.github/workflows/`
- `Memory.md` (−1871 lines) — generated notes, added to `.gitignore` earlier
- `model-pipeline/MLOPS_DESIGN.md` (−138 lines) — outdated design doc
- `Data-Pipeline/data/processed/64km.dvc` (−6 lines), `Data-Pipeline/data/processed/fused.dvc` (−6 lines) — replaced by `Data-Pipeline/dvc/processed_64km.dvc`
- `README.md` → renamed to `OVERVIEW.md` (108 lines changed)

## 3. New additions

- `project_summary.md` — interview-prep doc (this session)
- `Data-Pipeline/tests/test_fusion/test_priority_resolver.py` (+132 lines) — new fusion priority resolver tests

---

## 4. Lint / test cleanup (dozens of single-line changes)

These are almost all unused-import removal or small type fixes to satisfy the lint stage that's currently blocking CI/CD:

**Scripts** (Data-Pipeline/scripts/, ~20 files, 1-14 line changes each):
- `backfill/historical_backfill.py`, `detection/emergency.py`, `detection/fire_detector.py`
- `export/export_spatial.py`, `fire_monitor.py`
- `ingestion/ingest_field_telemetry.py`, `ingest_firms.py`, `ingest_goes.py`, `ingest_hrrr.py`, `ingest_landfire.py`, `ingest_srtm.py`
- `integration_test.py`, `seed_local_test.py`
- `processing/process_firms.py` (+5), `processing/process_weather.py`
- `utils/export_appeears_points.py`, `rate_limiter.py`, `schema_loader.py`
- `validation/detect_anomalies.py` (+4)

**Tests** (Data-Pipeline/tests/, ~25 files, 1-22 line changes each):
- `conftest.py`, `test_dags/test_dag_structure.py` (18 lines)
- `test_detection/`, `test_dvc/test_dvc.py` (9 lines), `test_export/test_dual_track.py`
- `test_fusion/test_fuse_static_only.py`, `test_fusion_properties.py` (+22), `test_temporal_lag.py`
- `test_infrastructure/test_gce_deploy.py` (−3)
- `test_ingestion/test_bug_fixes.py` (+66 net — expanded), `test_firms.py`, `test_hrrr.py` (17 lines), `test_weather.py`
- `test_processing/`, `test_utils/` (7 test files)
- `test_validation/test_anomalies.py`, `test_bias_analysis.py`

**Workflows**:
- `.github/workflows/ci.yaml` (30 line changes)
- `.github/workflows/model_ci.yml` (30 line changes)

**Config**:
- `.dockerignore` (−2), `.gitignore` (11 lines — added Memory.md etc.)
- `Data-Pipeline/.gitignore` (13 lines)
- `Data-Pipeline/docker/Dockerfile` (−13)
- `model-pipeline/Dockerfile` (3 lines), `model-pipeline/pyproject.toml` (3 lines)

---

## 5. Data artifacts (automatic — regenerate, don't manually merge)

These are DVC metadata updates produced by the pipeline — regenerate by running the DAG on master rather than merging manually:
- `Data-Pipeline/dvc.lock` (14 line changes)
- `Data-Pipeline/dvc/processed_64km.dvc` (6 line changes)
- `model-pipeline/experimentation/california.ipynb` (4 line changes — execution counts)

---

## 6. Suggested application order on master

```bash
# 1. Apply the functional fixes first (small, easy to review)
git checkout master
git pull origin master

# 2. Cherry-pick the session commit if CI allows, OR manually apply from this doc:
git cherry-pick origin/dev_ack  # the 1ac74f7 commit

# 3. If cherry-pick hits the same CI failure, split into logical commits:
#    a. functional/*: items 1.1-1.8 above (small surface area, clean review)
#    b. cleanup/*: item 2 (deletions)
#    c. lint-fixes/*: item 4 (the CI-blocker fixes)
#    d. tests/*: new test_priority_resolver.py + test expansions

# 4. After merge, verify CI passed Stage 8 (Cloud Run deploy fires automatically)
#    and redeploy Cloud Function manually since that's not in CI:
cd Data-Pipeline && ./cloud/deploy.sh

# 5. Rebuild + deploy frontend (no CI path exists for frontend Cloud Run):
cd ../Frontend
docker build --platform linux/amd64 -t gcr.io/wildfire-mlops-123/wildfire-frontend:latest .
docker push gcr.io/wildfire-mlops-123/wildfire-frontend:latest
gcloud run deploy wildfire-frontend \
  --image gcr.io/wildfire-mlops-123/wildfire-frontend:latest \
  --region us-central1 --port 3000 --allow-unauthenticated
```

---

## 7. What was blocking the merge

The `Pipeline CI` workflow failed. Most likely causes based on what's in the diff:
- **Lint stage** — unused imports or whitespace; the lint-fixes in §4 above are specifically the fixes
- **Unit tests** — new `test_priority_resolver.py` or modified tests may require fixtures that exist only in `dev_ack`

Fastest unblock: cherry-pick the commit onto a branch off master, run CI locally (`pytest Data-Pipeline/tests -x`), fix whatever fails, then PR.
