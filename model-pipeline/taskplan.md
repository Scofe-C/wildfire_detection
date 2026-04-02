# OBJ-1 Model Pipeline Refactoring — Task Plan
**Status:** Post-backfill implementation  
**Models:** LightGBM (primary) · XGBoost (secondary)  
**Date:** 2026-03-31

---

## Reuse vs Rebuild Summary

| File | Verdict | Reason |
|---|---|---|
| `configs/feature_schema.yaml` | **REBUILD** | Wrong target name, wrong index column, all feature names wrong |
| `configs/model_config.yaml` | **MODIFY** | Update `backfill_dir` path to `historical_data/` |
| `src/data/loader.py` | **MODIFY** | Add time-based train/test split (no temporal split = data leakage) |
| `src/data/schema.py` | **REUSE** | Validation engine is sound — only the YAML it reads is broken |
| `src/models/base.py` | **REUSE** | Good abstract interface — extend with `tune_hyperparameters()` |
| `src/models/obj1_xgboost/model.py` | **REBUILD** | ERA5 column names, silent 0-fill, no tuning, no SHAP |
| `src/models/obj1_lightgbm/model.py` | **CREATE NEW** | Does not exist — LightGBM primary model |
| `src/models/registry.py` | **MODIFY** | Add rollback() method; add Vertex AI Model Registry push |
| `src/pipeline/orchestrator.py` | **MODIFY HEAVILY** | No train/test split, no HP tuning, no SHAP, no rollback trigger |
| `src/validation/metrics.py` | **REUSE** | Comprehensive — AUC-PR, F1, FNR all present |
| `src/validation/model_selector.py` | **MODIFY** | Add multi-model comparison (LightGBM vs XGBoost), add rollback trigger |
| `src/validation/visualizations.py` | **REUSE** | PR curve, confusion matrix, model comparison plots all present |
| `src/bias/detector.py` | **REUSE** | FairLearn bias gate is correctly implemented |
| `src/bias/nri_loader.py` | **MODIFY** | `sjoin_nearest` has no distance constraint — can match cells 100+ km away |
| `src/bias/mitigation.py` | **REUSE + INTEGRATE** | Code exists but is never called by orchestrator |
| `src/bias/report.py` | **REUSE** | Report generation is correct |
| `src/tracking/mlflow_logger.py` | **MODIFY** | Add SHAP value logging, add hyperparameter logging |
| `src/tracking/vertex_sync.py` | **MODIFY** | Add Vertex AI Model Registry push (currently only Experiments) |
| `src/notifications/alerter.py` | **REUSE + INTEGRATE** | Slack alerts exist but rollback alert is never triggered |
| `.github/workflows/model_ci.yml` | **REBUILD** | Stages 5–9 not wired; stages 1–4 may work |

---

## Phase 1 — Schema & Config (Prerequisite for everything)

### Task 1.1 — Rebuild `feature_schema.yaml`
**File:** `configs/feature_schema.yaml`  
**Verdict:** REBUILD  
**Course rule:** "Loading Data from the Data Pipeline" — features must match pipeline output exactly

Fix three breaking issues:
- `h3_index` → `grid_id` in index_columns
- `fire_detected` → `fire_detected_binary` in target
- Replace all feature entries with actual parquet column names

**Correct features to include:**

```
Weather (required):     temperature_2m, relative_humidity_2m, wind_speed_10m,
                        wind_direction_10m, precipitation, soil_moisture_0_to_7cm,
                        vpd, fire_weather_index

Fire / FIRMS (required): active_fire_count, mean_frp, median_frp, max_confidence,
                         nearest_fire_distance_km

Terrain (required):     fuel_model_fbfm40, elevation_m, slope_degrees, aspect_degrees

Derived (required):     days_since_last_precipitation, cumulative_wind_run_24h,
                        drought_index_proxy

Optional:               canopy_cover_pct, ndvi
```

> `vegetation_type`, `data_quality_flag`, `dominant_fuel_fraction`, `region`,
> `latitude`, `longitude`, `resolution_km` are metadata — exclude from model features.

---

### Task 1.2 — Update `model_config.yaml`
**File:** `configs/model_config.yaml`  
**Verdict:** MODIFY (small change)

Change `backfill_dir` from the placeholder path to the historical data output:
```yaml
# Before
backfill_dir: "../data-pipeline/data/processed/backfill"

# After
backfill_dir: "historical_data/64km"
```

---

## Phase 2 — Data Layer

### Task 2.1 — Add temporal train/test split to `loader.py`
**File:** `src/data/loader.py`  
**Verdict:** MODIFY  
**Course rule:** "Training and Selecting the Best Model" — must use held-out test set, no data leakage

Add a `load_train_test_split()` function that:
1. Loads all parquet files from `historical_data/`
2. Sorts by `timestamp` (ascending)
3. Splits at the 80th percentile of dates (train = older 80%, test = newer 20%)
4. Returns `(X_train, X_test, y_train, y_test, meta_train, meta_test)`

**Why time-based (not random):** A random split would allow the model to train on data from
January 2025 and test on data from June 2024 — the model would effectively "see the future"
during training. Time-based split preserves causal ordering.

The existing `load_backfill()` and `split_features_target()` functions stay unchanged.

---

## Phase 3 — Models

### Task 3.1 — Create LightGBM primary model
**File:** `src/models/obj1_lightgbm/model.py` (CREATE NEW)  
**File:** `src/models/obj1_lightgbm/__init__.py` (CREATE NEW)  
**Course rule:** Primary model with hyperparameter tuning + SHAP

LightGBM is better than XGBoost for this dataset because:
- Native categorical support for `fuel_model_fbfm40` (no one-hot encoding needed)
- 3–10× faster training → more hyperparameter tuning iterations
- Better handling of class imbalance on sparse spatial data
- Same SHAP TreeExplainer support

The class must:
1. Extend `BaseModel` (keep the existing interface — `predict()`, `validate()`, `explain()`)
2. Use the feature list from `feature_schema.yaml` (loaded at init, not hardcoded)
3. Use `is_unbalance=True` for class imbalance (equivalent to XGBoost `scale_pos_weight`)
4. Implement `tune_hyperparameters(X_train, y_train)` using `RandomizedSearchCV`:
   - Search space: `num_leaves`, `n_estimators`, `learning_rate`, `min_child_samples`,
     `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`
   - Scoring: `average_precision` (AUC-PR, matches validation metric)
   - cv=3, n_iter=30, n_jobs=-1
5. Implement `explain(X)` using `shap.TreeExplainer`:
   - Returns both `feature_importance` (native) and `shap_mean_abs` (SHAP)
   - Log warning and fall back to native importance if `shap` not installed
6. Implement `get_params()` returning all non-None model params for MLflow logging

**Key rule:** If any required feature is missing from input data → raise `ValueError` immediately.
Do NOT fill with zeros or defaults — silent fills produced a model trained entirely on zeros
in the current codebase.

---

### Task 3.2 — Rebuild XGBoost secondary model
**File:** `src/models/obj1_xgboost/model.py`  
**Verdict:** REBUILD  
**Course rule:** Secondary model for comparison; winner deployed

Same interface as LightGBM above. Key differences:
- Uses `scale_pos_weight` instead of `is_unbalance`
- Hyperparameter search space: `max_depth`, `n_estimators`, `learning_rate`,
  `subsample`, `colsample_bytree`, `min_child_weight`
- Remove ALL ERA5 preprocessing logic (`u10`, `v10`, `t2m`, `d2m`, `tp` — these
  columns do not exist in the pipeline output)
- Remove silent 0-fill: if feature missing → raise `ValueError`

---

## Phase 4 — Orchestrator

### Task 4.1 — Add training phase to `orchestrator.py`
**File:** `src/pipeline/orchestrator.py`  
**Verdict:** MODIFY HEAVILY  
**Course rules:** Train/test split · HP tuning · SHAP · Baseline comparison · Rollback

The current `run_pipeline()` function has no training step — it assumes the model is
pre-loaded and just runs inference + validation. This must be restructured:

**New flow in `run_pipeline()`:**

```
1. Load data
   └── load_train_test_split() → X_train, X_test, y_train, y_test, meta_train, meta_test

2. Hyperparameter tuning (on train set only)
   └── model.tune_hyperparameters(X_train, y_train) → best_params
   └── tracker.log_params(best_params)

3. Train
   └── model.train(X_train, y_train)
   └── tracker.log_params(model.get_params())

4. Validate on HELD-OUT TEST SET (not training data)
   └── model.predict(X_test) → predictions
   └── validate_model(y_test, y_prob) → metrics, passed_val
   └── tracker.log_validation_result(metrics, passed_val)
   └── If NOT passed → alert_validation_failure() → check rollback → return

5. SHAP explainability
   └── model.explain(X_test.sample(min(500, len(X_test))))
   └── tracker.log_shap(shap_summary)
   └── Check SHAP drift vs previous run (soil_moisture minimum importance)
   └── If drift detected → alert_shap_drift()

6. Visualizations
   └── generate_all_visualizations(y_test, y_prob, ...)

7. Bias gate (on test set predictions only)
   └── spatial_join_predictions(pred_df, nri)  ← uses grid_id not h3_index
   └── run_bias_gate_from_dataframe(joined)
   └── If NOT passed → apply mitigation OR alert_bias_gate_failure() → rollback

8. Registry push (only if both gates pass)
   └── registry.save_local(model, version, metadata)
   └── registry.tag_previous(current_version)
   └── registry.push_to_gcs(version)
   └── vertex_sync.push_to_model_registry(model, version, metrics)
   └── alert_success()
```

**Rollback trigger:** If validation fails OR bias gate fails → call `registry.rollback()`
and `alert_rollback()`. This is missing entirely from the current code.

---

### Task 4.2 — Add multi-model selection to `model_selector.py`
**File:** `src/validation/model_selector.py`  
**Verdict:** MODIFY  
**Course rule:** "Training and Selecting the Best Model" — compare candidates, deploy winner

Add `select_best_model(candidates: dict[str, BaseModel], X_test, y_test)` that:
1. Validates each model on the test set
2. Returns the model with highest AUC-PR above threshold
3. If both models fail threshold → trigger rollback to previous version
4. Logs a comparison table to MLflow

---

## Phase 5 — Bias Detection Fixes

### Task 5.1 — Fix spatial join in `nri_loader.py`
**File:** `src/bias/nri_loader.py`  
**Verdict:** MODIFY  
**Course rule:** Bias gate must be accurate — wrong spatial join = wrong FNR per group

`sjoin_nearest` with no distance constraint will match H3 cell centroids to NRI census
tracts dozens of kilometres away. Fix:

1. First attempt: `gpd.sjoin(pred_gdf, nri, how="left", predicate="intersects")`
   — matches cells that actually fall within a census tract polygon
2. For unmatched cells (H3 cell centroid between tract boundaries):
   fall back to `sjoin_nearest` with `max_distance=0.1` (degrees, ~11 km)
3. Remaining unmatched after fallback → `"Unknown"` (acceptable for remote/border cells)

Also fix column name: current code passes `h3_col="h3_index"` but data uses `grid_id`.
Default should be `h3_col="grid_id"`.

---

### Task 5.2 — Integrate `mitigation.py` into orchestrator
**File:** `src/bias/mitigation.py` (reuse), `src/pipeline/orchestrator.py` (wire up)  
**Verdict:** REUSE + INTEGRATE  
**Course rule:** Bias mitigation must be applied, not just detected

The mitigation strategies exist but are never called. When the bias gate fails:
1. Apply `compute_class_weights()` (primary — re-weight Very High vulnerability positives)
2. Re-train with adjusted weights
3. Re-run bias gate on new model
4. If still failing → log as unresolvable, alert, block deployment

---

## Phase 6 — Registry & Rollback

### Task 6.1 — Add `rollback()` to `registry.py`
**File:** `src/models/registry.py`  
**Verdict:** MODIFY  
**Course rule:** "CI/CD" — must be able to roll back to previous version

Add `rollback()` method that:
1. Reads `PREVIOUS_VERSION` marker file
2. Verifies the version directory and `metadata.json` exist
3. Returns the previous version string (caller loads the model)
4. Logs the rollback event

Also add `push_to_vertex_model_registry()` method (Vertex AI Model Registry, not just
Vertex AI Experiments which is what `vertex_sync.py` currently does).

---

## Phase 7 — MLflow Tracking

### Task 7.1 — Add SHAP + hyperparameter logging to `mlflow_logger.py`
**File:** `src/tracking/mlflow_logger.py`  
**Verdict:** MODIFY (small additions)

Add two methods:
- `log_shap(shap_summary: dict[str, float])` — logs `shap_{feature}` as MLflow metrics
- `log_hyperparameters(params: dict)` — wrapper around `log_params` with type coercion

These are called by the orchestrator after tuning and after SHAP explain.

---

## Phase 8 — CI/CD

### Task 8.1 — Rebuild `.github/workflows/model_ci.yml`
**File:** `.github/workflows/model_ci.yml`  
**Verdict:** REBUILD (stages 5–9 not wired)  
**Course rule:** Full CI/CD pipeline with all gates

The 9 required stages:

| Stage | Trigger | Action |
|---|---|---|
| 1. Lint | Push | `ruff check src/` |
| 2. Unit tests | Push | `pytest tests/` |
| 3. Container build | Push | Build Docker image |
| 4. Integration tests | Push | Run test suite with real data sample |
| 5. Model validation | Merge to main | Train + test split + AUC-PR gate |
| 6. Bias gate | After stage 5 pass | FairLearn FNR disparity ≤ 0.05 |
| 7. Artifact push | After stage 6 pass | `registry.push_to_gcs()` |
| 8. Vertex AI sync | After stage 7 | `vertex_sync.push_to_model_registry()` |
| 9. Deploy | Manual approval | Cloud Run update |

Stage 9 requires a manual approval gate in GitHub Actions (environment protection rule)
so production deployments are never automatic.

**Rollback workflow** (separate file `rollback.yml`):
- Manual trigger with `version` input
- Calls `registry.rollback()` and redeploys previous Cloud Run revision

---

## Phase 9 — Verification

### Task 9.1 — Update `model_config.yaml` thresholds
After running on real backfill data, the current thresholds may need calibration:

```yaml
validation:
  auc_pr_threshold: 0.75    # Verify this is achievable on LA fire data
  decision_threshold: 0.5   # May need lowering for rare fire events (try 0.3)

bias_gate:
  max_disparity: 0.05       # Strict — may need to be 0.10 for initial runs
```

### Task 9.2 — End-to-end smoke test
Run the full pipeline on a small date slice (one month) before the full 8-month training run:
```bash
python -m src.pipeline.orchestrator \
  --backfill-dir historical_data/64km \
  --start-date 2025-01-01 \
  --end-date 2025-01-31
```
Verify: MLflow run created · model saved to `models/ignition/` · SHAP values logged ·
bias report written · no crashes.

---

## Dependency Order

```
Task 1.1 (schema)
    └── Task 2.1 (loader)
        └── Task 4.1 (orchestrator)  ←── Task 3.1 (LightGBM)
            └── Task 4.2 (selector)  ←── Task 3.2 (XGBoost)
            └── Task 5.2 (mitigation integration)
            └── Task 7.1 (MLflow SHAP)
            └── Task 6.1 (rollback)
                └── Task 8.1 (CI/CD)
                    └── Task 9.2 (smoke test)

Task 1.2 (model_config)   — parallel, any time
Task 5.1 (nri_loader fix) — parallel, any time before bias gate
```

---

## What Satisfies Each Course Rule

| Course Requirement | Task(s) |
|---|---|
| Load data from pipeline | 1.1, 1.2, 2.1 |
| Time-based train/test split (no leakage) | 2.1, 4.1 |
| Hyperparameter tuning | 3.1, 3.2, 4.1 |
| Primary model (LightGBM) | 3.1 |
| Secondary model comparison (XGBoost) | 3.2, 4.2 |
| SHAP explainability | 3.1, 3.2, 4.1, 7.1 |
| Bias detection + mitigation | 5.1, 5.2 |
| MLflow experiment tracking | 4.1, 7.1 |
| GCP model registry | 6.1 (Vertex AI Model Registry) |
| CI/CD pipeline with all gates | 8.1 |
| Rollback mechanism | 6.1, 8.1 |
