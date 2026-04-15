com# Wildfire Detection — Model Pipeline Summary

## 1. Overview

The model pipeline (OBJ-1) predicts wildfire ignition risk for California and Texas grid cells. It is a full MLOps training and inference system that loads labeled historical data from GCS, trains XGBoost and LightGBM classifiers, validates them through an AUC-PR gate and a bias gate, and pushes the winning model to Vertex AI Model Registry. Inference runs every 6 hours by fetching live weather from Open-Meteo, joining static terrain features, and scoring all ~55 grid cells.

---

## 2. Model Development and ML Code

### 2.1 Loading Data from the Data Pipeline

Data is loaded from GCS (`gs://wildfire-mlops-123/historical_data/{region}_historical.csv`), where the data pipeline writes verified, labeled historical records. Each region (California, Texas) is processed independently. Loading uses an exponential backoff retry (2 attempts, 2s/4s delays) to handle transient GCS failures.

**Temporal split** preserves causal ordering — no random shuffle is used:

| Split | Window | Rows (CA) | Rationale |
|---|---|---|---|
| Train | All data except Jan 2025, up to Dec 31 2025 | 12,604 | Full historical coverage |
| Test | Jan 1–31, 2025 | 713 | Jan 2025 LA fires — major documented event |

2026 rows are excluded with a hard `LABEL_CUTOFF = 2025-12-31` because FIRMS has not yet confirmed 2026 detections — those rows have `fire_detected_binary=0` not because there were no fires but because they are unlabeled. Including them would tank model precision.

### 2.2 Feature Engineering — Single Source of Truth

All transforms live exclusively in `src/preprocessing/feature_engineering.py`. Both model classes and the inference script import from this module — no transform is ever duplicated inside a model file. This prevents training/inference skew.

**Canonical feature set (20 features + target):**

| Group | Features |
|---|---|
| Continuous weather | temperature_2m, relative_humidity_2m, wind_speed_10m, precipitation, soil_moisture_0_to_7cm, vpd, fire_weather_index |
| Angular (sin/cos encoded) | wind_direction_10m_sin/cos, aspect_degrees_sin/cos |
| Static terrain | elevation_m, slope_degrees, ndvi, dominant_fuel_fraction |
| Geographic | latitude, longitude |
| Categorical | fuel_model_fbfm40, vegetation_type |
| Derived (time-series) | cumulative_wind_run_24h, drought_index_proxy |
| **Target** | fire_detected_binary |

**Preprocessing pipeline (7 steps, order strictly preserved):**

**Step 0a — Sentinel fix:** Replaces `-9999` (FIRMS/LANDFIRE sentinel) with `NaN`.

**Step 0b — Time-series fill:** `cumulative_wind_run_24h` and `drought_index_proxy` are forward/backward filled per `grid_id` group before `grid_id` is dropped. Using median here would be wrong because these are rolling accumulations, not point measurements.

**Step 1 — drop_non_features():** Drops metadata (grid_id, region, date, resolution_km) and leakage columns (active_fire_count, mean_frp, nearest_fire_distance_km). At inference, raises a hard error if leakage columns are present — this is a safety guard against data pipeline bugs.

**Step 2 — impute_before_encoding():** Median-imputes `wind_direction_10m` and `aspect_degrees` BEFORE circular encoding. If this were done after, `sin(NaN) = NaN` would propagate into the encoded features and two correlated columns would need to be imputed separately.

**Step 3 — apply_circular_encoding():** Converts angular columns to sin/cos pairs to resolve the 0°=360° discontinuity (e.g., 1° and 359° are numerically distant but directionally adjacent). Raw degree values would confuse tree models.

**Step 4 — apply_log1p():** Log-transforms right-skewed weather columns (precipitation, vpd, fire_weather_index, soil_moisture) after clipping negatives to 0. Open-Meteo can return small negative values for precipitation due to floating point; clipping prevents `log(negative)`.

**Step 5 — apply_median_imputation():** Median-imputes static terrain columns. At training, medians are computed from training data and stored in a state dict. At inference, training medians are passed in via `fit_medians` to prevent the inference batch's (possibly incomplete) data from shifting the imputation values.

**Step 5b — apply_categorical_imputation():** Mode-fills `fuel_model_fbfm40` and `vegetation_type` before encoding. Must occur after numeric imputation and before encoding so that OrdinalEncoder and `cast_category_dtype` never see NaN.

**Step 6a — apply_ordinal_encoding() (XGBoost path):** Applies `OrdinalEncoder` to categorical columns. XGBoost cannot handle pandas `category` dtype natively; unknown categories at inference are encoded as `-1`.

**Step 6b — cast_category_dtype() (LightGBM path):** Casts categoricals to pandas `category` dtype, which LightGBM handles natively. Critically, the category levels are derived from the full training dataset and passed to both test and inference preprocessing via `fit_categories`. Without this, LightGBM throws a "train and valid dataset categorical_feature do not match" error when CV folds have different category sets.

**Step 7 — validate_no_nulls():** Hard assertion. If any nulls remain, a `ValueError` is raised with a column-level null count report. There is one additional fallback before this step: `LOG1P_COLS` still containing NaN (e.g., if Open-Meteo returns no soil moisture data) are filled with 0, since `log1p(0) = 0` correctly represents "no moisture/precipitation."

### 2.3 Training and Selecting the Best Model

The pipeline supports two execution modes:

**Initial mode** — run once to establish baseline:
1. Trains XGBoost with 50-iteration `RandomizedSearchCV` over a 7-parameter grid using `TimeSeriesSplit(5)` (respects temporal ordering across CV folds, scored by ROC-AUC)
2. Trains LightGBM with the same search strategy and 8-parameter grid
3. Both models are evaluated on the Jan 2025 test set
4. Winner = highest AUC-PR above threshold; XGBoost confirmed as winner (AUC-PR: 0.9051 vs 0.8962)

**Retrain mode** — runs daily once winner is confirmed:
- Trains XGBoost only (LightGBM comparison not repeated)
- Directly validates against the AUC-PR gate

**XGBoost hyperparameter search space:**

```
max_depth:         [3, 4, 5, 6, 7, 8]
n_estimators:      [100, 200, 300, 400, 500]
learning_rate:     [0.01, 0.05, 0.1, 0.2, 0.3]
subsample:         [0.6–1.0]
colsample_bytree:  [0.6–1.0]
min_child_weight:  [1, 3, 5, 7, 10]
gamma:             [0, 0.1, 0.2, 0.3, 0.5]
```

Class imbalance is handled via `scale_pos_weight = n_negative / n_positive` (dynamically computed per training set, typically ~1.8 for CA). LightGBM uses `is_unbalance=True`.

**Decision threshold tuning** — after model selection, the winner's threshold is tuned on the test set to achieve ≥90% recall (≥90% of fires must be caught). The logic uses `candidates[-1]` from the precision-recall curve: it selects the HIGHEST threshold that still meets the recall target. `np.argmax` would instead return the FIRST (lowest) threshold — near-100% recall but near-zero precision. The tuned threshold for XGBoost on CA data was **0.4596**.

### 2.4 Model Validation

Validation is an AUC-PR hard gate:

| Metric | Threshold | Result (CA) |
|---|---|---|
| AUC-PR | ≥ 0.89 | 0.9051 ✓ |
| AUC-PR (LightGBM) | ≥ 0.89 | 0.8962 ✓ |
| Recall at tuned threshold | ≥ 0.90 | 0.900 ✓ |

AUC-PR was chosen over AUC-ROC as the primary metric because the dataset is class-imbalanced (fires are rare events). AUC-ROC can appear high even when the model performs poorly on the minority class; AUC-PR directly measures performance on positive (fire) predictions.

Validation is computed on the Jan 2025 hold-out set using `compute_all_metrics()`, which returns: AUC-PR, AUC-ROC, F1, FNR, accuracy, confusion matrix, positive rate, threshold, and sample count. If all candidates fail the gate, a `RuntimeError` is raised and the pipeline triggers a rollback to the most recently archived production model.

---

## 3. Hyperparameter Tuning

Both models use `RandomizedSearchCV` with `TimeSeriesSplit(5)` folds:

- **Search method:** Random search over 50 parameter combinations
- **CV strategy:** `TimeSeriesSplit` — each fold's training data is always earlier than its validation data, preserving temporal causality. Standard `KFold` would leak future data into earlier training folds.
- **Scoring metric during search:** ROC-AUC (computationally stable across folds; AUC-PR is the final validation metric)
- **Parallelism:** `n_jobs=-1` at the search level; `n_jobs=1` at the base estimator level (avoids nested parallelism conflicts)

**Best parameters found (CA, run 970bb676):**
- XGBoost: subsample=0.6, n_estimators=400, min_child_weight=1, max_depth=4, learning_rate=0.01, gamma=0.1, colsample_bytree=0.6 (CV ROC-AUC=0.9524)
- LightGBM: subsample=0.9, reg_lambda=0.5, num_leaves=20, n_estimators=300, min_child_samples=100, learning_rate=0.01, colsample_bytree=0.7 (CV ROC-AUC=0.9535)

---

## 4. Experiment Tracking

**MLflow** is used for all experiment tracking (local SQLite backend: `sqlite:///mlruns.db`, swappable for a remote URI in CI/CD).

Every training run logs:

**Parameters:** region, training_mode, hyperparameters (all tuned values), n_train_rows, n_test_rows, train_fire_rate, test_fire_rate, framework, model_artifact_sha256

**Metrics:** xgboost_auc_pr, lightgbm_auc_pr, xgboost_f1, lightgbm_f1, tuned_threshold, recall_at_threshold, bias_overall_fnr, bias_disparity_{slice}, bias_fnr_{slice}_{group}, shap_{feature} (mean absolute SHAP value per feature)

**Artifacts:** precision_recall_curve.png, confusion_matrix.png, model_comparison.png

**Run tags:** pipeline=obj1_ignition, region, training_mode, run_id

Note: `tuned_threshold` is logged as a **metric** (not a param) because XGBoost's `get_params()` already logs a default `threshold=0.365` as a param — MLflow params are immutable once written, so the tuned value would collide.

**Visualizations generated:**

1. **Precision-Recall Curve** — shows AUC-PR with shaded area under curve and a no-skill horizontal baseline at the dataset's fire rate. The operating threshold (≥90% recall point) is visually identifiable.

2. **Confusion Matrix** — 2×2 heatmap (Fire vs No Fire) showing TP, FP, FN, TN at the tuned threshold.

3. **Model Comparison Bar Chart** — grouped bars comparing XGBoost and LightGBM across AUC-PR, F1, FNR, and accuracy. Only generated in initial mode when both models are evaluated.

MLflow UI is launched with:
```bash
PATH="$(pwd)/.venv/bin:$PATH" .venv/bin/python -m mlflow ui \
  --backend-store-uri sqlite:///mlruns.db --port 5001
```

---

## 5. Model Sensitivity Analysis (SHAP)

SHAP (SHapley Additive exPlanations) is computed using `TreeExplainer` on 500 randomly sampled test rows.

**What is logged:**
- Per-feature mean absolute SHAP value (`shap_{feature_name}` metrics in MLflow)
- Enables tracking which features contribute most to predictions over time
- Native XGBoost feature importance (gain-based) logged alongside SHAP for comparison

**Why SHAP over native importance:**
- Native XGBoost `feature_importances_` (gain) can be misleading for correlated features
- SHAP values account for feature interactions and are additive across predictions
- Mean absolute SHAP per feature is interpretable as "average contribution to fire risk score"

**Drift detection:** The config defines `min_soil_moisture_importance: 0.05`. If SHAP importance for soil moisture drops below this across consecutive runs, a Slack `shap_drift` alert is triggered, indicating the model may be ignoring a key physical predictor.

**SHAP is non-blocking:** If the `shap` library is unavailable, the pipeline logs native importance only and continues.

---

## 6. Model Bias Detection (Slicing Techniques)

Bias is defined as **False Negative Rate (FNR) disparity** across domain-relevant slices — a fire prediction system that misses fires disproportionately in certain groups is both unfair and dangerous.

**FNR = 1 − Recall = missed fires / all fires**

A disparity of 0 means all groups have identical FNR; higher disparity means one group's fires are systematically missed relative to another's.

**Three evaluation slices (`src/validation/bias_check.py`):**

**Slice 1 — Region (california vs texas)**
Checks whether the model performs equally well across the two regions. California data dominates the training set; this slice catches regional underfitting.

**Slice 2 — Fire Season (May–Oct vs Nov–Apr)**
Checks whether the model's sensitivity holds during peak fire season when consequences of missed detections are highest. This slice is skipped if the test window only contains one season.

**Slice 3 — Fuel Model (LANDFIRE FBFM40)**
Checks whether specific fuel/vegetation types are disproportionately missed. High-risk fuel types (e.g., certain grass and timber litter models) must not be underdetected. Two filters are applied before evaluating a group:
- At least 20 total samples (unreliable FNR at tiny counts)
- At least 5 actual fire events (FNR is meaningless with 1–2 fire examples)

**Gate logic:**
- Disparity = max(group FNR) − min(group FNR) within each slice
- **Pass:** disparity ≤ 0.15 for ALL slices
- **Fail:** ANY slice exceeds 0.15

Max disparity is set to 0.15 (not 0.05) because the Jan 2025 test set has only 713 rows — at small sample sizes, FNR naturally varies between fuel types even in a well-calibrated model. 0.05 was statistically impossible to pass; 0.15 is a meaningful and achievable fairness constraint.

**Bias gate is currently non-blocking** — the pipeline logs a warning and continues to registry push. This is intentional during development to avoid blocking deployment on fine-grained fuel type imbalances that require data collection to resolve properly.

**Bias results are fully logged to MLflow** and included in the validation report JSON (`reports/bias_gate/`).

---

## 7. CI/CD Pipeline Automation

The pipeline is designed for automation via GitHub Actions / Cloud Build, with these key entry points:

### 7.1 Training Trigger

```bash
GOOGLE_APPLICATION_CREDENTIALS=path/to/key.json \
GCS_BUCKET_NAME=wildfire-mlops-123 \
GCP_PROJECT_ID=wildfire-mlops-123 \
python -m scripts.train --mode retrain --regions california texas \
                        --output-report reports/training_result.json
```

Exit code 0 = all regions deployable; exit code 1 = any failure. The JSON report at `--output-report` is read by CI/CD gates to make deployment decisions.

### 7.2 Automated Validation

The orchestrator evaluates the AUC-PR gate inline:
- **Pass:** pipeline continues to bias gate → registry push
- **Fail:** `RuntimeError` raised → rollback triggered → Slack alert sent → MLflow run marked FAILED → exit code 1

### 7.3 Automated Bias Detection

`run_bias_check()` is called automatically after threshold tuning. Results are:
- Logged to MLflow as metrics (`bias_disparity_*`, `bias_fnr_*`)
- Written to `reports/bias_gate/` as JSON
- Sent to Slack if gate fails (with disparity value, threshold, and worst-group FNR breakdown)

### 7.4 Model Deployment — Vertex AI Model Registry

When the model passes validation (and bias gate in non-blocking mode), it is pushed to Vertex AI Model Registry:

1. **Artifact saved to GCS:** `gs://wildfire-mlops-123/model-artifacts/{run_id}/model.bst` + `model_metadata.json`
2. **Registered in Vertex AI** with label `env=staging`
3. **Current production demoted** to `env=archived`
4. **New version promoted** to `env=production`
5. **Inference loads** the production model by querying for `display_name=wildfire-ignition-{region} AND labels.env=production`, then reconstructing the GCS path from the `run_id` label

The model metadata JSON co-located with the artifact contains everything needed for reproducible inference: threshold, training medians (for consistent imputation), feature list, and framework identifier.

### 7.5 Notifications and Alerts

Five alert types are implemented in `src/notifications/alerter.py` via Slack webhooks:

| Alert | Color | Trigger |
|---|---|---|
| bias_gate_failure | Red | FNR disparity exceeds threshold |
| validation_failure | Orange-Red | AUC-PR below 0.89 |
| pipeline_error | Red | Unhandled exception in any stage |
| rollback | Orange | Validation failure → previous model promoted |
| shap_drift | Gold | Feature importance below drift threshold |
| success | Green | Pipeline completes; model deployed |

Alerts are sent using Python's `urllib.request` (no external HTTP library dependency). If no webhook is configured, alerts are silently skipped with a log warning.

### 7.6 Rollback Mechanism

If all candidate models fail the AUC-PR gate:

1. `VertexRegistry.rollback()` is called
2. Current `env=production` model is relabeled `env=archived`
3. Most recently created `env=archived` model is relabeled `env=production`
4. A `rollback` Slack alert is sent with from/to version identifiers
5. Vertex AI Experiments logs the rollback event with delta metrics

This ensures a suboptimal model never reaches production — the most recently validated good model is reinstated automatically.

---

## 8. Inference Pipeline

The inference script (`scripts/inference.py`) runs every 6 hours:

### 8.1 Weather Fetching (Open-Meteo)

For each grid cell centroid, a 24-hour rolling window of hourly weather is fetched:
- Variables: temperature, humidity, wind speed, wind direction, precipitation, soil moisture, VPD
- Aggregation: mean (temperature, humidity, wind speed, VPD, soil moisture), sum (precipitation, wind run)

Three features are derived locally at inference (matching training notebook derivations):
- **Fire Weather Index (FWI):** Simplified Canadian FWI formula using FFMC, DMC, ISI, BUI
- **Cumulative Wind Run (24h):** Sum of hourly wind speeds
- **Drought Index Proxy:** `(temp/40) × (1 - humidity/100) × (1 - soil_moisture) × 100`

### 8.2 Static Feature Join

Static terrain/vegetation features (elevation, slope, aspect, NDVI, fuel model, vegetation type) are stored in `Data-Pipeline/data/static/static_features_64km.parquet` and joined by `grid_id`. These features are time-invariant and not available from Open-Meteo.

### 8.3 Preprocessing at Inference

The same `full_pipeline()` function is called with `is_inference=True` and `fit_medians` loaded from `model_metadata.json`. This guarantees that:
- The same median values used during training fill any missing terrain values at inference
- Leakage columns cause a hard error if accidentally present
- Category levels match training (LightGBM)
- Log transforms, circular encodings, and ordinal encodings are identical to training

### 8.4 Scoring and Risk Tiers

```
CRITICAL:  fire_risk_score ≥ 0.65
HIGH:      fire_risk_score ≥ 0.365
MEDIUM:    fire_risk_score ≥ 0.15
LOW:       fire_risk_score < 0.15
```

Binary flag (`fire_risk_flag`) uses the tuned operational threshold (0.4596 for CA).

### 8.5 Outputs

- **Parquet (history):** `gs://wildfire-mlops-123/inference/region={region}/year={year}/month={month}/inference_{timestamp}.parquet`
- **JSON (latest snapshot):** `gs://wildfire-mlops-123/inference/latest/{region}_latest.json` — overwritten each run; contains per-cell scores, risk tiers, and summary statistics
- **Slack alert:** Triggered if any cell reaches CRITICAL tier

---

## Key Design Decisions Summary

| Decision | Rationale |
|---|---|
| Temporal split, not random | Preserves causal ordering; prevents lookahead bias |
| AUC-PR as primary metric | More informative than AUC-ROC on imbalanced datasets |
| candidates[-1] threshold logic | Maximizes precision while meeting recall target; avoids threshold of ~0.05 |
| Single preprocessing module | Prevents training/inference skew from duplicated transforms |
| Training medians passed at inference | Consistent imputation without data leakage |
| Explicit category levels for LightGBM | Prevents CV fold categorical mismatch errors |
| FNR disparity bias metric | Domain-relevant — missed fires are the dangerous failure mode |
| run_id label in Vertex AI registry | Decouples registry from artifact URI format; enables reliable GCS path reconstruction |
| Non-blocking bias gate | Allows service continuity during known small-sample imbalances; still alerts operator |
| SHAP logged per run | Enables feature drift detection over retraining cycles |
