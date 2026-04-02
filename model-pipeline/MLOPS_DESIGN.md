# MLOps Design: Wildfire Detection Inference & Retraining

## Architecture Overview

Two decoupled loops running at different cadences:

```
FIRMS API (every 6h)                    Open-Meteo API (every 5 min)
      |                                         |
      v                                         v
  fire_detected_binary (label)          weather features (no label)
      |                                   + static features (cached)
      v                                         |
  Append ~140 rows/day                          v
  to historical data              model.predict_proba() on all ~35 CA cells
      |                                         |
      v                                         v
  Retrain LightGBM (every 24h)        Fire risk score per cell, refreshed every 5 min
      |
      v
  Push new model to registry
```

---

## Training Loop (every 24 hours)

- FIRMS gives ground truth labels 4× per day (every 6 hours)
- Each window produces ~35 California grid cell rows with `fire_detected_binary`
- 4 windows × 35 cells = ~140 new labeled rows per day appended to historical data
- Full LightGBM retrain on growing historical dataset
- Best params from hyperparameter search logged to MLflow

## Inference Loop (every 5 minutes)

- Fetch Open-Meteo for all ~35 CA grid cells (weather only, no label)
- Load cached static features (LANDFIRE/SRTM — unchanged)
- Run `model.predict_proba()` → fire risk probability per cell
- No FIRMS data needed at inference time
- Provides early warning **between** satellite passes (FIRMS has 3–6h NASA processing latency)

---

## Key Design Decisions

### Why 5-minute inference cadence?
FIRMS satellites (MODIS/VIIRS) pass over a location twice per day with 3–6 hour processing latency. The ML model running on weather + terrain provides fire risk scores continuously, filling the gap between satellite passes. This is the primary value proposition over just using FIRMS directly.

### Why retrain daily instead of every 6 hours?
- 140 rows/day is a small enough batch that daily retraining on the full dataset is fast
- Avoids model drift from retraining on a single 6-hour window
- Gives time to validate new data quality before retraining

### Why LightGBM and not a neural network?
- Tabular data with ~21 features — tree models outperform NNs in this regime
- Fast retraining on small growing dataset
- `scale_pos_weight` handles class imbalance natively
- Handles NaN natively — no imputation needed at inference

---

## Training/Inference Feature Consistency (Critical)

The model was trained on weather features **aggregated over 6-hour windows** using a **24-hour lookback**. At inference time, the same aggregation must be applied — do NOT feed raw instantaneous Open-Meteo readings.

| Feature | Training computation | Inference requirement |
|---------|---------------------|-----------------------|
| `temperature_2m` | Mean of last 24h hourly readings | Rolling 24h mean, updated every 5 min |
| `relative_humidity_2m` | Mean of last 24h | Rolling 24h mean |
| `wind_speed_10m` | Mean of last 24h | Rolling 24h mean |
| `wind_direction_10m` | Circular mean of last 24h | Rolling circular mean |
| `precipitation` | Sum of last 24h | Rolling 24h sum |
| `soil_moisture_0_to_7cm` | Mean of last 24h | Rolling 24h mean |
| `vpd` | Mean of last 24h | Rolling 24h mean |
| `fire_weather_index` | Computed from aggregated values | Recompute from rolling aggregates |
| `cumulative_wind_run_24h` | Sum of `wind_speed × 1h` over 24h | Same — sum over rolling 24h window |
| `drought_index_proxy` | Composite from 24h aggregates | Recompute from rolling aggregates |
| `days_since_last_precipitation` | From 24h window + forward-fill | **See caveat below** |

---

## Caveats & Known Issues

### 1. `days_since_last_precipitation` inference skew (Critical)
**Problem:** During training (backfill), `days_since_last_precipitation` accumulates across windows via forward-fill — so a 30-day drought correctly shows as 30. At inference with a fresh 24h lookback, if no rain is detected in the window, the feature caps at 1 day regardless of actual drought length.

**Impact:** The model underestimates drought severity at inference during prolonged dry spells — exactly when fire risk is highest.

**Fix options:**
1. Maintain a persistent state store (Redis/database) that tracks `last_rain_timestamp` per grid cell, updated each inference call
2. Fetch a longer lookback (e.g. 30 days) for this feature specifically at inference time
3. Drop the feature if ablation (Cell 15 in notebook) shows ROC-AUC is unaffected

### 2. Training/test distribution shift
The test set (Jan 2025 LA fires) has a 36% fire rate — far higher than typical months (~5–10%). The ROC-AUC of 0.937 is likely optimistic. Cross-validation score of 0.746 is more representative of real-world performance on normal months. Expect lower precision in production during non-fire-season periods.

### 3. 6-hour training windows vs 5-minute inference windows
Open-Meteo returns hourly data. At inference, you must aggregate the last 24 hours of hourly readings into the same statistics used during training. Do not feed instantaneous readings — this will cause feature distribution mismatch.

### 4. Static features are frozen
LANDFIRE and SRTM are loaded from a one-time cache. If wildfire burns significantly change vegetation/fuel in a grid cell (e.g. post-fire bare ground), the model won't reflect this until the static cache is manually rebuilt. LANDFIRE updates annually — rebuild the static cache after each annual release.

### 5. HRRR is not used in training
The historical backfill uses Open-Meteo only. HRRR (15-min, 3km resolution) is only used in production on watchdog triggers for active fire spread tracking — it feeds a different use case than the ignition prediction model trained in the notebook.

### 6. Leakage columns must never appear at inference
The following columns are derived from FIRMS fire detections and must never be fed to the model at inference (no FIRMS data available at inference time anyway):
- `active_fire_count`
- `mean_frp`, `median_frp`
- `max_confidence`
- `nearest_fire_distance_km`

### 7. Decision threshold is separate from the model
The model was trained optimizing ROC-AUC (0.9374). The operational threshold of **0.239** was chosen post-training to achieve ≥90% recall on the LA fires test set. This threshold must be versioned and stored alongside the model in MLflow — it is not a model parameter.

| Threshold | Fire Recall | Fire Precision | Fires Missed (out of 240) |
|-----------|------------|----------------|--------------------------|
| 0.50 (default) | 0.79 | 0.79 | ~50 |
| 0.35 | 0.88 | 0.72 | ~29 |
| **0.239 (chosen)** | **0.93** | **0.68** | **~17** |

---

## MLflow Tracking Plan

Each retraining run should log:

| Item | MLflow artifact/param |
|------|----------------------|
| Best hyperparameters | `mlflow.log_params(best_params)` |
| ROC-AUC on test set | `mlflow.log_metric("roc_auc")` |
| PR-AUC on test set | `mlflow.log_metric("pr_auc")` |
| Decision threshold | `mlflow.log_param("threshold")` |
| Recall at threshold | `mlflow.log_metric("recall_at_threshold")` |
| Training data size | `mlflow.log_param("n_train_rows")` |
| Trained model | `mlflow.lightgbm.log_model(clf)` |
| Feature list | `mlflow.log_param("features")` |
| `scale_pos_weight` | `mlflow.log_param("scale_pos_weight")` |
