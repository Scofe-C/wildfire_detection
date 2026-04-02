# Column Comparison: Historical CSV vs Fused Features vs ML Training Features

## Summary of Differences

| | california_historical.csv | fused_features_latest.parquet | fused_features_ml_latest.parquet | Notebook ML Features |
|---|---|---|---|---|
| Total columns | 33 | 32 | 32 | 21 |
| Has `date` | Yes | No | No | No |
| Fire context col order | After aspect_degrees | After aspect_degrees | After fire_detected_binary | Dropped (leakage) |
| Angular cols | Raw degrees | Raw degrees | Raw degrees | Encoded to sin/cos |
| Skewed cols | Raw values | Raw values | Raw values | log1p transformed |
| `canopy_cover_pct` | Present | Present | Present | Dropped (>50% null) |
| `resolution_km` | Present | Present | Present | Dropped (constant) |

---

## Full Column-by-Column Comparison

| Column | CA Historical CSV | Fused Latest | Fused ML | Notebook ML | Notes |
|--------|:-----------------:|:------------:|:--------:|:-----------:|-------|
| `grid_id` | ✅ | ✅ | ✅ | ❌ | Dropped — ID, not a feature |
| `region` | ✅ | ✅ | ✅ | ❌ | Dropped — ID/metadata |
| `latitude` | ✅ | ✅ | ✅ | ✅ | Kept — weak geo signal |
| `longitude` | ✅ | ✅ | ✅ | ✅ | Kept — weak geo signal |
| `timestamp` | ✅ | ✅ | ✅ | ❌ | Used for temporal split, not a feature |
| `resolution_km` | ✅ | ✅ | ✅ | ❌ | Dropped — constant (always 64) |
| `date` | ✅ | ❌ | ❌ | ❌ | Export-only string, not in fused or ML |
| `temperature_2m` | ✅ | ✅ | ✅ | ✅ | |
| `relative_humidity_2m` | ✅ | ✅ | ✅ | ✅ | |
| `wind_speed_10m` | ✅ | ✅ | ✅ | ✅ | |
| `wind_direction_10m` | ✅ | ✅ | ✅ | ❌ | Replaced by sin/cos encoding |
| `wind_direction_10m_sin` | ❌ | ❌ | ❌ | ✅ | Derived from wind_direction_10m |
| `wind_direction_10m_cos` | ❌ | ❌ | ❌ | ✅ | Derived from wind_direction_10m |
| `precipitation` | ✅ | ✅ | ✅ | ✅ | log1p transformed in notebook |
| `soil_moisture_0_to_7cm` | ✅ | ✅ | ✅ | ✅ | log1p transformed in notebook |
| `vpd` | ✅ | ✅ | ✅ | ✅ | log1p transformed in notebook |
| `fire_weather_index` | ✅ | ✅ | ✅ | ✅ | log1p transformed in notebook |
| `fuel_model_fbfm40` | ✅ | ✅ | ✅ | ✅ | Cast to category dtype |
| `canopy_cover_pct` | ✅ | ✅ | ✅ | ❌ | Dropped — >50% null after sentinel cleanup |
| `vegetation_type` | ✅ | ✅ | ✅ | ✅ | Cast to category dtype |
| `ndvi` | ✅ | ✅ | ✅ | ✅ | Median imputed |
| `elevation_m` | ✅ | ✅ | ✅ | ✅ | Median imputed |
| `slope_degrees` | ✅ | ✅ | ✅ | ✅ | Median imputed |
| `aspect_degrees` | ✅ | ✅ | ✅ | ❌ | Replaced by sin/cos encoding |
| `aspect_degrees_sin` | ❌ | ❌ | ❌ | ✅ | Derived from aspect_degrees |
| `aspect_degrees_cos` | ❌ | ❌ | ❌ | ✅ | Derived from aspect_degrees |
| `dominant_fuel_fraction` | ✅ | ✅ | ✅ | ✅ | Median imputed |
| `days_since_last_precipitation` | ✅ | ✅ | ✅ | ✅ | Forward-filled by grid_id |
| `cumulative_wind_run_24h` | ✅ | ✅ | ✅ | ✅ | Forward-filled by grid_id |
| `drought_index_proxy` | ✅ | ✅ | ✅ | ✅ | Forward-filled by grid_id |
| `data_quality_flag` | ✅ | ✅ | ✅ | ❌ | Dropped — metadata, not a feature |
| `active_fire_count` | ✅ | ✅ | ✅ | ❌ | Dropped — **leakage** |
| `mean_frp` | ✅ | ✅ | ✅ | ❌ | Dropped — **leakage** |
| `median_frp` | ✅ | ✅ | ✅ | ❌ | Dropped — **leakage** |
| `max_confidence` | ✅ | ✅ | ✅ | ❌ | Dropped — **leakage** |
| `nearest_fire_distance_km` | ✅ | ✅ | ✅ | ❌ | Dropped — **leakage** |
| `fire_detected_binary` | ✅ | ✅ | ✅ | ✅ (label) | Target variable |

---

## Why Each Column Was Dropped in the Notebook

| Column | Reason |
|--------|--------|
| `grid_id`, `region`, `date` | ID/metadata — no predictive value |
| `timestamp` | Used only for temporal train/test split |
| `resolution_km` | Constant — always 64, zero variance |
| `data_quality_flag` | Pipeline quality metadata, not a weather/terrain signal |
| `canopy_cover_pct` | >50% null after -9999 sentinel cleanup (non-vegetated cells) |
| `wind_direction_10m` | Replaced by `sin/cos` — raw degrees have a 0=360 discontinuity that confuses tree models |
| `aspect_degrees` | Same reason as wind_direction_10m |
| `active_fire_count`, `mean_frp`, `median_frp`, `max_confidence`, `nearest_fire_distance_km` | **Leakage** — all derived from the same FIRMS satellite detections that define `fire_detected_binary`. Using them would mean training on the answer. |

---

## Fused Latest vs Fused ML

The only difference between these two files is **column order** of the fire context columns:

- `fused_features_latest.parquet` — fire context columns appear in natural order after `aspect_degrees`
- `fused_features_ml_latest.parquet` — `fire_detected_binary` appears earlier; fire context columns (`active_fire_count`, `mean_frp`, etc.) are shifted to **T-1 (previous time window)** to prevent leakage during training

Both have 32 columns. The ML variant is the correct one to use for model training if you want the pipeline to handle the temporal lag for you. In the notebook we dropped the fire context columns entirely instead.

---

## Final Notebook ML Feature Set (21 features used for training)

```
Continuous weather:    temperature_2m, relative_humidity_2m, wind_speed_10m,
                       precipitation*, soil_moisture_0_to_7cm*, vpd*,
                       fire_weather_index*

Angular (encoded):     wind_direction_10m_sin, wind_direction_10m_cos,
                       aspect_degrees_sin, aspect_degrees_cos

Static terrain:        elevation_m, slope_degrees, dominant_fuel_fraction,
                       ndvi, latitude, longitude

Static categorical:    fuel_model_fbfm40, vegetation_type

Weather derived:       days_since_last_precipitation, cumulative_wind_run_24h,
                       drought_index_proxy

Label:                 fire_detected_binary
```
`*` = log1p transformed
