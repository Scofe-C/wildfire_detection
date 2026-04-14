// Mock OBJ-1 ignition predictions and OBJ-2 spread simulations for Fire Detection Map.
// Structured to match expected backend API shape from model-pipeline/ outputs.
// grid_ids match mockGridData.js (CALIFORNIA_CELLS + TEXAS_CELLS).

// --- OBJ-1: Ignition Probability Predictions ---
// Shape mirrors model-pipeline/obj1_ignition/inference output per H3 cell.
export const OBJ1_PREDICTIONS = {
  // California
  '8228e9ffffffffff': { grid_id: '8228e9ffffffffff', probability: 0.14, tier: 'LOW',      features: { temperature_2m: 22.1, relative_humidity_2m: 38.2, wind_speed_10m: 12.4, vpd: 1.82, fire_weather_index: 18.4 }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:12Z' },
  '822897ffffffffff': { grid_id: '822897ffffffffff', probability: 0.08, tier: 'LOW',      features: { temperature_2m: 14.2, relative_humidity_2m: 72.1, wind_speed_10m: 8.2,  vpd: 0.61, fire_weather_index: 6.2  }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:13Z' },
  '8228a7ffffffffff': { grid_id: '8228a7ffffffffff', probability: 0.12, tier: 'LOW',      features: { temperature_2m: 9.8,  relative_humidity_2m: 81.4, wind_speed_10m: 6.1,  vpd: 0.34, fire_weather_index: 4.1  }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:14Z' },
  '8228b7ffffffffff': { grid_id: '8228b7ffffffffff', probability: 0.09, tier: 'LOW',      features: { temperature_2m: 12.4, relative_humidity_2m: 76.2, wind_speed_10m: 7.4,  vpd: 0.48, fire_weather_index: 5.3  }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:15Z' },
  '8228c7ffffffffff': { grid_id: '8228c7ffffffffff', probability: 0.19, tier: 'MEDIUM',   features: { temperature_2m: 15.8, relative_humidity_2m: 58.4, wind_speed_10m: 14.2, vpd: 0.94, fire_weather_index: 12.8 }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:16Z' },
  '8228d7ffffffffff': { grid_id: '8228d7ffffffffff', probability: 0.28, tier: 'MEDIUM',   features: { temperature_2m: 24.8, relative_humidity_2m: 22.1, wind_speed_10m: 18.4, vpd: 2.98, fire_weather_index: 31.2 }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:17Z' },
  '8228e1ffffffffff': { grid_id: '8228e1ffffffffff', probability: 0.44, tier: 'HIGH',     features: { temperature_2m: 26.4, relative_humidity_2m: 18.4, wind_speed_10m: 22.1, vpd: 3.42, fire_weather_index: 42.8 }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:18Z' },
  '8228f1ffffffffff': { grid_id: '8228f1ffffffffff', probability: 0.38, tier: 'HIGH',     features: { temperature_2m: 25.1, relative_humidity_2m: 21.8, wind_speed_10m: 19.8, vpd: 3.01, fire_weather_index: 36.4 }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:19Z' },
  '82281bffffffffff': { grid_id: '82281bffffffffff', probability: 0.52, tier: 'HIGH',     features: { temperature_2m: 27.8, relative_humidity_2m: 16.2, wind_speed_10m: 24.6, vpd: 4.12, fire_weather_index: 51.4 }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:20Z' },
  '82282bffffffffff': { grid_id: '82282bffffffffff', probability: 0.21, tier: 'MEDIUM',   features: { temperature_2m: 10.4, relative_humidity_2m: 74.8, wind_speed_10m: 9.2,  vpd: 0.42, fire_weather_index: 7.8  }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:21Z' },
  '82283bffffffffff': { grid_id: '82283bffffffffff', probability: 0.15, tier: 'MEDIUM',   features: { temperature_2m: 7.2,  relative_humidity_2m: 82.1, wind_speed_10m: 6.8,  vpd: 0.24, fire_weather_index: 3.8  }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:22Z' },
  '82284bffffffffff': { grid_id: '82284bffffffffff', probability: 0.18, tier: 'MEDIUM',   features: { temperature_2m: 6.4,  relative_humidity_2m: 71.2, wind_speed_10m: 11.4, vpd: 0.38, fire_weather_index: 8.2  }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:23Z' },
  '82285bffffffffff': { grid_id: '82285bffffffffff', probability: 0.24, tier: 'MEDIUM',   features: { temperature_2m: 14.8, relative_humidity_2m: 62.4, wind_speed_10m: 8.4,  vpd: 0.78, fire_weather_index: 14.2 }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:24Z' },
  '82286bffffffffff': { grid_id: '82286bffffffffff', probability: 0.06, tier: 'LOW',      features: { temperature_2m: 18.4, relative_humidity_2m: 28.2, wind_speed_10m: 14.2, vpd: 1.82, fire_weather_index: 9.4  }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:25Z' },
  '82287bffffffffff': { grid_id: '82287bffffffffff', probability: 0.71, tier: 'CRITICAL', features: { temperature_2m: 28.2, relative_humidity_2m: 12.4, wind_speed_10m: 28.4, vpd: 4.98, fire_weather_index: 68.4 }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:26Z' },
  '82288bffffffffff': { grid_id: '82288bffffffffff', probability: 0.31, tier: 'MEDIUM',   features: { temperature_2m: 21.4, relative_humidity_2m: 24.8, wind_speed_10m: 16.2, vpd: 2.41, fire_weather_index: 28.4 }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:27Z' },
  '82289bffffffffff': { grid_id: '82289bffffffffff', probability: 0.11, tier: 'LOW',      features: { temperature_2m: 4.8,  relative_humidity_2m: 86.2, wind_speed_10m: 5.2,  vpd: 0.14, fire_weather_index: 2.1  }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:28Z' },
  '8228aabfffffffff': { grid_id: '8228aabfffffffff', probability: 0.17, tier: 'MEDIUM',   features: { temperature_2m: 8.2,  relative_humidity_2m: 79.4, wind_speed_10m: 7.8,  vpd: 0.28, fire_weather_index: 5.2  }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:29Z' },
  '8228abbfffffffff': { grid_id: '8228abbfffffffff', probability: 0.10, tier: 'LOW',      features: { temperature_2m: 11.8, relative_humidity_2m: 78.2, wind_speed_10m: 6.8,  vpd: 0.41, fire_weather_index: 4.8  }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:30Z' },
  '8228acbfffffffff': { grid_id: '8228acbfffffffff', probability: 0.23, tier: 'MEDIUM',   features: { temperature_2m: 13.4, relative_humidity_2m: 61.8, wind_speed_10m: 12.4, vpd: 0.84, fire_weather_index: 13.8 }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:31Z' },
  // Texas
  '8244d9ffffffffff': { grid_id: '8244d9ffffffffff', probability: 0.07, tier: 'LOW',      features: { temperature_2m: 18.4, relative_humidity_2m: 72.4, wind_speed_10m: 14.2, vpd: 0.72, fire_weather_index: 8.4  }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:32Z' },
  '8244e9ffffffffff': { grid_id: '8244e9ffffffffff', probability: 0.16, tier: 'MEDIUM',   features: { temperature_2m: 19.8, relative_humidity_2m: 54.2, wind_speed_10m: 16.8, vpd: 1.21, fire_weather_index: 14.8 }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:33Z' },
  '8244f9ffffffffff': { grid_id: '8244f9ffffffffff', probability: 0.22, tier: 'MEDIUM',   features: { temperature_2m: 21.4, relative_humidity_2m: 48.2, wind_speed_10m: 18.4, vpd: 1.74, fire_weather_index: 19.4 }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:34Z' },
  '824409ffffffffff': { grid_id: '824409ffffffffff', probability: 0.18, tier: 'MEDIUM',   features: { temperature_2m: 14.8, relative_humidity_2m: 58.4, wind_speed_10m: 14.8, vpd: 0.94, fire_weather_index: 12.4 }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:35Z' },
  '824419ffffffffff': { grid_id: '824419ffffffffff', probability: 0.24, tier: 'MEDIUM',   features: { temperature_2m: 17.8, relative_humidity_2m: 51.8, wind_speed_10m: 17.2, vpd: 1.41, fire_weather_index: 17.8 }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:36Z' },
  '824429ffffffffff': { grid_id: '824429ffffffffff', probability: 0.41, tier: 'HIGH',     features: { temperature_2m: 24.8, relative_humidity_2m: 22.4, wind_speed_10m: 21.4, vpd: 3.12, fire_weather_index: 38.4 }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:37Z' },
  '824439ffffffffff': { grid_id: '824439ffffffffff', probability: 0.28, tier: 'MEDIUM',   features: { temperature_2m: 22.4, relative_humidity_2m: 18.4, wind_speed_10m: 19.8, vpd: 2.84, fire_weather_index: 24.8 }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:38Z' },
  '824449ffffffffff': { grid_id: '824449ffffffffff', probability: 0.67, tier: 'CRITICAL', features: { temperature_2m: 28.8, relative_humidity_2m: 11.4, wind_speed_10m: 28.4, vpd: 5.12, fire_weather_index: 72.4 }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:39Z' },
  '824459ffffffffff': { grid_id: '824459ffffffffff', probability: 0.34, tier: 'MEDIUM',   features: { temperature_2m: 12.8, relative_humidity_2m: 32.4, wind_speed_10m: 22.8, vpd: 1.48, fire_weather_index: 28.4 }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:40Z' },
  '824469ffffffffff': { grid_id: '824469ffffffffff', probability: 0.12, tier: 'LOW',      features: { temperature_2m: 22.4, relative_humidity_2m: 68.4, wind_speed_10m: 16.4, vpd: 1.02, fire_weather_index: 10.4 }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:41Z' },
  '824479ffffffffff': { grid_id: '824479ffffffffff', probability: 0.09, tier: 'LOW',      features: { temperature_2m: 24.8, relative_humidity_2m: 64.8, wind_speed_10m: 12.8, vpd: 1.28, fire_weather_index: 9.8  }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:42Z' },
  '824489ffffffffff': { grid_id: '824489ffffffffff', probability: 0.38, tier: 'HIGH',     features: { temperature_2m: 14.4, relative_humidity_2m: 28.8, wind_speed_10m: 24.2, vpd: 1.98, fire_weather_index: 32.4 }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:43Z' },
  '824499ffffffffff': { grid_id: '824499ffffffffff', probability: 0.29, tier: 'MEDIUM',   features: { temperature_2m: 20.4, relative_humidity_2m: 34.4, wind_speed_10m: 18.8, vpd: 2.08, fire_weather_index: 22.4 }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:44Z' },
  '8244a9ffffffffff': { grid_id: '8244a9ffffffffff', probability: 0.48, tier: 'HIGH',     features: { temperature_2m: 27.4, relative_humidity_2m: 20.4, wind_speed_10m: 22.4, vpd: 3.84, fire_weather_index: 44.8 }, model_version: 'xgboost_ignition_v1.3.2', inference_ts: '2025-01-15T18:04:45Z' },
};

// --- OBJ-2: Fire Spread Simulation Results ---
// Shape mirrors model-pipeline/obj2_spread/Cell2Fire + Rothermel output.
// NOTE: Cell2Fire is manual-trigger only — not in Airflow DAG.
// Simulations exist only for cells with fire_detected_binary=1 or active_fire_count > 0.
export const OBJ2_SPREAD = {
  // Santa Ynez Range — CRITICAL, fire_detected=1, active_fires=3
  '82287bffffffffff': {
    source_cell: '82287bffffffffff',
    source_name: 'Santa Ynez Range',
    simulation_ts: '2025-01-15T16:30:00Z',
    trigger: 'manual',  // not automated
    model: 'Cell2Fire v2.1 + Rothermel (1972)',
    wind_direction_deg: 45,   // NE
    wind_speed_m_s: 7.9,
    spread_rate_m_per_min: 18.4,
    spread_area_km2: 124.8,
    time_horizon_hrs: 12,
    containment_probability: 0.31,
    affected_cells: [
      '82287bffffffffff',  // source
      '8228e1ffffffffff',  // Santa Barbara (downwind, HIGH)
      '82281bffffffffff',  // San Gabriel Mountains (HIGH)
      '82285bffffffffff',  // Sequoia Foothills (adjacent)
    ],
    perimeter_coords: [
      // Approximate polygon around Santa Ynez + downwind extent
      { lat: 35.1, lon: -120.8 },
      { lat: 35.2, lon: -120.0 },
      { lat: 34.8, lon: -119.5 },
      { lat: 34.5, lon: -119.6 },
      { lat: 34.4, lon: -120.2 },
      { lat: 34.7, lon: -120.7 },
    ],
    confidence: 0.71,
    notes: 'Diablo wind event. High ROS due to FBFM 4 fuel model. Manually triggered — not in DAG.',
  },
  // Big Bend / Trans-Pecos — CRITICAL, fire_detected=1, active_fires=4
  '824449ffffffffff': {
    source_cell: '824449ffffffffff',
    source_name: 'Big Bend / Trans-Pecos',
    simulation_ts: '2025-01-15T17:15:00Z',
    trigger: 'manual',
    model: 'Cell2Fire v2.1 + Rothermel (1972)',
    wind_direction_deg: 220,  // SW
    wind_speed_m_s: 7.9,
    spread_rate_m_per_min: 22.1,
    spread_area_km2: 218.4,
    time_horizon_hrs: 12,
    containment_probability: 0.18,
    affected_cells: [
      '824449ffffffffff',  // source
      '824429ffffffffff',  // Midland / Permian (HIGH, downwind)
      '8244a9ffffffffff',  // Laredo / Eagle Pass (HIGH)
      '824499ffffffffff',  // San Angelo (adjacent)
    ],
    perimeter_coords: [
      { lat: 30.8, lon: -104.2 },
      { lat: 31.2, lon: -103.2 },
      { lat: 30.6, lon: -102.8 },
      { lat: 29.8, lon: -103.0 },
      { lat: 29.6, lon: -103.8 },
      { lat: 30.0, lon: -104.4 },
    ],
    confidence: 0.68,
    notes: 'Isolated terrain with sparse fuel. SW wind driving spread NE toward Permian Basin. Manually triggered.',
  },
  // San Gabriel Mountains — HIGH, active_fires=2 (simulation available as precautionary)
  '82281bffffffffff': {
    source_cell: '82281bffffffffff',
    source_name: 'San Gabriel Mountains',
    simulation_ts: '2025-01-15T15:45:00Z',
    trigger: 'manual',
    model: 'Cell2Fire v2.1 + Rothermel (1972)',
    wind_direction_deg: 270,  // W
    wind_speed_m_s: 6.8,
    spread_rate_m_per_min: 11.2,
    spread_area_km2: 67.4,
    time_horizon_hrs: 6,
    containment_probability: 0.54,
    affected_cells: [
      '82281bffffffffff',  // source
      '8228f1ffffffffff',  // Orange County Hills
      '8228e9ffffffffff',  // LA Basin
    ],
    perimeter_coords: [
      { lat: 34.5, lon: -117.8 },
      { lat: 34.6, lon: -117.2 },
      { lat: 34.3, lon: -117.0 },
      { lat: 34.0, lon: -117.3 },
      { lat: 34.1, lon: -117.8 },
    ],
    confidence: 0.58,
    notes: 'Santa Ana wind conditions. FBFM 9 chaparral. Precautionary simulation — no confirmed ignition.',
  },
};

// Metadata about the simulation run environment
export const MAP_META = {
  grid_resolution_km: 64,
  mode: 'QUIET',
  obj1_model: 'xgboost_ignition_v1.3.2',
  obj2_model: 'Cell2Fire v2.1 + Rothermel (1972)',
  obj2_trigger: 'manual',  // not automated — see COMPONENT_STATUS
  last_obj1_inference: '2025-01-15T18:04:45Z',
  last_obj2_simulation: '2025-01-15T17:15:00Z',
  regions: ['california', 'texas'],
};
