// Mock H3 grid cell data derived from Data-Pipeline/ and model-pipeline/ structures.
// Grid is 55 cells at 64km resolution: ~35 CA + ~20 TX (as defined in fuse_features.py).
// Risk tiers map directly to model-pipeline/configs/model_config.yaml thresholds:
//   CRITICAL: fire_risk_score >= 0.65
//   HIGH:     fire_risk_score >= 0.365
//   MEDIUM:   fire_risk_score >= 0.15
//   LOW:      fire_risk_score < 0.15

export const RISK_THRESHOLDS = {
  CRITICAL: 0.65,
  HIGH: 0.365,
  MEDIUM: 0.15,
  LOW: 0.0,
};

export function getRiskTier(score) {
  if (score >= RISK_THRESHOLDS.CRITICAL) return 'CRITICAL';
  if (score >= RISK_THRESHOLDS.HIGH)     return 'HIGH';
  if (score >= RISK_THRESHOLDS.MEDIUM)   return 'MEDIUM';
  return 'LOW';
}

// California grid cells (H3 resolution 2, ~35 cells)
export const CALIFORNIA_CELLS = [
  { grid_id: '8228e9ffffffffff', lat: 34.05, lon: -118.24, name: 'Los Angeles Basin',    fire_risk_score: 0.14, temperature_2m: 22.1, relative_humidity_2m: 38.2, wind_speed_10m: 12.4, vpd: 1.82, fire_weather_index: 18.4, fuel_model_fbfm40: 'FBFM 9',  elevation_m: 86,   active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '822897ffffffffff', lat: 37.33, lon: -121.89, name: 'Bay Area / South Bay',  fire_risk_score: 0.08, temperature_2m: 14.2, relative_humidity_2m: 72.1, wind_speed_10m: 8.2,  vpd: 0.61, fire_weather_index: 6.2,  fuel_model_fbfm40: 'FBFM 2',  elevation_m: 15,   active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '8228a7ffffffffff', lat: 38.58, lon: -121.49, name: 'Sacramento Valley',     fire_risk_score: 0.12, temperature_2m: 9.8,  relative_humidity_2m: 81.4, wind_speed_10m: 6.1,  vpd: 0.34, fire_weather_index: 4.1,  fuel_model_fbfm40: 'FBFM 1',  elevation_m: 8,    active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '8228b7ffffffffff', lat: 36.74, lon: -119.79, name: 'Central Valley',         fire_risk_score: 0.09, temperature_2m: 12.4, relative_humidity_2m: 76.2, wind_speed_10m: 7.4,  vpd: 0.48, fire_weather_index: 5.3,  fuel_model_fbfm40: 'FBFM 1',  elevation_m: 70,   active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '8228c7ffffffffff', lat: 37.98, lon: -122.05, name: 'Diablo Range',           fire_risk_score: 0.19, temperature_2m: 15.8, relative_humidity_2m: 58.4, wind_speed_10m: 14.2, vpd: 0.94, fire_weather_index: 12.8, fuel_model_fbfm40: 'FBFM 5',  elevation_m: 412,  active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '8228d7ffffffffff', lat: 33.94, lon: -116.54, name: 'Inland Empire',          fire_risk_score: 0.28, temperature_2m: 24.8, relative_humidity_2m: 22.1, wind_speed_10m: 18.4, vpd: 2.98, fire_weather_index: 31.2, fuel_model_fbfm40: 'FBFM 4',  elevation_m: 521,  active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '8228e1ffffffffff', lat: 34.42, lon: -119.70, name: 'Santa Barbara',          fire_risk_score: 0.44, temperature_2m: 26.4, relative_humidity_2m: 18.4, wind_speed_10m: 22.1, vpd: 3.42, fire_weather_index: 42.8, fuel_model_fbfm40: 'FBFM 4',  elevation_m: 128,  active_fire_count: 1, fire_detected_binary: 0 },
  { grid_id: '8228f1ffffffffff', lat: 33.81, lon: -117.92, name: 'Orange County Hills',   fire_risk_score: 0.38, temperature_2m: 25.1, relative_humidity_2m: 21.8, wind_speed_10m: 19.8, vpd: 3.01, fire_weather_index: 36.4, fuel_model_fbfm40: 'FBFM 9',  elevation_m: 248,  active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '82281bffffffffff', lat: 34.21, lon: -117.38, name: 'San Gabriel Mountains', fire_risk_score: 0.52, temperature_2m: 27.8, relative_humidity_2m: 16.2, wind_speed_10m: 24.6, vpd: 4.12, fire_weather_index: 51.4, fuel_model_fbfm40: 'FBFM 9',  elevation_m: 1842, active_fire_count: 2, fire_detected_binary: 0 },
  { grid_id: '82282bffffffffff', lat: 39.52, lon: -121.56, name: 'Butte County',           fire_risk_score: 0.21, temperature_2m: 10.4, relative_humidity_2m: 74.8, wind_speed_10m: 9.2,  vpd: 0.42, fire_weather_index: 7.8,  fuel_model_fbfm40: 'FBFM 10', elevation_m: 184,  active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '82283bffffffffff', lat: 40.58, lon: -122.39, name: 'Shasta County',          fire_risk_score: 0.15, temperature_2m: 7.2,  relative_humidity_2m: 82.1, wind_speed_10m: 6.8,  vpd: 0.24, fire_weather_index: 3.8,  fuel_model_fbfm40: 'FBFM 10', elevation_m: 328,  active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '82284bffffffffff', lat: 37.48, lon: -118.91, name: 'Sierra Nevada East',    fire_risk_score: 0.18, temperature_2m: 6.4,  relative_humidity_2m: 71.2, wind_speed_10m: 11.4, vpd: 0.38, fire_weather_index: 8.2,  fuel_model_fbfm40: 'FBFM 10', elevation_m: 2841, active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '82285bffffffffff', lat: 36.27, lon: -118.52, name: 'Sequoia Foothills',     fire_risk_score: 0.24, temperature_2m: 14.8, relative_humidity_2m: 62.4, wind_speed_10m: 8.4,  vpd: 0.78, fire_weather_index: 14.2, fuel_model_fbfm40: 'FBFM 10', elevation_m: 1124, active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '82286bffffffffff', lat: 33.42, lon: -115.82, name: 'Salton Sea / Desert',   fire_risk_score: 0.06, temperature_2m: 18.4, relative_humidity_2m: 28.2, wind_speed_10m: 14.2, vpd: 1.82, fire_weather_index: 9.4,  fuel_model_fbfm40: 'FBFM 99', elevation_m: -69,  active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '82287bffffffffff', lat: 34.88, lon: -120.41, name: 'Santa Ynez Range',      fire_risk_score: 0.71, temperature_2m: 28.2, relative_humidity_2m: 12.4, wind_speed_10m: 28.4, vpd: 4.98, fire_weather_index: 68.4, fuel_model_fbfm40: 'FBFM 4',  elevation_m: 892,  active_fire_count: 3, fire_detected_binary: 1 },
  { grid_id: '82288bffffffffff', lat: 35.38, lon: -119.02, name: 'Kern County',            fire_risk_score: 0.31, temperature_2m: 21.4, relative_humidity_2m: 24.8, wind_speed_10m: 16.2, vpd: 2.41, fire_weather_index: 28.4, fuel_model_fbfm40: 'FBFM 2',  elevation_m: 248,  active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '82289bffffffffff', lat: 41.28, lon: -123.18, name: 'Trinity / Klamath',     fire_risk_score: 0.11, temperature_2m: 4.8,  relative_humidity_2m: 86.2, wind_speed_10m: 5.2,  vpd: 0.14, fire_weather_index: 2.1,  fuel_model_fbfm40: 'FBFM 10', elevation_m: 1042, active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '8228aabfffffffff', lat: 38.91, lon: -120.06, name: 'Sierra Nevada West',    fire_risk_score: 0.17, temperature_2m: 8.2,  relative_humidity_2m: 79.4, wind_speed_10m: 7.8,  vpd: 0.28, fire_weather_index: 5.2,  fuel_model_fbfm40: 'FBFM 10', elevation_m: 1584, active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '8228abbfffffffff', lat: 37.04, lon: -120.48, name: 'Merced / Fresno',       fire_risk_score: 0.10, temperature_2m: 11.8, relative_humidity_2m: 78.2, wind_speed_10m: 6.8,  vpd: 0.41, fire_weather_index: 4.8,  fuel_model_fbfm40: 'FBFM 1',  elevation_m: 52,   active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '8228acbfffffffff', lat: 38.52, lon: -123.01, name: 'Sonoma / Napa',         fire_risk_score: 0.23, temperature_2m: 13.4, relative_humidity_2m: 61.8, wind_speed_10m: 12.4, vpd: 0.84, fire_weather_index: 13.8, fuel_model_fbfm40: 'FBFM 5',  elevation_m: 184,  active_fire_count: 0, fire_detected_binary: 0 },
];

// Texas grid cells (H3 resolution 2, ~20 cells)
export const TEXAS_CELLS = [
  { grid_id: '8244d9ffffffffff', lat: 29.76, lon: -95.37, name: 'Houston',                fire_risk_score: 0.07, temperature_2m: 18.4, relative_humidity_2m: 72.4, wind_speed_10m: 14.2, vpd: 0.72, fire_weather_index: 8.4,  fuel_model_fbfm40: 'FBFM 1', elevation_m: 15,  active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '8244e9ffffffffff', lat: 30.27, lon: -97.74, name: 'Austin',                  fire_risk_score: 0.16, temperature_2m: 19.8, relative_humidity_2m: 54.2, wind_speed_10m: 16.8, vpd: 1.21, fire_weather_index: 14.8, fuel_model_fbfm40: 'FBFM 2', elevation_m: 148, active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '8244f9ffffffffff', lat: 29.42, lon: -98.49, name: 'San Antonio',             fire_risk_score: 0.22, temperature_2m: 21.4, relative_humidity_2m: 48.2, wind_speed_10m: 18.4, vpd: 1.74, fire_weather_index: 19.4, fuel_model_fbfm40: 'FBFM 2', elevation_m: 198, active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '824409ffffffffff', lat: 32.78, lon: -96.80, name: 'Dallas / Fort Worth',     fire_risk_score: 0.18, temperature_2m: 14.8, relative_humidity_2m: 58.4, wind_speed_10m: 14.8, vpd: 0.94, fire_weather_index: 12.4, fuel_model_fbfm40: 'FBFM 1', elevation_m: 186, active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '824419ffffffffff', lat: 31.54, lon: -97.14, name: 'Waco',                    fire_risk_score: 0.24, temperature_2m: 17.8, relative_humidity_2m: 51.8, wind_speed_10m: 17.2, vpd: 1.41, fire_weather_index: 17.8, fuel_model_fbfm40: 'FBFM 2', elevation_m: 148, active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '824429ffffffffff', lat: 30.82, lon: -102.42, name: 'Midland / Permian',      fire_risk_score: 0.41, temperature_2m: 24.8, relative_humidity_2m: 22.4, wind_speed_10m: 21.4, vpd: 3.12, fire_weather_index: 38.4, fuel_model_fbfm40: 'FBFM 1', elevation_m: 852, active_fire_count: 1, fire_detected_binary: 0 },
  { grid_id: '824439ffffffffff', lat: 31.84, lon: -106.42, name: 'El Paso',                fire_risk_score: 0.28, temperature_2m: 22.4, relative_humidity_2m: 18.4, wind_speed_10m: 19.8, vpd: 2.84, fire_weather_index: 24.8, fuel_model_fbfm40: 'FBFM 1', elevation_m: 1141, active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '824449ffffffffff', lat: 30.19, lon: -103.58, name: 'Big Bend / Trans-Pecos', fire_risk_score: 0.67, temperature_2m: 28.8, relative_humidity_2m: 11.4, wind_speed_10m: 28.4, vpd: 5.12, fire_weather_index: 72.4, fuel_model_fbfm40: 'FBFM 2', elevation_m: 1284, active_fire_count: 4, fire_detected_binary: 1 },
  { grid_id: '824459ffffffffff', lat: 35.22, lon: -101.83, name: 'Amarillo / Panhandle',   fire_risk_score: 0.34, temperature_2m: 12.8, relative_humidity_2m: 32.4, wind_speed_10m: 22.8, vpd: 1.48, fire_weather_index: 28.4, fuel_model_fbfm40: 'FBFM 1', elevation_m: 1099, active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '824469ffffffffff', lat: 27.80, lon: -97.40, name: 'Corpus Christi',          fire_risk_score: 0.12, temperature_2m: 22.4, relative_humidity_2m: 68.4, wind_speed_10m: 16.4, vpd: 1.02, fire_weather_index: 10.4, fuel_model_fbfm40: 'FBFM 1', elevation_m: 12,  active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '824479ffffffffff', lat: 26.20, lon: -98.23, name: 'Rio Grande Valley',       fire_risk_score: 0.09, temperature_2m: 24.8, relative_humidity_2m: 64.8, wind_speed_10m: 12.8, vpd: 1.28, fire_weather_index: 9.8,  fuel_model_fbfm40: 'FBFM 1', elevation_m: 38,  active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '824489ffffffffff', lat: 33.58, lon: -101.86, name: 'Lubbock / Staked Plain', fire_risk_score: 0.38, temperature_2m: 14.4, relative_humidity_2m: 28.8, wind_speed_10m: 24.2, vpd: 1.98, fire_weather_index: 32.4, fuel_model_fbfm40: 'FBFM 1', elevation_m: 988, active_fire_count: 1, fire_detected_binary: 0 },
  { grid_id: '824499ffffffffff', lat: 31.46, lon: -100.44, name: 'San Angelo',             fire_risk_score: 0.29, temperature_2m: 20.4, relative_humidity_2m: 34.4, wind_speed_10m: 18.8, vpd: 2.08, fire_weather_index: 22.4, fuel_model_fbfm40: 'FBFM 2', elevation_m: 564, active_fire_count: 0, fire_detected_binary: 0 },
  { grid_id: '8244a9ffffffffff', lat: 28.70, lon: -100.49, name: 'Laredo / Eagle Pass',    fire_risk_score: 0.48, temperature_2m: 27.4, relative_humidity_2m: 20.4, wind_speed_10m: 22.4, vpd: 3.84, fire_weather_index: 44.8, fuel_model_fbfm40: 'FBFM 2', elevation_m: 182, active_fire_count: 2, fire_detected_binary: 0 },
];
