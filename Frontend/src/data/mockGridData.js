// Mock H3 grid cell data — full fused feature schema (35 columns).
// Matches Data-Pipeline/data/processed/fused/ output + model-pipeline inference.
// Grid is ~55 cells at 64km resolution: ~35 CA + ~20 TX.
//
// Risk tiers map to model-pipeline/configs/model_config.yaml thresholds:
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

// ─── Extended features per cell ──────────────────────────────────────────────
// Keys align with Data-Pipeline schema_config.yaml:
//   wind_direction_10m (°), precipitation (mm), soil_moisture_0_to_7cm (m³/m³),
//   canopy_cover_pct (%), canopy_base_height_m (m), canopy_bulk_density (kg/m³),
//   vegetation_type, slope_degrees (°), aspect_degrees (°),
//   mean_frp (MW), dominant_fuel_fraction (0-1)

const EXT = {
  // ── California ──
  '8228e9ffffffffff': { wind_direction_10m: 270, precipitation: 0.0, soil_moisture_0_to_7cm: 0.12, canopy_cover_pct: 15, canopy_base_height_m: 8.2, canopy_bulk_density: 0.04, vegetation_type: 'Shrubland',      slope_degrees: 2.4,  aspect_degrees: 180, mean_frp: 0,     dominant_fuel_fraction: 0.72 },
  '822897ffffffffff': { wind_direction_10m: 315, precipitation: 1.2, soil_moisture_0_to_7cm: 0.28, canopy_cover_pct: 22, canopy_base_height_m: 10.0,canopy_bulk_density: 0.06, vegetation_type: 'Mixed Forest',   slope_degrees: 4.1,  aspect_degrees: 225, mean_frp: 0,     dominant_fuel_fraction: 0.58 },
  '8228a7ffffffffff': { wind_direction_10m: 200, precipitation: 2.8, soil_moisture_0_to_7cm: 0.35, canopy_cover_pct: 8,  canopy_base_height_m: 12.0,canopy_bulk_density: 0.02, vegetation_type: 'Agriculture',    slope_degrees: 0.8,  aspect_degrees: 90,  mean_frp: 0,     dominant_fuel_fraction: 0.45 },
  '8228b7ffffffffff': { wind_direction_10m: 190, precipitation: 1.4, soil_moisture_0_to_7cm: 0.32, canopy_cover_pct: 5,  canopy_base_height_m: 14.0,canopy_bulk_density: 0.01, vegetation_type: 'Agriculture',    slope_degrees: 0.5,  aspect_degrees: 120, mean_frp: 0,     dominant_fuel_fraction: 0.40 },
  '8228c7ffffffffff': { wind_direction_10m: 245, precipitation: 0.0, soil_moisture_0_to_7cm: 0.18, canopy_cover_pct: 28, canopy_base_height_m: 6.5, canopy_bulk_density: 0.08, vegetation_type: 'Chaparral',      slope_degrees: 18.2, aspect_degrees: 210, mean_frp: 0,     dominant_fuel_fraction: 0.68 },
  '8228d7ffffffffff': { wind_direction_10m: 290, precipitation: 0.0, soil_moisture_0_to_7cm: 0.06, canopy_cover_pct: 12, canopy_base_height_m: 7.0, canopy_bulk_density: 0.03, vegetation_type: 'Chaparral',      slope_degrees: 14.8, aspect_degrees: 195, mean_frp: 0,     dominant_fuel_fraction: 0.74 },
  '8228e1ffffffffff': { wind_direction_10m: 310, precipitation: 0.0, soil_moisture_0_to_7cm: 0.05, canopy_cover_pct: 34, canopy_base_height_m: 4.2, canopy_bulk_density: 0.09, vegetation_type: 'Chaparral',      slope_degrees: 22.4, aspect_degrees: 170, mean_frp: 42.8,  dominant_fuel_fraction: 0.81 },
  '8228f1ffffffffff': { wind_direction_10m: 275, precipitation: 0.0, soil_moisture_0_to_7cm: 0.08, canopy_cover_pct: 30, canopy_base_height_m: 5.8, canopy_bulk_density: 0.07, vegetation_type: 'Chaparral',      slope_degrees: 16.5, aspect_degrees: 240, mean_frp: 0,     dominant_fuel_fraction: 0.76 },
  '82281bffffffffff': { wind_direction_10m: 270, precipitation: 0.0, soil_moisture_0_to_7cm: 0.04, canopy_cover_pct: 52, canopy_base_height_m: 2.8, canopy_bulk_density: 0.14, vegetation_type: 'Conifer Forest', slope_degrees: 28.4, aspect_degrees: 200, mean_frp: 124.6, dominant_fuel_fraction: 0.85 },
  '82282bffffffffff': { wind_direction_10m: 220, precipitation: 0.4, soil_moisture_0_to_7cm: 0.24, canopy_cover_pct: 45, canopy_base_height_m: 5.4, canopy_bulk_density: 0.11, vegetation_type: 'Mixed Forest',   slope_degrees: 12.1, aspect_degrees: 160, mean_frp: 0,     dominant_fuel_fraction: 0.62 },
  '82283bffffffffff': { wind_direction_10m: 210, precipitation: 3.2, soil_moisture_0_to_7cm: 0.38, canopy_cover_pct: 58, canopy_base_height_m: 6.2, canopy_bulk_density: 0.12, vegetation_type: 'Conifer Forest', slope_degrees: 15.4, aspect_degrees: 145, mean_frp: 0,     dominant_fuel_fraction: 0.71 },
  '82284bffffffffff': { wind_direction_10m: 255, precipitation: 0.2, soil_moisture_0_to_7cm: 0.16, canopy_cover_pct: 35, canopy_base_height_m: 4.8, canopy_bulk_density: 0.10, vegetation_type: 'Conifer Forest', slope_degrees: 32.1, aspect_degrees: 280, mean_frp: 0,     dominant_fuel_fraction: 0.67 },
  '82285bffffffffff': { wind_direction_10m: 240, precipitation: 0.0, soil_moisture_0_to_7cm: 0.14, canopy_cover_pct: 48, canopy_base_height_m: 3.6, canopy_bulk_density: 0.12, vegetation_type: 'Mixed Forest',   slope_degrees: 20.8, aspect_degrees: 190, mean_frp: 0,     dominant_fuel_fraction: 0.70 },
  '82286bffffffffff': { wind_direction_10m: 180, precipitation: 0.0, soil_moisture_0_to_7cm: 0.02, canopy_cover_pct: 2,  canopy_base_height_m: 20.0,canopy_bulk_density: 0.00, vegetation_type: 'Desert Scrub',   slope_degrees: 1.2,  aspect_degrees: 0,   mean_frp: 0,     dominant_fuel_fraction: 0.30 },
  '82287bffffffffff': { wind_direction_10m: 45,  precipitation: 0.0, soil_moisture_0_to_7cm: 0.03, canopy_cover_pct: 38, canopy_base_height_m: 2.4, canopy_bulk_density: 0.15, vegetation_type: 'Chaparral',      slope_degrees: 25.6, aspect_degrees: 175, mean_frp: 284.2, dominant_fuel_fraction: 0.88 },
  '82288bffffffffff': { wind_direction_10m: 260, precipitation: 0.0, soil_moisture_0_to_7cm: 0.07, canopy_cover_pct: 10, canopy_base_height_m: 9.0, canopy_bulk_density: 0.03, vegetation_type: 'Grassland',      slope_degrees: 3.8,  aspect_degrees: 150, mean_frp: 0,     dominant_fuel_fraction: 0.55 },
  '82289bffffffffff': { wind_direction_10m: 195, precipitation: 5.8, soil_moisture_0_to_7cm: 0.42, canopy_cover_pct: 62, canopy_base_height_m: 7.8, canopy_bulk_density: 0.13, vegetation_type: 'Conifer Forest', slope_degrees: 24.2, aspect_degrees: 310, mean_frp: 0,     dominant_fuel_fraction: 0.75 },
  '8228aabfffffffff': { wind_direction_10m: 230, precipitation: 0.8, soil_moisture_0_to_7cm: 0.26, canopy_cover_pct: 55, canopy_base_height_m: 5.0, canopy_bulk_density: 0.11, vegetation_type: 'Conifer Forest', slope_degrees: 22.8, aspect_degrees: 250, mean_frp: 0,     dominant_fuel_fraction: 0.69 },
  '8228abbfffffffff': { wind_direction_10m: 185, precipitation: 1.0, soil_moisture_0_to_7cm: 0.30, canopy_cover_pct: 6,  canopy_base_height_m: 13.0,canopy_bulk_density: 0.02, vegetation_type: 'Agriculture',    slope_degrees: 1.0,  aspect_degrees: 100, mean_frp: 0,     dominant_fuel_fraction: 0.42 },
  '8228acbfffffffff': { wind_direction_10m: 340, precipitation: 0.0, soil_moisture_0_to_7cm: 0.15, canopy_cover_pct: 32, canopy_base_height_m: 4.5, canopy_bulk_density: 0.09, vegetation_type: 'Mixed Forest',   slope_degrees: 10.4, aspect_degrees: 185, mean_frp: 0,     dominant_fuel_fraction: 0.64 },
  // ── Texas ──
  '8244d9ffffffffff': { wind_direction_10m: 165, precipitation: 4.2, soil_moisture_0_to_7cm: 0.34, canopy_cover_pct: 18, canopy_base_height_m: 11.0,canopy_bulk_density: 0.05, vegetation_type: 'Mixed Forest',   slope_degrees: 1.2,  aspect_degrees: 90,  mean_frp: 0,     dominant_fuel_fraction: 0.48 },
  '8244e9ffffffffff': { wind_direction_10m: 180, precipitation: 1.4, soil_moisture_0_to_7cm: 0.20, canopy_cover_pct: 25, canopy_base_height_m: 8.0, canopy_bulk_density: 0.06, vegetation_type: 'Woodland',       slope_degrees: 5.8,  aspect_degrees: 200, mean_frp: 0,     dominant_fuel_fraction: 0.55 },
  '8244f9ffffffffff': { wind_direction_10m: 195, precipitation: 0.8, soil_moisture_0_to_7cm: 0.16, canopy_cover_pct: 14, canopy_base_height_m: 9.5, canopy_bulk_density: 0.04, vegetation_type: 'Shrubland',      slope_degrees: 3.4,  aspect_degrees: 170, mean_frp: 0,     dominant_fuel_fraction: 0.52 },
  '824409ffffffffff': { wind_direction_10m: 175, precipitation: 2.0, soil_moisture_0_to_7cm: 0.22, canopy_cover_pct: 20, canopy_base_height_m: 10.0,canopy_bulk_density: 0.05, vegetation_type: 'Woodland',       slope_degrees: 2.8,  aspect_degrees: 140, mean_frp: 0,     dominant_fuel_fraction: 0.50 },
  '824419ffffffffff': { wind_direction_10m: 190, precipitation: 0.6, soil_moisture_0_to_7cm: 0.18, canopy_cover_pct: 16, canopy_base_height_m: 9.0, canopy_bulk_density: 0.04, vegetation_type: 'Grassland',      slope_degrees: 2.2,  aspect_degrees: 160, mean_frp: 0,     dominant_fuel_fraction: 0.60 },
  '824429ffffffffff': { wind_direction_10m: 210, precipitation: 0.0, soil_moisture_0_to_7cm: 0.04, canopy_cover_pct: 4,  canopy_base_height_m: 15.0,canopy_bulk_density: 0.01, vegetation_type: 'Desert Scrub',   slope_degrees: 1.8,  aspect_degrees: 220, mean_frp: 38.4,  dominant_fuel_fraction: 0.35 },
  '824439ffffffffff': { wind_direction_10m: 250, precipitation: 0.0, soil_moisture_0_to_7cm: 0.03, canopy_cover_pct: 3,  canopy_base_height_m: 18.0,canopy_bulk_density: 0.00, vegetation_type: 'Desert Scrub',   slope_degrees: 4.2,  aspect_degrees: 260, mean_frp: 0,     dominant_fuel_fraction: 0.28 },
  '824449ffffffffff': { wind_direction_10m: 220, precipitation: 0.0, soil_moisture_0_to_7cm: 0.02, canopy_cover_pct: 8,  canopy_base_height_m: 5.5, canopy_bulk_density: 0.03, vegetation_type: 'Desert Scrub',   slope_degrees: 8.4,  aspect_degrees: 190, mean_frp: 312.8, dominant_fuel_fraction: 0.38 },
  '824459ffffffffff': { wind_direction_10m: 200, precipitation: 0.0, soil_moisture_0_to_7cm: 0.06, canopy_cover_pct: 3,  canopy_base_height_m: 20.0,canopy_bulk_density: 0.00, vegetation_type: 'Grassland',      slope_degrees: 0.8,  aspect_degrees: 180, mean_frp: 0,     dominant_fuel_fraction: 0.82 },
  '824469ffffffffff': { wind_direction_10m: 155, precipitation: 2.4, soil_moisture_0_to_7cm: 0.28, canopy_cover_pct: 10, canopy_base_height_m: 12.0,canopy_bulk_density: 0.02, vegetation_type: 'Grassland',      slope_degrees: 0.6,  aspect_degrees: 110, mean_frp: 0,     dominant_fuel_fraction: 0.70 },
  '824479ffffffffff': { wind_direction_10m: 145, precipitation: 3.8, soil_moisture_0_to_7cm: 0.30, canopy_cover_pct: 12, canopy_base_height_m: 11.0,canopy_bulk_density: 0.03, vegetation_type: 'Shrubland',      slope_degrees: 1.0,  aspect_degrees: 100, mean_frp: 0,     dominant_fuel_fraction: 0.48 },
  '824489ffffffffff': { wind_direction_10m: 215, precipitation: 0.0, soil_moisture_0_to_7cm: 0.05, canopy_cover_pct: 2,  canopy_base_height_m: 20.0,canopy_bulk_density: 0.00, vegetation_type: 'Grassland',      slope_degrees: 0.4,  aspect_degrees: 200, mean_frp: 18.2,  dominant_fuel_fraction: 0.85 },
  '824499ffffffffff': { wind_direction_10m: 205, precipitation: 0.0, soil_moisture_0_to_7cm: 0.08, canopy_cover_pct: 6,  canopy_base_height_m: 14.0,canopy_bulk_density: 0.01, vegetation_type: 'Grassland',      slope_degrees: 2.0,  aspect_degrees: 175, mean_frp: 0,     dominant_fuel_fraction: 0.72 },
  '8244a9ffffffffff': { wind_direction_10m: 235, precipitation: 0.0, soil_moisture_0_to_7cm: 0.04, canopy_cover_pct: 8,  canopy_base_height_m: 10.0,canopy_bulk_density: 0.02, vegetation_type: 'Shrubland',      slope_degrees: 3.2,  aspect_degrees: 215, mean_frp: 86.4,  dominant_fuel_fraction: 0.58 },
};

// ─── Annotations for cells with notable conditions ───────────────────────────

const NOTES = {
  '82287bffffffffff': 'DIABLO WIND EVENT — Active fire spreading NE toward Santa Barbara. Evacuation order issued for Santa Ynez valley. Spot fires confirmed at 2 locations downwind.',
  '824449ffffffffff': 'ISOLATED TERRAIN FIRE — Access limited to aerial support only. SW wind driving spread NE toward Permian Basin. Local ranchers reported rapid grass fire growth.',
  '82281bffffffffff': 'SANTA ANA CONDITIONS — Precautionary monitoring. No confirmed ignition but extreme fire weather. FBFM 9 chaparral fuel load at critical moisture deficit.',
  '8228e1ffffffffff': 'Spot fire observed 2km NE of main Santa Barbara perimeter. Ground crew deployed. Canopy torching reported in isolated groves.',
  '824429ffffffffff': 'Permian Basin grassland fire. Low fuel load but high wind driving rapid lateral spread. Oil infrastructure in potential path.',
  '8244a9ffffffffff': 'Border region fire activity. Coordination with MX fire services underway. High FRP readings suggest intense surface fire.',
  '824489ffffffffff': 'Staked Plains grass fire. Flat terrain allows unimpeded spread. Lubbock county issued burn ban.',
};

// ─── California grid cells (H3 res-2, ~64km) ────────────────────────────────

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

// ─── Texas grid cells ────────────────────────────────────────────────────────

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

// ─── Merge extended features + notes into cell arrays ────────────────────────

function enrichCell(cell) {
  const ext = EXT[cell.grid_id] || {};
  return {
    ...cell,
    wind_direction_10m: ext.wind_direction_10m ?? 180,
    precipitation: ext.precipitation ?? 0,
    soil_moisture_0_to_7cm: ext.soil_moisture_0_to_7cm ?? 0.15,
    canopy_cover_pct: ext.canopy_cover_pct ?? 10,
    canopy_base_height_m: ext.canopy_base_height_m ?? 10,
    canopy_bulk_density: ext.canopy_bulk_density ?? 0.03,
    vegetation_type: ext.vegetation_type ?? 'Unknown',
    slope_degrees: ext.slope_degrees ?? 2,
    aspect_degrees: ext.aspect_degrees ?? 0,
    mean_frp: ext.mean_frp ?? 0,
    dominant_fuel_fraction: ext.dominant_fuel_fraction ?? 0.5,
    ndvi: null,
    cumulative_wind_run_24h: +(cell.wind_speed_10m * 24).toFixed(1),
    drought_index_proxy: +Math.max(0, Math.min(1, (cell.temperature_2m / 40) - (cell.relative_humidity_2m / 200) + (cell.vpd / 5))).toFixed(3),
    data_quality_flag: 0,
    notes: NOTES[cell.grid_id] || null,
  };
}

// Re-export enriched arrays (backward-compatible — all original fields preserved)
const _CA = CALIFORNIA_CELLS.map(enrichCell);
const _TX = TEXAS_CELLS.map(enrichCell);
export { _CA as CALIFORNIA_CELLS_ENRICHED, _TX as TEXAS_CELLS_ENRICHED };
