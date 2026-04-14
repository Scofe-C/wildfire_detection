// Backend API base URL.
// In Docker: nginx proxies /api → obj3-dashboard:8000 (use relative paths).
// In dev: point directly to localhost:8000 (CORS enabled server-side).
export const API_BASE = import.meta.env.VITE_API_BASE || '';

export function apiUrl(path) {
  return `${API_BASE}${path}`;
}

/** Normalize a grid cell from the API to the shape the frontend expects.
 *  Handles: latitude→lat, longitude→lon, missing name, null values. */
export function normalizeCell(c) {
  return {
    ...c,
    lat: c.lat ?? c.latitude ?? 0,
    lon: c.lon ?? c.longitude ?? 0,
    name: c.name || c.grid_id?.slice(0, 12) || '?',
    fire_risk_score: c.fire_risk_score ?? 0,
    temperature_2m: c.temperature_2m ?? null,
    relative_humidity_2m: c.relative_humidity_2m ?? null,
    wind_speed_10m: c.wind_speed_10m ?? null,
    vpd: c.vpd ?? null,
    fire_weather_index: c.fire_weather_index ?? null,
    active_fire_count: c.active_fire_count ?? 0,
    fire_detected_binary: c.fire_detected_binary ?? 0,
    elevation_m: c.elevation_m ?? null,
    fuel_model_fbfm40: c.fuel_model_fbfm40 ?? '—',
    mean_frp: c.mean_frp ?? 0,
  };
}

/** Format a number to n decimal places, return '—' for null/undefined. */
export function fmt(v, decimals = 2) {
  if (v == null || isNaN(v)) return '—';
  return Number(v).toFixed(decimals);
}
