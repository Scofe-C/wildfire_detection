// Mock data derived from Data-Pipeline/ repository structure.
// All stage names, field names, source names, and metric values match the actual codebase.

export const PIPELINE_META = {
  dag_id: 'wildfire_data_pipeline',
  schedule: '0 */6 * * *',
  last_run: '2025-01-15T18:00:00Z',
  next_run: '2025-01-16T00:00:00Z',
  operational_mode: 'QUIET',         // QUIET | ACTIVE | EMERGENCY
  grid_resolution_km: 64,
  regions: ['california', 'texas'],
  fire_season_active: false,         // Jun–Nov
};

export const INGESTION_STAGES = [
  {
    id: 'ingest_firms',
    label: 'NASA FIRMS',
    module: 'scripts/ingestion/ingest_firms.py',
    source: 'NASA FIRMS API',
    sensor: 'VIIRS + MODIS',
    status: 'success',
    last_run: '2025-01-15T18:02:11Z',
    records_fetched: 142,
    output_path: 'data/raw/firms/firms_raw_20250115_1800.csv',
    key_columns: ['latitude', 'longitude', 'bright_ti4', 'bright_ti5', 'frp', 'confidence', 'acq_date', 'acq_time', 'sensor_source'],
  },
  {
    id: 'ingest_weather',
    label: 'Open-Meteo',
    module: 'scripts/ingestion/ingest_weather.py',
    source: 'Open-Meteo API',
    sensor: null,
    status: 'success',
    last_run: '2025-01-15T18:02:44Z',
    records_fetched: 55,
    output_path: 'data/raw/weather/weather_raw_ca_tx_20250115_1800.csv',
    key_columns: ['grid_id', 'timestamp', 'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'wind_direction_10m', 'precipitation', 'vpd', 'fire_weather_index'],
  },
  {
    id: 'ingest_landfire',
    label: 'LANDFIRE',
    module: 'scripts/ingestion/ingest_landfire.py',
    source: 'USGS LANDFIRE',
    sensor: 'FBFM40 + Canopy',
    status: 'cached',
    last_run: '2025-01-01T00:00:00Z',
    records_fetched: null,
    output_path: 'data/cache/landfire/',
    key_columns: ['fuel_model_fbfm40', 'canopy_cover_pct', 'dominant_fuel_fraction', 'vegetation_type'],
  },
  {
    id: 'ingest_srtm',
    label: 'USGS SRTM',
    module: 'scripts/ingestion/ingest_srtm.py',
    source: 'USGS SRTM (3DEP)',
    sensor: 'DEM 30m',
    status: 'cached',
    last_run: '2025-01-01T00:00:00Z',
    records_fetched: null,
    output_path: 'data/cache/srtm/',
    key_columns: ['elevation_m', 'slope_degrees', 'aspect_degrees'],
  },
  {
    id: 'ingest_goes',
    label: 'GOES-R ABI',
    module: 'scripts/ingestion/ingest_goes.py',
    source: 'NOAA GOES-R (S3)',
    sensor: 'ABI Band 7/14',
    status: 'stub',
    last_run: null,
    records_fetched: 0,
    output_path: 'data/raw/goes/',
    key_columns: [],
  },
];

export const PROCESSING_STAGES = [
  {
    id: 'process_firms',
    label: 'FIRMS Processing',
    module: 'scripts/processing/process_firms.py',
    status: 'success',
    last_run: '2025-01-15T18:03:20Z',
    input_rows: 142,
    output_rows: 8,
    output_path: 'data/processed/firms/firms_features_ca_tx_latest.parquet',
    ops: ['snap_to_h3_grid', 'clip_frp_p99.5', 'groupby_grid_id', 'agg_mean_median_max'],
    output_columns: ['grid_id', 'active_fire_count', 'mean_frp', 'median_frp', 'max_confidence', 'fire_detected_binary', 'nearest_fire_distance_km'],
  },
  {
    id: 'process_weather',
    label: 'Weather Processing',
    module: 'scripts/processing/process_weather.py',
    status: 'success',
    last_run: '2025-01-15T18:03:35Z',
    input_rows: 55,
    output_rows: 55,
    output_path: 'data/processed/weather/weather_features_ca_tx_latest.parquet',
    ops: ['6hr_temporal_agg', 'derive_vpd', 'derive_drought_proxy', 'derive_cumulative_wind_run'],
    output_columns: ['grid_id', 'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'wind_direction_10m', 'precipitation', 'soil_moisture_0_to_7cm', 'vpd', 'fire_weather_index', 'days_since_last_precipitation', 'cumulative_wind_run_24h', 'drought_index_proxy'],
  },
  {
    id: 'process_static',
    label: 'Static Features',
    module: 'scripts/processing/process_static.py',
    status: 'success',
    last_run: '2025-01-15T18:03:40Z',
    input_rows: null,
    output_rows: 55,
    output_path: 'data/processed/static/',
    ops: ['agg_mode_fuel', 'agg_mean_terrain', 'circular_mean_aspect'],
    output_columns: ['fuel_model_fbfm40', 'canopy_cover_pct', 'vegetation_type', 'elevation_m', 'slope_degrees', 'aspect_degrees', 'dominant_fuel_fraction'],
  },
];

export const FUSION_STAGE = {
  id: 'fuse_features',
  label: 'Feature Fusion',
  module: 'scripts/fusion/fuse_features.py',
  function: 'fuse_features(firms_df, weather_df, static_df, previous_fused_path)',
  status: 'success',
  last_run: '2025-01-15T18:04:05Z',
  grid_cells_total: 55,
  grid_cells_ca: 35,
  grid_cells_tx: 20,
  fire_cells: 8,
  output_columns_count: 32,
  output_path_latest: 'data/processed/fused/fused_features_latest.parquet',
  output_path_ml: 'data/processed/fused/fused_features_ml_latest.parquet',
  output_path_partitioned: 'data/processed/fused/64km/region={region}/year={year}/month={month}/features_{date}.parquet',
  join_strategy: 'LEFT JOIN FIRMS → LEFT JOIN Weather → LEFT JOIN Static',
  temporal_lag_applied: true,
  fill_strategies: {
    forward_fill: ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'vpd', 'fire_weather_index', 'ndvi'],
    zero_fill: ['precipitation'],
    default_zero: ['active_fire_count', 'mean_frp', 'fire_detected_binary'],
  },
};

export const VALIDATION_STAGE = {
  id: 'validate_schema',
  label: 'Schema Validation',
  module: 'scripts/validation/validate_schema.py',
  function: 'run_validation(df, registry, resolution_km, enforce_row_count=True)',
  status: 'success',
  last_run: '2025-01-15T18:04:18Z',
  checks_total: 6,
  checks_passed: 6,
  checks_failed: 0,
  details: [
    { check: 'column_existence', status: 'pass', detail: 'All 28 features present' },
    { check: 'non_nullable_constraints', status: 'pass', detail: '9 required columns non-null' },
    { check: 'null_rates_le_15pct', status: 'pass', detail: 'Max null rate: 8.2% (ndvi)' },
    { check: 'min_max_bounds', status: 'pass', detail: 'All values within defined bounds' },
    { check: 'allowed_values', status: 'pass', detail: 'resolution_km=64 ✓, region ∈ {ca,tx} ✓' },
    { check: 'row_count_tolerance', status: 'pass', detail: '55 rows ± 5% of 55 expected ✓' },
  ],
  anomaly_detection: {
    module: 'scripts/validation/detect_anomalies.py',
    method: 'Seasonal z-score (Welford online)',
    threshold_fire_season: 4.0,
    threshold_off_season: 3.5,
    monitored_features: ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'precipitation', 'active_fire_count', 'mean_frp'],
    anomalies_detected: 0,
    slack_alert_sent: false,
  },
};

export const EXPORT_STAGE = {
  id: 'export_spatial',
  label: 'Export & Version',
  module: 'scripts/export/export_spatial.py',
  status: 'success',
  last_run: '2025-01-15T18:04:32Z',
  outputs: [
    { type: 'Parquet', path: 'data/processed/64km/region=california/year=2025/month=01/features_20250115.parquet', size_kb: 42 },
    { type: 'Parquet', path: 'data/processed/64km/region=texas/year=2025/month=01/features_20250115.parquet', size_kb: 28 },
    { type: 'NPZ (spatial array)', path: 'data/processed/spatial/h3_grid_20250115.npz', size_kb: 18 },
    { type: 'NPZ (adjacency)', path: 'data/processed/spatial/h3_adjacency_20250115.npz', size_kb: 9 },
  ],
  dvc_tracked: true,
  gcs_bucket: 'gs://wildfire-mlops-data/',
};

export const DATA_QUALITY_FLAGS = [
  { flag: 0, label: 'All sources fresh', count: 0 },
  { flag: 1, label: 'Open-Meteo primary', count: 47 },
  { flag: 2, label: 'NWS fallback', count: 5 },
  { flag: 3, label: 'HRRR rapid (emergency)', count: 0 },
  { flag: 4, label: 'Interpolated / forward-filled', count: 3 },
  { flag: 5, label: 'Partial data (excluded)', count: 0 },
];

// Historical pipeline run timeline (last 12 runs)
export const PIPELINE_HISTORY = [
  { run: '2025-01-15 18:00', status: 'success', duration_s: 142 },
  { run: '2025-01-15 12:00', status: 'success', duration_s: 138 },
  { run: '2025-01-15 06:00', status: 'success', duration_s: 145 },
  { run: '2025-01-15 00:00', status: 'success', duration_s: 151 },
  { run: '2025-01-14 18:00', status: 'success', duration_s: 139 },
  { run: '2025-01-14 12:00', status: 'warning', duration_s: 210 },
  { run: '2025-01-14 06:00', status: 'success', duration_s: 144 },
  { run: '2025-01-14 00:00', status: 'success', duration_s: 137 },
  { run: '2025-01-13 18:00', status: 'success', duration_s: 141 },
  { run: '2025-01-13 12:00', status: 'success', duration_s: 148 },
  { run: '2025-01-13 06:00', status: 'failed', duration_s: 22 },
  { run: '2025-01-13 00:00', status: 'success', duration_s: 140 },
];
