// Mock data derived from model-pipeline/ repository structure.
// All model names, metric values, threshold values, feature names, and config keys
// match the actual codebase (configs/model_config.yaml, src/models/, src/validation/).

// ─── OBJ-1: XGBoost / LightGBM Ignition Classifier ───────────────────────────

export const OBJ1_RUNS = [
  {
    run_id: '970bb676',
    model: 'xgboost_ignition',
    region: 'california',
    train_cutoff: '2024-12-31',
    test_period: 'Jan 2025',
    status: 'production',
    metrics: {
      auc_pr: 0.9051,
      auc_roc: 0.9412,
      f1: 0.7834,
      fnr: 0.097,
      accuracy: 0.9823,
      positive_rate: 0.0152,
      threshold_tuned: 0.4596,
      threshold_default: 0.365,
      confusion_matrix: { tn: 8821, fp: 94, fn: 34, tp: 315 },
    },
    best_params: {
      subsample: 0.6,
      n_estimators: 400,
      min_child_weight: 1,
      max_depth: 4,
      learning_rate: 0.01,
      gamma: 0.1,
      colsample_bytree: 0.6,
    },
    gates: {
      auc_pr_gate: { threshold: 0.89, value: 0.9051, passed: true },
      fnr_disparity_gate: { threshold: 0.15, value: 0.09, passed: true },
      recall_gate: { threshold: 0.90, value: 0.903, passed: true },
    },
    mlflow_experiment: 'wildfire-ignition-v1',
    vertex_model: 'wildfire-ignition-california',
    vertex_stage: 'production',
    trained_at: '2025-01-10T09:41:22Z',
  },
  {
    run_id: 'a3f1c291',
    model: 'lightgbm_ignition',
    region: 'california',
    train_cutoff: '2024-12-31',
    test_period: 'Jan 2025',
    status: 'staging',
    metrics: {
      auc_pr: 0.8961,
      auc_roc: 0.9388,
      f1: 0.7612,
      fnr: 0.108,
      accuracy: 0.9801,
      positive_rate: 0.0152,
      threshold_tuned: 0.239,
      threshold_default: 0.239,
      confusion_matrix: { tn: 8814, fp: 101, fn: 38, tp: 311 },
    },
    best_params: {
      n_estimators: 300,
      max_depth: 5,
      learning_rate: 0.05,
      num_leaves: 31,
      subsample: 0.8,
      colsample_bytree: 0.8,
    },
    gates: {
      auc_pr_gate: { threshold: 0.89, value: 0.8961, passed: false },
      fnr_disparity_gate: { threshold: 0.15, value: 0.12, passed: true },
      recall_gate: { threshold: 0.90, value: 0.891, passed: false },
    },
    mlflow_experiment: 'wildfire-ignition-v1',
    vertex_model: 'wildfire-ignition-california',
    vertex_stage: 'staging',
    trained_at: '2025-01-10T11:03:45Z',
  },
  {
    run_id: 'b7e52d18',
    model: 'xgboost_ignition',
    region: 'texas',
    train_cutoff: '2024-12-31',
    test_period: 'Jan 2025',
    status: 'production',
    metrics: {
      auc_pr: 0.9124,
      auc_roc: 0.9453,
      f1: 0.7991,
      fnr: 0.088,
      accuracy: 0.9847,
      positive_rate: 0.0178,
      threshold_tuned: 0.4201,
      threshold_default: 0.365,
      confusion_matrix: { tn: 5621, fp: 62, fn: 19, tp: 198 },
    },
    best_params: {
      subsample: 0.7,
      n_estimators: 400,
      min_child_weight: 3,
      max_depth: 5,
      learning_rate: 0.05,
      gamma: 0.2,
      colsample_bytree: 0.7,
    },
    gates: {
      auc_pr_gate: { threshold: 0.89, value: 0.9124, passed: true },
      fnr_disparity_gate: { threshold: 0.15, value: 0.07, passed: true },
      recall_gate: { threshold: 0.90, value: 0.912, passed: true },
    },
    mlflow_experiment: 'wildfire-ignition-v1',
    vertex_model: 'wildfire-ignition-texas',
    vertex_stage: 'production',
    trained_at: '2025-01-10T10:18:33Z',
  },
];

// SHAP feature importance (mean |SHAP|) for production XGBoost CA run
export const SHAP_IMPORTANCE = [
  { feature: 'vpd', importance: 0.2341, rank: 1 },
  { feature: 'fire_weather_index', importance: 0.1987, rank: 2 },
  { feature: 'temperature_2m', importance: 0.1542, rank: 3 },
  { feature: 'drought_index_proxy', importance: 0.1234, rank: 4 },
  { feature: 'fuel_model_fbfm40', importance: 0.0978, rank: 5 },
  { feature: 'relative_humidity_2m', importance: 0.0876, rank: 6 },
  { feature: 'wind_speed_10m', importance: 0.0812, rank: 7 },
  { feature: 'slope_degrees', importance: 0.0621, rank: 8 },
  { feature: 'cumulative_wind_run_24h', importance: 0.0543, rank: 9 },
  { feature: 'elevation_m', importance: 0.0489, rank: 10 },
  { feature: 'dominant_fuel_fraction', importance: 0.0412, rank: 11 },
  { feature: 'soil_moisture_0_to_7cm', importance: 0.0387, rank: 12 },
];

// Bias analysis across slices (FNR disparity gate)
export const BIAS_ANALYSIS = {
  metric: 'false_negative_rate',
  max_disparity_threshold: 0.15,
  min_group_size: 20,
  min_fire_count: 5,
  slices: [
    { group: 'california', fnr: 0.097, n: 349, fire_count: 349, pass: true },
    { group: 'texas', fnr: 0.088, n: 217, fire_count: 217, pass: true },
    { group: 'fire_season (Jun-Nov)', fnr: 0.091, n: 412, fire_count: 412, pass: true },
    { group: 'off_season (Dec-May)', fnr: 0.109, n: 154, fire_count: 154, pass: true },
    { group: 'fuel_fbfm_1_2 (grass)', fnr: 0.101, n: 187, fire_count: 187, pass: true },
    { group: 'fuel_fbfm_9_10 (timber)', fnr: 0.094, n: 203, fire_count: 203, pass: true },
    { group: 'fuel_fbfm_40+ (chaparral)', fnr: 0.112, n: 176, fire_count: 176, pass: true },
  ],
  max_observed_disparity: 0.021,
  gate_passed: true,
};

// PR curve data points for visualization
export const PR_CURVE_CA = [
  { recall: 1.00, precision: 0.152 },
  { recall: 0.98, precision: 0.421 },
  { recall: 0.96, precision: 0.558 },
  { recall: 0.94, precision: 0.641 },
  { recall: 0.92, precision: 0.703 },
  { recall: 0.903, precision: 0.748 },  // tuned threshold
  { recall: 0.88, precision: 0.791 },
  { recall: 0.85, precision: 0.834 },
  { recall: 0.80, precision: 0.872 },
  { recall: 0.75, precision: 0.903 },
  { recall: 0.70, precision: 0.921 },
  { recall: 0.60, precision: 0.946 },
  { recall: 0.50, precision: 0.961 },
  { recall: 0.40, precision: 0.972 },
  { recall: 0.30, precision: 0.981 },
  { recall: 0.20, precision: 0.989 },
  { recall: 0.10, precision: 0.995 },
];

// ─── OBJ-2: Fire Spread Simulator ────────────────────────────────────────────

export const OBJ2_SIMULATIONS = [
  {
    sim_id: 'sim_20250115_ca_003',
    ignition_grid_id: '8228e9ffffffffff',
    region: 'california',
    timestamp: '2025-01-15T14:00:00Z',
    ignition_probability: 0.72,
    status: 'completed',
    n_simulations: 100,
    fire_period_length_hr: 1.0,
    inputs: {
      fuel_model_fbfm40: 'FBFM 9',
      elevation_m: 420,
      slope_degrees: 18.4,
      canopy_base_height_m: 3.2,
      canopy_bulk_density_kgm3: 0.12,
      temperature_c: 28.4,
      relative_humidity_pct: 18.2,
      wind_speed_kmh: 34.1,
      wind_direction_deg: 245,
    },
    outputs: {
      spread_direction_deg: 68,
      spread_speed_kmh: 4.2,
      dead_fuel_moisture_pct: 5.1,
      crown_fire_status: 'passive_crown',
      burn_probability_mean: 0.41,
    },
    validation: {
      buffered_iou: 0.42,
      dice_coefficient: 0.58,
      buffered_iou_threshold: 0.35,
      dice_threshold: 0.50,
      reference: 'CAL FIRE FRAP historical perimeters',
      passed: true,
    },
    physics_model: 'Rothermel (1972) + Scott & Burgan FBFM40',
  },
];

// ─── OBJ-3: Gemini Reporter State Machine ─────────────────────────────────────

export const OBJ3_STATE = {
  operational_mode: 'QUIET',         // QUIET | ACTIVE | EMERGENCY
  emergency_sub_state: null,         // ACTIVE_FIRE | INTERIM | POST_FIRE | FINAL
  risk_level: 'LOW',                 // LOW | MODERATE | HIGH | CRITICAL
  firms_hotspot_count: 0,
  is_deployable: true,
  mode_disagreement: false,
  llm_backend: 'gemini_dev',         // ollama | gemini_dev | vertex_ai
  llm_model: 'gemini-2.5-flash',
  corpus_chars_loaded: 48320,
  corpus_max_chars: 500000,
  reports_generated_today: 2,
  last_report_at: '2025-01-15T12:00:00Z',
  report_confidence: 0.88,
  confidence_threshold: 0.70,
  min_grounding_sources: 3,
};

// Mode decision matrix for display
export const MODE_MATRIX = [
  { risk: 'LOW/MODERATE',  hotspots: '0',   deployable: true,  mode: 'QUIET',     disagreement: false },
  { risk: 'HIGH/CRITICAL', hotspots: '0',   deployable: true,  mode: 'ACTIVE',    disagreement: false },
  { risk: 'LOW/MODERATE',  hotspots: '>0',  deployable: true,  mode: 'ACTIVE',    disagreement: true  },
  { risk: 'HIGH/CRITICAL', hotspots: '>0',  deployable: true,  mode: 'EMERGENCY', disagreement: false },
  { risk: 'ANY',           hotspots: 'ANY', deployable: false, mode: 'QUIET',     disagreement: false },
];

// Watchdog configuration (from schema_config.yaml)
export const WATCHDOG_CONFIG = {
  current_mode: 'quiet',
  modes: {
    quiet:    { poll_interval_min: 30, pipeline_interval_hr: 6,   resolution_km: 64 },
    active:   { poll_interval_min: 15, pipeline_interval_hr: 2,   resolution_km: 64 },
    emergency:{ poll_interval_min: 5,  pipeline_interval_hr: 0.5, resolution_km: 22 },
  },
  false_alarm_gates: {
    min_neighbor_detections: 2,
    min_consecutive_scans: 2,
    viirs_lookback_hours: 3,
    viirs_bypass_frp_mw: 50.0,
    industrial_exclusion_radius_km: 2,
    revert_after_minutes: 30,
  },
  emergency_trigger: {
    min_frp_mw: 200.0,
    min_expanding_scans: 2,
    deactivate_frp_mw: 50.0,
    deactivate_low_frp_scans: 2,
    deactivate_no_expansion_scans: 3,
  },
};
