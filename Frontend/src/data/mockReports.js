// Mock AI-generated incident reports derived from model-pipeline/src/models/obj3_gemini/
// Report types, schema fields, and state machine modes match the actual codebase.
// Schema aligned with src/models/obj3_gemini/schemas/ (ICS-209 structure).

export const MOCK_REPORTS = [
  {
    report_id: 'rpt_20250115_quiet_ca',
    mode: 'QUIET',
    sub_state: null,
    region: 'california',
    generated_at: '2025-01-15T12:00:00Z',
    llm_backend: 'gemini_dev',
    llm_model: 'gemini-2.5-flash',
    confidence: 0.91,
    grounding_sources: 4,
    schema_type: 'IncidentBrief',
    title: 'Daily Incident Brief — California — 2025-01-15',
    content: {
      situation_summary: 'No active fire incidents confirmed in the California monitoring area. Risk conditions remain LOW across 19 of 20 monitored grid cells. The Santa Ynez Range cell (82287b) shows elevated VPD (4.98 kPa) and fire weather index (68.4), flagged as CRITICAL by ignition model. No FIRMS hotspots confirmed over the past 18 hours.',
      weather_outlook: 'Offshore flow pattern persists over Southern California. Temperature 22–28°C, relative humidity 12–22%, wind speeds 22–28 km/h from NW. VPD remains elevated. No precipitation forecast through 2025-01-20.',
      risk_summary: {
        critical_cells: 1,
        high_cells: 4,
        medium_cells: 6,
        low_cells: 9,
        highest_risk_cell: 'Santa Ynez Range (82287b)',
        highest_risk_score: 0.71,
      },
      key_features_driving_risk: ['vpd (4.98 kPa)', 'fire_weather_index (68.4)', 'wind_speed_10m (28.4 km/h)', 'fuel_model_fbfm40 = FBFM 4 (Chaparral)'],
      recommended_actions: [
        'Pre-position suppression resources in Santa Barbara County',
        'Issue Fire Weather Watch for Santa Ynez Range area',
        'Increase GOES-R scan frequency to 5-minute intervals over 82287b',
        'Alert CAL FIRE dispatch for proactive patrol',
      ],
      model_attribution: 'XGBoost ignition classifier (run 970bb676), AUC-PR=0.9051, threshold=0.4596',
    },
  },
  {
    report_id: 'rpt_20250115_quiet_tx',
    mode: 'QUIET',
    sub_state: null,
    region: 'texas',
    generated_at: '2025-01-15T12:00:00Z',
    llm_backend: 'gemini_dev',
    llm_model: 'gemini-2.5-flash',
    confidence: 0.88,
    grounding_sources: 4,
    schema_type: 'IncidentBrief',
    title: 'Daily Incident Brief — Texas — 2025-01-15',
    content: {
      situation_summary: 'One CRITICAL cell detected in the Big Bend / Trans-Pecos region (cell 824449). Four active FIRMS hotspots confirmed with FRP 24–76 MW. Model ignition probability 0.67. Conditions consistent with active range fire. Cell 8244a9 (Laredo / Eagle Pass) elevated to HIGH (score 0.48) due to offshore drought conditions.',
      weather_outlook: 'Strong ridge over western Texas. Temperature 24–29°C, relative humidity 11–20%, wind 22–28 km/h from WSW. Drought index proxy at 82.4 for Trans-Pecos region. No precipitation for 12+ days.',
      risk_summary: {
        critical_cells: 1,
        high_cells: 3,
        medium_cells: 5,
        low_cells: 5,
        highest_risk_cell: 'Big Bend / Trans-Pecos (824449)',
        highest_risk_score: 0.67,
      },
      key_features_driving_risk: ['drought_index_proxy (82.4)', 'vpd (5.12 kPa)', 'wind_speed_10m (28.4 km/h)', 'active_fire_count=4', 'fuel_model_fbfm40 = FBFM 2'],
      recommended_actions: [
        'Escalate Big Bend to ACTIVE monitoring mode',
        'Coordinate with Texas A&M Forest Service for suppression pre-positioning',
        'Activate HRRR rapid-refresh weather ingestion for Trans-Pecos',
        'Notify National Park Service — Big Bend National Park sector',
      ],
      model_attribution: 'XGBoost ignition classifier (run b7e52d18), AUC-PR=0.9124, threshold=0.4201',
    },
  },
  {
    report_id: 'rpt_20250110_active_ca',
    mode: 'ACTIVE',
    sub_state: null,
    region: 'california',
    generated_at: '2025-01-10T18:00:00Z',
    llm_backend: 'gemini_dev',
    llm_model: 'gemini-2.5-flash',
    confidence: 0.84,
    grounding_sources: 5,
    schema_type: 'TacticalOperations',
    title: 'Tactical Operations Brief — California — 2025-01-10',
    content: {
      situation_summary: 'Elevated risk conditions across Southern California. Model disagreement detected: cell 82281b (San Gabriel Mountains) showing ignition probability 0.52 with HIGH risk tier despite absence of FIRMS hotspot confirmation. VPD 4.12 kPa — above 90th percentile seasonal baseline. Wind event forecast 24–36 hours.',
      weather_outlook: 'Santa Ana wind event beginning 2025-01-11. NE winds 30–45 km/h, gusts to 70 km/h. RH dropping to 8–12%. Red Flag Warning criteria met for LA, Ventura, and San Bernardino counties.',
      risk_summary: {
        critical_cells: 0,
        high_cells: 7,
        medium_cells: 8,
        low_cells: 5,
        highest_risk_cell: 'San Gabriel Mountains (82281b)',
        highest_risk_score: 0.52,
      },
      key_features_driving_risk: ['vpd (4.12 kPa)', 'wind_speed_10m (24.6 km/h)', 'fire_weather_index (51.4)', 'fuel_model_fbfm40 = FBFM 9'],
      recommended_actions: [
        'Issue Red Flag Warning for LA and Ventura Counties',
        'Place OES Region I task forces on standby',
        'Activate aerial tanker base at Fox Field and Santa Barbara',
        'Increase pipeline cadence to 2-hour intervals (ACTIVE mode)',
        'Monitor VIIRS + GOES for hotspot confirmation',
      ],
      model_attribution: 'XGBoost ignition classifier (run 970bb676), disagreement flag=True (model HIGH, FIRMS 0)',
    },
  },
];

export const REPORT_MODE_COLORS = {
  QUIET: 'text-accent-green',
  ACTIVE: 'text-accent-orange',
  EMERGENCY: 'text-risk-critical',
};

export const REPORT_MODE_BG = {
  QUIET: 'bg-accent-green/10 border-accent-green/30',
  ACTIVE: 'bg-accent-orange/10 border-accent-orange/30',
  EMERGENCY: 'bg-risk-critical/10 border-risk-critical/30',
};
