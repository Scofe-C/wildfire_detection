# Model Pipeline — Wildfire Prediction & Disaster Response

## 1. Overview

This directory contains the ML model pipeline infrastructure for the Wildfire Prediction & Disaster Response platform. It handles everything after the data pipeline produces features and before a model reaches production: validation, bias detection, experiment tracking, visualization, alerting, and CI/CD.

**OBJ-1 (XGBoost)** is a placeholder — the infrastructure is fully implemented and tested, and once a model is plugged in, validation, bias gating, tracking, and deployment happen automatically. **OBJ-2 (Cell2Fire)** is fully implemented as a C++ subprocess wrapper with weather CSV formatting, raster clipping, burn probability parsing, and Dice coefficient validation. **OBJ-3 (Gemini Disaster Reporting)** is fully implemented with 3 swappable LLM backends, 4 structured report types, and Jinja2 rendering.

```
data-pipeline/                          model-pipeline/ (this directory)
──────────────                          ────────────────
data/processed/backfill/*.parquet  ───> src/data/loader.py (load + validate)
                                              │
                                              ▼
                                        src/models/ (predict)
                                              │
                                              ▼
                                        src/validation/ (AUC-PR > 0.75?)
                                              │
                                              ▼
                                        src/bias/ (FNR disparity < 5%?)
                                              │
                                       PASS ──┴── FAIL
                                        │          │
                                        ▼          ▼
                                  Push to       Block deploy,
                                  registry      alert + RCA
```

---

## 2. Project Structure

```
model-pipeline/
├── configs/
│   ├── feature_schema.yaml         # what columns the model expects from data pipeline
│   └── model_config.yaml           # thresholds, paths, tracking, alert config
├── src/
│   ├── data/                       # load + validate parquet from data pipeline
│   ├── models/                     # abstract interface + OBJ-1 stub + OBJ-2 (Cell2Fire) + OBJ-3 (Gemini)
│   ├── validation/                 # metrics, model selection gate, visualizations
│   ├── bias/                       # Fairlearn FNR gate + FEMA NRI spatial join
│   ├── tracking/                   # MLflow local + Vertex AI Experiments
│   ├── notifications/              # Slack webhook alerts
│   └── pipeline/                   # end-to-end orchestrator
├── tests/                          # pytest suite
├── data/static/fema_nri/           # FEMA NRI shapefile (downloaded, not committed)
├── models/ignition/                # trained model artifacts (DVC-tracked)
├── reports/                        # validation reports, bias reports, plots
├── .github/workflows/model_ci.yml  # 9-stage CI/CD
├── dvc.yaml                        # DVC stages for validate + bias gate
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

---

## 3. Setup

```bash
cd model-pipeline
pip install -r requirements.txt

# verify everything works
PYTHONPATH=. pytest tests/ -v --cov=src
```

---

## 4. Module Reference

### 4.1 `configs/feature_schema.yaml` — Data Contract

Defines every column the model pipeline expects from `data-pipeline/data/processed/backfill/*.parquet`.

**What it does:**
- Lists index columns (`h3_index`, `timestamp`), the target (`fire_detected`), and all features with their dtype, source, valid bounds, and whether they are required or optional.
- Acts as the single source of truth between the two pipelines.

**Expected outcome:**
- When the data loader reads parquet files, it validates them against this schema.
- Required features missing → error. Optional features missing → logged as info, no error.

**What to do next:**
- When teammates add SMAP, 3DEP, or LANDFIRE to the data pipeline, change `required: false` to `required: true` for those features here.

---

### 4.2 `configs/model_config.yaml` — Pipeline Configuration

All thresholds, file paths, GCS settings, tracking config, and alert config in one place.

**Key values:**
- `paths.backfill_dir: ../data-pipeline/data/processed/backfill` — where to read input data
- `validation.auc_pr_threshold: 0.75` — model must beat this to pass Stage 5
- `bias_gate.max_disparity: 0.05` — max FNR difference across vulnerability groups (Stage 6)
- `shap.min_soil_moisture_importance: 0.10` — drift alert if SHAP drops below 10%

---

### 4.3 `src/data/schema.py` — Schema Loader & Validator

**Functions:**

| Function | Input | Output | Purpose |
|---|---|---|---|
| `load_schema(config_path)` | path to `feature_schema.yaml` (or None for default) | `FeatureSchema` object | Parses the YAML into an immutable schema with properties like `required_features`, `target_name`, `index_column_names` |
| `validate_dataframe(df, schema)` | a pandas DataFrame + schema | `list[str]` of errors (empty = valid) | Checks: all required columns present, all values within declared bounds. Does NOT raise — caller decides to fail or warn |

**Expected outcome:**
- `validate_dataframe` returns `[]` for valid data.
- Returns strings like `"Missing required feature: frp_max (source: firms)"` or `"'temperature_max' out of bounds: expected [200.0, 350.0], got [150.0, 400.0]"` for invalid data.

---

### 4.4 `src/data/loader.py` — Parquet Loader

**Functions:**

| Function | Input | Output | Purpose |
|---|---|---|---|
| `load_backfill(backfill_dir, schema, strict)` | path to parquet directory | validated `DataFrame` | Reads all `.parquet` files, concatenates, validates against schema. `strict=True` raises on errors, `strict=False` warns only |
| `split_features_target(df, schema)` | validated DataFrame | `(X, y, metadata)` tuple | Separates feature columns, target column, and index columns into three DataFrames |
| `load_and_split(backfill_dir, schema, strict)` | path | `(X, y, metadata)` | Convenience — calls `load_backfill` then `split_features_target` |

**Expected outcome:**
- `X`: DataFrame with only feature columns (7 required + any optional present)
- `y`: Series of `fire_detected` labels (0/1)
- `metadata`: DataFrame with `h3_index` and `timestamp` (needed for spatial join in bias gate)

**Raises `DataLoadError` when:**
- Backfill directory doesn't exist
- No `.parquet` files found
- Required features missing (in strict mode)

---

### 4.5 `src/models/base.py` — Abstract Model Interface

Defines the contract that OBJ-1, OBJ-2, and OBJ-3 must implement.

**Methods every model must have:**

| Method | Input | Output | Purpose |
|---|---|---|---|
| `load_model(model_path)` | path to saved weights | None | Load model from disk |
| `predict(X)` | feature DataFrame | DataFrame with `prediction` (0/1) and `probability` [0,1] columns | Run inference |
| `validate(X, y)` | features + labels | `dict` with `auc_pr`, `f1`, `fnr`, etc. | Compute metrics |
| `explain(X)` | features | `dict` with SHAP values, feature importance | Explainability |
| `compute_artifact_hash(model_path)` | path | SHA-256 hex string | Reproducibility tracking |

**What to do next:**
- Teammates: subclass `BaseModel` in `obj1_xgboost/`, implement all 4 methods.
- ~~Owner: subclass in `obj3_gemini/` for Gemini disaster reporting.~~ ✅ **Done** — see section 4.9.
- ~~Teammates: subclass `BaseModel` in `obj2_spread/`.~~ ✅ **Done** — see section 4.8.

---

### 4.6 `src/models/registry.py` — Model Registry

**Methods:**

| Method | Purpose |
|---|---|
| `save_local(model_artifact_path, version, metadata)` | Copy model + metadata JSON to `models/ignition/{version}/` |
| `push_to_gcs(version)` | Upload versioned model to GCS via `gsutil` |
| `tag_previous(current_version)` | Write `PREVIOUS_VERSION` marker for rollback |
| `get_previous_version()` | Read the marker to know which version to roll back to |
| `list_versions()` | List all locally saved versions that have `metadata.json` |

**Expected outcome:**
- After a successful pipeline run: `models/ignition/1.0.0/metadata.json` exists with `run_id`, `auc_pr`, `bias_gate`, and `saved_at` fields.

---

### 4.7 `src/models/obj1_xgboost/placeholder.py` — OBJ-1 Stub

**Status:** Not implemented. Raises `NotImplementedError` on all methods.

**What teammates need to do:**
1. Load pre-trained ECMWF PoF XGBoost weights in `load_model()`.
2. Return a DataFrame with `prediction` and `probability` columns from `predict()`.
3. Compute AUC-PR, F1, FNR in `validate()` (can use `src.validation.metrics`).
4. Run SHAP TreeExplainer in `explain()`.

---

### 4.8 `src/models/obj2_spread/cell2fire_spread.py` — OBJ-2 Cell2Fire Fire Spread

**Status:** Fully implemented. Physics-based C++ simulator wrapped as a `BaseModel`. Runs Monte Carlo fire spread simulations from DEM + fuel + weather inputs, outputs burn probability grids.

**Helper functions:**

| Function | Purpose |
|---|---|
| `format_weather_csv(weather_df, output_path)` | Maps pipeline weather columns (e.g. `wind_speed_10m`, `temperature_2m`) → Cell2Fire CSV format (`ws`, `wd`, `tmp`, `rh`) |
| `clip_raster_to_aoi(raster_path, bounds, output_path, target_resolution)` | Clips DEM/fuel GeoTIFFs to a bounding box with optional resampling |
| `parse_burn_probability(output_dir, n_simulations)` | Reads Cell2Fire `ForestGrid*.csv` output grids → 2D burn probability array |
| `burn_grid_to_geodataframe(burn_prob, transform, crs, threshold)` | Converts burn probability grid → `GeoDataFrame` of burned polygons |
| `compute_dice_coefficient(predicted_mask, actual_mask)` | Dice = 2·\|P∩A\| / (\|P\|+\|A\|), range [0, 1]. Primary validation metric |

**`Cell2FireSpread` (implements `BaseModel`):**

| Method | What it does |
|---|---|
| `load_model(path)` | Loads simulation config JSON (ignition points, AOI bounds, raster paths) + base params from `model_config.yaml`. Warns if C++ binary not on PATH |
| `predict(X)` | Prepares weather CSV + clips rasters → runs C++ binary via subprocess → parses burn grids → maps burn probability back to H3 cells in `X` |
| `validate(X, y)` | Dice coefficient vs actual burn labels (CAL FIRE perimeters intersected with H3 grid) + AUC-PR/F1/FNR for pipeline compatibility |
| `explain(X)` | Parameter sensitivity sweep over `sweep_space` in `model_config.yaml`; returns most-influential parameter by burn-area range |

**Inputs required:**
- DEM: GeoTIFF float32 (elevation, metres) — USGS 3DEP
- Fuel: GeoTIFF int16 (LANDFIRE FBFM40 codes)
- Weather: pipeline-format DataFrame (columns auto-mapped)
- Ignition: `ignition_points` list of `(row, col)` tuples in simulation config JSON

**Validation gate:** Dice coefficient >= `obj2.cell2fire.validation.minimum_dice` (default 0.50)

**Setup:**
1. Install Cell2Fire C++ binary (`Cell2Fire` on PATH, or set `obj2.cell2fire.binary_path` in `model_config.yaml`)
2. Place DEM and fuel GeoTIFFs at paths configured under `obj2.cell2fire.raster_inputs`
3. Create a simulation config JSON:
```json
{
    "ignition_points": [[120, 85], [121, 86]],
    "aoi_bounds": [-121.5, 38.5, -120.5, 39.5],
    "params": {"n_simulations": 200}
}
```
4. Call `model.load_model("path/to/simulation_config.json")`

---

### 4.9 `src/models/obj3_gemini/` — OBJ-3 Gemini Disaster Reporting Engine

**Status:** Fully implemented. Multi-phase architecture with 3 swappable LLM backends, 4 report types, structured output via Pydantic, and Jinja2 rendering.

```
src/models/obj3_gemini/
├── __init__.py                    # re-exports GeminiDisasterReporter, GeneratedReport, etc.
├── reporter.py                    # main orchestrator (subclasses BaseModel)
├── state_machine.py               # mode resolution: QUIET / ACTIVE / EMERGENCY
├── context_builder.py             # multi-source context assembly → ContextBundle
├── corpus_loader.py               # RAG corpus loading + Vertex AI context cache
├── renderer.py                    # Jinja2 → Markdown / HTML / PDF rendering
├── adapters/
│   ├── base_adapter.py            # abstract LLMAdapter interface
│   ├── ollama_adapter.py          # Phase 1 — local Ollama server
│   ├── gemini_dev_adapter.py      # Phase 2 — Gemini Developer API (free-tier)
│   └── vertex_adapter.py          # Phase 3 — Vertex AI + context caching
└── schemas/
    ├── base_schema.py             # BaseReport + nested types (RiskCell, Recommendation, …)
    ├── daily_schema.py            # DailyReport   (QUIET mode)
    ├── high_risk_schema.py        # HighRiskReport (ACTIVE mode)
    ├── incident_schema.py         # IncidentReport (EMERGENCY — active fire)
    └── final_schema.py            # FinalReport    (EMERGENCY — post-incident close-out)
```

---

#### 4.9.1 `state_machine.py` — Mode Resolution

Pure-logic module: no I/O, no LLM calls.

| Function / Class | Input | Output | Purpose |
|---|---|---|---|
| `resolve_mode(pipeline_result)` | dict with `risk_level` + `firms_hotspot_count` | `(OperationalMode, EmergencySubState \| None)` | Maps risk level / hotspot count to QUIET, ACTIVE, or EMERGENCY mode |
| `mode_to_report_type(mode, sub_state)` | mode + optional sub-state | `str` — one of `"daily"`, `"high_risk"`, `"incident"`, `"final"` | Maps mode to the report type string |
| `AdminToggle` | config dict | `.is_on` / `.enable()` / `.disable()` | Controls whether the human input channel is active. Persists to YAML (Phase 1) or Firestore (Phase 3 stub) |

**Mode → report mapping:**
| Risk Level | FIRMS Hotspots | Mode | Report Type |
|---|---|---|---|
| LOW | 0 | QUIET | `daily` |
| MODERATE / HIGH | 0 | ACTIVE | `high_risk` |
| CRITICAL _or_ any | > 0 | EMERGENCY | `incident` (or `final` if sub-state is FINAL) |

---

#### 4.9.2 `context_builder.py` — Context Assembly

Assembles all input sources into a single `ContextBundle` before any LLM call. No LLM calls happen here.

| Function | Purpose |
|---|---|
| `build_system_prompt(report_type, schema)` | Constructs role definition + JSON schema + hallucination rules + disclaimer injection |
| `build_ml_block(pipeline_result, max_chars)` | Serialises XGBoost top cells, Cell2Fire GeoJSON, Propagator summary, bias gate |
| `build_data_block(pipeline_result, max_chars)` | Serialises environmental telemetry (OWM/SMAP), FIRMS hotspots, FEMA NRI tracts |
| `build_human_block(human_inputs, toggle)` | Formats operator/management text notes + uploaded files. Returns `""` if toggle is OFF |
| `build_instruction(report_type, incident_id, dt)` | Final directive: "Generate a {type} report …" |
| `assemble(...)` | Orchestrates all builders → `ContextBundle` |

**Key data classes:**

| Class | Description |
|---|---|
| `ContextBundle` | Complete payload sent to the adapter: `system_prompt`, `corpus_ref`, `corpus_text`, `ml_block`, `data_block`, `human_block`, `instruction`, `report_type`, `incident_id` |
| `HumanInput` | Operator/management input with text notes, uploaded files, and source (operator / management) |
| `UploadedFile` | Filename, raw bytes, MIME type |

---

#### 4.9.3 `corpus_loader.py` — RAG Corpus Loading

Loads reference documents (`.pdf`, `.txt`) from the versioned `corpus/{version}/` directory.

| Function | Purpose |
|---|---|
| `load_corpus_texts(corpus_dir, version)` | Reads all PDF/TXT files from `corpus/{version}/` → `list[CorpusDocument]` |
| `get_corpus_as_text(corpus_docs, max_chars)` | Concatenates docs as plain text (Phase 1/2 fallback). PDFs included as filename reference only |
| `estimate_corpus_tokens(corpus_docs)` | Rough token estimate: total bytes ÷ 4 |
| `get_or_create_cache(client, model, …)` | Phase 3: creates or reuses a Vertex AI context cache (`wildfire-rag-corpus-v{version}`) |

**Raises `CorpusLoadError`** if the directory is missing or contains no documents.
**Raises `CacheCreationError`** if Vertex AI cache creation fails or corpus is < 2048 tokens.

---

#### 4.9.4 `adapters/` — LLM Backend Implementations

All adapters implement the `LLMAdapter` interface:

| Method | Signature | Purpose |
|---|---|---|
| `generate(context_bundle, schema)` | `ContextBundle, dict → str` | Send context to LLM, return raw JSON string |
| `is_available()` | `→ bool` | Health check: backend reachable and model available |

**Phase 1 — `OllamaAdapter`** (local development):
- Calls a local Ollama server via `ollama.chat()`.
- Schema enforcement via the `format` parameter.
- Default model: `qwen2.5:14b`. Configurable via `ollama.model` in config.
- Retries on JSON parse failure (`max_retries` configurable).
- Requires: `pip install ollama` + running Ollama server.

**Phase 2 — `GeminiDevAdapter`** (free-tier cloud):
- Calls Gemini Developer API with `GEMINI_API_KEY` env var.
- Uses `response_mime_type="application/json"` + `response_schema` for structured output.
- Default model: `gemini-2.5-flash`. Free-tier limits: 10 RPM / 500 RPD.
- Requires: `pip install google-generativeai` + API key from https://aistudio.google.com/apikey.

**Phase 3 — `VertexAdapter`** (production GCP):
- Calls Vertex AI via `google-genai` SDK with `vertexai=True`.
- Supports context caching: `load_corpus_cache()` uploads corpus once, subsequent calls reference the cache name.
- Uses `cached_content` parameter in generation config to avoid re-transmitting corpus.
- Default model: `gemini-2.5-flash`. Location: `us-central1`.
- Requires: `pip install google-genai` + `GOOGLE_CLOUD_PROJECT` env var + `gcloud auth`.

**Backend selection:** set `llm_backend` in `reporting_config.yaml` to `"ollama"`, `"gemini_dev"`, or `"vertex_ai"`.

---

#### 4.9.5 `schemas/` — Pydantic Report Schemas

All schemas inherit from `BaseReport`. All require:
- `disclaimer` == `"AI-generated. Not for operational use without human review."` (enforced by `field_validator`)
- `report_confidence` ∈ [0.0, 1.0]
- `human_review_required` flag

**Nested supporting types:** `RiskCell`, `Recommendation`, `VulnerableGroup`, `ResourceRequirement`, `ProjectedLoss`, `TimelineEvent`, `ResourceDeployed`.

| Schema | Mode | Key fields |
|---|---|---|
| `DailyReport` | QUIET | `summary`, `monitored_area_count`, `weather_summary`, `notable_changes` |
| `HighRiskReport` | ACTIVE | `risk_summary`, `top_risk_cells` (1–5 `RiskCell`), `contributing_factors`, `preventive_recommendations` (≥ 2), `escalation_trigger` |
| `IncidentReport` | EMERGENCY | `incident_name`, `incident_status` (ACTIVE/CONTAINED/CONTROLLED/OUT), `affected_communities`, `spread_summary`, `resource_requirements`, `projected_losses`, `vulnerable_populations`, `immediate_actions` (≥ 3) |
| `FinalReport` | EMERGENCY (FINAL) | `incident_name`, `linked_incident_id`, `incident_timeline` (≥ 3 events), `resources_deployed`, `losses_summary`, `lessons_learned` (≥ 2), `recommendations_for_future` (≥ 2). `human_review_required` is always `True` |

---

#### 4.9.6 `renderer.py` — Jinja2 Rendering

Deterministic rendering — zero LLM involvement, all values pre-computed in the Pydantic model.

| Function | Purpose |
|---|---|
| `render_markdown(report, template_dir)` | Renders daily / high_risk reports via Jinja2 `.md.j2` templates |
| `render_html(report, template_dir)` | Renders incident / final reports via Jinja2 `.html.j2` templates (autoescaped) |
| `markdown_to_html(md_str)` | Converts Markdown → HTML via `python-markdown` |
| `render_pdf(html_str, css_string)` | Converts HTML → PDF via WeasyPrint (optional dependency, admin-only) |

Template files are expected at `templates/{report_type}.{md\|html}.j2`.

---

#### 4.9.7 `reporter.py` — Main Orchestrator

`GeminiDisasterReporter` subclasses `BaseModel` and overrides methods to work with `ContextBundle` / `ReportResult` instead of raw DataFrames.

**Overridden `BaseModel` methods:**

| Method | Input | Output | Purpose |
|---|---|---|---|
| `load_model(model_path)` | path to `reporting_config.yaml` | None | Loads config, creates adapter, runs health check, loads corpus (with Vertex AI caching if applicable), initialises admin toggle |
| `predict(context_bundle)` | `ContextBundle` | `ReportResult` | Sends context to LLM, parses response via Pydantic. Retries once on parse failure |
| `validate(report_result)` | `ReportResult` | `ValidationResult` | Checks: (1) schema valid, (2) sections complete, (3) confidence ≥ threshold, (4) `human_review_required` flag correct. Final reports always require review |
| `explain(report_result)` | `ReportResult` | `dict` | Returns confidence, data sources, review flag, latency — no LLM call |

**High-level convenience method:**

`generate_report(pipeline_result, human_inputs, mode, sub_state)` runs the full pipeline:

| Step | Description |
|---|---|
| 1–2 | Resolve operational mode (auto or override) |
| 3–4 | Assemble context via `context_builder.assemble()` |
| 5–6 | Call `predict()` → LLM generation + Pydantic parsing |
| 7 | `validate()` — 4-criterion check |
| 8 | Render to Markdown or HTML (based on report type) |
| 9 | Save JSON + rendered file to `reports/disaster_reports/` |
| 10 | Sync to GCS bucket (if configured) |
| 11 | Return `GeneratedReport` with all artefacts |

**Usage:**
```python
from src.models.obj3_gemini import GeminiDisasterReporter

reporter = GeminiDisasterReporter()
reporter.load_model("configs/reporting_config.yaml")

result = reporter.generate_report(
    pipeline_result={
        "risk_level": "HIGH",
        "firms_hotspot_count": 0,
        "xgboost_top_cells": [...],
        "telemetry": {"temperature": 38.2, "humidity": 12},
    },
    human_inputs=[],
)
print(result.validation.passed)     # True if all 4 checks pass
print(result.markdown_path)         # Path to rendered report
```

---

### 4.10 `src/validation/metrics.py` — Metric Computation

**Functions:**

| Function | Output | Purpose |
|---|---|---|
| `compute_auc_pr(y_true, y_prob)` | `float` | Primary metric. Area Under Precision-Recall Curve |
| `compute_f1(y_true, y_pred)` | `float` | F1 score at decision threshold |
| `compute_fnr(y_true, y_pred)` | `float` | False Negative Rate = FN / (FN + TP). Key metric for bias gate |
| `compute_confusion_matrix(y_true, y_pred)` | `dict` with `true_negatives`, `false_positives`, `false_negatives`, `true_positives` | Labeled confusion matrix |
| `compute_all_metrics(y_true, y_prob, threshold, inference_latency_ms)` | `dict` with all above + `accuracy`, `auc_roc`, `positive_rate`, `n_samples` | Full metrics suite for one evaluation run |
| `measure_inference_latency(predict_fn, X, n_runs)` | `float` (ms) | Average wall-clock inference time over `n_runs` |

**Expected outcome:**
- `compute_all_metrics` returns a dict like:
```python
{
    "auc_pr": 0.82, "f1": 0.75, "fnr": 0.08, "accuracy": 0.94,
    "auc_roc": 0.91, "positive_rate": 0.013, "threshold": 0.5, "n_samples": 1000,
    "confusion_matrix": {"true_negatives": 850, "false_positives": 20, "false_negatives": 12, "true_positives": 118},
}
```

---

### 4.11 `src/validation/visualizations.py` — Plot Generation

**Functions:**

| Function | Output File | Purpose |
|---|---|---|
| `plot_precision_recall_curve(y_true, y_prob, output_path)` | `precision_recall_curve.png` | PR curve with AUC annotation and no-skill baseline |
| `plot_confusion_matrix(y_true, y_pred, output_path)` | `confusion_matrix.png` | Heatmap with "No Fire" / "Fire" labels |
| `plot_model_comparison(metrics, output_path)` | `model_comparison.png` | Side-by-side bar chart (e.g., XGBoost vs FWI baseline) |
| `generate_all_visualizations(...)` | all three PNGs | Convenience — calls all three, returns `dict[str, Path]` |

**Expected outcome:**
- Three PNG files in `reports/visualizations/`, each 150 DPI.
- All three are logged as MLflow artifacts.

---

### 4.12 `src/validation/model_selector.py` — Validation Gate

**Functions:**

| Function | Output | Purpose |
|---|---|---|
| `validate_model(y_true, y_prob, config_path)` | `(metrics_dict, passed_bool)` | Computes metrics, checks `auc_pr >= 0.75`. Returns whether gate passed |
| `save_validation_report(result, output_dir)` | `Path` to JSON | Writes `reports/validation/validation_report.json` with all metrics, gate results, viz paths |
| `main()` | exit code 0 or 1 | CLI entry point for DVC stage `validate_model`. Reads predictions parquet, runs gate, exits non-zero on failure |

**Expected outcome:**
- `reports/validation/validation_report.json`:
```json
{
    "model_name": "xgboost_pof",
    "passed_validation": true,
    "passed_bias_gate": true,
    "is_deployable": true,
    "metrics": { "auc_pr": 0.82 }
}
```

---

### 4.13 `src/bias/nri_loader.py` — FEMA NRI Data

**Functions:**

| Function | Output | Purpose |
|---|---|---|
| `load_nri(cache_dir)` | `GeoDataFrame` | Loads FEMA NRI census tract shapefile from `data/static/fema_nri/` |
| `compute_vulnerability_quartiles(nri)` | `GeoDataFrame` with `nri_vulnerability_quartile` column | Splits SOVI_SCORE into 4 quartiles: Low, Medium, High, Very High |
| `spatial_join_predictions(predictions, nri)` | `GeoDataFrame` | Converts H3 cell IDs to lat/lng points, joins to nearest NRI census tract. Adds vulnerability quartile to each prediction row |

**Expected outcome:**
- Each prediction row gets a `nri_vulnerability_quartile` label.
- Unmatched rows (points outside any tract) get label "Unknown" and are excluded from bias analysis.

**Prerequisite:**
- Download FEMA NRI shapefile from https://hazards.fema.gov/nri/data-resources and extract into `data/static/fema_nri/`.

---

### 4.14 `src/bias/detector.py` — Bias Gate (BLOCKING)

**Functions:**

| Function | Output | Purpose |
|---|---|---|
| `false_negative_rate(y_true, y_pred)` | `float` | Computes FNR = 1 - recall. Used as the Fairlearn metric function |
| `run_bias_gate(y_true, y_pred, sensitive_features, config_path)` | `(report_dict, passed_bool)` | Builds `MetricFrame`, computes FNR per vulnerability group, checks `disparity < 0.05` |
| `run_bias_gate_from_dataframe(df)` | `(report_dict, passed_bool)` | Convenience — extracts columns from a spatially-joined DataFrame |
| `main()` | exit code 0 or 1 | CLI entry point for DVC stage `model_bias_gate`. Exits non-zero if gate fails |

**Expected outcome:**
- `report_dict`:
```python
{
    "metric": "false_negative_rate",
    "overall_fnr": 0.08,
    "per_group_fnr": {"Low": 0.06, "Medium": 0.07, "High": 0.09, "Very High": 0.10},
    "disparity_between_groups": 0.04,
    "max_allowed_disparity": 0.05,
    "gate_result": "PASS"
}
```
- If `gate_result == "FAIL"`: deployment is blocked, alert fires, RCA report generated.

---

### 4.15 `src/bias/mitigation.py` — Bias Mitigation Strategies

Called when the bias gate fails. Three strategies in escalation order:

| Function | Strategy | When to use |
|---|---|---|
| `compute_class_weights(y, sensitive_features)` | Give 2x sample weight to positive samples in Very High vulnerability quartile | First attempt after bias gate failure |
| `apply_spatial_smote(X, y, metadata)` | Generate synthetic positive samples constrained to H3 spatial neighborhoods. NOT standard SMOTE — preserves spatial autocorrelation | If class_weight alone doesn't fix disparity |
| `apply_correlation_remover(X, sensitive_scores)` | Fairlearn `CorrelationRemover` to decorrelate features from SOVI score | Last resort if above two are insufficient |

**Expected outcome:**
- `compute_class_weights` → numpy array of per-sample weights, pass as `sample_weight` to XGBoost `fit()`.
- `apply_spatial_smote` → `(X_new, y_new, metadata_new)` with more positive samples.
- `apply_correlation_remover` → transformed `X` with reduced SOVI correlation.

---

### 4.16 `src/bias/report.py` — Bias Report Generator

**Functions:**

| Function | Output | Purpose |
|---|---|---|
| `generate_bias_report(bias_result, run_id, model_version, ...)` | `dict` | Builds structured report. Includes RCA section (failure type, recommended mitigations, delta from previous run) when gate fails |
| `save_bias_report(report, output_dir)` | `Path` | Writes `reports/bias_gate/bias_gate_report.json` |
| `load_previous_report(report_dir)` | `dict` or `None` | Loads last report for delta comparison |

**Expected outcome on failure:**
```json
{
    "gate_result": "FAIL",
    "rca": {
        "failure_type": "bias_gate",
        "observed_disparity": 0.12,
        "threshold": 0.05,
        "recommended_mitigations": [
            "1. class_weight adjustment (2x for Very High SOVI cells)",
            "2. Spatial-SMOTE if insufficient (NOT standard SMOTE)",
            "3. CorrelationRemover on SOVI-correlated features",
            "4. Re-run pipeline and verify disparity < 5%"
        ]
    }
}
```

---

### 4.17 `src/tracking/mlflow_logger.py` — Local Experiment Tracking

**Key methods of `MLflowLogger`:**

| Method | What it logs |
|---|---|
| `start_run(run_name, tags)` | Creates new MLflow run, returns `run_id` |
| `log_metrics(metrics)` | AUC-PR, F1, FNR, accuracy, latency |
| `log_params(params)` | Model name, version, hyperparameters |
| `log_input_statistics(stats)` | Per-feature mean/std/min/max for covariate drift detection |
| `log_bias_gate_result(bias_report)` | Gate PASS/FAIL, FNR disparity, per-group FNR |
| `log_validation_result(metrics, passed)` | Validation metrics + gate decision |
| `log_visualization(viz_paths)` | PR curve, confusion matrix, comparison chart PNGs |
| `log_model_hash(hash)` | SHA-256 of model artifact |

**Standalone function:**
- `compute_input_statistics(X)` → `dict[feature_name → {mean, std, min, max}]` for drift detection.

**Expected outcome:**
- Local SQLite DB at `mlruns.db` with full experiment history.
- View with: `mlflow ui --backend-store-uri sqlite:///mlruns.db`

---

### 4.18 `src/tracking/vertex_sync.py` — Vertex AI Experiments Sync

**Key methods of `VertexAISync`:**

| Method | Purpose |
|---|---|
| `sync_run(run_id, metrics, params)` | Upload a run's metrics and params to Vertex AI Experiments |
| `sync_rollback_event(run_id, reason_code, delta_auc_pr, delta_fnr_disparity)` | Record a rollback event with deltas |

**Expected outcome:**
- Metrics visible in Vertex AI Experiments console under experiment `wildfire-model-pipeline`.
- Requires `google-cloud-aiplatform` and valid GCP credentials.

---

### 4.19 `src/notifications/alerter.py` — Slack Alerts

**Key methods of `SlackAlerter`:**

| Method | When it fires |
|---|---|
| `alert_validation_failure(run_id, auc_pr, threshold)` | AUC-PR below 0.75 |
| `alert_bias_gate_failure(run_id, disparity, threshold, per_group)` | FNR disparity above 5% |
| `alert_pipeline_error(run_id, error_message, stage)` | Unhandled exception in any stage |
| `alert_rollback(run_id, reason, from_version, to_version)` | Model rolled back to previous version |
| `alert_shap_drift(run_id, feature, importance, threshold)` | Soil moisture SHAP importance below 10% |
| `alert_success(run_id, model_version, auc_pr)` | Pipeline passed all gates, model pushed |

**Setup:**
- Set env var `SLACK_WEBHOOK_URL` locally.
- In CI/CD, configure via GitHub Secret `SLACK_WEBHOOK_URL`.
- If not configured, alerts are silently skipped (logged as warning).

---

### 4.20 `src/pipeline/orchestrator.py` — End-to-End Pipeline

`run_pipeline(model, config, run_id, baseline_metrics)` executes the full sequence:

| Step | What happens | On failure |
|---|---|---|
| 1 | Load data via `load_and_split()` | `DataLoadError` → alert + abort |
| 2 | Start MLflow run, log input statistics | — |
| 3 | Run `model.predict(X)`, measure latency | `NotImplementedError` if stub |
| 4 | `validate_model()` — check AUC-PR >= 0.75 | Slack alert, return early |
| 5 | Generate 3 visualization PNGs | — |
| 6 | Load FEMA NRI, spatial join, `run_bias_gate()` | If NRI missing: gate = SKIPPED. If disparity > 5%: Slack alert, return early |
| 7 | `registry.save_local()` + `tag_previous()` | — |
| 8 | `VertexAISync.sync_run()` (non-blocking) | Warning logged, continues |

**Returns:** `PipelineResult` dataclass with `is_deployable`, `metrics`, `bias_report`, `visualization_paths`, `error`.

**Usage:**
```python
from src.pipeline.orchestrator import run_pipeline
from src.models.obj1_xgboost.placeholder import XGBoostFireRisk

model = XGBoostFireRisk()
model.load_model("models/ignition/1.0.0/model.json")

result = run_pipeline(
    model=model,
    baseline_metrics={"auc_pr": 0.68, "f1": 0.61},  # FWI baseline for comparison chart
)
print(result.is_deployable)  # True only if validation + bias gate both pass
```

---

## 5. CI/CD Pipeline (GitHub Actions)

Located at `.github/workflows/model_ci.yml` (repo root level).

| Stage | Gate | Blocking |
|---|---|---|
| 1. Lint + type check | `ruff check` + `mypy` zero errors | Yes |
| 2. Unit tests | `pytest --cov-fail-under=90` | Yes |
| 3. Container build | `docker buildx` multi-arch success | Yes |
| 4. Integration test | Smoke test with synthetic data | Yes |
| 5. Model validation | AUC-PR >= 0.75 | Yes |
| 6. Bias gate | FNR disparity < 5% | **Yes (BLOCKING)** |
| 7. Artifact push | Stages 5+6 must pass | Yes |
| 8. Vertex AI sync | Non-blocking | No |
| 9. Deploy | Cloud Run service update | Yes |

Stages 5-9 are placeholder `echo` commands until OBJ-1 is implemented. OBJ-2 and OBJ-3 are ready to be wired in once OBJ-1 provides the primary metrics.

---

## 6. DVC Stages

`dvc.yaml` defines two stages that can be run with `dvc repro`:

- **`validate_model`** — reads `reports/validation/predictions.parquet`, runs validation gate, writes report to `reports/validation/`.
- **`model_bias_gate`** — reads spatially-joined predictions, runs Fairlearn FNR disparity check, writes bias report to `reports/bias_gate/` (git-committed, `cache: false`).

These stages depend on `models/ignition` and `data/static/fema_nri` — they won't run until those exist.

---

## 7. What To Do Next

### Implementation status

| Objective | Status | Notes |
|---|---|---|
| OBJ-1 XGBoost ignition model | **Placeholder** | Implement `src/models/obj1_xgboost/placeholder.py` |
| OBJ-2 Cell2Fire spread simulation | ✅ **Done** | See section 4.8 — requires C++ binary + rasters |
| OBJ-3 Gemini disaster reporting | ✅ **Done** | See section 4.9 — requires Vertex AI / Ollama setup |

### For Teammates — OBJ-1 (XGBoost)

1. Replace `src/models/obj1_xgboost/placeholder.py` with real `XGBoostFireRisk`.
2. `load_model()` → load XGBoost `.json` or `.ubj` weights.
3. `predict(X)` → return DataFrame with `prediction` and `probability` columns.
4. `validate(X, y)` → call `src.validation.metrics.compute_all_metrics()`.
5. `explain(X)` → call `shap.TreeExplainer`.
6. Save trained model to `models/ignition/{version}/`.
7. Run the orchestrator to verify end-to-end.

### Running OBJ-2 (Cell2Fire)

Prerequisites before calling `model.predict()`:
1. Build and install the Cell2Fire C++ binary, ensure it is on `PATH` (or set `obj2.cell2fire.binary_path` in `configs/model_config.yaml`).
2. Place DEM and LANDFIRE FBFM40 fuel GeoTIFFs at the paths configured under `obj2.cell2fire.raster_inputs`.
3. Create a simulation config JSON with `ignition_points`, `aoi_bounds`, and any `params` overrides (see section 4.8 for the JSON schema).

### Before Demo

1. OBJ-1 implemented and passing the orchestrator.
2. Validation gate passing (AUC-PR > 0.75).
3. Bias gate passing (FNR disparity < 5%).
4. Three visualization PNGs in `reports/visualizations/`.
5. MLflow viewable: `mlflow ui --backend-store-uri sqlite:///mlruns.db`.
6. CI/CD stages 1-6 green on GitHub Actions.
