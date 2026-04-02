"""run_report.py — Generate one real disaster report end-to-end.

Usage
-----
    # Use the existing mock fixture (always works, no live data needed):
    python scripts/run_report.py

    # Use the real captured fixture:
    python scripts/run_report.py --fixture tests/obj3/fixtures/real_pipeline_result.json

    # Use a specific scenario fixture:
    python scripts/run_report.py --fixture tests/obj3/fixtures/fixture_c_emergency.json

    # Override LLM backend (default: reads from reporting_config.yaml):
    python scripts/run_report.py --backend ollama
    python scripts/run_report.py --backend gemini_dev

    # Override the risk scenario for a quick demo (no fixture needed):
    python scripts/run_report.py --demo high_risk
    python scripts/run_report.py --demo emergency
    python scripts/run_report.py --demo low_risk

Exit criterion
--------------------------
After this script completes, you should see:
    reports/disaster_reports/<type>_<incident_id>_<timestamp>.json
    reports/disaster_reports/<type>_<incident_id>_<timestamp>.md  (or .html)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_report")

# ---------------------------------------------------------------------------
# Built-in demo scenarios
# ---------------------------------------------------------------------------

_DEMO_SCENARIOS: dict[str, dict] = {
    "low_risk": {
        "run_id": "demo-low-001",
        "is_deployable": True,
        "risk_level": "LOW",
        "firms_hotspot_count": 0,
        "firms_hotspots": [],
        "xgboost_top_cells": [
            {"h3_index": "8928308280fffff", "probability": 0.15, "lat": 37.4, "lon": -119.5},
        ],
        "cell2fire_geojson": None,
        "obj2_simulation": None,
        "propagator_summary": None,
        "telemetry": {
            "temperature_max": 22.0,
            "wind_speed_mph": 8.0,
            "relative_humidity": 65.0,
            "soil_moisture": 0.28,
        },
        "fema_nri_tracts": [],
        "bias_report": {"gate_result": "PASS", "observed_disparity": 0.02},
        "metrics": {"auc_pr": 0.82, "f1": 0.74, "fnr": 0.12},
        "source_status": {
            "FIRMS": {"status": "OK", "detail": ""},
            "Open-Meteo": {"status": "OK", "detail": ""},
            "SMAP": {"status": "OK", "detail": "Via Open-Meteo soil moisture"},
        },
    },
    "high_risk": {
        "run_id": "demo-high-001",
        "is_deployable": True,
        "risk_level": "HIGH",
        "firms_hotspot_count": 4,
        "firms_hotspots": [
            {"lat": 34.12, "lon": -118.32, "frp": 85.3, "confidence": "high",
             "acq_datetime": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%MZ")},
            {"lat": 34.14, "lon": -118.30, "frp": 62.1, "confidence": "nominal",
             "acq_datetime": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%MZ")},
            {"lat": 34.10, "lon": -118.35, "frp": 41.7, "confidence": "nominal",
             "acq_datetime": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%MZ")},
            {"lat": 34.16, "lon": -118.28, "frp": 28.5, "confidence": "low",
             "acq_datetime": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%MZ")},
        ],
        "xgboost_top_cells": [
            {"h3_index": "8928308280fffff", "probability": 0.81, "lat": 34.12, "lon": -118.32},
            {"h3_index": "8928308281fffff", "probability": 0.74, "lat": 34.14, "lon": -118.30},
        ],
        "cell2fire_geojson": None,
        "obj2_simulation": {
            "ignition_cell": "8928308280fffff",
            "ignition_probability": 0.72,
            "spread_direction_deg": 77.1,
            "spread_speed_kmh": 2.27,
            "crown_fire_status": "passive_crown",
            "byram_intensity_kwm": 1446.1,
            "dead_fuel_moisture_pct": 11.3,
            "foliar_moisture_content_pct": 115.0,
            "dominant_factor": "wind",
            "inputs_used": {
                "wind_speed_10m_ms": 9.64,
                "midflame_wind_mph": 8.63,
                "ignition_cell_slope_deg": 2.2,
                "ignition_cell_fbfm40": 122.0,
            },
            "warnings": [],
        },
        "propagator_summary": "Ignition cell: 8928308280fffff. spreading at 2.27 km/h (1.4 mph). direction 77.1°. crown fire: passive_crown. Byram intensity: 1446.1 kW/m. dominant factor: wind.",
        "telemetry": {
            "temperature_max": 38.5,
            "wind_speed_mph": 22.0,
            "relative_humidity": 14.0,
            "soil_moisture": 0.06,
            "dead_fuel_moisture_pct": 11.3,
        },
        "fema_nri_tracts": [
            {"tract_id": "06037701000", "nri_score": 78.2, "county": "Los Angeles"},
        ],
        "bias_report": {"gate_result": "PASS", "observed_disparity": 0.03},
        "metrics": {"auc_pr": 0.85, "f1": 0.77, "fnr": 0.10},
        "source_status": {
            "FIRMS": {"status": "OK", "detail": ""},
            "Open-Meteo": {"status": "OK", "detail": ""},
            "SMAP": {"status": "OK", "detail": "Via Open-Meteo soil moisture"},
        },
    },
    "emergency": {
        "run_id": "demo-emergency-001",
        "is_deployable": True,
        "risk_level": "CRITICAL",
        "firms_hotspot_count": 15,
        "firms_hotspots": [
            {"lat": 34.12 + i * 0.01, "lon": -118.32 + i * 0.01,
             "frp": 300.0 - i * 15, "confidence": "high",
             "acq_datetime": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%MZ")}
            for i in range(15)
        ],
        "xgboost_top_cells": [
            {"h3_index": f"892830828{i}fffff", "probability": round(0.97 - i * 0.04, 2),
             "lat": 34.12 + i * 0.01, "lon": -118.32 + i * 0.01}
            for i in range(5)
        ],
        "cell2fire_geojson": '{"type":"FeatureCollection","features":[]}',
        "obj2_simulation": {
            "ignition_cell": "8928308280fffff",
            "ignition_probability": 0.95,
            "spread_direction_deg": 45.0,
            "spread_speed_kmh": 5.8,
            "crown_fire_status": "active_crown",
            "byram_intensity_kwm": 4200.0,
            "dead_fuel_moisture_pct": 5.2,
            "foliar_moisture_content_pct": 80.0,
            "dominant_factor": "wind",
            "inputs_used": {
                "wind_speed_10m_ms": 17.0,
                "midflame_wind_mph": 15.2,
                "ignition_cell_slope_deg": 8.5,
                "ignition_cell_fbfm40": 165.0,
            },
            "warnings": ["Extreme fire behavior expected"],
        },
        "propagator_summary": (
            "EMERGENCY: 15 active hotspots. Peak FRP 300 MW. "
            "Spreading NE at 5.8 km/h (3.6 mph). Active crown fire. "
            "Byram intensity 4200 kW/m. ~420 acres."
        ),
        "telemetry": {
            "temperature_max": 43.0,
            "wind_speed_mph": 38.0,
            "relative_humidity": 6.0,
            "soil_moisture": 0.03,
            "dead_fuel_moisture_pct": 5.2,
        },
        "fema_nri_tracts": [
            {"tract_id": "06037701000", "nri_score": 91.5, "county": "Los Angeles"},
            {"tract_id": "06037702000", "nri_score": 84.2, "county": "Los Angeles"},
        ],
        "bias_report": {"gate_result": "PASS", "observed_disparity": 0.03},
        "metrics": {"auc_pr": 0.88, "f1": 0.81, "fnr": 0.08},
        "source_status": {
            "FIRMS": {"status": "OK", "detail": ""},
            "Open-Meteo": {"status": "STALE", "detail": "OWM returned 429, using cached data"},
            "SMAP": {"status": "OK", "detail": "Via Open-Meteo soil moisture"},
        },
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate one disaster report end-to-end.")
    parser.add_argument(
        "--fixture", default=None,
        help="Path to pipeline_result JSON fixture file. Overrides --demo.",
    )
    parser.add_argument(
        "--demo", default="high_risk",
        choices=list(_DEMO_SCENARIOS),
        help="Built-in scenario to use when no --fixture is provided (default: high_risk).",
    )
    parser.add_argument(
        "--backend", default=None,
        choices=["ollama", "gemini_dev", "vertex_ai"],
        help="Override llm_backend from config.",
    )
    parser.add_argument(
        "--config", default=str(_ROOT / "configs" / "reporting_config.yaml"),
        help="Path to reporting_config.yaml.",
    )
    args = parser.parse_args()

    # 1. Load pipeline_result
    if args.fixture:
        fixture_path = Path(args.fixture)
        if not fixture_path.is_absolute():
            fixture_path = _ROOT / fixture_path
        if not fixture_path.exists():
            logger.error("Fixture not found: %s", fixture_path)
            return 1
        with open(fixture_path, encoding="utf-8") as fh:
            pipeline_result = json.load(fh)
        logger.info("Loaded fixture: %s", fixture_path)
    else:
        pipeline_result = _DEMO_SCENARIOS[args.demo]
        logger.info("Using built-in demo scenario: %s", args.demo)

    logger.info(
        "Pipeline result — risk_level=%s, firms_hotspot_count=%d, is_deployable=%s",
        pipeline_result.get("risk_level", "?"),
        pipeline_result.get("firms_hotspot_count", 0),
        pipeline_result.get("is_deployable", True),
    )

    # 2. Optionally patch backend in config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        return 1

    patched_config_path = config_path
    if args.backend:
        # Write a temporary patched config
        import tempfile

        import yaml  # noqa: PLC0415
        with open(config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        cfg["llm_backend"] = args.backend
        tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115  # delete=False requires explicit close before use on Windows
            suffix=".yaml", mode="w", encoding="utf-8", delete=False
        )
        yaml.dump(cfg, tmp)
        tmp.flush()
        tmp.close()
        patched_config_path = Path(tmp.name)
        logger.info("Backend overridden to: %s", args.backend)

    # 3. Load reporter
    try:
        from src.models.obj3_gemini.reporter import GeminiDisasterReporter  # noqa: PLC0415
    except ImportError as exc:
        logger.error("Failed to import reporter: %s", exc)
        return 1

    reporter = GeminiDisasterReporter()

    logger.info("Loading model from: %s", patched_config_path)
    try:
        reporter.load_model(patched_config_path)
    except RuntimeError as exc:
        logger.error("load_model failed: %s", exc)
        logger.error(
            "If using Ollama: make sure 'ollama serve' is running and "
            "the model is pulled (ollama pull qwen3:8b)."
        )
        return 1

    # 4. Generate report
    logger.info("Generating report...")
    t0 = datetime.now(tz=UTC)
    try:
        result = reporter.generate_report(pipeline_result)
    except Exception as exc:
        logger.exception("generate_report failed: %s", exc)
        return 1

    elapsed = (datetime.now(tz=UTC) - t0).total_seconds()

    # 5. Print summary
    rr = result.report_result
    val = result.validation

    logger.info("=" * 60)
    logger.info("REPORT GENERATED in %.1fs", elapsed)
    logger.info("  report_type:     %s", rr.report_type)
    logger.info("  incident_id:     %s", rr.incident_id)
    logger.info("  latency_ms:      %.0f", rr.latency_ms)
    logger.info("  error:           %s", rr.error or "none")
    logger.info("  schema_valid:    %s", val.schema_valid)
    logger.info("  sections_ok:     %s", val.sections_complete)
    logger.info("  confidence_ok:   %s", val.confidence_ok)
    logger.info("  review_flag_ok:  %s", val.review_flag_correct)
    logger.info("  validation.passed: %s", val.passed)

    if rr.parsed_report:
        logger.info("  report_confidence: %.2f", rr.parsed_report.report_confidence)
        logger.info("  human_review_required: %s", rr.parsed_report.human_review_required)
        logger.info("  review_status: %s", rr.parsed_report.review_status)
        logger.info("  grounding_search_count: %d", rr.parsed_report.grounding_search_count)

    logger.info("")
    if result.json_path:
        logger.info("  JSON:  %s", result.json_path)
    if result.markdown_path:
        logger.info("  MD:    %s", result.markdown_path)
    if result.html_path:
        logger.info("  HTML:  %s", result.html_path)
    if result.gcs_paths:
        for p in result.gcs_paths:
            logger.info("  GCS:   %s", p)
    logger.info("=" * 60)

    if not val.passed:
        logger.warning("Validation did not fully pass — check the report for quality issues.")

    # Clean up temp config
    if args.backend and patched_config_path != config_path:
        patched_config_path.unlink(missing_ok=True)

    return 0 if rr.error is None else 1


if __name__ == "__main__":
    sys.exit(main())
