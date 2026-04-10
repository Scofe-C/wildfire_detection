#!/usr/bin/env python3
"""
Continuous Fire Monitoring Loop
===============================
Simulates the production watchdog → pipeline → model cycle locally.
No Airflow, Docker, or GCS required.

Usage:
    python scripts/fire_monitor.py                          # Quiet mode, run until Ctrl+C
    python scripts/fire_monitor.py --mode emergency          # Start in emergency mode
    python scripts/fire_monitor.py --interval 60             # Custom interval (seconds)
    python scripts/fire_monitor.py --cycles 3                # Run 3 cycles then exit
    python scripts/fire_monitor.py --with-api                # Also start control API on :8001
    python scripts/fire_monitor.py --region california       # Single region

Each cycle:
    quiet:     FIRMS → Weather(64km) → Fuse → (OBJ-3 daily report if model-pipeline available)
    active:    FIRMS → Weather(22km) → Fuse → (OBJ-1 → OBJ-2 → OBJ-3 incident report)
    emergency: FIRMS → Weather(22km) → Fuse → (OBJ-1 → OBJ-2 → OBJ-3 incident report)

Mode auto-escalation:
    - FIRMS hotspots with FRP > 50 MW → escalate to active
    - FIRMS hotspots with FRP > 100 MW or count > 5 → escalate to emergency
    - No hotspots for 2 consecutive cycles → de-escalate one level
    - User override (via API or .mode_override.json) always wins
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env files (FIRMS_MAP_KEY, GEMINI_API_KEY, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT.parent / "model-pipeline" / ".env")
except ImportError:
    pass

# Also add model-pipeline to path for OBJ-3 integration
# IMPORTANT: insert AFTER Data-Pipeline so Data-Pipeline's scripts/ takes priority
MODEL_PIPELINE_ROOT = PROJECT_ROOT.parent / "model-pipeline"
if MODEL_PIPELINE_ROOT.exists():
    sys.path.append(str(MODEL_PIPELINE_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fire_monitor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
STATIC_DIR = DATA_DIR / "static"
FIELD_TELEMETRY_DIR = RAW_DIR / "field_telemetry"
MODE_OVERRIDE_FILE = RAW_DIR / ".mode_override.json"

REGIONS = {
    "california": {"bbox": [-124.48, 32.53, -114.13, 42.01]},
    "texas": {"bbox": [-106.65, 25.84, -93.51, 36.50]},
}

MODE_CONFIG = {
    "quiet": {"resolution_km": 64, "lookback_hours": 24, "interval_s": 1800},
    "active": {"resolution_km": 22, "lookback_hours": 2, "interval_s": 900},
    "emergency": {"resolution_km": 22, "lookback_hours": 2, "interval_s": 300},
}

# Demo fire cells for testing when no real FIRMS data is available.
# These are H3 res 5 cells in the Angeles National Forest, California.
DEMO_FIRE_CELLS = ["8528308bfffffff", "852830abfffffff", "85283097fffffff"]

# Escalation thresholds
FRP_ACTIVE_THRESHOLD = 50.0
FRP_EMERGENCY_THRESHOLD = 100.0
HOTSPOT_EMERGENCY_COUNT = 5
DEESCALATE_AFTER_CYCLES = 2

# Shared state for API access
monitor_state: dict[str, Any] = {
    "mode": "quiet",
    "cycle_count": 0,
    "last_cycle_at": None,
    "next_cycle_at": None,
    "fire_cells_detected": [],
    "no_fire_streak": 0,
    "cycle_history": [],
    "field_telemetry_log": [],
    "running": True,
    "user_override": None,
}


# ---------------------------------------------------------------------------
# Mode management
# ---------------------------------------------------------------------------

def _read_mode_override() -> dict[str, Any] | None:
    """Read user mode override from file or shared state."""
    # Check shared state first (set by API)
    if monitor_state.get("user_override"):
        override = monitor_state["user_override"]
        monitor_state["user_override"] = None  # consume it
        return override

    # Check file-based override
    if MODE_OVERRIDE_FILE.exists():
        try:
            data = json.loads(MODE_OVERRIDE_FILE.read_text(encoding="utf-8"))
            MODE_OVERRIDE_FILE.unlink()  # consume it
            logger.info("Mode override from file: %s", data)
            return data
        except Exception as exc:
            logger.warning("Failed to read mode override: %s", exc)
    return None


def _determine_mode(
    current_mode: str,
    firms_hotspots: list[dict],
    field_telemetry: list[dict],
    no_fire_streak: int,
) -> tuple[str, int]:
    """Determine the current operational mode based on inputs.

    Returns (mode, updated_no_fire_streak).
    """
    # User override always wins
    override = _read_mode_override()
    if override:
        new_mode = override.get("mode", current_mode)
        logger.info(
            "Mode override applied: %s → %s (reason: %s)",
            current_mode, new_mode, override.get("reason", "user request"),
        )
        return new_mode, 0

    # Check field telemetry for fire confirmation
    field_fire = any(
        ft.get("confidence", 0) >= 70 and ft.get("frp", 0) > 0
        for ft in field_telemetry
    )

    # Check FIRMS detections
    if firms_hotspots:
        max_frp = max((h.get("frp", 0) for h in firms_hotspots), default=0)
        count = len(firms_hotspots)

        if max_frp >= FRP_EMERGENCY_THRESHOLD or count >= HOTSPOT_EMERGENCY_COUNT or field_fire:
            return "emergency", 0
        if max_frp >= FRP_ACTIVE_THRESHOLD:
            return "active", 0
        return current_mode, 0

    if field_fire:
        return max(current_mode, "active", key=lambda m: list(MODE_CONFIG).index(m)), 0

    # No fire detected — track streak for de-escalation
    no_fire_streak += 1
    if no_fire_streak >= DEESCALATE_AFTER_CYCLES and current_mode != "quiet":
        modes = list(MODE_CONFIG)
        idx = modes.index(current_mode)
        new_mode = modes[max(0, idx - 1)]
        logger.info(
            "De-escalating: %s → %s (no fire for %d cycles)",
            current_mode, new_mode, no_fire_streak,
        )
        return new_mode, 0

    return current_mode, no_fire_streak


# ---------------------------------------------------------------------------
# Pipeline cycle
# ---------------------------------------------------------------------------

def run_cycle(
    mode: str,
    regions: list[str],
    cycle_num: int,
    backend: str | None = None,
) -> dict[str, Any]:
    """Run one complete pipeline cycle.

    Returns dict with cycle results.
    """
    cfg = MODE_CONFIG[mode]
    execution_date = datetime.now(timezone.utc)
    result: dict[str, Any] = {
        "cycle": cycle_num,
        "mode": mode,
        "started_at": execution_date.isoformat(),
        "resolution_km": cfg["resolution_km"],
        "steps": {},
        "firms_hotspots": [],
        "field_telemetry_count": 0,
        "report_generated": False,
    }

    t_total = time.time()

    # --- Step 1: Load field telemetry ---
    field_payloads: list[dict] = []
    try:
        from scripts.ingestion.ingest_field_telemetry import (
            load_pending_field_telemetry,
            batch_field_telemetry_to_dataframe,
        )
        FIELD_TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
        field_payloads = load_pending_field_telemetry(FIELD_TELEMETRY_DIR)
        result["field_telemetry_count"] = len(field_payloads)
        if field_payloads:
            logger.info("  Loaded %d field telemetry observations", len(field_payloads))
    except Exception as e:
        logger.warning("  Field telemetry load failed: %s", e)

    # --- Step 2: FIRMS ingestion ---
    all_firms_hotspots: list[dict] = []
    for region in regions:
        t0 = time.time()
        try:
            from scripts.ingestion.ingest_firms import fetch_firms_data
            firms_path = fetch_firms_data(
                execution_date=execution_date,
                resolution_km=cfg["resolution_km"],
                lookback_hours=24,
                output_dir=str(RAW_DIR / "firms"),
                region=region,
            )
            # Read back to check for hotspots
            import pandas as pd
            firms_df = pd.read_csv(firms_path)
            hotspots = []
            if "latitude" in firms_df.columns and len(firms_df) > 0:
                for _, row in firms_df.head(20).iterrows():
                    hotspots.append({
                        "lat": round(float(row.get("latitude", 0)), 4),
                        "lon": round(float(row.get("longitude", 0)), 4),
                        "frp": float(row.get("frp", 0)),
                        "confidence": str(row.get("confidence", "")),
                    })
            all_firms_hotspots.extend(hotspots)
            result["steps"][f"firms_{region}"] = {
                "status": "OK", "time_s": round(time.time() - t0, 1),
                "hotspots": len(hotspots),
            }
            logger.info("  FIRMS %s: %d hotspots (%.1fs)", region, len(hotspots), time.time() - t0)
        except Exception as e:
            result["steps"][f"firms_{region}"] = {"status": "FAIL", "error": str(e)[:200]}
            logger.error("  FIRMS %s failed: %s", region, e)

    result["firms_hotspots"] = all_firms_hotspots

    # --- Step 3: Weather ingestion ---
    for region in regions:
        t0 = time.time()
        try:
            from scripts.ingestion.ingest_weather import fetch_weather_data
            from scripts.utils.grid_utils import generate_grid_for_bbox

            bbox = REGIONS[region]["bbox"]
            grid = generate_grid_for_bbox(bbox, cfg["resolution_km"])
            grid_centroids = grid[["grid_id", "latitude", "longitude"]]

            trigger_source = "cron" if mode == "quiet" else f"watchdog_{mode}"
            fire_cells = [h.get("grid_id") for h in all_firms_hotspots if h.get("grid_id")]

            # In active/emergency mode with no real FIRMS detections,
            # use demo fire_cells so weather only fetches focal area
            # instead of all ~3949 cells (which causes HTTP 414).
            if not fire_cells and mode != "quiet":
                fire_cells = DEMO_FIRE_CELLS
                logger.info("  Using demo fire_cells for focal weather fetch")

            weather_path = fetch_weather_data(
                grid_centroids=grid_centroids,
                execution_date=execution_date,
                lookback_hours=cfg["lookback_hours"],
                output_dir=str(RAW_DIR / "weather"),
                trigger_source=trigger_source,
                fire_cells=fire_cells or None,
                h3_ring_max=5,
                region=region,
            )
            result["steps"][f"weather_{region}"] = {
                "status": "OK", "time_s": round(time.time() - t0, 1),
            }
            logger.info("  Weather %s: OK (%.1fs)", region, time.time() - t0)
        except Exception as e:
            result["steps"][f"weather_{region}"] = {"status": "FAIL", "error": str(e)[:200]}
            logger.error("  Weather %s failed: %s", region, e)

    # --- Step 4: Processing + Fusion (best-effort) ---
    t0 = time.time()
    try:
        from scripts.processing.process_firms import process_firms_data
        from scripts.processing.process_weather import process_weather_data
        from scripts.fusion.fuse_features import fuse_features
        import pandas as pd

        firms_dfs, weather_dfs = [], []
        for region in regions:
            firms_step = result["steps"].get(f"firms_{region}", {})
            weather_step = result["steps"].get(f"weather_{region}", {})

            if firms_step.get("status") == "OK":
                firms_raw = sorted((RAW_DIR / "firms").glob(f"*{region}*.csv"), key=lambda p: p.stat().st_mtime)
                if firms_raw:
                    df = process_firms_data(str(firms_raw[-1]), cfg["resolution_km"])
                    df["region"] = region
                    firms_dfs.append(df)

            if weather_step.get("status") == "OK":
                weather_raw = sorted((RAW_DIR / "weather").glob(f"*{region}*.csv"), key=lambda p: p.stat().st_mtime)
                if weather_raw:
                    df = process_weather_data(str(weather_raw[-1]), cfg["resolution_km"])
                    df["region"] = region
                    weather_dfs.append(df)

        firms_all = pd.concat(firms_dfs, ignore_index=True) if firms_dfs else pd.DataFrame()
        weather_all = pd.concat(weather_dfs, ignore_index=True) if weather_dfs else pd.DataFrame()

        # Load static if available
        static_path = STATIC_DIR / f"static_features_{cfg['resolution_km']}km.parquet"
        static_df = pd.read_parquet(static_path) if static_path.exists() else pd.DataFrame()

        fused = fuse_features(
            firms_features=firms_all,
            weather_features=weather_all,
            static_features=static_df,
            execution_date=pd.Timestamp(str(execution_date)),
            resolution_km=cfg["resolution_km"],
        )

        output_path = PROCESSED_DIR / "fused" / "fused_features_latest.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fused.to_parquet(output_path, index=False)

        result["steps"]["fusion"] = {
            "status": "OK",
            "time_s": round(time.time() - t0, 1),
            "rows": len(fused),
        }
        logger.info("  Fusion: %d rows (%.1fs)", len(fused), time.time() - t0)
    except Exception as e:
        result["steps"]["fusion"] = {"status": "FAIL", "error": str(e)[:200]}
        logger.warning("  Fusion failed: %s", e)

    # --- Step 5: OBJ-3 report generation (if model-pipeline available) ---
    t0 = time.time()
    try:
        _generate_report(mode, all_firms_hotspots, cfg, result, backend=backend)
    except Exception as e:
        result["steps"]["obj3_report"] = {"status": "FAIL", "error": str(e)[:200]}
        logger.warning("  OBJ-3 report skipped: %s", e)

    result["total_time_s"] = round(time.time() - t_total, 1)
    result["completed_at"] = datetime.now(timezone.utc).isoformat()
    return result


def _generate_report(
    mode: str,
    firms_hotspots: list[dict],
    cfg: dict,
    result: dict,
    backend: str | None = None,
) -> None:
    """Generate an OBJ-3 report using demo scenarios as fallback."""
    if not MODEL_PIPELINE_ROOT.exists():
        result["steps"]["obj3_report"] = {"status": "SKIP", "reason": "model-pipeline not found"}
        return

    # Map mode → demo scenario
    demo_map = {"quiet": "low_risk", "active": "high_risk", "emergency": "emergency"}
    demo_name = demo_map.get(mode, "low_risk")

    try:
        from src.models.obj3_gemini.reporter import GeminiDisasterReporter
        import yaml

        config_path = MODEL_PIPELINE_ROOT / "configs" / "reporting_config.yaml"

        # Override backend if specified
        if backend:
            with open(config_path, encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            config_data["llm_backend"] = backend
            # Write temp config
            tmp_config = MODEL_PIPELINE_ROOT / "configs" / ".reporting_config_tmp.yaml"
            with open(tmp_config, "w", encoding="utf-8") as f:
                yaml.safe_dump(config_data, f, default_flow_style=False)
            config_path = tmp_config

        reporter = GeminiDisasterReporter()
        reporter.load_model(config_path)

        # Use demo scenario data (always available, no trained model needed)
        # Import the demo scenarios from run_report.py
        sys.path.insert(0, str(MODEL_PIPELINE_ROOT / "scripts"))
        from run_report import _DEMO_SCENARIOS
        pipeline_result = _DEMO_SCENARIOS[demo_name]

        # Override with real FIRMS data if available
        if firms_hotspots:
            pipeline_result["firms_hotspot_count"] = len(firms_hotspots)
            pipeline_result["firms_hotspots"] = firms_hotspots

        gen = reporter.generate_report(pipeline_result=pipeline_result)

        rr = gen.report_result
        result["steps"]["obj3_report"] = {
            "status": "OK" if rr.error is None else "FAIL",
            "time_s": round(rr.latency_ms / 1000, 1),
            "report_type": rr.report_type,
            "incident_id": rr.incident_id,
            "confidence": rr.parsed_report.report_confidence if rr.parsed_report else None,
            "json_path": str(gen.json_path) if gen.json_path else None,
        }
        result["report_generated"] = rr.error is None
        logger.info(
            "  OBJ-3: %s report (confidence=%.0f%%, %.1fs)",
            rr.report_type,
            (rr.parsed_report.report_confidence * 100) if rr.parsed_report else 0,
            rr.latency_ms / 1000,
        )
    except ImportError:
        result["steps"]["obj3_report"] = {
            "status": "SKIP",
            "reason": "LLM backend not available (install ollama or set GEMINI_API_KEY)",
        }
        logger.info("  OBJ-3: skipped (LLM backend not available)")
    except Exception as e:
        result["steps"]["obj3_report"] = {"status": "FAIL", "error": str(e)[:200]}
        logger.warning("  OBJ-3 failed: %s", e)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous fire monitoring loop")
    parser.add_argument("--mode", choices=["quiet", "active", "emergency"], default="quiet")
    parser.add_argument("--interval", type=int, default=None, help="Override cycle interval (seconds)")
    parser.add_argument("--cycles", type=int, default=None, help="Run N cycles then exit (default: infinite)")
    parser.add_argument("--region", default=None, help="Single region (default: all)")
    parser.add_argument("--with-api", action="store_true", help="Start control API on :8001")
    parser.add_argument("--api-port", type=int, default=8001, help="API port (default: 8001)")
    parser.add_argument("--skip-obj3", action="store_true", help="Skip OBJ-3 report generation")
    parser.add_argument("--backend", default=None, help="LLM backend for OBJ-3 (ollama, gemini_dev)")
    args = parser.parse_args()

    monitor_state["mode"] = args.mode
    regions = [args.region] if args.region else list(REGIONS.keys())

    # Start API server in background thread if requested
    if args.with_api:
        try:
            from scripts.fire_monitor_api import start_api_background
            start_api_background(port=args.api_port, state=monitor_state)
            logger.info("Control API started on http://127.0.0.1:%d", args.api_port)
        except ImportError:
            logger.warning("fire_monitor_api.py not found — API not started")

    # Graceful shutdown
    def _signal_handler(sig: int, frame: Any) -> None:
        logger.info("\nShutting down fire monitor...")
        monitor_state["running"] = False

    signal.signal(signal.SIGINT, _signal_handler)

    logger.info("=" * 60)
    logger.info("FIRE MONITOR STARTED")
    logger.info("  Mode: %s", args.mode)
    logger.info("  Regions: %s", regions)
    logger.info("  Cycles: %s", args.cycles or "infinite (Ctrl+C to stop)")
    if args.with_api:
        logger.info("  Dashboard: http://127.0.0.1:%d", args.api_port)
    logger.info("  Field telemetry dir: %s", FIELD_TELEMETRY_DIR)
    logger.info("=" * 60)

    cycle_num = 0

    while monitor_state["running"]:
        cycle_num += 1
        if args.cycles and cycle_num > args.cycles:
            break

        mode = monitor_state["mode"]
        cfg = MODE_CONFIG[mode]
        interval = args.interval or cfg["interval_s"]

        logger.info("\n" + "=" * 60)
        logger.info("CYCLE %d  |  Mode: %s  |  Resolution: %dkm", cycle_num, mode.upper(), cfg["resolution_km"])
        logger.info("=" * 60)

        # Run pipeline cycle
        cycle_result = run_cycle(mode, regions, cycle_num, backend=args.backend)

        # Update state
        monitor_state["cycle_count"] = cycle_num
        monitor_state["last_cycle_at"] = cycle_result["completed_at"]
        monitor_state["fire_cells_detected"] = [
            h.get("lat", 0) for h in cycle_result.get("firms_hotspots", [])
        ]

        # Keep last 20 cycles
        monitor_state["cycle_history"].append(cycle_result)
        if len(monitor_state["cycle_history"]) > 20:
            monitor_state["cycle_history"] = monitor_state["cycle_history"][-20:]

        # Auto-escalate/de-escalate
        new_mode, new_streak = _determine_mode(
            mode,
            cycle_result.get("firms_hotspots", []),
            [],  # field telemetry already loaded in cycle
            monitor_state["no_fire_streak"],
        )
        monitor_state["no_fire_streak"] = new_streak
        if new_mode != mode:
            logger.info("MODE CHANGE: %s → %s", mode.upper(), new_mode.upper())
        monitor_state["mode"] = new_mode

        # Print cycle summary
        steps_ok = sum(1 for s in cycle_result["steps"].values() if s.get("status") == "OK")
        steps_total = len(cycle_result["steps"])
        logger.info(
            "\nCycle %d complete: %d/%d steps OK, %.1fs total, mode=%s%s",
            cycle_num, steps_ok, steps_total, cycle_result["total_time_s"],
            new_mode.upper(),
            " [REPORT GENERATED]" if cycle_result.get("report_generated") else "",
        )

        # Wait for next cycle
        if args.cycles and cycle_num >= args.cycles:
            break

        next_at = datetime.now(timezone.utc).timestamp() + interval
        monitor_state["next_cycle_at"] = datetime.fromtimestamp(next_at, tz=timezone.utc).isoformat()

        logger.info("Next cycle in %ds (%s)...", interval, monitor_state["next_cycle_at"][:19])
        # Interruptible sleep
        for _ in range(interval):
            if not monitor_state["running"]:
                break
            time.sleep(1)

    logger.info("\nFire monitor stopped after %d cycles.", cycle_num)


if __name__ == "__main__":
    main()
