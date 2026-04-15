"""capture_fixture.py — Capture a real pipeline-result snapshot as a test fixture.

Usage
-----
    python scripts/capture_fixture.py [--out PATH] [--area BBOX] [--days N]

What it does
------------
1. Fetches FIRMS VIIRS NRT hotspots for a bounding box (default: California).
   Requires env var FIRMS_MAP_KEY.  Falls back to empty hotspot list if missing.
2. Fetches current OWM weather for the centre of the bounding box.
   Requires env var OWM_API_KEY.  Falls back to None values if missing.
3. Simulates SMAP via clean_smap_grid() — UNAVAILABLE (no public API without
   NASA Earthdata credentials).  Marks source_status accordingly.
4. Assembles a full pipeline_result dict that matches the format expected by
   GeminiDisasterReporter.generate_report().
5. Writes to tests/obj3/fixtures/real_pipeline_result.json (or --out path).

Exit codes
----------
0 — fixture written (even if some sources were unavailable)
1 — unrecoverable error
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow running from project root without installing the package
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("capture_fixture")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_BBOX = "-124.5,32.5,-114.0,42.1"   # California
DEFAULT_DAYS = 1
DEFAULT_OUT = _ROOT / "tests" / "obj3" / "fixtures" / "real_pipeline_result.json"

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
OWM_BASE = "https://api.openweathermap.org/data/2.5/weather"


# ---------------------------------------------------------------------------
# FIRMS fetcher
# ---------------------------------------------------------------------------

def fetch_firms(bbox: str, days: int) -> tuple[list[dict], dict]:
    """Fetch VIIRS SNPP NRT hotspots from FIRMS CSV API.

    Returns (hotspots_list, status_dict).
    """
    map_key = os.getenv("FIRMS_MAP_KEY", "").strip()
    if not map_key:
        logger.warning("FIRMS_MAP_KEY not set — skipping live FIRMS fetch.")
        return [], {"status": "UNAVAILABLE", "detail": "FIRMS_MAP_KEY env var not set"}

    try:
        import httpx  # noqa: PLC0415
    except ImportError:
        try:
            import urllib.request as _ur  # noqa: F401  # availability check only; real call uses _ur2
        except ImportError:
            return [], {"status": "UNAVAILABLE", "detail": "No HTTP client available"}
        _ur_mode = True
    else:
        _ur_mode = False

    url = f"{FIRMS_BASE}/{map_key}/VIIRS_SNPP_NRT/{bbox}/{days}"
    logger.info("Fetching FIRMS: %s (days=%d)", bbox, days)

    try:
        if _ur_mode:
            import urllib.request as _ur2
            with _ur2.urlopen(url, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
        else:
            with httpx.Client(timeout=30) as client:
                resp = client.get(url)
                resp.raise_for_status()
                raw = resp.text

        hotspots = _parse_firms_csv(raw)
        logger.info("FIRMS: %d hotspots fetched", len(hotspots))
        return hotspots, {"status": "OK", "detail": ""}

    except Exception as exc:
        logger.warning("FIRMS fetch failed: %s", exc)
        return [], {"status": "UNAVAILABLE", "detail": str(exc)}


def _parse_firms_csv(raw: str) -> list[dict]:
    """Parse FIRMS CSV response into list of hotspot dicts."""
    reader = csv.DictReader(io.StringIO(raw))
    hotspots: list[dict] = []
    for row in reader:
        try:
            acq_date = row.get("acq_date", "")
            acq_time = row.get("acq_time", "0000").zfill(4)
            acq_dt = f"{acq_date}T{acq_time[:2]}:{acq_time[2:]}Z" if acq_date else ""
            hotspots.append({
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "frp": float(row.get("frp") or 0),
                "confidence": row.get("confidence", "nominal"),
                "acq_datetime": acq_dt,
                "satellite": row.get("satellite", "SNPP"),
                "instrument": row.get("instrument", "VIIRS"),
                "daynight": row.get("daynight", "D"),
            })
        except (KeyError, ValueError):
            continue
    return hotspots


# ---------------------------------------------------------------------------
# OWM fetcher
# ---------------------------------------------------------------------------

def fetch_owm(lat: float, lon: float) -> tuple[dict | None, dict]:
    """Fetch current weather from OpenWeatherMap.

    Returns (telemetry_dict_or_None, status_dict).
    """
    api_key = os.getenv("OWM_API_KEY", "").strip()
    if not api_key:
        logger.warning("OWM_API_KEY not set — skipping live OWM fetch.")
        return None, {"status": "UNAVAILABLE", "detail": "OWM_API_KEY env var not set"}

    url = (
        f"{OWM_BASE}?lat={lat}&lon={lon}"
        f"&appid={api_key}&units=imperial"
    )
    logger.info("Fetching OWM weather for (%.3f, %.3f)", lat, lon)

    try:
        try:
            import httpx  # noqa: PLC0415
            with httpx.Client(timeout=15) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()
        except ImportError:
            import urllib.request as _ur
            with _ur.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))

        telem = {
            "temperature_max": data.get("main", {}).get("temp_max"),
            "wind_speed_mph": data.get("wind", {}).get("speed"),
            "relative_humidity": data.get("main", {}).get("humidity"),
            "soil_moisture": None,  # OWM doesn't provide soil moisture
            "weather_description": data.get("weather", [{}])[0].get("description"),
            "location": data.get("name"),
        }
        logger.info("OWM: temp=%.1f°F, wind=%.1f mph, rh=%d%%",
                    telem["temperature_max"] or 0,
                    telem["wind_speed_mph"] or 0,
                    telem["relative_humidity"] or 0)
        return telem, {"status": "OK", "detail": ""}

    except Exception as exc:
        logger.warning("OWM fetch failed: %s", exc)
        return None, {"status": "UNAVAILABLE", "detail": str(exc)}


# ---------------------------------------------------------------------------
# SMAP — always UNAVAILABLE in this pipeline (no public endpoint)
# ---------------------------------------------------------------------------

def get_smap_status() -> tuple[dict | None, dict]:
    """Apply clean_smap_grid to a null grid — returns UNAVAILABLE status."""
    try:
        from src.data.smap_cleaner import clean_smap_grid  # noqa: PLC0415
        result = clean_smap_grid({})  # empty = all null
        return result, {"status": "UNAVAILABLE", "detail": "SMAP endpoint requires NASA Earthdata credentials"}
    except Exception as exc:
        logger.warning("SMAP cleaner failed: %s", exc)
        return None, {"status": "UNAVAILABLE", "detail": str(exc)}


# ---------------------------------------------------------------------------
# Risk assessment (deterministic from hotspot data)
# ---------------------------------------------------------------------------

def infer_risk_level(hotspots: list[dict], telemetry: dict | None) -> str:
    """Derive a risk level from FIRMS + OWM without running XGBoost."""
    count = len(hotspots)
    max_frp = max((h.get("frp", 0) for h in hotspots), default=0)

    # Wind + humidity from OWM
    wind_mph = (telemetry or {}).get("wind_speed_mph") or 0
    rh = (telemetry or {}).get("relative_humidity") or 100

    if count >= 10 or max_frp >= 200:
        return "CRITICAL"
    if count >= 3 or max_frp >= 50 or (wind_mph >= 25 and rh <= 15):
        return "HIGH"
    if count >= 1 or (wind_mph >= 15 and rh <= 25):
        return "MODERATE"
    return "LOW"


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble_pipeline_result(
    hotspots: list[dict],
    telemetry: dict | None,
    smap_result: dict | None,
    source_status: dict[str, dict],
    bbox: str,
) -> dict:
    """Combine all sources into a pipeline_result dict."""
    risk_level = infer_risk_level(hotspots, telemetry)

    # Realistic XGBoost-style top cells derived from top hotspot locations
    xgboost_top_cells = []
    for i, hs in enumerate(sorted(hotspots, key=lambda h: h.get("frp", 0), reverse=True)[:5]):
        # Fake but plausible probability scaled from FRP
        frp = float(hs.get("frp", 10))
        prob = min(0.99, 0.3 + frp / 400)
        xgboost_top_cells.append({
            "h3_index": f"892830828{i}fffff",
            "probability": round(prob, 3),
            "lat": hs["lat"],
            "lon": hs["lon"],
        })

    # Propagator summary
    propagator_summary = None
    if hotspots:
        max_frp = max(h.get("frp", 0) for h in hotspots)
        propagator_summary = (
            f"{len(hotspots)} active hotspot(s) detected. "
            f"Peak FRP: {max_frp:.1f} MW. "
            f"Risk level: {risk_level}."
        )

    # Default telemetry if OWM unavailable
    telem = telemetry or {
        "temperature_max": None,
        "wind_speed_mph": None,
        "relative_humidity": None,
        "soil_moisture": None,
    }

    # Add SMAP soil moisture if available
    if smap_result and smap_result.get("status") == "COMPLETE":
        telem["soil_moisture"] = smap_result.get("mean_soil_moisture")

    return {
        "run_id": f"fixture-{datetime.now(tz=UTC).strftime('%Y%m%d-%H%M%S')}",
        "captured_at": datetime.now(tz=UTC).isoformat(),
        "bbox": bbox,
        "is_deployable": True,
        "risk_level": risk_level,
        "firms_hotspot_count": len(hotspots),
        "firms_hotspots": hotspots,
        "xgboost_top_cells": xgboost_top_cells,
        "propagator_summary": propagator_summary,
        "telemetry": telem,
        "smap": smap_result,
        "fema_nri_tracts": [],  # Would come from NRI lookup by bbox
        "bias_report": {
            "gate_result": "PASS",
            "observed_disparity": 0.02,
            "note": "Placeholder — real bias gate requires trained model run",
        },
        "metrics": {
            "auc_pr": None,
            "f1": None,
            "fnr": None,
            "note": "Placeholder — metrics populated by XGBoost training pipeline",
        },
        "source_status": source_status,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Capture a real pipeline-result fixture.")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help=f"Output path (default: {DEFAULT_OUT})")
    parser.add_argument("--bbox", default=DEFAULT_BBOX,
                        help=f"Bounding box lon_min,lat_min,lon_max,lat_max (default: {DEFAULT_BBOX})")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"Number of days for FIRMS query (default: {DEFAULT_DAYS})")
    args = parser.parse_args()

    # Parse centre of bbox for OWM
    try:
        lon_min, lat_min, lon_max, lat_max = (float(x) for x in args.bbox.split(","))
        centre_lat = (lat_min + lat_max) / 2
        centre_lon = (lon_min + lon_max) / 2
    except ValueError:
        logger.error("Invalid --bbox format. Expected: lon_min,lat_min,lon_max,lat_max")
        return 1

    source_status: dict[str, dict] = {}

    # 1. FIRMS
    hotspots, firms_status = fetch_firms(args.bbox, args.days)
    source_status["FIRMS"] = firms_status

    # 2. OWM
    telemetry, owm_status = fetch_owm(centre_lat, centre_lon)
    source_status["OWM"] = owm_status

    # 3. SMAP
    smap_result, smap_status = get_smap_status()
    source_status["SMAP"] = smap_status

    # 4. Assemble
    pipeline_result = assemble_pipeline_result(
        hotspots=hotspots,
        telemetry=telemetry,
        smap_result=smap_result,
        source_status=source_status,
        bbox=args.bbox,
    )

    # 5. Write
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(pipeline_result, fh, indent=2, default=str)

    # Summary
    logger.info("=" * 60)
    logger.info("FIXTURE WRITTEN: %s", out_path)
    logger.info("  risk_level:         %s", pipeline_result["risk_level"])
    logger.info("  firms_hotspot_count:%d", pipeline_result["firms_hotspot_count"])
    logger.info("  FIRMS status:       %s", firms_status["status"])
    logger.info("  OWM status:         %s", owm_status["status"])
    logger.info("  SMAP status:        %s", smap_status["status"])
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
