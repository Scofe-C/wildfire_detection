#!/usr/bin/env python3
"""
Generate Fake Field Telemetry
=============================
Creates synthetic drone/firefighter/ICS-209 observations for testing
the fire monitoring pipeline.

Usage:
    python scripts/generate_fake_telemetry.py --source drone --count 5
    python scripts/generate_fake_telemetry.py --source firefighter --lat 34.12 --lon -118.32
    python scripts/generate_fake_telemetry.py --scenario spreading
    python scripts/generate_fake_telemetry.py --scenario contained
    python scripts/generate_fake_telemetry.py --scenario false_alarm
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "field_telemetry"

# California fire-prone areas for realistic fake data
CA_FIRE_ZONES = [
    {"name": "Angeles NF", "lat": 34.25, "lon": -117.85},
    {"name": "Los Padres NF", "lat": 34.68, "lon": -119.72},
    {"name": "San Bernardino NF", "lat": 34.15, "lon": -117.10},
    {"name": "Mendocino NF", "lat": 39.45, "lon": -122.85},
    {"name": "Sequoia NF", "lat": 36.45, "lon": -118.65},
]


def _make_observation(
    source_type: str,
    lat: float,
    lon: float,
    confidence: int,
    frp: float | None,
    report_text: str | None = None,
    timestamp: str | None = None,
) -> dict:
    return {
        "source_type": source_type,
        "priority": 1,
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "confidence": confidence,
        "frp": frp,
        "report_text": report_text,
        "spatial_trust_radius_km": 5.0 if source_type == "drone" else 10.0,
    }


def generate_random(source_type: str, count: int, lat: float | None, lon: float | None) -> list[dict]:
    """Generate random field telemetry observations."""
    obs = []
    zone = random.choice(CA_FIRE_ZONES)
    base_lat = lat or zone["lat"]
    base_lon = lon or zone["lon"]

    for i in range(count):
        offset_lat = random.uniform(-0.05, 0.05)
        offset_lon = random.uniform(-0.05, 0.05)

        if source_type == "drone":
            obs.append(_make_observation(
                source_type="drone",
                lat=base_lat + offset_lat,
                lon=base_lon + offset_lon,
                confidence=random.randint(75, 99),
                frp=round(random.uniform(20, 300), 1),
                report_text=f"Drone observation #{i+1}: thermal anomaly detected",
            ))
        elif source_type == "firefighter":
            obs.append(_make_observation(
                source_type="firefighter",
                lat=base_lat + offset_lat,
                lon=base_lon + offset_lon,
                confidence=random.randint(60, 95),
                frp=round(random.uniform(10, 200), 1) if random.random() > 0.3 else None,
                report_text=random.choice([
                    "Active flame front visible from ridgeline, wind pushing NE",
                    "Spot fire across road, ~2 acres, requesting air support",
                    "Smoke visible but no active flame, smoldering in duff layer",
                    "Structure threat: 3 homes within 500m of fire perimeter",
                    "Crew conducting burnout operation on south flank",
                ]),
            ))
        else:  # ics209
            obs.append(_make_observation(
                source_type="ics209",
                lat=base_lat,
                lon=base_lon,
                confidence=90,
                frp=round(random.uniform(50, 500), 1),
                report_text="ICS-209 status update: incident ongoing, resources deployed",
            ))
    return obs


def generate_scenario(scenario: str) -> list[dict]:
    """Generate a multi-observation scenario."""
    zone = random.choice(CA_FIRE_ZONES)
    now = datetime.now(timezone.utc)
    obs = []

    if scenario == "spreading":
        # Fire moving NE over time, increasing FRP
        for i in range(6):
            ts = (now - timedelta(minutes=30 * (5 - i))).isoformat()
            obs.append(_make_observation(
                source_type="drone" if i % 2 == 0 else "firefighter",
                lat=zone["lat"] + i * 0.008,  # ~800m NE per step
                lon=zone["lon"] + i * 0.008,
                confidence=min(99, 70 + i * 5),
                frp=round(50 + i * 40, 1),  # increasing FRP
                report_text=f"Fire spreading NE, observation {i+1}/6, ~{(i+1)*50} acres",
                timestamp=ts,
            ))

    elif scenario == "contained":
        # Decreasing FRP, increasing confidence
        for i in range(4):
            ts = (now - timedelta(minutes=60 * (3 - i))).isoformat()
            obs.append(_make_observation(
                source_type="firefighter",
                lat=zone["lat"] + random.uniform(-0.01, 0.01),
                lon=zone["lon"] + random.uniform(-0.01, 0.01),
                confidence=min(99, 80 + i * 5),
                frp=round(max(5, 150 - i * 40), 1),  # decreasing FRP
                report_text=f"Containment progress: {25*(i+1)}% contained, FRP decreasing",
                timestamp=ts,
            ))

    elif scenario == "false_alarm":
        # Low confidence, no FRP, industrial source
        obs.append(_make_observation(
            source_type="firefighter",
            lat=zone["lat"],
            lon=zone["lon"],
            confidence=30,
            frp=None,
            report_text="Investigated — smoke from agricultural burn / industrial site. Not a wildfire.",
        ))

    else:
        print(f"Unknown scenario: {scenario}")
        sys.exit(1)

    return obs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fake field telemetry for testing")
    parser.add_argument("--source", choices=["drone", "firefighter", "ics209"], default="drone")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--region", default="california", help="Use a random location in this region")
    parser.add_argument("--scenario", choices=["spreading", "contained", "false_alarm"], default=None,
                        help="Generate a predefined scenario instead of random data")
    args = parser.parse_args()

    if args.scenario:
        observations = generate_scenario(args.scenario)
        label = args.scenario
    else:
        observations = generate_random(args.source, args.count, args.lat, args.lon)
        label = f"{args.source}_{args.count}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"fake_{label}_{ts}.json"
    filepath = OUTPUT_DIR / filename

    filepath.write_text(json.dumps(observations, indent=2), encoding="utf-8")
    print(f"Generated {len(observations)} observations -> {filepath}")
    for obs in observations:
        print(f"  [{obs['source_type']}] ({obs['latitude']}, {obs['longitude']}) "
              f"conf={obs['confidence']} frp={obs.get('frp', '--')}")


if __name__ == "__main__":
    main()
