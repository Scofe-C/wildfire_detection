"""
Dixie Fire 2021 - OBJ-2 Python Rothermel simulation
====================================================
Simulates fire spread from the Dixie Fire ignition point (Feather River
Canyon, Plumas County, CA) on 2021-07-13 using actual fire-weather
conditions recorded on the day of ignition.

Run from model-pipeline root:
    python run_obj2_dixie.py

Outputs
-------
  - Deterministic single-run result (Rothermel physics)
  - Monte Carlo N=100 (perturbed weather -> burn probabilities)
  - Side-by-side comparison table
  - Saved to: reports/simulations/dixie_2021_mc.json
"""
from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from src.models.obj2_spread.fire_spread_simulator import PythonFireSpreadSimulator

# ---------------------------------------------------------------------------
# Dixie Fire ignition - Feather River Canyon, Plumas County CA
# Coordinates: 40.013 degN, 121.555 degW (near Cresta Dam)
# Date: 2021-07-13 ~14:00 local (21:00 UTC)
# Source: CAL FIRE incident report + RAWS Feather River station
# ---------------------------------------------------------------------------
IGNITION_LAT = 40.013
IGNITION_LON = -121.555

# Fire-weather conditions recorded at RAWS Cresta (nearest station)
# on 2021-07-13 ~14:00 PDT - the afternoon of ignition
WEATHER = {
    # Wind: 20-25 km/h from SW (225 deg) - typical Feather River drainage wind
    "wind_speed_10m":          22.0,    # km/h - Open-Meteo units
    "wind_direction_10m":      225.0,   # degrees (from SW)
    # Extreme heat + low humidity
    "temperature_2m":          41.0,    #  degC (106 degF)
    "relative_humidity_2m":    9.0,     # % - critically dry
    "vpd":                     6.2,     # kPa - very high VPD
    "days_since_last_precipitation": 18.0,  # 18 days dry
}

# Feather River Canyon terrain - steep, east-facing canyon walls
# Slope 30-45 deg typical; aspect ~100-120 deg (east-facing)
TERRAIN = {
    "slope_degrees":  35.0,
    "aspect_degrees": 110.0,    # east-facing -> uphill toward SW (into the wind)
}

# FBFM40 fuel in Sierra Nevada mixed conifer at this elevation (~900m)
# TU5 (Timber Understory 5) - heavy timber with understory, high load
# w0=0.0644 lb/ft², delta=1.0 ft, sigma=1800 1/ft, Mx=0.25
FUEL_CODE = 165   # TU5

# Canopy structure (Sierra Nevada mixed conifer)
# CBH ~5m (base height), CBD ~0.20 kg/m³ (dense canopy)
CANOPY = {
    "canopy_base_height_m":  5.0,
    "canopy_bulk_density":   0.20,
}

# ---------------------------------------------------------------------------
# Build 22km H3 DataFrame
# ---------------------------------------------------------------------------

def _build_dixie_df(ignition_cell: str) -> pd.DataFrame:
    """Build a synthetic 22km H3 DataFrame for the Dixie Fire area."""
    try:
        import h3
    except ImportError:
        raise ImportError("pip install h3")

    # Ignition cell + ring-1 neighbours
    all_cells = list(h3.grid_disk(ignition_cell, 1))

    rows = []
    for cell in all_cells:
        lat, lon = h3.cell_to_latlng(cell)
        # Small terrain variation across neighbours (realistic canyon topology)
        slope_var   = TERRAIN["slope_degrees"]  + np.random.normal(0, 4)
        aspect_var  = (TERRAIN["aspect_degrees"] + np.random.normal(0, 10)) % 360

        row = {
            "grid_id":                       cell,
            "latitude":                      lat,
            "longitude":                     lon,
            # Weather (same for all cells - single synoptic state)
            **WEATHER,
            # Terrain with local variation
            "slope_degrees":                 max(0, slope_var),
            "aspect_degrees":                aspect_var,
            # Fuel
            "fuel_model_fbfm40":             float(FUEL_CODE),
            # Canopy
            **CANOPY,
            # FIRMS fire detection on ignition cell only
            "fire_detected_binary":          1 if cell == ignition_cell else 0,
            "active_fire_count":             3 if cell == ignition_cell else 0,
            "max_confidence":                90 if cell == ignition_cell else 0,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def _latlon_to_h3(lat: float, lon: float, resolution: int = 5) -> str:
    try:
        import h3
        return h3.latlng_to_cell(lat, lon, resolution)
    except ImportError:
        raise ImportError("pip install h3")


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def _bar(prob: float, width: int = 25) -> str:
    filled = int(round(prob * width))
    return "#" * filled + "." * (width - filled)


def _print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _print_deterministic(result: dict) -> None:
    _print_header("DETERMINISTIC  (single Rothermel run)")
    print(f"  ignition cell  : {result['ignition_cell']}")
    print(f"  wind           : {result['inputs_used']['wind_speed_10m_ms']*3.6:.1f} km/h "
          f"from {result['inputs_used']['wind_from_direction_deg']:.0f} deg")
    print(f"  temperature    : {result['inputs_used']['temperature_c']:.1f} degC")
    print(f"  RH             : {result['inputs_used']['relative_humidity_pct']:.1f}%")
    print(f"  DFMC           : {result['dead_fuel_moisture_pct']:.1f}%  "
          f"(extinction={int(result['inputs_used'].get('relative_humidity_pct',0))}%)")
    print()
    print(f"  spread direction : {result['spread_direction_deg']:.1f} deg  "
          f"(wind->{result['wind_spread_direction_deg']:.1f} deg, "
          f"slope->{result.get('slope_spread_direction_deg','n/a')} deg)")
    print(f"  spread speed     : {result['spread_speed_kmh']:.3f} km/h  "
          f"[{result['dominant_factor']} dominated]")
    print(f"  fireline intensity: {result['byram_intensity_kwm']:.1f} kW/m")
    print(f"  crown fire status : {result['crown_fire_status'].upper()}")
    print()
    print("  Per-neighbour spread rates:")
    for nb in sorted(result["neighbour_details"],
                     key=lambda x: x["spread_rate_kmh"], reverse=True):
        kmh  = nb["spread_rate_kmh"]
        hrs  = (25.0 / kmh) if kmh > 0 else float("inf")
        hrs_str = f"{hrs:.1f}h" if hrs < 500 else "never (1h)"
        crown_tag = f" [{nb['crown_status']}]" if nb["crown_status"] != "surface" else ""
        bar   = _bar(min(kmh / 15.0, 1.0), 20)
        print(f"    {nb['neighbour_id']}  {kmh:6.3f} km/h  "
              f"-> arrives {hrs_str:>10}  {bar}{crown_tag}")


def _print_monte_carlo(mc: dict) -> None:
    _print_header(
        f"MONTE CARLO  N={mc['n_simulations']}  "
        f"horizon={mc['horizon_hours']:.0f}h  "
        f"threshold={mc['burn_speed_threshold_kmh']:.2f} km/h"
    )
    cfg = mc["perturbation_config"]["base_weather"]
    print(f"  base wind  : {cfg['wind_speed_kmh']:.1f} km/h  dir={cfg['wind_direction_deg']:.0f} deg")
    print(f"  base RH    : {cfg['relative_humidity_pct']:.1f}%   temp={cfg['temperature_c']:.1f} degC")
    print()
    print(f"  +-- Spread speed distribution ----------------------------------------+")
    print(f"  |  mean : {mc['spread_speed_kmh_mean']:>7.4f} km/h                                       |")
    print(f"  |  p90  : {mc['spread_speed_kmh_p90']:>7.4f} km/h  (severe scenario)                |")
    print(f"  |  p95  : {mc['spread_speed_kmh_p95']:>7.4f} km/h  (extreme scenario)               |")
    print(f"  |  max  : {mc['spread_speed_kmh_max']:>7.4f} km/h  (worst 1 of {mc['n_simulations']} runs)            |")
    print(f"  |  std  : {mc['spread_speed_kmh_std']:>7.4f} km/h                                       |")
    print(f"  +---------------------------------------------------------------------+")
    print()
    print(f"  Direction  : {mc['dominant_direction_deg']:.1f} deg +/- {mc['direction_uncertainty_deg']:.1f} deg")
    print(f"  Crown fire : {mc['crown_fire_probability']:.1%} of runs initiated crown fire")
    print()
    print(f"  Neighbour burn probabilities  (within {mc['horizon_hours']:.0f}h):")
    nb_probs = mc["neighbor_burn_probabilities"]
    sorted_nb = sorted(nb_probs.items(), key=lambda x: x[1], reverse=True)
    for cell_id, prob in sorted_nb:
        bar = _bar(prob, 25)
        print(f"    {cell_id}  {prob:5.1%}  {bar}")


def _print_hybrid(hybrid: dict) -> None:
    _print_header(
        f"HYBRID  (det {int(hybrid['det_weight']*100)}% + MC {int((1-hybrid['det_weight'])*100)}%)"
        f"  -->  RISK: {hybrid['risk_level']}"
    )
    print(f"  hybrid spread speed  : {hybrid['hybrid_spread_speed_kmh']:.4f} km/h")
    print(f"    det component      : {hybrid['det_spread_speed_kmh']:.4f} km/h  x {hybrid['det_weight']:.0%}")
    print(f"    MC mean component  : {hybrid['mc_spread_speed_mean']:.4f} km/h  x {1-hybrid['det_weight']:.0%}")
    print(f"  speed CI (p90)       : {hybrid['speed_ci_high_kmh']:.4f} km/h  (severe-scenario ceiling)")
    print(f"  hybrid direction     : {hybrid['hybrid_spread_direction_deg']:.1f} deg")
    print(f"  crown fire prob      : {hybrid['hybrid_crown_fire_probability']:.1%}")
    print(f"  det crown            : {hybrid['det_crown_fire_status'].upper()}")
    print(f"  MC crown prob        : {hybrid['mc_crown_fire_probability']:.1%}")
    print()
    print(f"  Hybrid burn probabilities per neighbour  (within {hybrid['horizon_hours']:.0f}h):")
    sorted_cells = sorted(
        hybrid["hybrid_burn_probabilities"].items(), key=lambda x: x[1], reverse=True
    )
    mc_probs = hybrid["mc_burn_probabilities"]
    for cell_id, h_prob in sorted_cells:
        bar       = _bar(h_prob, 25)
        mc_p      = mc_probs.get(cell_id, 0.0)
        print(f"    {cell_id}  hybrid={h_prob:5.1%}  mc={mc_p:5.1%}  {bar}")


def _print_comparison(det: dict, mc: dict, hybrid: dict) -> None:
    _print_header("DETERMINISTIC  vs  MONTE CARLO p90  vs  HYBRID")
    d_speed = det["spread_speed_kmh"]
    h_speed = hybrid["hybrid_spread_speed_kmh"]
    print(f"  {'Metric':<30} {'Deterministic':>14}  {'MC p90':>10}  {'HYBRID':>10}")
    print(f"  {'-'*30} {'-'*14}  {'-'*10}  {'-'*10}")
    print(f"  {'Spread speed (km/h)':<30} {d_speed:>14.4f}  "
          f"{mc['spread_speed_kmh_p90']:>10.4f}  "
          f"{h_speed:>10.4f}")
    print(f"  {'Direction (deg)':<30} {det['spread_direction_deg']:>14.1f}  "
          f"{'+/-'+str(round(mc['direction_uncertainty_deg'],1)):>10}  "
          f"{hybrid['hybrid_spread_direction_deg']:>10.1f}")
    print(f"  {'Byram intensity (kW/m)':<30} {det['byram_intensity_kwm']:>14.1f}  "
          f"{'n/a':>10}  {'n/a':>10}")
    det_crown_str = 'YES' if det['crown_fire_status'] != 'surface' else 'NO'
    print(f"  {'Crown fire':<30} {det_crown_str:>14}  "
          f"{mc['crown_fire_probability']:>10.1%}  "
          f"{hybrid['hybrid_crown_fire_probability']:>10.1%}")
    print(f"  {'Max burn probability':<30} {'n/a':>14}  "
          f"{mc['max_neighbor_burn_probability']:>10.1%}  "
          f"{hybrid['max_hybrid_burn_probability']:>10.1%}")
    print(f"  {'Risk level':<30} {'n/a':>14}  {'n/a':>10}  "
          f"{hybrid['risk_level']:>10}")
    print()
    d_hrs   = 25.0 / d_speed  if d_speed  > 0 else float("inf")
    p90_hrs = 25.0 / mc["spread_speed_kmh_p90"] if mc["spread_speed_kmh_p90"] > 0 else float("inf")
    h_hrs   = 25.0 / h_speed  if h_speed  > 0 else float("inf")
    print(f"  Time to reach nearest neighbour (25 km):")
    print(f"    Deterministic : {d_hrs:.1f} h")
    print(f"    MC p90        : {p90_hrs:.1f} h")
    print(f"    Hybrid        : {h_hrs:.1f} h")
    print()
    nb_mc     = mc["neighbor_burn_probabilities"]
    nb_hybrid = hybrid["hybrid_burn_probabilities"]
    n_mc_nz     = sum(1 for p in nb_mc.values()     if p > 0)
    n_hybrid_nz = sum(1 for p in nb_hybrid.values() if p > 0)
    print(f"  Cells with non-zero burn probability:")
    print(f"    MC     : {n_mc_nz}/{len(nb_mc)}")
    print(f"    Hybrid : {n_hybrid_nz}/{len(nb_hybrid)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(0)   # reproducible terrain variation

    log.info("Dixie Fire 2021 - OBJ-2 Python Rothermel simulation")
    log.info("Ignition: %.4f degN  %.4f degW  (Feather River Canyon)", IGNITION_LAT, IGNITION_LON)
    log.info("Conditions: wind %.0f km/h from %.0f deg, temp %.0f degC, RH %.0f%%, VPD %.1f kPa",
             WEATHER["wind_speed_10m"], WEATHER["wind_direction_10m"],
             WEATHER["temperature_2m"], WEATHER["relative_humidity_2m"], WEATHER["vpd"])

    # Resolve ignition H3 cell (res-5 = 22km)
    ignition_cell = _latlon_to_h3(IGNITION_LAT, IGNITION_LON, resolution=5)
    log.info("H3 ignition cell (res-5): %s", ignition_cell)

    # Build DataFrame
    df = _build_dixie_df(ignition_cell)
    log.info("DataFrame: %d cells", len(df))

    sim = PythonFireSpreadSimulator()

    # ── Step 1: Deterministic ────────────────────────────────────────────────
    log.info("Running deterministic simulation ...")
    det = sim.simulate(df, ignition_cell, ignition_prob=0.85)

    # ── Step 2: Monte Carlo N=100 ────────────────────────────────────────────
    log.info("Running Monte Carlo N=100 ...")
    mc = sim.simulate_monte_carlo(
        df, ignition_cell,
        ignition_prob=0.85,
        n_simulations=100,
        horizon_hours=24.0,
        rng_seed=42,
    )

    # ── Step 3: Hybrid (det=40% + MC=60%) ───────────────────────────────────
    log.info("Running Hybrid (det_weight=0.4, N=100) ...")
    hybrid = sim.simulate_hybrid(
        df, ignition_cell,
        ignition_prob=0.85,
        n_simulations=100,
        horizon_hours=24.0,
        det_weight=0.4,
        rng_seed=42,
    )

    # ── Print results ────────────────────────────────────────────────────────
    _print_deterministic(det)
    _print_monte_carlo(mc)
    _print_hybrid(hybrid)
    _print_comparison(det, mc, hybrid)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = Path("reports/simulations")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dixie_2021_hybrid.json"
    with open(out_path, "w") as f:
        json.dump({
            "fire":          "Dixie Fire 2021",
            "ignition_lat":  IGNITION_LAT,
            "ignition_lon":  IGNITION_LON,
            "ignition_cell": ignition_cell,
            "weather":       WEATHER,
            "terrain":       TERRAIN,
            "fuel_code":     FUEL_CODE,
            "deterministic": {k: det[k] for k in [
                "spread_direction_deg", "spread_speed_kmh",
                "dead_fuel_moisture_pct", "byram_intensity_kwm",
                "crown_fire_status", "dominant_factor", "neighbour_details",
            ]},
            "monte_carlo": {k: mc[k] for k in [
                "n_simulations", "horizon_hours",
                "spread_speed_kmh_mean", "spread_speed_kmh_p50",
                "spread_speed_kmh_p90", "spread_speed_kmh_p95",
                "spread_speed_kmh_max", "spread_speed_kmh_std",
                "dominant_direction_deg", "direction_uncertainty_deg",
                "crown_fire_probability", "neighbor_burn_probabilities",
                "max_neighbor_burn_probability", "perturbation_config",
            ]},
            "hybrid": hybrid,
        }, f, indent=2)
    log.info("Saved -> %s", out_path)
    print(f"\nReport saved: {out_path}\n")


if __name__ == "__main__":
    main()
