"""
test_physics_all_fires.py  —  Multi-Fire Physics Validation Suite
===================================================================
Validates the Rothermel-based simulator against 5 well-documented
historical wildfires spanning all major fuel types, wind regimes,
and terrain conditions.

Fires covered:
  1. Palisades Fire   (Jan 7 2025, LA)        — SH7 chaparral, Santa Ana
  2. Camp Fire        (Nov 8 2018, Pulga CA)   — TU5 timber, Diablo wind
  3. Creek Fire       (Sep 5 2020, Sierra)     — TU5/TL9 timber, Mono wind
  4. Thomas Fire      (Dec 4 2017, Ventura)    — SH7/SH9 chaparral, Santa Ana
  5. Carr Fire        (Jul 23 2018, Redding)   — GR9/TU5 grass-timber, NW wind

Run from:  model-pipeline/
    python test_physics_all_fires.py
"""
import sys
sys.path.insert(0, "src")

import math
import h3
import pandas as pd
from models.obj2_spread.fire_spread_simulator import PythonFireSpreadSimulator

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build synthetic DataFrame for a single fire scenario
# ─────────────────────────────────────────────────────────────────────────────

def make_df(lat, lon, conditions: dict) -> tuple:
    """Return (df, ignition_id) with ignition + ring-1 neighbours."""
    ignition_id = h3.latlng_to_cell(lat, lon, 2)
    neighbours = sorted(set(h3.grid_disk(ignition_id, 1)) - {ignition_id})
    all_ids = [ignition_id] + neighbours
    n = len(all_ids)

    rows = {col: [val] * n for col, val in conditions.items()}
    df = pd.DataFrame({"grid_id": all_ids, **rows})
    for i, cell_id in enumerate(all_ids):
        clat, clon = h3.cell_to_latlng(cell_id)
        df.loc[i, "latitude"] = clat
        df.loc[i, "longitude"] = clon
    return df, ignition_id


def angular_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


# ─────────────────────────────────────────────────────────────────────────────
# Fire scenario definitions
# ─────────────────────────────────────────────────────────────────────────────

FIRES = [
    # ── 1. Palisades Fire (Jan 7 2025) ──────────────────────────────────────
    {
        "name": "Palisades Fire (Jan 7 2025, LA)",
        "lat": 34.05, "lon": -118.52,
        "ignition_prob": 0.90,
        "conditions": {
            "wind_speed_10m":            90.0,   # km/h — Santa Ana gusts
            "wind_direction_10m":        65.0,   # ° ENE
            "relative_humidity_2m":      8.0,    # % extremely dry
            "temperature_2m":            24.0,   # °C
            "days_since_last_precipitation": 45.0,
            "fuel_model_fbfm40":         147.0,  # SH7 chaparral
            "canopy_base_height_m":      0.8,
            "canopy_bulk_density":       0.08,
            "slope_degrees":             20.0,
            "aspect_degrees":            225.0,
            "canopy_cover_pct":          60.0,
            "elevation_m":               250.0,
            "vpd":                       5.5,
            "active_fire_count":         0,
        },
        "expected_spread_dir":  245.0,   # WSW (downwind from ENE 65°)
        "checks": {
            "Direction ±30° of WSW 245°":    lambda r: angular_diff(r["spread_direction_deg"], 245.0) <= 30.0,
            "Speed 2–20 km/h":               lambda r: 2.0 <= r["spread_speed_kmh"] <= 20.0,
            "Dead fuel moisture < 10%":      lambda r: r["dead_fuel_moisture_pct"] <= 10.0,
            "Crown fire (low CBH+extreme wind)": lambda r: r["crown_fire_status"] in ("passive_crown", "active_crown"),
            "Byram intensity > 1000 kW/m":   lambda r: r["byram_intensity_kwm"] > 1000,
        },
    },

    # ── 2. Camp Fire (Nov 8 2018, Pulga CA) ─────────────────────────────────
    {
        "name": "Camp Fire (Nov 8 2018, Pulga CA)",
        "lat": 39.810, "lon": -121.470,
        "ignition_prob": 0.90,
        "conditions": {
            "wind_speed_10m":            85.0,   # km/h — Diablo gusts
            "wind_direction_10m":        15.0,   # ° NNE (Diablo)
            "relative_humidity_2m":      23.0,   # %
            "temperature_2m":            11.0,   # °C
            "days_since_last_precipitation": 10.0,
            "fuel_model_fbfm40":         165.0,  # TU5 timber understory
            "canopy_base_height_m":      3.0,
            "canopy_bulk_density":       0.12,
            "slope_degrees":             28.0,
            "aspect_degrees":            220.0,  # SW-facing canyon
            "canopy_cover_pct":          75.0,
            "elevation_m":               800.0,
            "vpd":                       2.5,
            "active_fire_count":         0,
        },
        "expected_spread_dir":  195.0,   # SSW
        "checks": {
            "Direction ±30° of SSW 195°":    lambda r: angular_diff(r["spread_direction_deg"], 195.0) <= 30.0,
            "Speed 4–15 km/h":               lambda r: 4.0 <= r["spread_speed_kmh"] <= 15.0,
            "Dead fuel moisture 3–10%":      lambda r: 3.0 <= r["dead_fuel_moisture_pct"] <= 10.0,
            "Crown fire (timber+Diablo wind)":   lambda r: r["crown_fire_status"] in ("passive_crown", "active_crown"),
            "Byram intensity > 2000 kW/m":   lambda r: r["byram_intensity_kwm"] > 2000,
        },
    },

    # ── 3. Creek Fire (Sep 5 2020, Sierra Nevada) ────────────────────────────
    # Largest single-ignition fire in CA history at the time.
    # Conditions: Mono Wind event, extreme dry, Sierra mixed conifer.
    # Ignition near Shaver Lake, rapid spread NW under NE/ENE wind.
    {
        "name": "Creek Fire (Sep 5 2020, Shaver Lake CA)",
        "lat": 37.10, "lon": -119.30,
        "ignition_prob": 0.85,
        "conditions": {
            "wind_speed_10m":            70.0,   # km/h — Mono wind gusts
            "wind_direction_10m":        55.0,   # ° NE (Mono wind from Great Basin)
            "relative_humidity_2m":      10.0,   # % late summer drought
            "temperature_2m":            32.0,   # °C — hot September
            "days_since_last_precipitation": 60.0,  # long dry season
            "fuel_model_fbfm40":         165.0,  # TU5 — Sierra mixed conifer
            "canopy_base_height_m":      4.0,
            "canopy_bulk_density":       0.15,
            "slope_degrees":             22.0,
            "aspect_degrees":            235.0,  # SW-facing
            "canopy_cover_pct":          70.0,
            "elevation_m":               1500.0,
            "vpd":                       4.5,
            "active_fire_count":         0,
        },
        "expected_spread_dir":  235.0,   # SW (downwind from NE 55°)
        "checks": {
            "Direction ±35° of SW 235°":     lambda r: angular_diff(r["spread_direction_deg"], 235.0) <= 35.0,
            "Speed 3–18 km/h":               lambda r: 3.0 <= r["spread_speed_kmh"] <= 18.0,
            "Dead fuel moisture 1–8%":       lambda r: 1.0 <= r["dead_fuel_moisture_pct"] <= 8.0,
            "Crown fire (dry timber+hot)":   lambda r: r["crown_fire_status"] in ("passive_crown", "active_crown"),
            "Byram intensity > 2000 kW/m":   lambda r: r["byram_intensity_kwm"] > 2000,
        },
    },

    # ── 4. Thomas Fire (Dec 4 2017, Ventura CA) ──────────────────────────────
    # Largest CA fire at the time. Strong Santa Ana, dry chaparral.
    # Ignition near Thomas Aquinas College, Ventura County.
    {
        "name": "Thomas Fire (Dec 4 2017, Ventura CA)",
        "lat": 34.42, "lon": -118.88,
        "ignition_prob": 0.88,
        "conditions": {
            "wind_speed_10m":            95.0,   # km/h — extreme Santa Ana
            "wind_direction_10m":        60.0,   # ° ENE
            "relative_humidity_2m":      6.0,    # % extremely dry
            "temperature_2m":            26.0,   # °C
            "days_since_last_precipitation": 50.0,
            "fuel_model_fbfm40":         149.0,  # SH9 high-load chaparral
            "canopy_base_height_m":      1.0,
            "canopy_bulk_density":       0.09,
            "slope_degrees":             25.0,
            "aspect_degrees":            240.0,  # SW-facing
            "canopy_cover_pct":          65.0,
            "elevation_m":               350.0,
            "vpd":                       6.0,
            "active_fire_count":         0,
        },
        "expected_spread_dir":  240.0,   # WSW (downwind from ENE 60°)
        "checks": {
            "Direction ±30° of WSW 240°":    lambda r: angular_diff(r["spread_direction_deg"], 240.0) <= 30.0,
            "Speed 3–20 km/h":               lambda r: 3.0 <= r["spread_speed_kmh"] <= 20.0,
            "Dead fuel moisture < 8%":       lambda r: r["dead_fuel_moisture_pct"] <= 8.0,
            "Crown fire or high-intensity surface": lambda r: r["crown_fire_status"] in ("passive_crown", "active_crown"),
            "Byram intensity > 1500 kW/m":   lambda r: r["byram_intensity_kwm"] > 1500,
        },
    },

    # ── 5. Carr Fire (Jul 23 2018, Redding CA) ──────────────────────────────
    # Extreme fire whirl ("fire tornado") event. Grassland/timber interface.
    # Near Whiskeytown Lake, Shasta County. NW wind, hot dry afternoon.
    {
        "name": "Carr Fire (Jul 23 2018, Redding CA)",
        "lat": 40.65, "lon": -122.55,
        "ignition_prob": 0.85,
        "conditions": {
            "wind_speed_10m":            55.0,   # km/h — NW afternoon wind
            "wind_direction_10m":        330.0,  # ° NW
            "relative_humidity_2m":      12.0,   # % hot dry summer
            "temperature_2m":            38.0,   # °C — extreme heat
            "days_since_last_precipitation": 55.0,
            "fuel_model_fbfm40":         109.0,  # GR9 high-load grass (valley/interface)
            "canopy_base_height_m":      2.5,
            "canopy_bulk_density":       0.07,
            "slope_degrees":             15.0,
            "aspect_degrees":            150.0,  # SE-facing
            "canopy_cover_pct":          30.0,
            "elevation_m":               200.0,
            "vpd":                       5.8,
            "active_fire_count":         0,
        },
        "expected_spread_dir":  150.0,   # SSE (downwind from NW 330°)
        "checks": {
            "Direction ±35° of SSE 150°":    lambda r: angular_diff(r["spread_direction_deg"], 150.0) <= 35.0,
            "Speed 3–25 km/h":               lambda r: 3.0 <= r["spread_speed_kmh"] <= 25.0,
            "Dead fuel moisture 1–8%":       lambda r: 1.0 <= r["dead_fuel_moisture_pct"] <= 8.0,
            "Non-zero spread (burnable grass)": lambda r: r["spread_speed_kmh"] > 0,
            "Byram intensity > 800 kW/m":    lambda r: r["byram_intensity_kwm"] > 800,
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Run all fire scenarios
# ─────────────────────────────────────────────────────────────────────────────

sim = PythonFireSpreadSimulator()

total_checks = 0
total_passed = 0
fire_results = []

print("\n" + "=" * 66)
print("  MULTI-FIRE PHYSICS VALIDATION SUITE")
print("=" * 66)

for fire in FIRES:
    df, ignition_id = make_df(fire["lat"], fire["lon"], fire["conditions"])
    result = sim.simulate(df, ignition_grid_id=ignition_id,
                          ignition_prob=fire["ignition_prob"])

    print(f"\n{'─'*66}")
    print(f"  {fire['name']}")
    print(f"{'─'*66}")
    print(f"  spread_direction : {result['spread_direction_deg']:.1f}°  "
          f"(expected ~{fire['expected_spread_dir']:.0f}°)")
    print(f"  spread_speed     : {result['spread_speed_kmh']:.3f} km/h")
    print(f"  dead_fuel_moist  : {result['dead_fuel_moisture_pct']:.1f}%")
    print(f"  byram_intensity  : {result['byram_intensity_kwm']:.1f} kW/m")
    print(f"  crown_fire       : {result['crown_fire_status']}")
    print()

    fire_pass = True
    for desc, check_fn in fire["checks"].items():
        passed = check_fn(result)
        icon = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {icon}  {desc}")
        total_checks += 1
        total_passed += (1 if passed else 0)
        fire_pass = fire_pass and passed

    fire_results.append((fire["name"], fire_pass))

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 66)
print(f"  RESULTS: {total_passed}/{total_checks} checks passed across {len(FIRES)} fires")
print("=" * 66)

all_pass = True
for name, passed in fire_results:
    icon = "✅" if passed else "❌"
    print(f"  {icon}  {name}")
    all_pass = all_pass and passed

print()
if all_pass:
    print("  ✅ ALL FIRES PASSED — Rothermel physics validated across fuel types")
else:
    print("  ❌ SOME FIRES FAILED — review formulas or PASS criteria")
print("=" * 66 + "\n")
