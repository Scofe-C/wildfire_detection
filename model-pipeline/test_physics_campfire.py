"""
test_physics_campfire.py  —  Physics Validation: Camp Fire (Nov 8 2018)
========================================================================
The Camp Fire is the most precisely documented California wildfire and the
best physics benchmark available.

Known historical conditions (CAL FIRE investigation / CPUC report):
  - Ignition:    ~06:15 AM, Pulga, CA  (lat=39.810, lon=-121.470)
  - Wind:        NNE ~15°, gusts 80–96 km/h (Diablo wind event)
  - RH:          23% (low but NOT extreme — November morning)
  - Temp:        11°C (cool November morning)
  - Fuel:        TU5 (timber understory, code 165) — dominant Feather River Canyon
  - Slope:       28° (Feather River Canyon ridge terrain)
  - Aspect:      220° (SW-facing canyon wall — wind-aligned downslope)
  - Days dry:    ~10 (last rain early Nov 2018)

Historical fire behavior (verified from CAL FIRE report):
  - Spread direction: SSW ~195° (downwind from NNE 15° wind)
  - Spread to Paradise (~11 km): reached in ~90 minutes → HEAD FIRE ~7.3 km/h
  - Crown fire:  YES — passive/active crown in timber fuel under extreme wind

PASS criteria (stricter than Palisades — Camp Fire is better documented):
  - Direction : 165–225° (SSW ± 30°)
  - Speed     : 4.0–15.0 km/h (head fire ~7.3 km/h avg; peak crown fire ~13 km/h)
  - Moisture  : 3–10%  (1-hr EMC at RH=23%, 11°C — ~4.8% by Simard/Nelson formula)
  - Crown fire: passive_crown or active_crown
  - Byram     : > 2000 kW/m (timber fire — higher intensity than chaparral)

Run from:  model-pipeline/
    python test_physics_campfire.py
"""
import sys
sys.path.insert(0, "src")

import h3
import pandas as pd
from models.obj2_spread.fire_spread_simulator import PythonFireSpreadSimulator

# ── Ignition cell — Pulga, CA (Camp Fire origin) ─────────────────────────────
IGNITION_LAT, IGNITION_LON = 39.810, -121.470
REAL_IGNITION_ID = h3.latlng_to_cell(IGNITION_LAT, IGNITION_LON, 2)   # H3 res-2 (64km)

# Ring-1 neighbours
real_neighbours = sorted(set(h3.grid_disk(REAL_IGNITION_ID, 1)) - {REAL_IGNITION_ID})
all_ids = [REAL_IGNITION_ID] + real_neighbours

# ── Build synthetic DataFrame with Camp Fire conditions ───────────────────────
conditions = {
    # Weather — Diablo wind event, Nov 8 2018 06:15 AM
    "wind_speed_10m":            [85.0]  * len(all_ids),  # km/h — NNE Diablo gusts
    "wind_direction_10m":        [15.0]  * len(all_ids),  # ° — NNE (Diablo typical)
    "relative_humidity_2m":      [23.0]  * len(all_ids),  # % — low but not extreme
    "temperature_2m":            [11.0]  * len(all_ids),  # °C — cool November morning
    "days_since_last_precipitation": [10.0] * len(all_ids),

    # Fuel — TU5 (code 165): Timber understory, Feather River Canyon conifers
    "fuel_model_fbfm40":         [165.0] * len(all_ids),

    # Canopy — Ponderosa pine / mixed conifer (crown fire expected)
    "canopy_base_height_m":      [3.0]   * len(all_ids),  # m — elevated canopy
    "canopy_bulk_density":       [0.12]  * len(all_ids),  # kg/m³ — dense conifer

    # Terrain — Feather River Canyon ridge, SW-facing
    "slope_degrees":             [28.0]  * len(all_ids),  # ° — steep canyon
    "aspect_degrees":            [220.0] * len(all_ids),  # ° — SW-facing (downwind)

    # Supporting columns
    "canopy_cover_pct":          [75.0]  * len(all_ids),
    "elevation_m":               [800.0] * len(all_ids),
    "vpd":                       [2.5]   * len(all_ids),
    "active_fire_count":         [0]     * len(all_ids),
}

df = pd.DataFrame({"grid_id": all_ids, **conditions})

# Set accurate lat/lon for each H3 cell
for i, cell_id in enumerate(all_ids):
    lat, lon = h3.cell_to_latlng(cell_id)
    df.loc[i, "latitude"]  = lat
    df.loc[i, "longitude"] = lon

# ── Run simulator ─────────────────────────────────────────────────────────────
sim    = PythonFireSpreadSimulator()
result = sim.simulate(df, ignition_grid_id=REAL_IGNITION_ID, ignition_prob=0.90)

# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "="*62)
print("  CAMP FIRE PHYSICS VALIDATION  (Nov 8 2018, Pulga CA)")
print("="*62)
print(f"  Ignition cell         : {REAL_IGNITION_ID}")
print(f"  Ignition location     : {IGNITION_LAT}°N, {IGNITION_LON}°W (Pulga, CA)")
print(f"  Wind                  : 85 km/h from 15° (NNE) — Diablo wind")
print(f"  RH                    : 23%  |  Temp: 11°C  |  Days dry: 10")
print(f"  Fuel                  : TU5 (code 165) — Feather River Canyon timber")
print(f"  Slope / Aspect        : 28° / 220° SW-facing canyon")
print()
print(f"  ── Simulator Output ──────────────────────────────────")
print(f"  spread_direction_deg  : {result['spread_direction_deg']:.1f}°")
print(f"  spread_speed_kmh      : {result['spread_speed_kmh']:.3f} km/h")
print(f"  dead_fuel_moisture    : {result['dead_fuel_moisture_pct']:.1f}%")
print(f"  byram_intensity_kwm   : {result['byram_intensity_kwm']:.1f} kW/m")
print(f"  crown_fire_status     : {result['crown_fire_status']}")
print()
print(f"  ── Historical Record ─────────────────────────────────")
print(f"  Expected direction    : ~195° (SSW, downwind from NNE)")
print(f"  Expected speed        : 4–15 km/h (avg ~7.3 km/h; peak active crown ~13 km/h)")
print(f"  Expected crown fire   : passive_crown or active_crown")
print(f"  Expected intensity    : > 2000 kW/m (timber)")
print()

# ── PASS / FAIL checks ────────────────────────────────────────────────────────
WIND_DIR           = 15.0
EXPECTED_SPREAD    = (WIND_DIR + 180) % 360   # 195° SSW

def angular_diff(a, b):
    return abs((a - b + 180) % 360 - 180)

checks = {
    "Direction within ±30° of SSW (195°)":
        angular_diff(result["spread_direction_deg"], EXPECTED_SPREAD) <= 30.0,

    "Speed in documented range 4.0–15.0 km/h":
        4.0 <= result["spread_speed_kmh"] <= 15.0,

    "Dead fuel moisture 3–10% (1-hr EMC at RH=23%, 11°C)":
        3.0 <= result["dead_fuel_moisture_pct"] <= 10.0,

    "Crown fire triggered (conifer + extreme wind)":
        result["crown_fire_status"] in ("passive_crown", "active_crown"),

    "Byram intensity > 2000 kW/m (dense timber)":
        result["byram_intensity_kwm"] > 2000,
}

print("  VALIDATION CHECKS vs Camp Fire historical record:")
print("  " + "-"*57)
all_pass = True
for desc, passed in checks.items():
    icon     = "✅ PASS" if passed else "❌ FAIL"
    all_pass = all_pass and passed
    print(f"  {icon}  {desc}")

print()

# Side-by-side comparison table
print("  COMPARISON TABLE:")
print(f"  {'Metric':<28} {'Simulated':>12}  {'Historical':>16}")
print("  " + "-"*58)
print(f"  {'Spread direction (°)':<28} {result['spread_direction_deg']:>11.1f}°  {'~195° (SSW)':>16}")
print(f"  {'Spread speed (km/h)':<28} {result['spread_speed_kmh']:>11.3f}   {'~7.3 km/h':>16}")
print(f"  {'Dead fuel moisture (%)':<28} {result['dead_fuel_moisture_pct']:>11.1f}%  {'~12–18%':>16}")
print(f"  {'Crown fire':<28} {result['crown_fire_status']:>12}  {'passive/active':>16}")
print()

if all_pass:
    print("  ✅ ALL CHECKS PASSED — Rothermel physics validated for Camp Fire")
else:
    print("  ❌ SOME CHECKS FAILED — review formulas or input conditions")
    # Helpful diagnostics
    if result["spread_speed_kmh"] < 4.0:
        print("  ℹ  Speed too low — check fuel moisture or wind conversion")
    elif result["spread_speed_kmh"] > 12.0:
        print("  ℹ  Speed too high — possible unit issue or slope overcorrection")
    if result["crown_fire_status"] == "surface":
        print("  ℹ  Crown fire not triggered — check CBH/CBD values and Byram intensity")

print("="*62 + "\n")
