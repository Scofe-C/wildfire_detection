"""
test_pipeline_integration.py  —  Test 2: Pipeline Integration
==============================================================
Verifies the full path:
  fused parquet → enrich CBH/CBD → OBJ-1 mock → simulate → save JSON + CSV

Checks:
  1. Fused parquet loads with all critical columns present
  2. Static features (CBH/CBD) merge in correctly
  3. OBJ-1 mock: top fire-risk cell (highest fire_weather_index or lowest RH)
  4. Simulator runs end-to-end without errors
  5. Output is non-zero (spread_speed_kmh > 0)
  6. JSON + CSV saved correctly with expected structure
  7. Both 64km and 22km resolutions work (22km skipped if parquet not found)

Run from:  model-pipeline/
    python test_pipeline_integration.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import pandas as pd

ROOT          = Path(__file__).resolve().parent.parent
DATA_PIPELINE = ROOT / "Data-Pipeline"

PARQUET_64KM  = DATA_PIPELINE / "data/processed/fused/64km/region=california/year=2026/month=03/fused_2026-03-31.parquet"
STATIC_64KM   = DATA_PIPELINE / "data/static/static_features_64km.parquet"
STATIC_22KM   = DATA_PIPELINE / "data/static/static_features_22km.parquet"
OUT_DIR       = Path("simulation_output")
OUT_DIR.mkdir(exist_ok=True)

CRITICAL_COLS = [
    "grid_id", "latitude", "longitude",
    "wind_speed_10m", "wind_direction_10m",
    "relative_humidity_2m", "temperature_2m",
    "fuel_model_fbfm40", "slope_degrees", "aspect_degrees",
]

results = {}   # check_name → (passed: bool, detail: str)

def check(name, passed, detail=""):
    icon = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {icon}  {name}")
    if detail:
        print(f"         {detail}")
    results[name] = passed

def enrich_static(df, static_path):
    """Merge CBH/CBD from static parquet if missing."""
    if not static_path.exists():
        return df, False
    st = pd.read_parquet(static_path)
    st["grid_id"] = st["grid_id"].astype(str)
    cols = [c for c in ["canopy_base_height_m","canopy_bulk_density","evt_national_class"]
            if c in st.columns]
    df = df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")
    df = df.merge(st[["grid_id"] + cols], on="grid_id", how="left")
    return df, True

def run_test(label, parquet_path, static_path, resolution_km):
    print(f"\n{'─'*60}")
    print(f"  {label}  ({resolution_km}km)")
    print(f"{'─'*60}")

    # ── Check 1: parquet loads ───────────────────────────────────────────────
    if not parquet_path.exists():
        check(f"[{label}] Parquet file exists", False, str(parquet_path))
        print(f"  ⚠  Skipping remaining checks for {label}")
        return

    df = pd.read_parquet(parquet_path)
    df["grid_id"] = df["grid_id"].astype(str)
    check(f"[{label}] Parquet loads", True, f"{len(df)} rows × {len(df.columns)} cols")

    # ── Check 2: critical columns present ────────────────────────────────────
    missing = [c for c in CRITICAL_COLS if c not in df.columns]
    check(f"[{label}] All critical columns present",
          len(missing) == 0,
          f"Missing: {missing}" if missing else "All present")

    # ── Check 3: static enrichment (CBH/CBD) ─────────────────────────────────
    df, enriched = enrich_static(df, static_path)
    cbh_valid = df["canopy_base_height_m"].notna().sum() if "canopy_base_height_m" in df.columns else 0
    cbd_valid = df["canopy_bulk_density"].notna().sum()   if "canopy_bulk_density"   in df.columns else 0
    check(f"[{label}] Static parquet found for enrichment", enriched,
          str(static_path) if not enriched else f"CBH valid={cbh_valid} | CBD valid={cbd_valid}")

    # ── Check 4: OBJ-1 mock — pick highest fire-risk cell ────────────────────
    valid = df[df["fuel_model_fbfm40"].notna() & (df["fuel_model_fbfm40"] != -9999)].copy()
    check(f"[{label}] Valid fuel cells found", len(valid) > 0, f"{len(valid)} cells with fuel data")

    if len(valid) == 0:
        print(f"  ⚠  No valid fuel cells — skipping simulation for {label}")
        return

    # Pick top fire-risk cell: lowest RH (dryest = most dangerous)
    # This simulates what OBJ-1 would output as top prediction
    if "relative_humidity_2m" in valid.columns and valid["relative_humidity_2m"].notna().any():
        top_cell = valid.sort_values("relative_humidity_2m").iloc[0]
    else:
        top_cell = valid.iloc[0]

    ignition_id   = str(top_cell["grid_id"])
    ignition_prob = 0.85   # mocked OBJ-1 output

    print(f"\n  OBJ-1 mock (top fire-risk cell):")
    print(f"    grid_id          = {ignition_id}")
    print(f"    ignition_prob    = {ignition_prob:.0%}")
    print(f"    fuel_model       = {top_cell.get('fuel_model_fbfm40', 'N/A')}")
    print(f"    wind_speed_10m   = {top_cell.get('wind_speed_10m', 'N/A')} km/h")
    print(f"    wind_dir         = {top_cell.get('wind_direction_10m', 'N/A')}°")
    print(f"    relative_humidity= {top_cell.get('relative_humidity_2m', 'N/A')}%")
    print(f"    temperature      = {top_cell.get('temperature_2m', 'N/A')}°C")
    print(f"    CBH              = {top_cell.get('canopy_base_height_m', 'N/A')} m")
    print(f"    CBD              = {top_cell.get('canopy_bulk_density', 'N/A')} kg/m³")

    # ── Check 5: simulator runs without error ─────────────────────────────────
    from models.obj2_spread.fire_spread_simulator import PythonFireSpreadSimulator
    try:
        sim    = PythonFireSpreadSimulator()
        result = sim.simulate(df, ignition_grid_id=ignition_id, ignition_prob=ignition_prob)
        check(f"[{label}] Simulator runs without error", True)
    except Exception as e:
        check(f"[{label}] Simulator runs without error", False, str(e))
        return

    # ── Check 6: non-zero spread ──────────────────────────────────────────────
    speed = result.get("spread_speed_kmh", 0.0)
    check(f"[{label}] Non-zero spread speed",
          speed > 0.0,
          f"spread_speed_kmh = {speed:.4f}  (0 = no fuel or extreme moisture)")

    print(f"\n  Simulation results:")
    print(f"    spread_direction  = {result['spread_direction_deg']:.1f}°")
    print(f"    spread_speed      = {result['spread_speed_kmh']:.3f} km/h")
    print(f"    fuel moisture     = {result['dead_fuel_moisture_pct']:.1f}%")
    print(f"    byram intensity   = {result['byram_intensity_kwm']:.1f} kW/m")
    print(f"    crown fire status = {result['crown_fire_status']}")
    print(f"    threatened cells  = {len(result.get('neighbour_details', []))}")

    # ── Check 7: save JSON + CSV ──────────────────────────────────────────────
    stem = f"test_{resolution_km}km_{ignition_id[:12]}"

    json_path = OUT_DIR / f"{stem}.json"
    json_out = {
        "resolution_km":   resolution_km,
        "obj1_input":      {"grid_id": ignition_id, "prob": ignition_prob},
        "summary": {
            "spread_direction_deg":   result["spread_direction_deg"],
            "spread_speed_kmh":       result["spread_speed_kmh"],
            "dead_fuel_moisture_pct": result["dead_fuel_moisture_pct"],
            "byram_intensity_kwm":    result["byram_intensity_kwm"],
            "crown_fire_status":      result["crown_fire_status"],
        },
        "threatened_cells": result.get("neighbour_details", []),
    }
    try:
        with open(json_path, "w") as f:
            json.dump(json_out, f, indent=2, default=str)
        json_ok = True
    except Exception as e:
        json_ok = False

    rows = result.get("neighbour_details") or \
           [{"note": "zero spread", **json_out["summary"]}]
    csv_path = OUT_DIR / f"{stem}.csv"
    try:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        csv_ok = True
    except Exception as e:
        csv_ok = False

    check(f"[{label}] JSON saved correctly", json_ok and json_path.exists(),
          str(json_path))
    check(f"[{label}] CSV saved correctly",  csv_ok  and csv_path.exists(),
          str(csv_path))

    # Check JSON has expected keys
    if json_path.exists():
        with open(json_path) as f:
            loaded = json.load(f)
        has_keys = all(k in loaded for k in ["obj1_input", "summary", "threatened_cells"])
        check(f"[{label}] JSON has correct structure", has_keys)


# ══════════════════════════════════════════════════════════════════════════════
# RUN BOTH RESOLUTIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  TEST 2: PIPELINE INTEGRATION TEST")
print("="*60)

# 64km — existing parquet
run_test("64km", PARQUET_64KM, STATIC_64KM, 64)

# 22km — search for the latest parquet if exists
parquet_22km_candidates = list(
    (DATA_PIPELINE / "data/processed/fused/22km/region=california").glob("**/fused_*.parquet")
) if (DATA_PIPELINE / "data/processed/fused/22km/region=california").exists() else []

if parquet_22km_candidates:
    latest_22km = sorted(parquet_22km_candidates)[-1]
    run_test("22km", latest_22km, STATIC_22KM, 22)
else:
    print(f"\n{'─'*60}")
    print("  22km  — SKIPPED")
    print("  No 22km fused parquet found.")
    print("  Run the pipeline at 22km resolution first to test this.")
    print(f"{'─'*60}")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
passed = sum(1 for v in results.values() if v)
total  = len(results)
print(f"  RESULTS: {passed}/{total} checks passed")
if passed == total:
    print("  ✅ ALL CHECKS PASSED — pipeline integration OK")
else:
    failed = [k for k, v in results.items() if not v]
    print("  ❌ FAILED CHECKS:")
    for f in failed:
        print(f"     - {f}")
print("="*60 + "\n")
