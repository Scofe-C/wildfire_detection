"""
test_obj1_to_obj2.py — Integration test: OBJ-1 scores → OBJ-2 spread simulation
==================================================================================
Simulates the real deployment flow:
  1. Load latest 22km fused parquet (same source as real-time pipeline)
  2. If a trained OBJ-1 model is available → score cells and attach fire_risk_score
     Otherwise → synthesize fire_risk_score from FIRMS/RH (demo fallback)
  3. Feed the scored DataFrame into OBJ-2 (evaluate_obj2 logic)
  4. Print full spread simulation output

Usage
-----
  cd model-pipeline
  python scripts/test_obj1_to_obj2.py                   # auto-detect model
  python scripts/test_obj1_to_obj2.py --mock             # force synthetic scores (no model needed)
  python scripts/test_obj1_to_obj2.py --top-n 3          # simulate top-3 OBJ-1 cells
  python scripts/test_obj1_to_obj2.py --cell 8529a39bfffffff --prob 0.72  # specific cell
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_PIPELINE))
sys.path.insert(0, str(MODEL_PIPELINE / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_obj1_to_obj2")

ROOT = Path(__file__).resolve().parents[2]
DATA_PIPELINE = ROOT / "Data-Pipeline"
PROCESSED_22KM = DATA_PIPELINE / "data" / "processed" / "22km"
FUSED_22KM = DATA_PIPELINE / "data" / "processed" / "fused" / "22km"
STATIC_22KM = DATA_PIPELINE / "data" / "static" / "static_features_22km.parquet"


# ---------------------------------------------------------------------------
# Step 1: Load latest 22km parquet (same logic as _load_22km in evaluate_obj2)
# ---------------------------------------------------------------------------

def load_latest_22km() -> pd.DataFrame:
    """Load the newest 22km fused parquet for CA + TX."""
    region_dfs = []
    for region in ("california", "texas"):
        best_file = None
        best_name = ""
        for base_dir, pat in [
            (PROCESSED_22KM, f"region={region}/**/features_*.parquet"),
            (FUSED_22KM,     f"region={region}/**/fused_*.parquet"),
        ]:
            if base_dir.exists():
                files = sorted(base_dir.glob(pat), reverse=True)
                if files and files[0].name > best_name:
                    best_file, best_name = files[0], files[0].name
        if best_file:
            df = pd.read_parquet(best_file)
            df["grid_id"] = df["grid_id"].astype(str)
            region_dfs.append(df)
            logger.info("Loaded (%s): %s  (%d rows)", region, best_file.name, len(df))

    if not region_dfs:
        raise FileNotFoundError(
            "No 22km parquet found. Run:\n"
            "  cd Data-Pipeline && python -m scripts.utils.run_pipeline_once --resolution-km 22"
        )

    df = pd.concat(region_dfs, ignore_index=True)
    logger.info("Combined CA+TX: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Step 2a: Score with real OBJ-1 model
# ---------------------------------------------------------------------------

def score_with_obj1(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to run the real OBJ-1 XGBoost model over the DataFrame.

    Returns DataFrame with fire_risk_score, fire_risk_flag, risk_tier added.
    Raises ImportError / FileNotFoundError if model not available.
    """
    from models.obj1_xgboost.model import XGBoostFireRiskModel
    from models.obj1_xgboost.feature_engineering import full_pipeline

    model = XGBoostFireRiskModel()
    model.load()  # loads from MLflow Model Registry or local path

    X, _ = full_pipeline(df, model_type="xgb", is_inference=True)
    preds = model.predict(X)

    scored = df.copy()
    scored["fire_risk_score"] = preds["probability"].values
    scored["fire_risk_flag"]  = preds["prediction"].values
    scored["risk_tier"] = scored["fire_risk_score"].apply(_assign_risk_tier)

    logger.info(
        "OBJ-1 scored: max=%.4f  CRITICAL=%d  HIGH=%d  MEDIUM=%d",
        scored["fire_risk_score"].max(),
        (scored["risk_tier"] == "CRITICAL").sum(),
        (scored["risk_tier"] == "HIGH").sum(),
        (scored["risk_tier"] == "MEDIUM").sum(),
    )
    return scored


# ---------------------------------------------------------------------------
# Step 2b: Synthetic OBJ-1 scores (demo/testing fallback)
# ---------------------------------------------------------------------------

def synthesize_obj1_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Build synthetic fire_risk_score from FIRMS + RH when model not available.

    Score formula (demo only):
        0.4 × norm(active_fire_count)  +
        0.4 × norm(1 / relative_humidity_2m)  +
        0.2 × norm(fire_weather_index)
    Clipped to [0, 1]. Named fire_risk_score so OBJ-2 picks it up automatically.
    """
    import numpy as np

    scored = df.copy()

    def _norm(s: pd.Series) -> pd.Series:
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    fire_cnt = pd.to_numeric(scored.get("active_fire_count", pd.Series(0, index=scored.index)), errors="coerce").fillna(0)
    rh       = pd.to_numeric(scored.get("relative_humidity_2m", pd.Series(50, index=scored.index)), errors="coerce").fillna(50).clip(1)
    fwi      = pd.to_numeric(scored.get("fire_weather_index", pd.Series(0, index=scored.index)), errors="coerce").fillna(0)

    score = (
        0.4 * _norm(fire_cnt) +
        0.4 * _norm(1.0 / rh) +
        0.2 * _norm(fwi)
    ).clip(0, 1)

    scored["fire_risk_score"] = score.round(4)
    scored["fire_risk_flag"]  = (score >= 0.365).astype(int)
    scored["risk_tier"] = scored["fire_risk_score"].apply(_assign_risk_tier)

    logger.info(
        "Synthetic OBJ-1 scores: max=%.4f  flagged=%d",
        scored["fire_risk_score"].max(),
        scored["fire_risk_flag"].sum(),
    )
    return scored


def _assign_risk_tier(score: float) -> str:
    if score >= 0.65:  return "CRITICAL"
    if score >= 0.365: return "HIGH"
    if score >= 0.15:  return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Step 3: Run OBJ-2 on the scored DataFrame
# ---------------------------------------------------------------------------

def run_obj2(df: pd.DataFrame, top_n: int = 1, generate_report: bool = False) -> None:
    """Run OBJ-2 spread simulation on top-N highest-scored cells from OBJ-1."""
    from models.obj2_spread.fire_spread_simulator import PythonFireSpreadSimulator

    # Enrich with static terrain/canopy if needed
    if STATIC_22KM.exists() and "slope_degrees" not in df.columns:
        static = pd.read_parquet(STATIC_22KM)
        static["grid_id"] = static["grid_id"].astype(str)
        df = df.merge(static, on="grid_id", how="left", suffixes=("", "_static"))
        logger.info("Enriched with static features")

    # Apply CBH clamp to prevent false crown fire
    if "canopy_base_height_m" in df.columns:
        df["canopy_base_height_m"] = df["canopy_base_height_m"].clip(lower=2.0)

    sim = PythonFireSpreadSimulator()

    # Get top-N cells by fire_risk_score
    top_cells = (
        df.nlargest(top_n, "fire_risk_score")[["grid_id", "fire_risk_score", "risk_tier"]]
        if "fire_risk_score" in df.columns
        else df.head(top_n)[["grid_id"]]
    )

    print("\n" + "=" * 70)
    print("  OBJ-1 -> OBJ-2 INTEGRATION TEST")
    print("=" * 70)
    print(f"  Data rows   : {len(df)}")
    print(f"  Simulating  : top-{top_n} OBJ-1 cell(s)")

    # Show top-5 risk cells
    if "fire_risk_score" in df.columns:
        top5 = df.nlargest(5, "fire_risk_score")[["grid_id", "fire_risk_score", "risk_tier"]]
        if "active_fire_count" in df.columns:
            top5 = top5.merge(df[["grid_id", "active_fire_count"]], on="grid_id", how="left")
        print(f"\n  Top-5 OBJ-1 risk cells:")
        print(top5.to_string(index=False))

    for _, cell_row in top_cells.iterrows():
        ign_id   = str(cell_row["grid_id"])
        ign_prob = float(cell_row.get("fire_risk_score", 0.3))
        tier     = cell_row.get("risk_tier", "?")

        print(f"\n{'-'*70}")
        print(f"  Cell: {ign_id}  |  OBJ-1 score: {ign_prob:.4f}  |  tier: {tier}")
        print(f"{'-'*70}")

        try:
            result = sim.simulate(df, ign_id, ign_prob)

            # MC
            mc = sim.simulate_monte_carlo(df, ign_id, ign_prob, n_simulations=100, horizon_hours=6.0)

            DET_W, MC_W = 0.4, 0.6
            det_speed   = result["spread_speed_kmh"]
            mc_p50      = mc.get("spread_speed_kmh_p50", det_speed)
            mc_p90      = mc.get("spread_speed_kmh_p90", det_speed)
            h_speed     = DET_W * det_speed + MC_W * mc_p90
            h_dir       = mc.get("dominant_direction_deg", result["spread_direction_deg"])

            inputs = result.get("inputs_used", {})
            wind_kmh = inputs.get("wind_speed_10m_ms", 0) * 3.6
            rh = inputs.get("relative_humidity_pct", 0)
            crown_prob = mc.get("crown_fire_probability", 0)

            print(f"\n  Fire behavior:")
            print(f"    direction     : {h_dir:.1f} deg ({result['dominant_factor']})")
            print(f"    intensity     : {result['byram_intensity_kwm']:.1f} kW/m ({result['crown_fire_status']})")
            print(f"    wind          : {wind_kmh:.1f} km/h from {inputs.get('wind_from_direction_deg', 0):.0f} deg")
            print(f"    moisture      : DFMC={result['dead_fuel_moisture_pct']:.1f}% (RH={rh:.0f}%)")
            print(f"    crown status  : {result['crown_fire_status']} (prob {crown_prob:.1%})")

            n_sims = mc.get("n_simulations", 100)
            print(f"\n  Monte Carlo spread forecast (N={n_sims}, 6h horizon):")
            print(f"    speed p50     : {mc_p50:.4f} km/h")
            print(f"    speed p90     : {mc_p90:.4f} km/h (worst-case)")
            print(f"    hybrid speed  : {h_speed:.4f} km/h (40% det + 60% MC p90)")

            print(f"\n    Distance projection:")
            for hr in [1, 2, 3, 6]:
                dist = h_speed * hr
                print(f"      t={hr}h : {dist:.2f} km")

            # ── OBJ-3 report generation ───────────────────────────────────
            if generate_report:
                try:
                    from pipeline.bridge import build_pipeline_result
                    from models.obj3_gemini.reporter import GeminiDisasterReporter

                    prob_col = df["fire_risk_score"] if "fire_risk_score" in df.columns else pd.Series(0.15, index=df.index)
                    predictions = pd.DataFrame({
                        "prediction": (prob_col >= 0.365).astype(int),
                        "probability": prob_col,
                    })
                    pipeline_result = build_pipeline_result(
                        obj1_predictions=predictions,
                        obj1_input=df,
                        obj2_simulation=result,
                    )
                    config_path = MODEL_PIPELINE / "configs" / "reporting_config.yaml"
                    reporter = GeminiDisasterReporter()
                    reporter.load_model(config_path)
                    gen = reporter.generate_report(pipeline_result=pipeline_result)
                    rpt_path = gen.get("json_path") or gen.get("markdown_path")
                    print(f"\n  OBJ-3 Report: {rpt_path}")
                except Exception as rpt_exc:
                    logger.error("OBJ-3 report failed: %s", rpt_exc)

        except Exception as exc:
            print(f"  [ERROR] OBJ-2 failed: {exc}")
            logger.exception("OBJ-2 simulation failed for %s", ign_id)

    print("\n" + "=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OBJ-1 → OBJ-2 integration test")
    parser.add_argument("--mock",   action="store_true", help="Force synthetic OBJ-1 scores (no model needed)")
    parser.add_argument("--top-n",  type=int, default=1, help="Simulate top-N OBJ-1 cells (default: 1)")
    parser.add_argument("--cell",   type=str, help="Override: specific H3 cell ID to simulate")
    parser.add_argument("--prob",   type=float, default=0.5, help="Ignition probability when --cell is used (default: 0.5)")
    args = parser.parse_args()

    # Load data
    df = load_latest_22km()

    # Override: single cell injection
    if args.cell:
        if "fire_risk_score" not in df.columns:
            df["fire_risk_score"] = 0.0
            df["risk_tier"] = "LOW"
            df["fire_risk_flag"] = 0
        df.loc[df["grid_id"] == args.cell, "fire_risk_score"] = args.prob
        df.loc[df["grid_id"] == args.cell, "risk_tier"] = _assign_risk_tier(args.prob)
        logger.info("Injected: cell=%s  prob=%.4f", args.cell, args.prob)
        run_obj2(df, top_n=1)
        return

    # Score cells
    if not args.mock:
        try:
            df = score_with_obj1(df)
            logger.info("Using real OBJ-1 model scores")
        except Exception as exc:
            logger.warning("OBJ-1 model unavailable (%s) — using synthetic scores", exc)
            df = synthesize_obj1_scores(df)
    else:
        logger.info("--mock: using synthetic OBJ-1 scores")
        df = synthesize_obj1_scores(df)

    run_obj2(df, top_n=args.top_n)


if __name__ == "__main__":
    main()
