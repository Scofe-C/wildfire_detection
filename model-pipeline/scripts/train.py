"""
Training CLI entry point — OBJ-1 wildfire ignition model.

Two modes:

  initial  — first-time training; trains XGBoost AND LightGBM per region,
             selects winner by AUC-PR.  Run once to establish baseline models.
             Produces: wildfire-ignition-california v1, wildfire-ignition-texas v1

  retrain  — daily retraining (default); trains XGBoost only per region.
             XGBoost confirmed as winner in initial run — no need to re-run LightGBM.
             Produces: wildfire-ignition-california vN+1, wildfire-ignition-texas vN+1

Pipeline per region:
  1. Load region data from GCS
  2. Temporal split (train < 2025-01-01, test = Jan 2025 LA fires)
  3. Tune + train XGBoost (+ LightGBM if initial mode)
  4. Select winner by AUC-PR (initial) or validate directly (retrain)
  5. Threshold tuning (≥90% recall, candidates[-1] logic)
  6. SHAP, visualizations
  7. Bias gate (fire_season + fuel_model_fbfm40 slices)
  8. Push to Vertex AI Model Registry as wildfire-ignition-{region}

Usage
-----
  # Daily retraining (default):
  python -m scripts.train

  # First-time / initial training:
  python -m scripts.train --mode initial

  # Single region:
  python -m scripts.train --regions california

  # With explicit GCS bucket and report output:
  python -m scripts.train --bucket wildfire-mlops-123 --output-report reports/training_result.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OBJ-1 wildfire ignition model.")
    parser.add_argument(
        "--mode",
        choices=["initial", "retrain"],
        default="retrain",
        help=(
            "initial: train XGBoost + LightGBM, select winner. "
            "retrain: train XGBoost only (default, runs daily)."
        ),
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["california", "texas"],
        help="Regions to train (default: california texas)",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="GCS bucket name (default: GCS_BUCKET_NAME env var or 'wildfire-mlops-123')",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to model_config.yaml (default: auto-detect)",
    )
    parser.add_argument(
        "--output-report",
        default=None,
        help="Write per-region result JSON to this path (used by CI/CD gates)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "Local dev mode: read CSVs from historical_data/ instead of GCS, "
            "save model to reports/local_run/ instead of Vertex AI."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    bucket = args.bucket or os.environ.get("GCS_BUCKET_NAME") or "wildfire-mlops-123"
    logger.info(
        "Training pipeline starting — mode: %s, regions: %s, bucket: gs://%s",
        args.mode, args.regions, bucket,
    )

    from src.pipeline.orchestrator import load_pipeline_config, run_training_pipeline

    base_config = load_pipeline_config(args.config)
    base_config.gcs_bucket = bucket

    all_results: dict[str, dict] = {}
    overall_deployable = True

    for region in args.regions:
        logger.info("=" * 60)
        logger.info("Training region: %s  [mode=%s]", region.upper(), args.mode)
        logger.info("=" * 60)

        # Clone config and set per-region fields
        import copy
        from pathlib import Path as _Path
        config = copy.copy(base_config)
        config.region = region
        config.is_initial_run = (args.mode == "initial")
        if args.local:
            _script_dir = _Path(__file__).resolve().parent
            _root = _script_dir.parent
            config.local_data_path = str(_root / "historical_data" / f"{region}_historical.csv")
            config.local_model_dir = str(_root / "reports" / "local_run")

        result = run_training_pipeline(config=config)

        region_dict = {
            "run_id": result.run_id,
            "region": region,
            "training_mode": args.mode,
            "winner_name": result.winner_name,
            "winner_version": result.winner_version,
            "validation_passed": result.validation_passed,
            "bias_gate_passed": result.bias_gate_passed,
            "is_deployable": result.is_deployable,
            "metrics": result.metrics,
            "registry_version": result.registry_version,
            "mlflow_run_id": result.mlflow_run_id,
            "error": result.error,
        }
        all_results[region] = region_dict

        if result.error:
            logger.error("[%s] Pipeline failed: %s", region, result.error)
            overall_deployable = False
        elif not result.is_deployable:
            logger.warning(
                "[%s] Model NOT deployable (validation=%s, bias_gate=%s)",
                region, result.validation_passed, result.bias_gate_passed,
            )
            overall_deployable = False
        else:
            logger.info(
                "[%s] Training complete — winner: %s, AUC-PR: %.4f, threshold: %.4f",
                region,
                result.winner_name,
                result.metrics.get("auc_pr", 0.0),
                result.metrics.get("threshold", 0.0),
            )

    # ── Write combined report ──────────────────────────────────────────────────
    report = {"regions": all_results, "overall_deployable": overall_deployable}

    if args.output_report:
        out_path = Path(args.output_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Training report written to %s", out_path)

    if not overall_deployable:
        sys.exit(1)

    logger.info("All regions trained and deployed successfully.")


if __name__ == "__main__":
    main()
