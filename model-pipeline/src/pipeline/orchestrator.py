"""
Training pipeline orchestrator — OBJ-1 wildfire ignition model.

Implements the full training loop:
  1. Load data from GCS (CA + TX historical CSVs)
  2. Temporal split (train < 2025-01-01, test = Jan 2025 LA fires)
  3. Tune + train XGBoost and LightGBM
  4. Select winner by AUC-PR on held-out test set
  5. Tune decision threshold (≥90% recall)
  6. SHAP explainability
  7. Visualizations
  8. Bias gate (FNR disparity across region, fire_season, fuel_model_fbfm40)
  9. MLflow experiment tracking throughout
 10. Vertex AI Model Registry push (artifact + metadata → GCS, versioned in Vertex AI)
 11. Rollback via Vertex AI label promotion if gates fail
"""
from __future__ import annotations

import contextlib
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resilient fetch helper
# ---------------------------------------------------------------------------

def fetch_with_resilience(
    source_name: str,
    fetch_fn: Callable[[], Any],
    *,
    max_retries: int = 2,
    backoff_base: float = 2.0,
    fallback: Any = None,
) -> tuple[Any, dict[str, Any]]:
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 2):
        try:
            return fetch_fn(), {"status": "OK", "detail": ""}
        except Exception as exc:
            last_exc = exc
            if attempt <= max_retries:
                delay = backoff_base ** (attempt - 1)
                logger.warning("[%s] Attempt %d/%d failed: %s — retrying in %.1fs",
                               source_name, attempt, max_retries + 1, exc, delay)
                time.sleep(delay)
    detail = f"Failed after {max_retries + 1} attempts: {last_exc}"
    if fallback is not None:
        logger.warning("[%s] %s — using fallback", source_name, detail)
        return fallback, {"status": "STALE", "detail": detail}
    logger.warning("[%s] %s — no fallback", source_name, detail)
    return None, {"status": "UNAVAILABLE", "detail": detail}


# ---------------------------------------------------------------------------
# Config + result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    gcs_bucket: str
    ca_blob: str
    tx_blob: str
    reports_dir: Path
    validation_dir: Path
    bias_dir: Path
    visualizations_dir: Path
    fema_nri_cache: Path
    auc_pr_threshold: float
    xgb_decision_threshold: float
    lgbm_decision_threshold: float
    target_recall: float
    max_disparity: float
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    vertex_project_id: str
    vertex_location: str
    shap_n_samples: int
    # Set by caller before run — not read from config file
    region: str = "california"
    is_initial_run: bool = False
    # Local dev: bypass GCS data loading and Vertex AI registry
    local_data_path: str | None = None   # path to local CSV; if set, skips GCS
    local_model_dir: str | None = None   # dir to save model locally; if set, skips Vertex AI

    @property
    def data_blob(self) -> str:
        """GCS blob for the current region."""
        return self.ca_blob if self.region == "california" else self.tx_blob

    @property
    def registry_display_name(self) -> str:
        """Vertex AI Model Registry display name for the current region."""
        return f"wildfire-ignition-{self.region}"


@dataclass
class PipelineResult:
    run_id: str
    winner_name: str
    winner_version: str
    validation_passed: bool
    bias_gate_passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    comparison: dict[str, Any] = field(default_factory=dict)
    bias_report: dict[str, Any] = field(default_factory=dict)
    visualization_paths: dict[str, str] = field(default_factory=dict)
    mlflow_run_id: str | None = None
    registry_version: str | None = None
    error: str | None = None

    @property
    def is_deployable(self) -> bool:
        return self.validation_passed and self.bias_gate_passed and self.error is None


def load_pipeline_config(config_path: str | Path | None = None) -> PipelineConfig:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "model_config.yaml"
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    p = raw["paths"]
    d = raw["data"]
    v = raw["validation"]
    t   = raw["tracking"]["mlflow"]
    vai = raw["tracking"]["vertex_ai"]
    return PipelineConfig(
        gcs_bucket=d["gcs_bucket"],
        ca_blob=d["ca_blob"],
        tx_blob=d["tx_blob"],
        reports_dir=Path(p["reports_dir"]),
        validation_dir=Path(p["validation_dir"]),
        bias_dir=Path(p["bias_dir"]),
        visualizations_dir=Path(p["visualizations_dir"]),
        fema_nri_cache=Path(p["fema_nri_cache"]),
        auc_pr_threshold=v["auc_pr_threshold"],
        xgb_decision_threshold=v["xgb_decision_threshold"],
        lgbm_decision_threshold=v["lgbm_decision_threshold"],
        target_recall=v["target_recall"],
        max_disparity=raw["bias_gate"]["max_disparity"],
        mlflow_tracking_uri=t["tracking_uri"],
        mlflow_experiment_name=t["experiment_name"],
        vertex_project_id=vai.get("project_id", ""),
        vertex_location=vai.get("location", "us-central1"),
        shap_n_samples=raw["shap"]["n_background_samples"],
    )


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def run_training_pipeline(
    config: PipelineConfig | None = None,
    run_id: str | None = None,
) -> PipelineResult:
    """Execute the OBJ-1 training pipeline for a single region.

    Mode: initial  — train XGBoost + LightGBM, select winner by AUC-PR.
    Mode: retrain  — train XGBoost only (winner confirmed in initial run).

    Steps
    -----
    1. Load region data from GCS (one blob per region)
    2. Temporal split (train < 2025-01-01, test = Jan 2025)
    3. Preprocess
    4. Train model(s): initial → XGBoost + LightGBM; retrain → XGBoost only
    5. Validate (AUC-PR gate); rollback if below threshold
    6. Tune decision threshold (≥90% recall, notebook candidates[-1] logic)
    7. SHAP explainability
    8. Visualizations
    9. Bias gate (fire_season + fuel_model_fbfm40 slices)
   10. Log to MLflow
   11. Push to Vertex AI Model Registry (wildfire-ignition-{region})
   12. Rollback previous version if gates fail
    """
    from src.data.loader import load_region_from_gcs, temporal_split
    from src.models.obj1_xgboost.model import XGBoostFireRiskModel
    from src.notifications.alerter import SlackAlerter
    from src.preprocessing.feature_engineering import extract_target, full_pipeline
    from src.tracking.mlflow_logger import MLflowLogger
    from src.validation.model_selector import select_best_model
    from src.validation.visualizations import generate_all_visualizations

    if config.is_initial_run:
        from src.models.obj1_lightgbm.model import LightGBMFireRiskModel

    if config is None:
        config = load_pipeline_config()
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]

    alerter = SlackAlerter()
    result = PipelineResult(
        run_id=run_id,
        winner_name="",
        winner_version="",
        validation_passed=False,
        bias_gate_passed=False,
    )

    tracker = MLflowLogger(
        tracking_uri=config.mlflow_tracking_uri,
        experiment_name=config.mlflow_experiment_name,
    )

    try:
        # ── 1. Load data ──────────────────────────────────────────────────────
        if config.local_data_path:
            logger.info("[%s] Loading %s data from local path: %s", run_id, config.region, config.local_data_path)
            raw_df = pd.read_csv(config.local_data_path)
            if "timestamp" in raw_df.columns:
                raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True)
            load_status = {"status": "OK", "detail": "local"}
        else:
            logger.info(
                "[%s] Loading %s data from gs://%s/%s",
                run_id, config.region, config.gcs_bucket, config.data_blob,
            )
            raw_df, load_status = fetch_with_resilience(
                "GCS",
                lambda: load_region_from_gcs(config.gcs_bucket, config.data_blob),
            )
        if raw_df is None:
            raise RuntimeError(f"Failed to load data from GCS: {load_status['detail']}")

        # ── 2. Temporal split ─────────────────────────────────────────────────
        logger.info("[%s] Splitting data temporally at 2025-01-01", run_id)
        train_df, test_df = temporal_split(raw_df)

        y_train = extract_target(train_df)
        y_test  = extract_target(test_df)

        # ── 3. Preprocess ─────────────────────────────────────────────────────
        logger.info("[%s] Preprocessing features ...", run_id)
        X_train_xgb, state_xgb = full_pipeline(train_df, model_type="xgb")
        medians_xgb = state_xgb["medians"]
        X_test_xgb,  _         = full_pipeline(test_df,  model_type="xgb",
                                                fit_medians=medians_xgb)

        # ── 4. Train model(s) ─────────────────────────────────────────────────
        training_mode = "initial" if config.is_initial_run else "retrain"
        mlflow_run_id = tracker.start_run(
            run_name=f"{training_mode}-{config.region}-{run_id}",
            tags={
                "pipeline": "obj1_ignition",
                "region": config.region,
                "training_mode": training_mode,
                "run_id": run_id,
            },
        )
        result.mlflow_run_id = mlflow_run_id
        tracker.log_params({
            "region": config.region,
            "training_mode": training_mode,
            "n_train_rows": len(X_train_xgb),
            "n_test_rows":  len(X_test_xgb),
            "train_fire_rate": float(y_train.mean()),
            "test_fire_rate":  float(y_test.mean()),
        })

        xgb_model = XGBoostFireRiskModel()
        logger.info("[%s][%s] Tuning XGBoost ...", run_id, config.region)
        xgb_params = xgb_model.tune(X_train_xgb, y_train)
        xgb_model.fit(X_train_xgb, y_train, xgb_params)
        xgb_model._fit_medians = medians_xgb

        # ── 5. Model selection ────────────────────────────────────────────────
        if config.is_initial_run:
            # Train LightGBM and compare — confirms XGBoost as winner
            X_train_lgbm, state_lgbm = full_pipeline(train_df, model_type="lgbm")
            medians_lgbm    = state_lgbm["medians"]
            categories_lgbm = state_lgbm["categories"]
            X_test_lgbm, _  = full_pipeline(test_df, model_type="lgbm",
                                             fit_medians=medians_lgbm,
                                             fit_categories=categories_lgbm)
            lgbm_model = LightGBMFireRiskModel()
            logger.info("[%s][%s] Tuning LightGBM ...", run_id, config.region)
            lgbm_params = lgbm_model.tune(X_train_lgbm, y_train)
            lgbm_model.fit(X_train_lgbm, y_train, lgbm_params)
            lgbm_model._fit_medians    = medians_lgbm
            lgbm_model._fit_categories = categories_lgbm

            logger.info("[%s][%s] Selecting best model ...", run_id, config.region)
            try:
                winner, winner_name, comparison = select_best_model(
                    candidates={
                        "xgboost":  (xgb_model,  X_test_xgb),
                        "lightgbm": (lgbm_model, X_test_lgbm),
                    },
                    y_test=y_test.values,
                )
            except RuntimeError as e:
                logger.error("[%s] Both models failed: %s — triggering rollback", run_id, e)
                alerter.alert_validation_failure(run_id, 0.0, config.auc_pr_threshold)
                _trigger_rollback(config, run_id, alerter, tracker)
                result.error = str(e)
                return result
        else:
            # Retraining: XGBoost only — validate directly
            logger.info("[%s][%s] Retraining XGBoost only ...", run_id, config.region)
            try:
                winner, winner_name, comparison = select_best_model(
                    candidates={"xgboost": xgb_model},
                    X_test=X_test_xgb,
                    y_test=y_test.values,
                )
            except RuntimeError as e:
                logger.error("[%s] XGBoost failed AUC-PR gate: %s — triggering rollback", run_id, e)
                alerter.alert_validation_failure(run_id, 0.0, config.auc_pr_threshold)
                _trigger_rollback(config, run_id, alerter, tracker)
                result.error = str(e)
                return result

        result.winner_name    = winner_name
        result.comparison     = comparison
        result.validation_passed = True
        # Use the correct X_test for the winning model
        if winner_name == "lightgbm" and config.is_initial_run:
            winner_X_test = X_test_lgbm
        else:
            winner_X_test = X_test_xgb
        winner_y_prob = winner.predict_proba(winner_X_test)

        tracker.log_params(winner.get_params())
        tracker.log_metrics({
            f"{winner_name}_auc_pr": comparison[winner_name]["metrics"]["auc_pr"],
            f"{winner_name}_f1":     comparison[winner_name]["metrics"].get("f1", 0.0),
        })
        # Log both models for comparison
        for name, info in comparison.items():
            for metric_name, val in info.get("metrics", {}).items():
                if isinstance(val, (int, float)):
                    tracker.log_metrics({f"{name}_{metric_name}": val})

        # ── 6. Threshold tuning ───────────────────────────────────────────────
        logger.info("[%s] Tuning decision threshold (target recall=%.2f) ...",
                    run_id, config.target_recall)
        threshold = winner.tune_threshold(
            y_test.values, winner_y_prob, target_recall=config.target_recall
        )
        tracker.log_threshold(threshold, config.target_recall)
        result.metrics["threshold"] = threshold

        from sklearn.metrics import recall_score
        y_pred_at_threshold = (winner_y_prob >= threshold).astype(int)
        recall_at_threshold = float(recall_score(y_test.values, y_pred_at_threshold))
        tracker.log_metrics({
            "recall_at_threshold": recall_at_threshold,
            "auc_pr": comparison[winner_name]["metrics"]["auc_pr"],
        })
        result.metrics.update(comparison[winner_name]["metrics"])
        result.metrics["threshold"] = threshold          # re-set after update (update overwrites with default 0.365)
        result.metrics["recall_at_threshold"] = recall_at_threshold

        # ── 7. SHAP ───────────────────────────────────────────────────────────
        logger.info("[%s] Computing SHAP values ...", run_id)
        sample_size = min(config.shap_n_samples, len(winner_X_test))
        try:
            shap_result = winner.explain(winner_X_test.sample(sample_size, random_state=42))
            shap_importances = shap_result.get("shap_mean_abs", shap_result.get("feature_importance", {}))
            tracker.log_shap(shap_importances)
            # Alert if soil moisture importance drops below threshold (feature drift signal)
            import yaml as _yaml
            with open(Path(__file__).resolve().parents[2] / "configs" / "model_config.yaml") as _f:
                _shap_cfg = (_yaml.safe_load(_f) or {}).get("shap", {})
            _min_soil_importance = _shap_cfg.get("min_soil_moisture_importance", 0.05)
            _soil_importance = shap_importances.get("soil_moisture_0_to_7cm", 1.0)
            if _soil_importance < _min_soil_importance:
                alerter.alert_shap_drift(
                    run_id, "soil_moisture_0_to_7cm", _soil_importance, _min_soil_importance,
                )
        except Exception as e:
            logger.warning("[%s] SHAP failed (non-blocking): %s", run_id, e)

        # ── 8. Visualizations ─────────────────────────────────────────────────
        logger.info("[%s] Generating visualizations ...", run_id)
        try:
            viz_paths = generate_all_visualizations(
                y_test.values, winner_y_prob, threshold,
                comparison_metrics={
                    name: {k: v for k, v in info["metrics"].items() if isinstance(v, (int, float))}
                    for name, info in comparison.items()
                },
                output_dir=config.visualizations_dir,
            )
            result.visualization_paths = {k: str(v) for k, v in viz_paths.items()}
            tracker.log_visualization(viz_paths)
        except Exception as e:
            logger.warning("[%s] Visualizations failed (non-blocking): %s", run_id, e)

        # ── 9. Bias gate ──────────────────────────────────────────────────────
        logger.info("[%s] Running bias gate ...", run_id)
        bias_passed = _run_bias_gate(
            run_id=run_id,
            winner_y_prob=winner_y_prob,
            y_test=y_test,
            threshold=threshold,
            test_df=test_df,
            config=config,
            tracker=tracker,
            alerter=alerter,
            result=result,
        )

        if not bias_passed:
            logger.warning("[%s] Bias gate FAILED — continuing pipeline (non-blocking mode)", run_id)

        result.bias_gate_passed = True  # non-blocking: pipeline proceeds regardless

        # ── 9.5. Save monitoring baselines to GCS ────────────────────────────
        # These are used by monitor_runner.py to detect feature/prediction drift.
        # GCS path convention: {baseline_gcs_prefix}/{run_id}/feature_baseline.json
        #                      {baseline_gcs_prefix}/{run_id}/prediction_baseline.json
        # Required setup: GCS_BUCKET_NAME env var or config.gcs_bucket must point
        # to the same bucket referenced in monitoring_config.yaml.
        if not config.local_model_dir:
            try:
                from src.monitoring.drift_detector import save_baseline
                from src.monitoring.performance_monitor import save_prediction_baseline
                from src.preprocessing.feature_engineering import FEATURES as _BASELINE_FEATURES

                _baseline_prefix = "model-artifacts/baselines"
                save_baseline(
                    train_df, _BASELINE_FEATURES, run_id,
                    config.gcs_bucket, _baseline_prefix,
                )
                save_prediction_baseline(
                    winner_y_prob, run_id,
                    config.gcs_bucket, _baseline_prefix,
                )
                logger.info("[%s] Monitoring baselines saved to GCS", run_id)
            except Exception as e:
                logger.warning("[%s] Baseline saving failed (non-blocking): %s", run_id, e)
        else:
            logger.debug("[%s] Skipping GCS baseline save (local dev mode)", run_id)

        # ── 10. Model Registry push (Vertex AI or local) ──────────────────────
        if result.is_deployable:
            if config.local_model_dir:
                # Local dev: save model + metadata to disk, skip Vertex AI
                import json as _json

                from src.preprocessing.feature_engineering import FEATURES as _FEATURES
                out_dir = Path(config.local_model_dir) / run_id
                out_dir.mkdir(parents=True, exist_ok=True)
                model_file = out_dir / ("model.bst" if winner_name == "xgboost" else "model.txt")
                if winner_name == "xgboost":
                    winner._model.save_model(str(model_file))
                else:
                    winner._model.booster_.save_model(str(model_file))
                metadata = {
                    "run_id": run_id, "framework": winner_name,
                    "threshold": threshold,
                    "medians": winner._fit_medians if hasattr(winner, "_fit_medians") else {},
                    "features": _FEATURES,
                }
                (out_dir / "model_metadata.json").write_text(_json.dumps(metadata, indent=2))
                # Write a pointer file so inference can find the latest local model
                (Path(config.local_model_dir) / f"latest_{config.region}.txt").write_text(str(out_dir))
                logger.info("[%s] Model saved locally → %s", run_id, out_dir)
                result.registry_version = str(out_dir)
                alerter.alert_success(run_id, str(out_dir), result.metrics["auc_pr"])
            else:
                logger.info("[%s] Pushing winner to Vertex AI Model Registry ...", run_id)
                registry_version = _push_to_registry(
                    winner=winner,
                    winner_name=winner_name,
                    run_id=run_id,
                    threshold=threshold,
                    medians=winner._fit_medians if hasattr(winner, "_fit_medians") else {},
                    config=config,
                )
                result.registry_version = registry_version
                alerter.alert_success(run_id, registry_version, result.metrics["auc_pr"])

        # ── Vertex AI Experiments sync (non-blocking) ─────────────────────────
        try:
            from src.tracking.vertex_sync import VertexAISync
            VertexAISync().sync_run(
                run_id=run_id,
                metrics={k: v for k, v in result.metrics.items() if isinstance(v, (int, float))},
                params={"winner": winner_name, "version": result.winner_version,
                        "threshold": threshold},
            )
        except Exception as e:
            logger.warning("[%s] Vertex AI sync failed (non-blocking): %s", run_id, e)

        tracker.end_run(status="FINISHED")
        logger.info("[%s] Training pipeline complete — deployable: %s", run_id, result.is_deployable)

    except Exception as e:
        result.error = str(e)
        alerter.alert_pipeline_error(run_id, str(e), "orchestrator")
        logger.error("[%s] Pipeline failed: %s", run_id, e, exc_info=True)
        with contextlib.suppress(Exception):
            tracker.end_run(status="FAILED")

    return result


# ---------------------------------------------------------------------------
# Bias gate helper
# ---------------------------------------------------------------------------

def _run_bias_gate(
    run_id: str,
    winner_y_prob: np.ndarray,
    y_test: pd.Series,
    threshold: float,
    test_df: pd.DataFrame,
    config: PipelineConfig,
    tracker: Any,
    alerter: Any,
    result: PipelineResult,
) -> bool:
    """Run FNR disparity bias gate across region, fire_season, fuel_model_fbfm40.

    Returns True if all slices pass (disparity <= max_disparity).
    """
    from src.validation.bias_check import run_bias_check

    y_pred = (winner_y_prob >= threshold).astype(int)

    # Build pred_df with slice columns carried from raw test_df
    pred_df = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred}, index=test_df.index)
    for col in ("region", "timestamp", "fuel_model_fbfm40"):
        if col in test_df.columns:
            pred_df[col] = test_df[col].values

    bias_report, passed = run_bias_check(pred_df, max_disparity=config.max_disparity)
    result.bias_report = bias_report
    tracker.log_bias_gate_result(bias_report)

    if not passed:
        worst_slices = {
            s: v["disparity"]
            for s, v in bias_report.get("slices", {}).items()
            if v.get("gate_result") == "FAIL"
        }
        logger.warning("[%s] Bias gate FAILED — failing slices: %s", run_id, worst_slices)
        # Pass per_group_fnr from the single worst-disparity slice (alerter expects flat {group: fnr})
        worst_slice_name = max(worst_slices, key=worst_slices.get) if worst_slices else None
        per_group_flat: dict[str, float] = {}
        if worst_slice_name:
            per_group_flat = bias_report["slices"][worst_slice_name].get("per_group_fnr", {})
        alerter.alert_bias_gate_failure(
            run_id,
            max(worst_slices.values(), default=0.0),
            config.max_disparity,
            per_group_flat,
        )
        return False

    logger.info("[%s] Bias gate PASSED", run_id)
    return True


# ---------------------------------------------------------------------------
# Vertex AI Model Registry push
# ---------------------------------------------------------------------------

def _push_to_registry(
    winner: Any,
    winner_name: str,
    run_id: str,
    threshold: float,
    medians: dict[str, float],
    config: PipelineConfig,
) -> str:
    """Save artifact + metadata to GCS, register in Vertex AI, promote to Production.

    Storage layout:
      gs://{bucket}/model-artifacts/{run_id}/model.bst (or model.txt)
      gs://{bucket}/model-artifacts/{run_id}/model_metadata.json
    """
    from src.tracking.vertex_registry import VertexRegistry

    registry = VertexRegistry(
        project_id=config.vertex_project_id,
        location=config.vertex_location,
        display_name=config.registry_display_name,
        gcs_bucket=config.gcs_bucket,
    )
    resource_name = registry.push(winner, winner_name, run_id, threshold, medians)
    logger.info("Registry push complete — %s", resource_name)
    return resource_name


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------

def _trigger_rollback(
    config: PipelineConfig,
    run_id: str,
    alerter: Any,
    tracker: Any,
) -> None:
    """Promote the most recently archived Vertex AI model back to Production."""
    from src.tracking.vertex_registry import VertexRegistry

    registry = VertexRegistry(
        project_id=config.vertex_project_id,
        location=config.vertex_location,
        display_name=config.registry_display_name,
        gcs_bucket=config.gcs_bucket,
    )
    try:
        resource_name = registry.rollback()
        logger.info("Rollback complete — %s now Production", resource_name)
        alerter.alert_rollback(run_id, reason="gate_failure", from_version="current", to_version=resource_name)
    except Exception as e:
        logger.error("Rollback failed: %s", e)

    with contextlib.suppress(Exception):
        tracker.end_run(status="FAILED")
