from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    backfill_dir: Path
    models_dir: Path
    validation_dir: Path
    bias_dir: Path
    visualizations_dir: Path
    fema_nri_cache: Path
    auc_pr_threshold: float
    decision_threshold: float
    max_disparity: float


@dataclass
class PipelineResult:
    run_id: str
    model_name: str
    model_version: str
    validation_passed: bool
    bias_gate_passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    bias_report: dict[str, Any] = field(default_factory=dict)
    visualization_paths: dict[str, str] = field(default_factory=dict)
    artifact_pushed: bool = False
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
    return PipelineConfig(
        backfill_dir=Path(p["backfill_dir"]),
        models_dir=Path(p["models_dir"]),
        validation_dir=Path(p["validation_dir"]),
        bias_dir=Path(p["bias_dir"]),
        visualizations_dir=Path(p["visualizations_dir"]),
        fema_nri_cache=Path(p["fema_nri_cache"]),
        auc_pr_threshold=raw["validation"]["auc_pr_threshold"],
        decision_threshold=raw["validation"]["decision_threshold"],
        max_disparity=raw["bias_gate"]["max_disparity"],
    )


def run_pipeline(
    model: Any,
    config: PipelineConfig | None = None,
    run_id: str | None = None,
    baseline_metrics: dict[str, float] | None = None,
) -> PipelineResult:
    from src.data.loader import load_and_split
    from src.notifications.alerter import SlackAlerter
    from src.tracking.mlflow_logger import MLflowLogger, compute_input_statistics
    from src.validation.metrics import measure_inference_latency
    from src.validation.model_selector import (
        validate_model,
    )
    from src.validation.visualizations import generate_all_visualizations

    if config is None:
        config = load_pipeline_config()
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]

    alerter = SlackAlerter()
    result = PipelineResult(
        run_id=run_id, model_name=model.model_name, model_version=model.version,
        validation_passed=False, bias_gate_passed=False,
    )

    try:
        # Load data
        logger.info("[%s] Loading data from %s", run_id, config.backfill_dir)
        X, y, metadata = load_and_split(config.backfill_dir)

        # Tracking
        tracker = MLflowLogger()
        tracker.start_run(
            run_name=f"{model.model_name}-{run_id}",
            tags={"model": model.model_name, "version": model.version},
        )
        input_stats = compute_input_statistics(X)
        tracker.log_input_statistics(input_stats)

        # Inference
        logger.info("[%s] Running inference", run_id)
        predictions = model.predict(X)
        y_prob = predictions["probability"].values
        latency_ms = measure_inference_latency(model.predict, X, n_runs=3)

        # Validation (CI/CD Stage 5)
        logger.info("[%s] Validating model", run_id)
        metrics, passed_val = validate_model(y.values, y_prob)
        metrics["inference_latency_ms"] = latency_ms
        result.metrics = metrics
        result.validation_passed = passed_val
        tracker.log_validation_result(metrics, passed_val)

        if not passed_val:
            alerter.alert_validation_failure(run_id, metrics["auc_pr"], config.auc_pr_threshold)
            tracker.end_run(status="FAILED")
            return result

        # Visualizations
        logger.info("[%s] Generating visualizations", run_id)
        comparison = None
        if baseline_metrics:
            comparison = {
                model.model_name: {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                "FWI Baseline": baseline_metrics,
            }
        viz_paths = generate_all_visualizations(
            y.values, y_prob, config.decision_threshold, comparison, config.visualizations_dir,
        )
        result.visualization_paths = {k: str(v) for k, v in viz_paths.items()}
        tracker.log_visualization(viz_paths)

        # Bias Gate (CI/CD Stage 6 — BLOCKING)
        logger.info("[%s] Running bias gate", run_id)
        try:
            from src.bias.detector import run_bias_gate_from_dataframe
            from src.bias.nri_loader import (
                compute_vulnerability_quartiles,
                load_nri,
                spatial_join_predictions,
            )
            from src.bias.report import generate_bias_report, save_bias_report

            nri = load_nri(cache_dir=config.fema_nri_cache)
            nri = compute_vulnerability_quartiles(nri)

            y_pred = (y_prob >= config.decision_threshold).astype(int)
            pred_df = metadata.copy()
            pred_df["y_true"] = y.values
            pred_df["y_pred"] = y_pred
            pred_df["y_prob"] = y_prob

            joined = spatial_join_predictions(pred_df, nri)
            bias_result, passed_bias = run_bias_gate_from_dataframe(joined)
            result.bias_report = bias_result
            result.bias_gate_passed = passed_bias

            report = generate_bias_report(bias_result, run_id, model.version, input_stats)
            save_bias_report(report, config.bias_dir)
            tracker.log_bias_gate_result(bias_result)

            if not passed_bias:
                alerter.alert_bias_gate_failure(
                    run_id, bias_result["disparity_between_groups"],
                    config.max_disparity, bias_result["per_group_fnr"],
                )
                tracker.end_run(status="FAILED")
                return result

        except Exception as e:
            logger.warning("[%s] Bias gate skipped: %s", run_id, e)
            result.bias_gate_passed = False
            result.bias_report = {"gate_result": "SKIPPED", "reason": str(e)}

        # Registry Push (CI/CD Stage 7)
        if result.is_deployable:
            logger.info("[%s] Pushing to model registry", run_id)
            from src.models.registry import ModelRegistry
            registry = ModelRegistry(local_models_dir=config.models_dir)
            registry.save_local(
                model_artifact_path=config.models_dir, version=model.version,
                metadata={"run_id": run_id, "auc_pr": metrics["auc_pr"], "bias_gate": "PASS"},
            )
            registry.tag_previous(model.version)
            result.artifact_pushed = True
            alerter.alert_success(run_id, model.version, metrics["auc_pr"])

        # Vertex AI Sync (CI/CD Stage 8 — non-blocking)
        try:
            from src.tracking.vertex_sync import VertexAISync
            VertexAISync().sync_run(
                run_id=run_id,
                metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                params={"model": model.model_name, "version": model.version,
                        "validation": str(result.validation_passed),
                        "bias_gate": str(result.bias_gate_passed)},
            )
        except Exception as e:
            logger.warning("[%s] Vertex AI sync failed (non-blocking): %s", run_id, e)

        tracker.end_run(status="FINISHED")
        logger.info("[%s] Pipeline done — deployable: %s", run_id, result.is_deployable)

    except Exception as e:
        result.error = str(e)
        alerter.alert_pipeline_error(run_id, str(e), "orchestrator")
        logger.error("[%s] Pipeline failed: %s", run_id, e, exc_info=True)

    return result
