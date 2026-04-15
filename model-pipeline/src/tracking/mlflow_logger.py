from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _load_tracking_config() -> dict[str, Any]:
    cfg = Path(__file__).resolve().parents[2] / "configs" / "model_config.yaml"
    with open(cfg, encoding="utf-8") as f:
        return yaml.safe_load(f)["tracking"]["mlflow"]


class MLflowLogger:
    def __init__(
        self,
        experiment_name: str | None = None,
        tracking_uri: str | None = None,
    ):
        import mlflow
        self._mlflow = mlflow

        config = _load_tracking_config()
        self._mlflow.set_tracking_uri(tracking_uri or config["tracking_uri"])
        self._mlflow.set_experiment(experiment_name or config["experiment_name"])
        self._run: Any = None  # mlflow.ActiveRun — untyped, annotated as Any
        self._registered_model_name: str = config.get("registered_model_name", "wildfire-ignition")

    def start_run(self, run_name: str | None = None, tags: dict[str, str] | None = None) -> str:
        self._run = self._mlflow.start_run(run_name=run_name, tags=tags)
        run_id = self._run.info.run_id
        logger.info("MLflow run started: %s", run_id)
        return run_id

    def end_run(self, status: str = "FINISHED"):
        self._mlflow.end_run(status=status)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        for k, v in metrics.items():
            if v is not None and isinstance(v, (int, float)):
                self._mlflow.log_metric(k, v, step=step)

    def log_params(self, params: dict[str, Any]):
        self._mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_artifact(self, local_path: str | Path, artifact_subdir: str | None = None):
        self._mlflow.log_artifact(str(local_path), artifact_subdir)

    def log_model_hash(self, model_hash: str):
        self._mlflow.log_param("model_artifact_sha256", model_hash)

    def log_input_statistics(self, stats: dict[str, dict[str, float]]):
        for feat, feat_stats in stats.items():
            for stat, val in feat_stats.items():
                self._mlflow.log_metric(f"input_{feat}_{stat}", val)

    def log_bias_gate_result(self, bias_report: dict[str, Any]):
        """Log bias gate results from bias_check.run_bias_check() report format."""
        self._mlflow.log_param("bias_gate_result", bias_report.get("gate_result", "UNKNOWN"))
        self._mlflow.log_metric("bias_overall_fnr", bias_report.get("overall_fnr", 0.0))
        for slice_name, slice_data in bias_report.get("slices", {}).items():
            disparity = slice_data.get("disparity", 0.0)
            self._mlflow.log_metric(f"bias_disparity_{slice_name}", disparity)
            for group, fnr in slice_data.get("per_group_fnr", {}).items():
                safe = group.replace(" ", "_").replace("/", "_")
                self._mlflow.log_metric(f"bias_fnr_{slice_name}_{safe}", fnr)

    def log_validation_result(self, metrics: dict[str, Any], passed: bool):
        self._mlflow.log_param("validation_passed", str(passed))
        self.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

    def log_visualization(self, viz_paths: dict[str, Path]):
        for _, path in viz_paths.items():
            if Path(path).exists():
                self.log_artifact(path, artifact_subdir="visualizations")

    def log_shap(self, shap_dict: dict[str, float]) -> None:
        """Log mean absolute SHAP values per feature as MLflow metrics.

        Each feature is logged as 'shap_{feature_name}' so they appear
        side-by-side in the MLflow UI and can be tracked for drift.
        """
        for feature, value in shap_dict.items():
            if isinstance(value, (int, float)) and value is not None:
                safe_name = f"shap_{feature}".replace(" ", "_").replace("/", "_")
                self._mlflow.log_metric(safe_name, float(value))

    def log_threshold(self, threshold: float, target_precision: float) -> None:
        """Log the operational decision threshold alongside the model.

        Logged as a metric (not a param) so it can be written after get_params()
        which may already have logged a default threshold value — MLflow params
        are immutable once written, metrics are not.
        """
        self._mlflow.log_metric("tuned_threshold", threshold)
        self._mlflow.log_metric("target_precision", target_precision)

def compute_input_statistics(X: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats = {}
    for col in X.columns:
        s = X[col].dropna()
        if len(s) > 0 and pd.api.types.is_numeric_dtype(s):
            stats[col] = {
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
            }
    return stats
