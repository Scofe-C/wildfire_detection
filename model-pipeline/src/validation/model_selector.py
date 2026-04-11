from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.validation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    model_name: str
    version: str
    metrics: dict[str, Any]
    passed_validation: bool
    passed_bias_gate: bool | None
    bias_report_path: str | None
    visualization_paths: dict[str, str]

    @property
    def is_deployable(self) -> bool:
        return self.passed_validation and (self.passed_bias_gate is True)


def _load_config(config_path: str | Path | None) -> dict:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "model_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def validate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    config_path: str | Path | None = None,
) -> tuple[dict[str, Any], bool]:
    config = _load_config(config_path)
    threshold = config["validation"].get("xgb_decision_threshold", 0.365)
    auc_pr_threshold = config["validation"]["auc_pr_threshold"]

    metrics = compute_all_metrics(y_true, y_prob, threshold=threshold)
    passed = metrics["auc_pr"] >= auc_pr_threshold

    if passed:
        logger.info("VALIDATION PASSED — AUC-PR: %.4f >= %.4f", metrics["auc_pr"], auc_pr_threshold)
    else:
        logger.warning("VALIDATION FAILED — AUC-PR: %.4f < %.4f", metrics["auc_pr"], auc_pr_threshold)

    return metrics, passed


def save_validation_report(result: ValidationResult, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "model_name": result.model_name,
        "version": result.version,
        "passed_validation": result.passed_validation,
        "passed_bias_gate": result.passed_bias_gate,
        "is_deployable": result.is_deployable,
        "metrics": result.metrics,
        "visualization_paths": result.visualization_paths,
    }
    if result.bias_report_path:
        report["bias_report_path"] = result.bias_report_path

    report_path = output_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Validation report saved: %s", report_path)
    return report_path


def select_best_model(
    candidates: dict[str, Any],
    y_test: np.ndarray,
    X_test: pd.DataFrame | None = None,
    config_path: str | Path | None = None,
) -> tuple[Any, str, dict[str, Any]]:
    """Compare multiple trained models on the test set and return the winner.

    Parameters
    ----------
    candidates : dict mapping model_name → model instance OR (model, X_test) tuple.
        When a tuple is provided, the per-model X_test is used (needed when
        XGBoost and LightGBM have differently preprocessed test sets).
        When a plain model is provided, the shared X_test parameter is used.
    y_test : true labels.
    X_test : shared preprocessed test features (used when candidates are plain models).
    config_path : path to model_config.yaml (None = auto-detect).

    Returns
    -------
    (winner_model, winner_name, comparison_dict)
        winner_model  — model instance with highest AUC-PR above threshold
        winner_name   — its key in candidates dict
        comparison_dict — metrics for all candidates (for MLflow logging)

    Raises
    ------
    RuntimeError if ALL candidates fail the AUC-PR threshold — caller must trigger rollback.
    """

    config = _load_config(config_path)
    threshold = config["validation"]["auc_pr_threshold"]

    comparison: dict[str, Any] = {}
    best_name: str | None = None
    best_auc_pr: float = -1.0
    best_model: Any = None

    for name, entry in candidates.items():
        # entry is either a plain model or a (model, X_test) tuple
        if isinstance(entry, tuple):
            model, X_test_model = entry
        else:
            model, X_test_model = entry, X_test
        try:
            y_prob = model.predict_proba(X_test_model)
            metrics, passed = validate_model(np.asarray(y_test), y_prob, config_path)
            comparison[name] = {"metrics": metrics, "passed": passed}
            logger.info(
                "Model '%s' — AUC-PR: %.4f, passed: %s", name, metrics["auc_pr"], passed
            )
            if passed and metrics["auc_pr"] > best_auc_pr:
                best_auc_pr = metrics["auc_pr"]
                best_name = name
                best_model = model
        except Exception as e:
            logger.error("Model '%s' evaluation failed: %s", name, e)
            comparison[name] = {"metrics": {}, "passed": False, "error": str(e)}

    if best_model is None:
        raise RuntimeError(
            f"All candidate models failed AUC-PR threshold ({threshold}). "
            f"Results: { {n: c.get('metrics', {}).get('auc_pr', 'N/A') for n, c in comparison.items()} }. "
            "Trigger rollback to previous production version."
        )

    logger.info("Winner: '%s' (AUC-PR=%.4f)", best_name, best_auc_pr)
    return best_model, best_name, comparison


def main() -> None:
    import argparse

    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default="reports/validation")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    preds = pd.read_parquet(args.predictions)
    metrics, passed = validate_model(preds["y_true"].values, preds["y_prob"].values, args.config)

    result = ValidationResult(
        model_name="xgboost_pof", version="0.0.0", metrics=metrics,
        passed_validation=passed, passed_bias_gate=None,
        bias_report_path=None, visualization_paths={},
    )
    save_validation_report(result, args.output_dir)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
