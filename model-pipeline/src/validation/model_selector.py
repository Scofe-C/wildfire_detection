from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
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
    threshold = config["validation"]["decision_threshold"]
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
