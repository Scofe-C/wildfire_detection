from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from fairlearn.metrics import MetricFrame
from sklearn.metrics import recall_score

logger = logging.getLogger(__name__)


class BiasGateError(Exception):
    pass


def false_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 1.0 - recall_score(y_true, y_pred, zero_division=0.0)


def run_bias_gate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    config_path: str | Path | None = None,
) -> tuple[dict[str, Any], bool]:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "model_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    max_disparity = config["bias_gate"]["max_disparity"]

    mf = MetricFrame(
        metrics=false_negative_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    per_group = mf.by_group.to_dict()
    disparity = mf.difference(method="between_groups")

    report: dict[str, Any] = {
        "metric": "false_negative_rate",
        "overall_fnr": float(mf.overall),
        "per_group_fnr": {str(k): float(v) for k, v in per_group.items()},
        "disparity_between_groups": float(disparity),
        "max_allowed_disparity": max_disparity,
        "gate_result": "PASS" if disparity <= max_disparity else "FAIL",
    }

    passed = disparity <= max_disparity
    level = logger.info if passed else logger.warning
    level("BIAS GATE %s — disparity: %.4f (threshold: %.4f)",
          "PASSED" if passed else "FAILED", disparity, max_disparity)
    for group, fnr in per_group.items():
        logger.info("  %s: FNR=%.4f", group, fnr)

    return report, passed


def run_bias_gate_from_dataframe(
    df: pd.DataFrame,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    sensitive_col: str = "nri_vulnerability_quartile",
    config_path: str | Path | None = None,
) -> tuple[dict[str, Any], bool]:
    mask = df[sensitive_col] != "Unknown"
    if mask.sum() < len(df):
        logger.warning("Excluding %d rows with unknown vulnerability", len(df) - mask.sum())
    df_filtered = df[mask]
    if len(df_filtered) == 0:
        raise BiasGateError("No samples with valid vulnerability assignments")

    return run_bias_gate(
        y_true=df_filtered[y_true_col].values,
        y_pred=df_filtered[y_pred_col].values,
        sensitive_features=df_filtered[sensitive_col].values,
        config_path=config_path,
    )


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default="reports/bias_gate")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    df = pd.read_parquet(args.predictions)
    report, passed = run_bias_gate_from_dataframe(df, config_path=args.config)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "bias_gate_report.json", "w") as f:
        json.dump(report, f, indent=2)

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
