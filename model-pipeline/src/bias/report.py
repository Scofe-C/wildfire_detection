from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def generate_bias_report(
    bias_result: dict[str, Any],
    run_id: str,
    model_version: str,
    input_data_stats: dict[str, Any] | None = None,
    previous_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "report_version": "1.0.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "model_version": model_version,
        "gate_result": bias_result["gate_result"],
        "metric": bias_result["metric"],
        "overall_fnr": bias_result["overall_fnr"],
        "per_group_fnr": bias_result["per_group_fnr"],
        "disparity_between_groups": bias_result["disparity_between_groups"],
        "max_allowed_disparity": bias_result["max_allowed_disparity"],
    }

    if bias_result["gate_result"] == "FAIL":
        rca: dict[str, Any] = {
            "failure_type": "bias_gate",
            "observed_disparity": bias_result["disparity_between_groups"],
            "threshold": bias_result["max_allowed_disparity"],
            "recommended_mitigations": [
                "1. class_weight adjustment (2x for Very High SOVI cells)",
                "2. Spatial-SMOTE if insufficient (NOT standard SMOTE)",
                "3. CorrelationRemover on SOVI-correlated features",
                "4. Re-run pipeline and verify disparity < 5%",
            ],
        }
        if input_data_stats:
            rca["input_data_stats"] = input_data_stats
        if previous_report:
            rca["delta_from_previous"] = {
                "previous_disparity": previous_report.get("disparity_between_groups"),
                "delta": (
                    bias_result["disparity_between_groups"]
                    - previous_report.get("disparity_between_groups", 0)
                ),
                "previous_run_id": previous_report.get("run_id"),
            }
        report["rca"] = rca

    return report


def save_bias_report(
    report: dict[str, Any],
    output_dir: str | Path,
    filename: str = "bias_gate_report.json",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / filename
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Bias report saved: %s (%s)", report_path, report["gate_result"])
    return report_path


def load_previous_report(
    report_dir: str | Path,
    filename: str = "bias_gate_report.json",
) -> dict[str, Any] | None:
    path = Path(report_dir) / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None
