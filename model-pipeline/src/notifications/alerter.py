from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

COLORS = {
    "bias_gate_failure": "#FF0000",
    "validation_failure": "#FF4500",
    "pipeline_error": "#FF0000",
    "rollback": "#FF8C00",
    "shap_drift": "#FFD700",
    "success": "#36A64F",
}


def _load_config() -> dict[str, Any]:
    cfg = Path(__file__).resolve().parents[2] / "configs" / "model_config.yaml"
    with open(cfg) as f:
        return yaml.safe_load(f)["notifications"]


class SlackAlerter:
    def __init__(self, webhook_url: str | None = None):
        self._webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        config = _load_config()
        self._enabled = config.get("enabled", True)
        self._alert_types = config.get("alert_on", [])

    def _send(self, payload: dict[str, Any]) -> bool:
        if not self._enabled or not self._webhook_url:
            logger.warning("Slack alert skipped — webhook not configured")
            return False
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._webhook_url, data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error("Slack alert failed: %s", e)
            return False

    def _build(
        self, alert_type: str, title: str, fields: dict[str, str], run_id: str | None = None,
    ) -> dict[str, Any]:
        ts = datetime.now(UTC).isoformat()
        att_fields = [{"title": k, "value": str(v), "short": True} for k, v in fields.items()]
        if run_id:
            att_fields.insert(0, {"title": "Run ID", "value": run_id, "short": True})
        return {
            "text": f":fire: *Wildfire Model Pipeline — {title}*",
            "attachments": [{
                "color": COLORS.get(alert_type, "#808080"),
                "fields": att_fields,
                "footer": f"Model Pipeline | {ts}",
            }],
        }

    def alert_bias_gate_failure(
        self, run_id: str, disparity: float, threshold: float, per_group: dict[str, float],
    ):
        if "bias_gate_failure" not in self._alert_types:
            return
        fields = {
            "Disparity": f"{disparity:.4f}", "Threshold": f"{threshold:.4f}",
            "Status": "BLOCKED",
        }
        for g, fnr in per_group.items():
            fields[f"FNR ({g})"] = f"{fnr:.4f}"
        self._send(self._build("bias_gate_failure", "Bias Gate FAILED", fields, run_id))

    def alert_validation_failure(self, run_id: str, auc_pr: float, threshold: float):
        if "validation_failure" not in self._alert_types:
            return
        self._send(self._build("validation_failure", "Validation FAILED", {
            "AUC-PR": f"{auc_pr:.4f}", "Threshold": f"{threshold:.4f}",
        }, run_id))

    def alert_pipeline_error(self, run_id: str, error_message: str, stage: str):
        if "pipeline_error" not in self._alert_types:
            return
        self._send(self._build("pipeline_error", f"Error in {stage}", {
            "Stage": stage, "Error": error_message[:200],
        }, run_id))

    def alert_rollback(
        self, run_id: str, reason: str, from_version: str, to_version: str,
        delta_auc_pr: float | None = None,
    ):
        if "rollback" not in self._alert_types:
            return
        fields: dict[str, str] = {"Reason": reason, "From": from_version, "To": to_version}
        if delta_auc_pr is not None:
            fields["Delta AUC-PR"] = f"{delta_auc_pr:+.4f}"
        self._send(self._build("rollback", "Model ROLLBACK", fields, run_id))

    def alert_shap_drift(self, run_id: str, feature: str, importance: float, threshold: float):
        if "shap_drift" not in self._alert_types:
            return
        self._send(self._build("shap_drift", "SHAP Drift", {
            "Feature": feature, "Importance": f"{importance:.4f}", "Threshold": f"{threshold:.4f}",
        }, run_id))

    def alert_success(self, run_id: str, model_version: str, auc_pr: float):
        self._send(self._build("success", "Pipeline SUCCESS", {
            "Version": model_version, "AUC-PR": f"{auc_pr:.4f}", "Bias Gate": "PASSED",
        }, run_id))
