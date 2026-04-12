"""Orchestrates drift check + alerting + auto-retrain trigger."""
from __future__ import annotations

import logging
import os
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "monitoring_config.yaml"
_COOLDOWN_FILE = Path(__file__).resolve().parents[2] / "reports" / ".last_retrain_ts"


def _load_monitoring_config() -> dict[str, Any]:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _check_cooldown(cooldown_hours: int) -> bool:
    """Return True if enough time has passed since the last retrain trigger."""
    if not _COOLDOWN_FILE.exists():
        return True
    last_ts = float(_COOLDOWN_FILE.read_text().strip())
    elapsed_hours = (time.time() - last_ts) / 3600
    return elapsed_hours >= cooldown_hours


def _record_retrain_trigger() -> None:
    _COOLDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
    _COOLDOWN_FILE.write_text(str(time.time()))


def _trigger_github_retrain(repo: str, workflow_file: str, github_token: str) -> bool:
    """POST to GitHub Actions workflow_dispatch API to trigger retraining."""
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_file}/dispatches"
    payload = '{"ref":"master","inputs":{"triggered_by":"drift_detection"}}'.encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            success = resp.status == 204
            logger.info("Retrain trigger response: %s", resp.status)
            return success
    except Exception as e:
        logger.error("Failed to trigger GitHub retrain: %s", e)
        return False


def resolve_latest_baseline_run_id(bucket: str, baseline_prefix: str) -> str | None:
    """List baseline directories in GCS and return the newest run_id.

    Baselines are stored at ``{baseline_prefix}/{run_id}/feature_baseline.json``.
    We list all blobs under the prefix, extract unique run_id subdirectories,
    and return the lexicographically last one (UUIDs and timestamps sort correctly).

    Returns None if no baselines exist.
    """
    try:
        from google.cloud import storage  # type: ignore[import]

        client = storage.Client()
        blobs = client.list_blobs(bucket, prefix=baseline_prefix + "/")
        run_ids: set[str] = set()
        for blob in blobs:
            # Path: baselines/{run_id}/feature_baseline.json
            parts = blob.name[len(baseline_prefix):].strip("/").split("/")
            if len(parts) >= 2:
                run_ids.add(parts[0])

        if not run_ids:
            logger.warning("No baselines found under gs://%s/%s", bucket, baseline_prefix)
            return None

        latest = sorted(run_ids)[-1]
        logger.info("Resolved latest baseline run_id: %s (from %d candidates)", latest, len(run_ids))
        return latest
    except Exception as e:
        logger.error("Failed to resolve latest baseline: %s", e)
        return None


def run_monitoring_check(
    run_id: str | None = None,
    gcs_bucket: str | None = None,
    baseline_run_id: str | None = None,
) -> dict[str, Any]:
    """Main monitoring entry point — called by Cloud Scheduler → Cloud Run.

    1. Load latest inference predictions from GCS
    2. Load training baseline stats from GCS
    3. Run PSI-based feature drift detection
    4. Run prediction distribution drift check
    5. Alert via Slack if WARNING or CRITICAL
    6. Trigger auto-retrain if CRITICAL (with cooldown)
    7. Log results to MLflow

    Returns a summary dict suitable for an HTTP JSON response.
    """
    from src.monitoring.drift_detector import DriftDetector, load_baseline
    from src.monitoring.performance_monitor import PerformanceMonitor, load_prediction_baseline
    from src.notifications.alerter import SlackAlerter
    from src.preprocessing.feature_engineering import FEATURES

    cfg = _load_monitoring_config()
    mon_cfg = cfg["monitoring"]
    retrain_cfg = cfg["retraining"]

    bucket = gcs_bucket or os.getenv("GCS_BUCKET_NAME", "wildfire-mlops-123")
    baseline_prefix = mon_cfg["baseline_gcs_prefix"]
    inference_prefix = mon_cfg["inference_gcs_prefix"]

    alerter = SlackAlerter()
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")

    result: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "feature_drift": None,
        "prediction_drift": None,
        "actions_taken": [],
    }

    # ── 1. Load baselines ──────────────────────────────────────────────────────
    # Resolve "latest" or empty baseline_run_id by scanning GCS
    if not baseline_run_id or baseline_run_id == "latest":
        baseline_run_id = resolve_latest_baseline_run_id(bucket, baseline_prefix)
        if not baseline_run_id:
            logger.warning("No baselines found in GCS — skipping drift check")
            result["error"] = "No baselines found in GCS. Run training first to generate baselines."
            return result
        logger.info("Using resolved baseline_run_id: %s", baseline_run_id)

    try:
        feature_baseline = load_baseline(bucket, baseline_prefix, baseline_run_id)
        pred_baseline = load_prediction_baseline(bucket, baseline_prefix, baseline_run_id)
    except Exception as e:
        logger.error("Failed to load baselines: %s", e)
        result["error"] = f"Baseline load failed: {e}"
        return result

    # ── 2. Load latest inference data from GCS ─────────────────────────────────
    try:
        import pandas as pd
        from google.cloud import storage  # type: ignore[import]

        client = storage.Client()
        blobs = sorted(
            client.list_blobs(bucket, prefix=inference_prefix),
            key=lambda b: b.time_created,
            reverse=True,
        )
        if not blobs:
            result["error"] = "No inference outputs found in GCS"
            return result

        latest_blob = blobs[0]
        content = latest_blob.download_as_bytes()
        import io
        inference_df = pd.read_parquet(io.BytesIO(content))
    except Exception as e:
        logger.error("Failed to load inference data: %s", e)
        result["error"] = f"Inference data load failed: {e}"
        return result

    # ── 3. Feature drift detection ─────────────────────────────────────────────
    detector = DriftDetector(
        warning_threshold=mon_cfg["psi_warning_threshold"],
        critical_threshold=mon_cfg["psi_critical_threshold"],
    )
    current_feature_data = {
        feat: inference_df[feat].dropna().to_numpy()
        for feat in FEATURES
        if feat in inference_df.columns
    }
    drift_report = detector.detect(feature_baseline, current_feature_data)
    result["feature_drift"] = {
        "overall_psi": drift_report.overall_psi,
        "verdict": drift_report.verdict,
        "drifted_features": drift_report.drifted_features,
        "n_features_checked": len(drift_report.feature_results),
    }

    # ── 4. Prediction distribution drift ──────────────────────────────────────
    if "fire_risk_score" in inference_df.columns:
        monitor = PerformanceMonitor(
            mean_shift_threshold=mon_cfg["prediction_mean_shift_threshold"],
            critical_rate_multiplier=mon_cfg["critical_rate_multiplier"],
        )
        scores = inference_df["fire_risk_score"].dropna().to_numpy()
        pred_report = monitor.check(pred_baseline, scores)
        result["prediction_drift"] = {
            "mean_shift": pred_report.mean_shift,
            "critical_rate_ratio": pred_report.critical_rate_ratio,
            "verdict": pred_report.verdict,
        }
        combined_verdict = (
            "CRITICAL"
            if "CRITICAL" in (drift_report.verdict, pred_report.verdict)
            else "WARNING"
            if "WARNING" in (drift_report.verdict, pred_report.verdict)
            else "OK"
        )
    else:
        combined_verdict = drift_report.verdict

    # ── 5. Alert if WARNING or CRITICAL ───────────────────────────────────────
    if combined_verdict != "OK":
        alerter.alert_data_drift(
            run_id=run_id,
            drifted_features=drift_report.drifted_features,
            overall_psi=drift_report.overall_psi,
            verdict=combined_verdict,
        )
        result["actions_taken"].append(f"slack_alert:{combined_verdict}")

    # ── 6. Auto-retrain on CRITICAL (with cooldown) ────────────────────────────
    if combined_verdict == "CRITICAL" and retrain_cfg.get("auto_trigger", False):
        cooldown_hours = retrain_cfg.get("cooldown_hours", 24)
        if _check_cooldown(cooldown_hours):
            github_token = os.getenv("GITHUB_TOKEN", "")
            if github_token:
                triggered = _trigger_github_retrain(
                    repo=retrain_cfg["github_repo"],
                    workflow_file=retrain_cfg["workflow_file"],
                    github_token=github_token,
                )
                if triggered:
                    _record_retrain_trigger()
                    result["actions_taken"].append("auto_retrain_triggered")
                    logger.info("[%s] Auto-retrain triggered via GitHub API", run_id)
            else:
                logger.warning("GITHUB_TOKEN not set — skipping auto-retrain trigger")
        else:
            logger.info("[%s] Retrain cooldown active — skipping auto-retrain", run_id)
            result["actions_taken"].append("retrain_skipped:cooldown")

    result["verdict"] = combined_verdict
    logger.info("[%s] Monitoring complete — verdict: %s", run_id, combined_verdict)
    return result