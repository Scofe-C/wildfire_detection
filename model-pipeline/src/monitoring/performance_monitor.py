"""Prediction distribution drift monitoring (proxy for model decay).

Since ground-truth fire labels arrive weeks after inference, we cannot compute
real-time AUC-PR. Instead, we track shifts in the predicted score distribution
as an early-warning signal that the model may be degrading.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PredictionDriftReport:
    baseline_mean: float
    current_mean: float
    mean_shift: float
    baseline_critical_rate: float
    current_critical_rate: float
    critical_rate_ratio: float
    verdict: str  # "OK" | "WARNING" | "CRITICAL"
    details: dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Tracks prediction score distribution to flag potential model decay."""

    def __init__(
        self,
        mean_shift_threshold: float = 0.1,
        critical_rate_multiplier: float = 2.0,
        critical_score_threshold: float = 0.75,
    ):
        self._mean_shift_threshold = mean_shift_threshold
        self._critical_rate_multiplier = critical_rate_multiplier
        self._critical_score_threshold = critical_score_threshold

    def check(
        self,
        baseline_stats: dict[str, Any],
        current_scores: np.ndarray,
    ) -> PredictionDriftReport:
        """Compare current prediction scores against baseline distribution stats.

        Parameters
        ----------
        baseline_stats:
            Dict with keys ``mean``, ``std``, ``critical_rate`` saved at training time.
        current_scores:
            Array of model output probabilities from the latest inference run.
        """
        baseline_mean = float(baseline_stats.get("mean", 0.0))
        baseline_critical_rate = float(baseline_stats.get("critical_rate", 0.0))

        current_mean = float(np.mean(current_scores))
        current_critical_rate = float(np.mean(current_scores >= self._critical_score_threshold))
        mean_shift = abs(current_mean - baseline_mean)
        critical_rate_ratio = (
            current_critical_rate / baseline_critical_rate
            if baseline_critical_rate > 0
            else 1.0
        )

        verdict = "OK"
        if (
            mean_shift > self._mean_shift_threshold
            or critical_rate_ratio > self._critical_rate_multiplier
        ):
            verdict = "CRITICAL"
        elif mean_shift > self._mean_shift_threshold * 0.5:
            verdict = "WARNING"

        return PredictionDriftReport(
            baseline_mean=baseline_mean,
            current_mean=current_mean,
            mean_shift=mean_shift,
            baseline_critical_rate=baseline_critical_rate,
            current_critical_rate=current_critical_rate,
            critical_rate_ratio=critical_rate_ratio,
            verdict=verdict,
            details={
                "baseline_std": baseline_stats.get("std", 0.0),
                "current_std": float(np.std(current_scores)),
            },
        )


def save_prediction_baseline(
    scores: np.ndarray,
    run_id: str,
    gcs_bucket: str,
    gcs_prefix: str,
    critical_score_threshold: float = 0.75,
) -> str:
    """Save prediction distribution stats to GCS."""
    import json

    from google.cloud import storage  # type: ignore[import]

    payload = {
        "run_id": run_id,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "critical_rate": float(np.mean(scores >= critical_score_threshold)),
        "n_samples": len(scores),
    }
    blob_path = f"{gcs_prefix}/{run_id}/prediction_baseline.json"
    client = storage.Client()
    client.bucket(gcs_bucket).blob(blob_path).upload_from_string(
        json.dumps(payload), content_type="application/json",
    )
    logger.info("Prediction baseline saved → gs://%s/%s", gcs_bucket, blob_path)
    return f"gs://{gcs_bucket}/{blob_path}"


def load_prediction_baseline(gcs_bucket: str, gcs_prefix: str, run_id: str) -> dict[str, Any]:
    """Load prediction baseline from GCS."""
    import json

    from google.cloud import storage  # type: ignore[import]

    blob_path = f"{gcs_prefix}/{run_id}/prediction_baseline.json"
    data = storage.Client().bucket(gcs_bucket).blob(blob_path).download_as_text()
    return json.loads(data)