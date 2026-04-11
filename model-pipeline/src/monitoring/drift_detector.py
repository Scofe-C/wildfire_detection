"""PSI-based data drift detection for wildfire model features."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_N_BINS = 10


@dataclass
class FeatureDriftResult:
    feature: str
    psi: float
    verdict: str  # "OK" | "WARNING" | "CRITICAL"
    baseline_counts: list[int] = field(default_factory=list)
    current_counts: list[int] = field(default_factory=list)


@dataclass
class DriftReport:
    overall_psi: float
    verdict: str  # "OK" | "WARNING" | "CRITICAL"
    feature_results: list[FeatureDriftResult] = field(default_factory=list)

    @property
    def drifted_features(self) -> list[str]:
        return [r.feature for r in self.feature_results if r.verdict != "OK"]


def _compute_psi(baseline: np.ndarray, current: np.ndarray, n_bins: int = _N_BINS) -> float:
    """Population Stability Index between two distributions.

    PSI < 0.1   → stable
    PSI 0.1-0.25 → moderate shift (WARNING)
    PSI > 0.25  → significant shift (CRITICAL)
    """
    eps = 1e-8
    # Build quantile-based bins from baseline
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(baseline, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
    current_counts, _ = np.histogram(current, bins=bin_edges)

    baseline_pct = baseline_counts / (baseline_counts.sum() + eps)
    current_pct = current_counts / (current_counts.sum() + eps)

    psi = float(np.sum((current_pct - baseline_pct) * np.log((current_pct + eps) / (baseline_pct + eps))))
    return psi


def _verdict(psi: float, warning_thresh: float, critical_thresh: float) -> str:
    if psi >= critical_thresh:
        return "CRITICAL"
    if psi >= warning_thresh:
        return "WARNING"
    return "OK"


class DriftDetector:
    """Compares current inference feature distributions against a saved training baseline."""

    def __init__(self, warning_threshold: float = 0.1, critical_threshold: float = 0.25):
        self._warning = warning_threshold
        self._critical = critical_threshold

    def detect(
        self,
        baseline_stats: dict[str, Any],
        current_data: dict[str, np.ndarray],
    ) -> DriftReport:
        """Run PSI drift detection per feature.

        Parameters
        ----------
        baseline_stats:
            Dict with key ``samples`` mapping feature name → list of baseline values,
            as saved by :func:`save_baseline`.
        current_data:
            Dict mapping feature name → numpy array of current inference values.
        """
        results: list[FeatureDriftResult] = []
        baseline_samples: dict[str, list[float]] = baseline_stats.get("samples", {})

        for feature, baseline_vals in baseline_samples.items():
            if feature not in current_data:
                logger.debug("Feature %s in baseline but not in current data — skipping", feature)
                continue
            baseline_arr = np.array(baseline_vals, dtype=float)
            current_arr = np.array(current_data[feature], dtype=float)

            if len(baseline_arr) < 2 or len(current_arr) < 2:
                continue

            psi = _compute_psi(baseline_arr, current_arr)
            verdict = _verdict(psi, self._warning, self._critical)
            results.append(FeatureDriftResult(
                feature=feature,
                psi=psi,
                verdict=verdict,
            ))
            logger.debug("PSI[%s] = %.4f (%s)", feature, psi, verdict)

        if not results:
            return DriftReport(overall_psi=0.0, verdict="OK", feature_results=[])

        overall_psi = float(np.mean([r.psi for r in results]))
        overall_verdict = _verdict(overall_psi, self._warning, self._critical)
        return DriftReport(overall_psi=overall_psi, verdict=overall_verdict, feature_results=results)


def save_baseline(
    train_df: Any,
    features: list[str],
    run_id: str,
    gcs_bucket: str,
    gcs_prefix: str,
) -> str:
    """Save per-feature sample distributions to GCS as a JSON baseline.

    Returns the GCS path written.
    """
    import json

    from google.cloud import storage  # type: ignore[import]

    samples: dict[str, list[float]] = {}
    for feat in features:
        if feat in train_df.columns:
            samples[feat] = train_df[feat].dropna().tolist()

    payload = {"run_id": run_id, "samples": samples}
    blob_path = f"{gcs_prefix}/{run_id}/feature_baseline.json"

    client = storage.Client()
    bucket = client.bucket(gcs_bucket)
    bucket.blob(blob_path).upload_from_string(
        json.dumps(payload), content_type="application/json",
    )
    logger.info("Feature baseline saved → gs://%s/%s", gcs_bucket, blob_path)
    return f"gs://{gcs_bucket}/{blob_path}"


def load_baseline(gcs_bucket: str, gcs_prefix: str, run_id: str) -> dict[str, Any]:
    """Load a previously saved feature baseline from GCS."""
    import json

    from google.cloud import storage  # type: ignore[import]

    blob_path = f"{gcs_prefix}/{run_id}/feature_baseline.json"
    client = storage.Client()
    data = client.bucket(gcs_bucket).blob(blob_path).download_as_text()
    return json.loads(data)
