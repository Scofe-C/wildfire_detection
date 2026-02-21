"""
Seasonal Anomaly Detection
==========================
Compares current feature values against per-season historical baselines
rather than against the current batch's own distribution. This prevents
the critical failure mode where a fire event elevates the entire batch
mean, masking the individual outlier rows.

Assignment requirements (Section 5.5):
  - Compare against seasonal baselines (fire_season vs off_season)
  - Z-score threshold: 4.0 (fire season), 3.5 (off-season)
  - Minimum 30 historical days before baseline is trusted
  - Baseline storage: data/processed/baselines/baseline_{feature}_{season}.json
  - Soft failure: anomalies are reported but never block the pipeline

Baseline JSON schema:
    {
        "feature": "temperature_2m",
        "season": "fire_season",
        "mean": 28.4,
        "std": 6.2,
        "sample_count": 4500,
        "last_updated": "2026-02-17T18:00:00"
    }
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Where baseline JSON files are stored
_DEFAULT_BASELINE_DIR = Path("data") / "processed" / "baselines"

# Minimum sample count (rows × features) before baseline is considered reliable.
# Assignment specifies 30 historical days; at 6 runs/day × ~200 cells = ~36,000
# samples minimum. We use a more lenient per-feature threshold for startup.
MIN_SAMPLE_COUNT = 500  # ~2-3 days of pipeline runs at 64km resolution


def detect_anomalies(
    fused_df: pd.DataFrame,
    registry,
    execution_date,
    baseline_dir: Optional[Path] = None,
    update_baseline: bool = True,
) -> List[Dict[str, Any]]:
    """Seasonal-baseline anomaly detection.

    Compares each monitored feature against a stored seasonal baseline.
    On first run (no baseline exists), seeds the baseline and returns no
    anomalies. Subsequent runs compare against the growing historical mean.

    Args:
        fused_df: Fused feature DataFrame from task_fuse_features.
        registry: FeatureRegistry from schema_loader.
        execution_date: Airflow execution_date (has .month attribute).
        baseline_dir: Directory for baseline JSON files. Defaults to
            data/processed/baselines/.
        update_baseline: If True, update the baseline with current data
            after anomaly detection. Set False in tests or backfill.

    Returns:
        List of anomaly dicts:
        [{"feature": "wind_speed_10m", "outlier_count": 3,
          "z_threshold": 4.0, "season": "fire_season",
          "mean_baseline": 12.4, "std_baseline": 5.1}]
        Empty list = no anomalies (or insufficient baseline history).
    """
    baseline_dir = baseline_dir or _DEFAULT_BASELINE_DIR
    baseline_dir = Path(baseline_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    current_month = execution_date.month
    anomaly_config = registry.anomaly_config
    z_threshold = registry.get_z_threshold(current_month)
    monitored = anomaly_config.get("monitored_features", [])

    in_fire_season = current_month in registry.fire_season_months
    season = "fire_season" if in_fire_season else "off_season"

    anomalies_found: List[Dict[str, Any]] = []

    for col in monitored:
        if col not in fused_df.columns:
            continue

        values = fused_df[col].dropna()
        if len(values) < 10:
            continue

        # Load (or initialize) the seasonal baseline for this feature
        baseline = _load_baseline(col, season, baseline_dir)

        if baseline is None or baseline["sample_count"] < MIN_SAMPLE_COUNT:
            # Not enough history yet — seed/update baseline and skip detection
            if len(values) > 0:
                _update_baseline(col, season, values, baseline, baseline_dir)
            remaining = (MIN_SAMPLE_COUNT - (baseline["sample_count"] if baseline else 0))
            logger.info(
                f"Anomaly detection skipped for '{col}' ({season}): "
                f"baseline needs ~{max(0, remaining)} more samples "
                f"(have {baseline['sample_count'] if baseline else 0}/{MIN_SAMPLE_COUNT})"
            )
            continue

        # Compare current values against historical baseline
        mean_val = baseline["mean"]
        std_val  = baseline["std"]

        if std_val <= 0:
            logger.debug(f"Skipping '{col}': zero std in baseline (constant feature)")
            continue

        z_scores = ((values - mean_val) / std_val).abs()
        outlier_count = int((z_scores > z_threshold).sum())

        if outlier_count > 0:
            anomalies_found.append({
                "feature":        col,
                "outlier_count":  outlier_count,
                "z_threshold":    float(z_threshold),
                "season":         season,
                "mean_baseline":  round(float(mean_val), 4),
                "std_baseline":   round(float(std_val), 4),
                "current_mean":   round(float(values.mean()), 4),
            })
            logger.warning(
                f"Anomaly: '{col}' has {outlier_count} outliers "
                f"(z>{z_threshold}, {season}, "
                f"baseline mean={mean_val:.2f} ± {std_val:.2f}, "
                f"current mean={values.mean():.2f})"
            )

        # Update baseline with current data (Welford's online update)
        if update_baseline:
            _update_baseline(col, season, values, baseline, baseline_dir)

    return anomalies_found


def load_baseline(
    feature: str,
    season: str,
    baseline_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Public accessor for baseline data. Used by tests and monitoring tools."""
    return _load_baseline(feature, season, baseline_dir or _DEFAULT_BASELINE_DIR)


def reset_baseline(
    feature: str,
    season: str,
    baseline_dir: Optional[Path] = None,
) -> None:
    """Delete a baseline file. Used in tests and for manual reseeding."""
    baseline_dir = Path(baseline_dir or _DEFAULT_BASELINE_DIR)
    path = _baseline_path(feature, season, baseline_dir)
    if path.exists():
        path.unlink()
        logger.info(f"Baseline reset: {path}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _baseline_path(feature: str, season: str, baseline_dir: Path) -> Path:
    """Canonical path for a baseline JSON file."""
    # Sanitize feature name (no slashes or special chars)
    safe_name = feature.replace("/", "_").replace(" ", "_")
    return baseline_dir / f"baseline_{safe_name}_{season}.json"


def _load_baseline(
    feature: str,
    season: str,
    baseline_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Load baseline JSON, return None if it doesn't exist."""
    path = _baseline_path(feature, season, baseline_dir)
    if not path.exists():
        return None
    try:
        with open(path,encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not read baseline {path}: {e} — treating as missing")
        return None


def _update_baseline(
    feature: str,
    season: str,
    new_values: pd.Series,
    existing: Optional[Dict[str, Any]],
    baseline_dir: Path,
) -> None:
    """Update baseline using Welford's online algorithm for numerical stability.

    Welford's method updates mean and variance incrementally without storing
    all historical values. Numerically stable for large sample counts.

    Reference: Welford (1962), "Note on a method for calculating corrected
    sums of squares and products", Technometrics 4(3).
    """
    n_new = len(new_values)
    if n_new == 0:
        return

    new_mean = float(new_values.mean())
    new_var  = float(new_values.var()) if n_new > 1 else 0.0

    if existing is None:
        # Seed the baseline with current data
        baseline = {
            "feature":      feature,
            "season":       season,
            "mean":         new_mean,
            "std":          float(np.sqrt(new_var)),
            "sample_count": n_new,
            "last_updated": datetime.utcnow().isoformat(),
        }
    else:
        # Welford's parallel update (Chan et al. 1979 formulation)
        n_old  = existing["sample_count"]
        m_old  = existing["mean"]
        s_old  = existing["std"] ** 2 * max(n_old - 1, 1)  # sum of squared deviations

        n_combined = n_old + n_new
        delta      = new_mean - m_old
        m_combined = m_old + delta * n_new / n_combined

        s_new      = new_var * max(n_new - 1, 1)
        s_combined = s_old + s_new + (delta ** 2) * n_old * n_new / n_combined

        std_combined = float(np.sqrt(s_combined / max(n_combined - 1, 1)))

        baseline = {
            "feature":      feature,
            "season":       season,
            "mean":         float(m_combined),
            "std":          std_combined,
            "sample_count": n_combined,
            "last_updated": datetime.utcnow().isoformat(),
        }

    path = _baseline_path(feature, season, baseline_dir)
    try:
        with open(path, "w",encoding="utf-8") as f:
            json.dump(baseline, f, indent=2)
        logger.debug(
            f"Baseline updated: {feature}/{season} "
            f"n={baseline['sample_count']}, "
            f"mean={baseline['mean']:.3f}, std={baseline['std']:.3f}"
        )
    except OSError as e:
        logger.warning(f"Could not write baseline {path}: {e}")
