"""
Bias check — inline FNR-per-slice using only pandas and sklearn.

No FairLearn, no geopandas, no external shapefiles required.

Slices evaluated:
  1. region            — california vs texas (training data imbalance risk)
  2. fire_season       — May–Oct vs Nov–Apr (must not fail when risk is highest)
  3. fuel_model_fbfm40 — fuel/vegetation type (high-risk types must not be underdetected)

All three are plain columns already present in the test DataFrame.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

logger = logging.getLogger(__name__)

FIRE_SEASON_MONTHS = {5, 6, 7, 8, 9, 10}   # May–October
MIN_GROUP_SIZE     = 20                       # skip groups too small for reliable FNR
MIN_FIRE_COUNT     = 5                        # skip groups with fewer than this many actual fire events


def _fnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """False Negative Rate = 1 - Recall.  0 = perfect, 1 = misses all fires."""
    return 1.0 - float(recall_score(y_true, y_pred, zero_division=0.0))


def run_bias_check(
    pred_df: pd.DataFrame,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    max_disparity: float = 0.05,
) -> tuple[dict[str, Any], bool]:
    """Compute FNR disparity across three domain-relevant slices.

    Parameters
    ----------
    pred_df : DataFrame containing predictions plus slice columns.
              Required columns: y_true, y_pred.
              Slice columns (used if present): region, timestamp, fuel_model_fbfm40.
    y_true_col : column name for true labels.
    y_pred_col : column name for binary predictions.
    max_disparity : maximum allowed FNR gap between any two groups within a slice.
                    Gate fails if ANY slice exceeds this.

    Returns
    -------
    (report, passed)
        report : dict with per-slice results for MLflow logging.
        passed : True if all slices are within max_disparity.
    """
    y_true = pred_df[y_true_col].values
    y_pred = pred_df[y_pred_col].values

    overall_fnr = _fnr(y_true, y_pred)
    report: dict[str, Any] = {
        "overall_fnr": overall_fnr,
        "max_disparity_threshold": max_disparity,
        "slices": {},
        "gate_result": "PASS",
    }
    gate_passed = True

    # ── Slice 1: Region ───────────────────────────────────────────────────────
    if "region" in pred_df.columns:
        slice_result = _compute_slice_fnr(pred_df, "region", y_true_col, y_pred_col)
        disparity = slice_result["disparity"]
        passed = disparity <= max_disparity
        slice_result["gate_result"] = "PASS" if passed else "FAIL"
        report["slices"]["region"] = slice_result

        level = logger.info if passed else logger.warning
        level(
            "BIAS [region] %s — disparity: %.4f  %s",
            slice_result["gate_result"], disparity,
            {k: f"{v:.3f}" for k, v in slice_result["per_group_fnr"].items()},
        )
        if not passed:
            gate_passed = False
            report["gate_result"] = "FAIL"

    # ── Slice 2: Fire season ──────────────────────────────────────────────────
    if "timestamp" in pred_df.columns:
        ts = pd.to_datetime(pred_df["timestamp"], utc=True)
        season = np.where(ts.dt.month.isin(FIRE_SEASON_MONTHS), "fire_season", "off_season")
        tmp = pred_df.copy()
        tmp["_season"] = season

        if len(np.unique(season)) > 1:
            slice_result = _compute_slice_fnr(tmp, "_season", y_true_col, y_pred_col)
            disparity = slice_result["disparity"]
            passed = disparity <= max_disparity
            slice_result["gate_result"] = "PASS" if passed else "FAIL"
            report["slices"]["fire_season"] = slice_result

            level = logger.info if passed else logger.warning
            level(
                "BIAS [fire_season] %s — disparity: %.4f  %s",
                slice_result["gate_result"], disparity,
                {k: f"{v:.3f}" for k, v in slice_result["per_group_fnr"].items()},
            )
            if not passed:
                gate_passed = False
                report["gate_result"] = "FAIL"
        else:
            logger.info("BIAS [fire_season] — only one season in test set, skipping")

    # ── Slice 3: Fuel / vegetation type ───────────────────────────────────────
    if "fuel_model_fbfm40" in pred_df.columns:
        fuel = pred_df["fuel_model_fbfm40"].astype(str)
        valid_groups = fuel.value_counts()
        valid_groups = valid_groups[valid_groups >= MIN_GROUP_SIZE].index
        tmp = pred_df[fuel.isin(valid_groups)].copy()

        if len(tmp) > 0 and tmp["fuel_model_fbfm40"].nunique() > 1:
            slice_result = _compute_slice_fnr(tmp, "fuel_model_fbfm40", y_true_col, y_pred_col)
            disparity = slice_result["disparity"]
            passed = disparity <= max_disparity
            slice_result["gate_result"] = "PASS" if passed else "FAIL"
            slice_result["n_groups_evaluated"] = len(valid_groups)
            report["slices"]["fuel_model_fbfm40"] = slice_result

            level = logger.info if passed else logger.warning
            level(
                "BIAS [fuel_model_fbfm40] %s — disparity: %.4f  (%d fuel types)",
                slice_result["gate_result"], disparity, len(valid_groups),
            )
            if not passed:
                gate_passed = False
                report["gate_result"] = "FAIL"

    overall = "PASSED" if gate_passed else "FAILED"
    logger.info("BIAS GATE %s — overall_fnr: %.4f", overall, overall_fnr)
    return report, gate_passed


def _compute_slice_fnr(
    df: pd.DataFrame,
    group_col: str,
    y_true_col: str,
    y_pred_col: str,
) -> dict[str, Any]:
    """Compute FNR per group and return disparity (max - min)."""
    per_group: dict[str, float] = {}
    for group, gdf in df.groupby(group_col):
        if len(gdf) < MIN_GROUP_SIZE:
            continue
        if gdf[y_true_col].sum() < MIN_FIRE_COUNT:
            continue
        per_group[str(group)] = _fnr(gdf[y_true_col].values, gdf[y_pred_col].values)

    if not per_group:
        return {"per_group_fnr": {}, "disparity": 0.0}

    values = list(per_group.values())
    return {
        "per_group_fnr": per_group,
        "disparity": float(max(values) - min(values)),
        "worst_group": max(per_group, key=per_group.get),
        "best_group":  min(per_group, key=per_group.get),
    }
