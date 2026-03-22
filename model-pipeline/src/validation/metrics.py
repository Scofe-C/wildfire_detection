from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_auc_pr(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return float(auc(recall, precision))


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, zero_division=0.0))


def compute_fnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fn, tp = cm[1, 0], cm[1, 1]
    if (fn + tp) == 0:
        return 0.0
    return float(fn / (fn + tp))


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "true_negatives": int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_positives": int(cm[1, 1]),
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    inference_latency_ms: float | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics: dict[str, Any] = {
        "auc_pr": compute_auc_pr(y_true, y_prob),
        "f1": compute_f1(y_true, y_pred),
        "fnr": compute_fnr(y_true, y_pred),
        "accuracy": float(np.mean(y_pred == y_true)),
        "confusion_matrix": compute_confusion_matrix(y_true, y_pred),
        "positive_rate": float(y_true.mean()),
        "threshold": threshold,
        "n_samples": len(y_true),
    }

    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc_roc"] = None

    if inference_latency_ms is not None:
        metrics["inference_latency_ms"] = inference_latency_ms

    return metrics


def measure_inference_latency(
    predict_fn: Callable, X: pd.DataFrame, n_runs: int = 3,
) -> float:
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        predict_fn(X)
        latencies.append((time.perf_counter() - start) * 1000)
    avg = float(np.mean(latencies))
    logger.info("Inference latency: %.1f ms avg (%d runs)", avg, n_runs)
    return avg
