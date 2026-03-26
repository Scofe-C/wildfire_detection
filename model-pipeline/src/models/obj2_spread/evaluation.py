"""
Evaluation metrics for Cell2Fire spread simulation.

Primary metric  : Buffered IoU (15% buffer, threshold sweep)
Secondary metric: Dice coefficient (legacy, kept for backwards compat)
Supporting      : Directional accuracy, area ratio
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dice coefficient (legacy)
# ---------------------------------------------------------------------------

def compute_dice_coefficient(
    predicted_mask: np.ndarray,
    actual_mask: np.ndarray,
) -> float:
    """Dice coefficient between predicted and actual burn masks.

    Dice = 2 * |P ∩ A| / (|P| + |A|)

    Parameters
    ----------
    predicted_mask : np.ndarray
        Boolean or 0/1 array of predicted burned cells.
    actual_mask : np.ndarray
        Boolean or 0/1 array of actual burned cells.

    Returns
    -------
    float in [0, 1]. Returns 1.0 if both masks are empty.
    """
    pred = np.asarray(predicted_mask, dtype=bool).ravel()
    actual = np.asarray(actual_mask, dtype=bool).ravel()

    if pred.shape != actual.shape:
        raise ValueError(
            f"Shape mismatch: predicted {pred.shape} vs actual {actual.shape}"
        )

    intersection = np.sum(pred & actual)
    total = np.sum(pred) + np.sum(actual)

    if total == 0:
        return 1.0

    return float(2.0 * intersection / total)


# ---------------------------------------------------------------------------
# Buffered IoU (primary)
# ---------------------------------------------------------------------------

def compute_buffered_iou(
    pred_burn_prob: np.ndarray,
    actual_perimeter_gdf: Any,
    transform: Any,
    threshold: float = 0.10,
    buffer_pct: float = 0.15,
) -> dict[str, Any]:
    """Compute buffered IoU between predicted burn area and actual perimeter.

    Expands predicted burn area by buffer_pct before computing IoU.
    This implements the team lead's guidance to accept 10-15% error range
    without requiring pixel-perfect perimeter match.

    Parameters
    ----------
    pred_burn_prob : np.ndarray
        2D array of burn probabilities from Cell2Fire.
    actual_perimeter_gdf : gpd.GeoDataFrame
        Actual fire perimeter polygon(s).
    transform : rasterio.Affine
        Georeferencing transform of the burn probability grid.
    threshold : float
        Burn probability threshold to classify cell as burned.
    buffer_pct : float
        Buffer expansion factor. 0.15 = 15% of sqrt(predicted area).

    Returns
    -------
    dict with keys:
        buffered_iou, directional_accuracy, angle_diff_degrees,
        area_ratio, area_ratio_ok, pred_area_km2, actual_area_km2,
        gate_passed, threshold_used, buffer_pct_used
    """
    import rasterio.features
    from shapely.geometry import shape
    from shapely.ops import unary_union

    # Vectorise predicted burn area
    pred_binary = (pred_burn_prob >= threshold).astype(np.uint8)
    shapes = list(rasterio.features.shapes(pred_binary, transform=transform))
    pred_polygons = [shape(s) for s, v in shapes if v == 1]

    if not pred_polygons:
        return {
            "buffered_iou": 0.0,
            "directional_accuracy": False,
            "angle_diff_degrees": 180.0,
            "area_ratio": 0.0,
            "area_ratio_ok": False,
            "pred_area_km2": 0.0,
            "actual_area_km2": 0.0,
            "gate_passed": False,
            "threshold_used": threshold,
            "buffer_pct_used": buffer_pct,
            "reason": f"No predicted burn area at threshold={threshold}",
        }

    pred_union = unary_union(pred_polygons)
    actual_union = unary_union(actual_perimeter_gdf.geometry.values)

    # Buffer predicted area
    buffer_dist = buffer_pct * math.sqrt(pred_union.area)
    pred_buffered = pred_union.buffer(buffer_dist)

    # Buffered IoU
    intersection = pred_buffered.intersection(actual_union).area
    union_area = pred_buffered.union(actual_union).area
    buffered_iou = intersection / union_area if union_area > 0 else 0.0

    # Directional accuracy — angle from ignition to centroids
    pred_centroid = pred_union.centroid
    actual_centroid = actual_union.centroid
    ignition = pred_polygons[0].centroid  # smallest cluster ≈ ignition point

    angle_pred = math.degrees(
        math.atan2(pred_centroid.y - ignition.y, pred_centroid.x - ignition.x)
    )
    angle_actual = math.degrees(
        math.atan2(actual_centroid.y - ignition.y, actual_centroid.x - ignition.x)
    )

    angle_diff = abs(angle_pred - angle_actual) % 360
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    directional_accuracy = angle_diff <= 45.0

    # Area ratio — predicted vs actual size
    area_ratio = pred_union.area / actual_union.area if actual_union.area > 0 else 0.0
    area_ratio_ok = 0.70 <= area_ratio <= 1.30

    # Convert degrees² → km² (rough: 1° ≈ 111km)
    pred_area_km2 = pred_union.area * (111 ** 2)
    actual_area_km2 = actual_union.area * (111 ** 2)

    # Gate: buffered_iou >= 0.35 AND direction correct AND area in range
    gate_passed = buffered_iou >= 0.35 and directional_accuracy and area_ratio_ok

    return {
        "buffered_iou": round(buffered_iou, 4),
        "directional_accuracy": directional_accuracy,
        "angle_diff_degrees": round(angle_diff, 1),
        "area_ratio": round(area_ratio, 3),
        "area_ratio_ok": area_ratio_ok,
        "pred_area_km2": round(pred_area_km2, 2),
        "actual_area_km2": round(actual_area_km2, 2),
        "gate_passed": gate_passed,
        "threshold_used": threshold,
        "buffer_pct_used": buffer_pct,
    }


def find_best_threshold(
    pred_burn_prob: np.ndarray,
    actual_perimeter_gdf: Any,
    transform: Any,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """Sweep thresholds and return the result with the best buffered IoU.

    Parameters
    ----------
    pred_burn_prob : np.ndarray
        2D burn probability array from Cell2Fire.
    actual_perimeter_gdf : gpd.GeoDataFrame
        Actual fire perimeter.
    transform : rasterio.Affine
        Georeferencing transform.
    thresholds : list[float] | None
        Thresholds to try. Defaults to [0.05, 0.10, 0.15, 0.20, 0.25, 0.30].

    Returns
    -------
    Best result dict from compute_buffered_iou.
    """
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    best: dict[str, Any] = {"buffered_iou": 0.0, "threshold_used": thresholds[0]}

    for t in thresholds:
        result = compute_buffered_iou(
            pred_burn_prob, actual_perimeter_gdf, transform, threshold=t
        )
        logger.info(
            "  threshold=%.2f → buffered_iou=%.4f | dir=%s | "
            "area_ratio=%.3f | gate=%s",
            t,
            result["buffered_iou"],
            result["directional_accuracy"],
            result["area_ratio"],
            "PASS" if result["gate_passed"] else "FAIL",
        )
        if result["buffered_iou"] > best["buffered_iou"]:
            best = result

    logger.info(
        "Best threshold: %.2f → Buffered IoU: %.4f | Gate: %s",
        best["threshold_used"],
        best["buffered_iou"],
        "PASS" if best.get("gate_passed") else "FAIL",
    )
    return best
