from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_class_weights(
    y: pd.Series,
    sensitive_features: pd.Series,
    base_scale_pos_weight: float = 65.0,
    high_vulnerability_multiplier: float = 2.0,
) -> np.ndarray:
    weights = np.ones(len(y), dtype=np.float64)
    pos_mask = y == 1
    weights[pos_mask] = base_scale_pos_weight

    high_vuln_pos = pos_mask & (sensitive_features == "Very High")
    weights[high_vuln_pos] = base_scale_pos_weight * high_vulnerability_multiplier

    logger.info(
        "Weights — base_pos: %.1f, very_high_pos: %.1f, affected: %d/%d",
        base_scale_pos_weight,
        base_scale_pos_weight * high_vulnerability_multiplier,
        high_vuln_pos.sum(), len(y),
    )
    return weights


def apply_spatial_smote(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    h3_col: str = "h3_index",
    target_ratio: float = 0.1,
    k_neighbors: int = 5,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    import h3 as h3lib

    n_positive = (y == 1).sum()
    n_negative = (y == 0).sum()
    n_target = int(n_negative * target_ratio) - n_positive
    if n_target <= 0:
        logger.info("No oversampling needed")
        return X, y, metadata

    logger.info("Spatial-SMOTE: generating %d synthetic samples", n_target)

    pos_mask = y == 1
    pos_indices = y[pos_mask].index.tolist()
    rng = np.random.default_rng(seed=42)

    synthetic_rows = []
    synthetic_meta = []

    for _ in range(n_target):
        seed_idx = rng.choice(pos_indices)
        seed_h3 = metadata.loc[seed_idx, h3_col]
        neighbors = list(h3lib.grid_disk(seed_h3, k_neighbors))

        neighbor_mask = metadata[h3_col].isin(neighbors) & pos_mask
        neighbor_indices = y[neighbor_mask].index.tolist()

        if len(neighbor_indices) < 2:
            row = X.loc[seed_idx].copy()
            row += rng.normal(0, 0.01, size=len(row))
        else:
            partner_idx = rng.choice(neighbor_indices)
            alpha = rng.uniform(0.1, 0.9)
            row = X.loc[seed_idx] * alpha + X.loc[partner_idx] * (1 - alpha)

        synthetic_rows.append(row)
        synthetic_meta.append({h3_col: seed_h3})

    X_syn = pd.DataFrame(synthetic_rows, columns=X.columns)
    y_syn = pd.Series(np.ones(n_target, dtype=y.dtype), name=y.name)
    meta_syn = pd.DataFrame(synthetic_meta)

    X_out = pd.concat([X, X_syn], ignore_index=True)
    y_out = pd.concat([y, y_syn], ignore_index=True)
    meta_out = pd.concat([metadata, meta_syn], ignore_index=True)

    logger.info("Spatial-SMOTE done — %d rows (%.1f%% pos)", len(y_out), 100 * y_out.mean())
    return X_out, y_out, meta_out


def apply_correlation_remover(
    X: pd.DataFrame,
    sensitive_scores: pd.Series,
    threshold: float = 0.3,
) -> pd.DataFrame:
    from fairlearn.preprocessing import CorrelationRemover

    correlations = X.corrwith(sensitive_scores).abs()
    correlated = correlations[correlations > threshold].index.tolist()
    if not correlated:
        logger.info("No features correlated above %.2f", threshold)
        return X

    logger.info("CorrelationRemover targeting: %s", correlated)

    X_aug = X.copy()
    X_aug["_sovi"] = sensitive_scores.values
    cr = CorrelationRemover(sensitive_feature_ids=["_sovi"])
    result = cr.fit_transform(X_aug)

    out_cols = [c for c in X_aug.columns if c != "_sovi"]
    return pd.DataFrame(result, columns=out_cols, index=X.index)
