"""
Data Bias Analysis — Subgroup Slicing
======================================
Evaluates whether feature distributions and fire detection rates differ
systematically across meaningful subgroups. Generates a structured report
that can be reviewed before model training to catch encoding-level biases.

Assignment requirements (Section 3):
  - Slice by demographic / categorical features relevant to the domain
  - Compute per-slice statistics and compare against overall population
  - Detect significant distributional shifts (KL divergence)
  - Document findings and mitigation steps

Slicing dimensions:
  1. Geographic region        — california vs. texas
  2. Fuel model class         — fuel_model_fbfm40 mapped to 5 fire-risk tiers
  3. Fire season              — fire_season (Jun–Nov) vs. off_season (Dec–May)
  4. Data quality tier        — tier A (flag 0–2) vs. tier B (flag 3–5)

Outputs written to ``data/processed/baselines/bias_report.json``.

Usage (offline, no Airflow needed):
    python scripts/validation/bias_analysis.py \
        --input  data/processed/backfill/ \
        --output data/processed/baselines/bias_report.json

Usage as a library (from tests or DAG):
    from scripts.validation.bias_analysis import run_bias_analysis
    report = run_bias_analysis(df, registry)
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# FBFM40 codes → fire-risk tier labels.
# Tiers follow the Scott & Burgan (2005) classification.
# Reference: https://www.fs.usda.gov/rm/pubs/rmrs_gtr153.pdf
FBFM40_TIER_MAP: Dict[int, str] = {
    # Non-burnable / water / urban
    **{c: "non_burnable" for c in [91, 92, 93, 98, 99]},
    # Grass group (GR) — low to moderate risk
    **{c: "grass" for c in range(101, 110)},
    # Grass-Shrub group (GS) — moderate risk
    **{c: "grass_shrub" for c in range(121, 125)},
    # Shrub group (SH) — moderate to high risk
    **{c: "shrub" for c in range(141, 150)},
    # Timber Understory group (TU) — moderate risk
    **{c: "timber_understory" for c in range(161, 166)},
    # Timber Litter group (TL) — high risk
    **{c: "timber_litter" for c in range(181, 190)},
    # Slash Blowdown group (SB) — high risk
    **{c: "slash_blowdown" for c in range(201, 205)},
}
FBFM40_TIER_UNKNOWN = "unknown"

# Features to compute per-slice statistics on.
# These are the fire-relevant numeric features most likely to exhibit bias.
NUMERIC_FEATURES_OF_INTEREST = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "active_fire_count",
    "mean_frp",
    "fire_detected_binary",
    "data_quality_flag",
]

# A slice is considered biased if KL divergence from the overall population
# exceeds this threshold. 0.1 nats is a practitioner heuristic for
# "meaningfully different" without being hypersensitive to minor shifts.
KL_DIVERGENCE_THRESHOLD = 0.1

# Fire detection rate difference (absolute) that we flag as imbalanced.
# 5 pp difference is large enough to affect recall across subgroups.
FIRE_RATE_DISPARITY_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_bias_analysis(
    df: pd.DataFrame,
    registry,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run full bias analysis across all slicing dimensions.

    Args:
        df: Fused feature DataFrame. Must contain at minimum: ``region``,
            ``data_quality_flag``, ``fire_detected_binary``, and at least one
            numeric feature from NUMERIC_FEATURES_OF_INTEREST.
        registry: FeatureRegistry instance (provides fire_season_months).
        output_path: If provided, write the JSON report to this path.

    Returns:
        A structured report dict with keys:
            - ``generated_at``: ISO timestamp
            - ``row_count``: total rows analysed
            - ``overall_stats``: population-level statistics
            - ``slices``: list of per-slice result dicts
            - ``findings``: list of human-readable finding strings
            - ``mitigations``: list of mitigation step strings
    """
    if df.empty:
        logger.warning("bias_analysis received an empty DataFrame — returning empty report")
        return _empty_report()

    logger.info(f"Running bias analysis on {len(df):,} rows")

    present_numeric = [c for c in NUMERIC_FEATURES_OF_INTEREST if c in df.columns]
    if not present_numeric:
        logger.warning(
            "None of the expected numeric features are present — bias analysis is limited. "
            f"Expected: {NUMERIC_FEATURES_OF_INTEREST}. Got: {list(df.columns)}"
        )

    df = _add_derived_columns(df, registry)

    overall_stats = _compute_slice_stats(df, present_numeric, label="overall")
    slice_results = []

    # --- Slice 1: Geographic region ---
    if "region" in df.columns:
        region_slices = _run_categorical_slices(
            df=df,
            column="region",
            label="geographic_region",
            numeric_features=present_numeric,
            overall_stats=overall_stats,
        )
        slice_results.extend(region_slices)
    else:
        logger.warning("'region' column absent — geographic slice skipped")

    # --- Slice 2: Fuel model tier ---
    if "fuel_model_tier" in df.columns:
        fuel_slices = _run_categorical_slices(
            df=df,
            column="fuel_model_tier",
            label="fuel_model_tier",
            numeric_features=present_numeric,
            overall_stats=overall_stats,
        )
        slice_results.extend(fuel_slices)
    else:
        logger.warning("'fuel_model_tier' column absent — fuel tier slice skipped")

    # --- Slice 3: Fire season ---
    if "fire_season_label" in df.columns:
        season_slices = _run_categorical_slices(
            df=df,
            column="fire_season_label",
            label="fire_season",
            numeric_features=present_numeric,
            overall_stats=overall_stats,
        )
        slice_results.extend(season_slices)

    # --- Slice 4: Data quality tier ---
    if "quality_tier" in df.columns:
        quality_slices = _run_categorical_slices(
            df=df,
            column="quality_tier",
            label="data_quality_tier",
            numeric_features=present_numeric,
            overall_stats=overall_stats,
        )
        slice_results.extend(quality_slices)

    findings, mitigations = _synthesize_findings(slice_results, overall_stats)

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "row_count": int(len(df)),
        "overall_stats": overall_stats,
        "slices": slice_results,
        "findings": findings,
        "mitigations": mitigations,
    }

    if output_path:
        _write_report(report, Path(output_path))

    return report


# ---------------------------------------------------------------------------
# Slice computation
# ---------------------------------------------------------------------------

def _add_derived_columns(df: pd.DataFrame, registry) -> pd.DataFrame:
    """Add computed grouping columns used for slicing.

    Columns added (if their source columns are present):
      - ``fuel_model_tier``: string tier from FBFM40_TIER_MAP
      - ``fire_season_label``: 'fire_season' | 'off_season'
      - ``quality_tier``: 'tier_a' (flag 0-2) | 'tier_b' (flag 3-5)
    """
    df = df.copy()

    # Fuel model tier
    if "fuel_model_fbfm40" in df.columns:
        df["fuel_model_tier"] = df["fuel_model_fbfm40"].map(
            lambda code: FBFM40_TIER_MAP.get(int(code), FBFM40_TIER_UNKNOWN)
            if pd.notna(code) else FBFM40_TIER_UNKNOWN
        )
        logger.debug(
            "Fuel model tier distribution: %s",
            df["fuel_model_tier"].value_counts().to_dict(),
        )

    # Fire season label (uses month from timestamp if available)
    fire_months = set(registry.fire_season_months)
    if "timestamp_utc" in df.columns:
        months = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce").dt.month
        df["fire_season_label"] = months.apply(
            lambda m: "fire_season" if m in fire_months else "off_season"
            if pd.notna(m) else None
        )
    elif "timestamp" in df.columns:
        months = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.month
        df["fire_season_label"] = months.apply(
            lambda m: "fire_season" if m in fire_months else "off_season"
            if pd.notna(m) else None
        )

    # Data quality tier
    if "data_quality_flag" in df.columns:
        df["quality_tier"] = df["data_quality_flag"].apply(
            lambda f: "tier_a" if pd.notna(f) and int(f) <= 2 else "tier_b"
            if pd.notna(f) else None
        )

    return df


def _run_categorical_slices(
    df: pd.DataFrame,
    column: str,
    label: str,
    numeric_features: List[str],
    overall_stats: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Compute per-value statistics for a categorical column and compare to overall.

    Args:
        df: Full DataFrame (already has derived columns).
        column: The grouping column to slice on.
        label: Human-readable slice dimension name (used in report).
        numeric_features: Numeric columns to summarise.
        overall_stats: Pre-computed population statistics for KL comparison.

    Returns:
        List of per-value result dicts.
    """
    results = []
    unique_values = df[column].dropna().unique()

    for value in sorted(str(v) for v in unique_values):
        mask = df[column].astype(str) == str(value)
        slice_df = df[mask]

        if len(slice_df) < 10:
            logger.debug(f"Slice {label}={value} has only {len(slice_df)} rows — skipping")
            continue

        stats = _compute_slice_stats(slice_df, numeric_features, label=f"{label}={value}")

        # KL divergence per numeric feature
        kl_divergences: Dict[str, float] = {}
        for feat in numeric_features:
            if feat not in slice_df.columns or feat not in df.columns:
                continue
            kl = _kl_divergence_approx(
                slice_df[feat].dropna(),
                df[feat].dropna(),
            )
            kl_divergences[feat] = round(kl, 4)

        # Fire detection rate disparity
        fire_rate_slice = float(
            slice_df["fire_detected_binary"].mean()
        ) if "fire_detected_binary" in slice_df.columns else None
        fire_rate_overall = float(
            df["fire_detected_binary"].mean()
        ) if "fire_detected_binary" in df.columns else None

        fire_rate_disparity = (
            abs(fire_rate_slice - fire_rate_overall)
            if fire_rate_slice is not None and fire_rate_overall is not None
            else None
        )

        # Null rate per feature
        null_rates: Dict[str, float] = {
            feat: round(float(slice_df[feat].isna().mean()), 4)
            for feat in numeric_features
            if feat in slice_df.columns
        }

        biased_features = [
            feat for feat, kl in kl_divergences.items()
            if kl > KL_DIVERGENCE_THRESHOLD
        ]

        results.append({
            "slice_dimension": label,
            "slice_value": str(value),
            "row_count": int(len(slice_df)),
            "pct_of_total": round(len(slice_df) / len(df) * 100, 2),
            "fire_rate": round(fire_rate_slice, 4) if fire_rate_slice is not None else None,
            "fire_rate_disparity": round(fire_rate_disparity, 4) if fire_rate_disparity is not None else None,
            "fire_rate_biased": (
                fire_rate_disparity is not None and
                fire_rate_disparity > FIRE_RATE_DISPARITY_THRESHOLD
            ),
            "feature_stats": stats.get("feature_stats", {}),
            "null_rates": null_rates,
            "kl_divergences": kl_divergences,
            "biased_features": biased_features,
            "has_bias": len(biased_features) > 0 or (
                fire_rate_disparity is not None and
                fire_rate_disparity > FIRE_RATE_DISPARITY_THRESHOLD
            ),
        })

    return results


def _compute_slice_stats(
    df: pd.DataFrame,
    numeric_features: List[str],
    label: str,
) -> Dict[str, Any]:
    """Compute descriptive statistics for a DataFrame slice.

    Args:
        df: Slice or full DataFrame.
        numeric_features: Columns to summarise.
        label: Label for logging.

    Returns:
        Dict with ``row_count`` and ``feature_stats`` per column.
    """
    feature_stats: Dict[str, Dict[str, Any]] = {}
    for feat in numeric_features:
        if feat not in df.columns:
            continue
        col = df[feat].dropna()
        if len(col) == 0:
            continue
        feature_stats[feat] = {
            "mean":   round(float(col.mean()), 4),
            "std":    round(float(col.std()), 4),
            "median": round(float(col.median()), 4),
            "p5":     round(float(col.quantile(0.05)), 4),
            "p95":    round(float(col.quantile(0.95)), 4),
            "null_rate": round(float(df[feat].isna().mean()), 4),
            "n":      int(len(col)),
        }

    logger.debug(f"Slice '{label}': {len(df):,} rows, {len(feature_stats)} features")
    return {"row_count": int(len(df)), "feature_stats": feature_stats}


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------

def _kl_divergence_approx(
    p_samples: pd.Series,
    q_samples: pd.Series,
    n_bins: int = 20,
) -> float:
    """Approximate KL divergence KL(P || Q) using equal-width histograms.

    Both series are binned over the combined range so their histograms are
    comparable. A small epsilon is added to avoid log(0). This is an
    approximation suitable for flagging large distributional shifts; it is not
    a rigorous statistical test.

    Args:
        p_samples: Slice distribution (query group).
        q_samples: Reference distribution (overall population).
        n_bins: Number of histogram bins.

    Returns:
        Approximate KL divergence in nats. Returns 0.0 if input is too small
        or degenerate (zero variance).
    """
    p = p_samples.dropna().values
    q = q_samples.dropna().values

    if len(p) < 5 or len(q) < 5:
        return 0.0

    combined_min = min(p.min(), q.min())
    combined_max = max(p.max(), q.max())

    if combined_max == combined_min:
        # Constant feature — no distributional difference to detect
        return 0.0

    bins = np.linspace(combined_min, combined_max, n_bins + 1)
    eps = 1e-10

    p_hist, _ = np.histogram(p, bins=bins)
    q_hist, _ = np.histogram(q, bins=bins)

    p_dist = p_hist / (p_hist.sum() + eps)
    q_dist = q_hist / (q_hist.sum() + eps)

    # Mask bins where both are zero to avoid 0 * log(0/0)
    mask = (p_dist > eps) | (q_dist > eps)
    p_dist = p_dist[mask] + eps
    q_dist = q_dist[mask] + eps

    kl = float(np.sum(p_dist * np.log(p_dist / q_dist)))
    return max(0.0, kl)


# ---------------------------------------------------------------------------
# Findings synthesis
# ---------------------------------------------------------------------------

def _synthesize_findings(
    slice_results: List[Dict[str, Any]],
    overall_stats: Dict[str, Any],
) -> tuple[List[str], List[str]]:
    """Convert numeric slice results into human-readable findings and mitigations.

    Args:
        slice_results: Output of all _run_categorical_slices calls.
        overall_stats: Population statistics for reference.

    Returns:
        Tuple of (findings list, mitigations list).
    """
    findings: List[str] = []
    mitigations: List[str] = []

    biased_slices = [s for s in slice_results if s.get("has_bias")]

    if not biased_slices:
        findings.append(
            "No significant bias detected across any slicing dimension. "
            "All KL divergences are below the threshold and fire detection "
            "rates are within acceptable disparity bounds."
        )
        return findings, mitigations

    for s in biased_slices:
        dim = s["slice_dimension"]
        val = s["slice_value"]
        n   = s["row_count"]
        pct = s["pct_of_total"]

        if s.get("biased_features"):
            feats = ", ".join(s["biased_features"])
            kl_vals = ", ".join(
                f"{f}={s['kl_divergences'][f]:.3f}"
                for f in s["biased_features"]
            )
            findings.append(
                f"[{dim}={val}] Feature distribution shift detected "
                f"({n:,} rows, {pct:.1f}% of data). "
                f"Features with KL > {KL_DIVERGENCE_THRESHOLD}: {feats}. "
                f"KL values: {kl_vals}."
            )

        if s.get("fire_rate_biased"):
            fire_rate = s["fire_rate"]
            disparity = s["fire_rate_disparity"]
            findings.append(
                f"[{dim}={val}] Fire detection rate disparity: "
                f"slice rate = {fire_rate:.3f}, "
                f"absolute difference from overall = {disparity:.3f} "
                f"(threshold = {FIRE_RATE_DISPARITY_THRESHOLD})."
            )

    # Mitigations — one per unique slice dimension that showed bias
    biased_dims = {s["slice_dimension"] for s in biased_slices}

    if "geographic_region" in biased_dims:
        mitigations.append(
            "Geographic region bias: Apply region-stratified sampling during "
            "model training (e.g., stratified K-fold by region). Ensure "
            "weather gap-fill codes (data_quality_flag 1-3) are used as a "
            "training feature so models can down-weight degraded rows in "
            "underserved regions rather than treating them as equivalent to "
            "fully observed rows."
        )

    if "fuel_model_tier" in biased_dims:
        mitigations.append(
            "Fuel model tier bias: Check that LANDFIRE FBFM40 rasters are "
            "present for all grid cells. non_burnable cells (FBFM 91-99) "
            "should typically be excluded from fire prediction targets. "
            "For remaining tiers, consider oversampling underrepresented "
            "fuel classes if they coincide with high fire-risk zones."
        )

    if "fire_season" in biased_dims:
        mitigations.append(
            "Fire season bias: Off-season fire detection rate is near zero by "
            "construction. Models should either be trained season-specifically "
            "or use fire_season_label as a binary feature. Class weights should "
            "be computed separately per season to avoid off-season noise "
            "diluting fire-season signal."
        )

    if "data_quality_tier" in biased_dims:
        mitigations.append(
            "Data quality tier bias: Tier B rows (data_quality_flag 3-5) "
            "exhibit systematic feature differences due to gap-filling. "
            "Two options: (1) exclude tier B rows from training and flag their "
            "predictions as lower confidence at inference; or (2) include them "
            "and add quality_tier as a training feature with appropriate "
            "sample weights."
        )

    return findings, mitigations


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _write_report(report: Dict[str, Any], output_path: Path) -> None:
    """Write the bias report to a JSON file.

    Args:
        report: Report dict from run_bias_analysis.
        output_path: Destination path (parent dirs are created if needed).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Bias report written to {output_path}")


def _empty_report() -> Dict[str, Any]:
    """Return a minimal valid report for empty input."""
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "row_count": 0,
        "overall_stats": {},
        "slices": [],
        "findings": ["Input DataFrame was empty — no analysis performed."],
        "mitigations": [],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run data bias analysis on the fused feature dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to Parquet file or directory of Parquet files.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/baselines/bias_report.json",
        help="Path to write the JSON report (default: data/processed/baselines/bias_report.json).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to schema_config.yaml (uses project default if omitted).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    args = _parse_args()

    from scripts.utils.schema_loader import FeatureRegistry
    registry = FeatureRegistry(config_path=args.config)

    input_path = Path(args.input)
    if input_path.is_dir():
        parts = list(input_path.rglob("*.parquet"))
        if not parts:
            logger.error(f"No Parquet files found in {input_path}")
            sys.exit(1)
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    else:
        df = pd.read_parquet(input_path)

    logger.info(f"Loaded {len(df):,} rows from {args.input}")

    report = run_bias_analysis(df, registry, output_path=Path(args.output))

    print(f"\n=== Bias Analysis Summary ({report['row_count']:,} rows) ===")
    for finding in report["findings"]:
        print(f"  FINDING: {finding}")
    for mitigation in report["mitigations"]:
        print(f"  MITIGATION: {mitigation}")
    print(f"\nFull report: {args.output}")