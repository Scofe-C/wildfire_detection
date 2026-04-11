from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd

from src.data.schema import FeatureSchema, load_schema, validate_dataframe

logger = logging.getLogger(__name__)

# Temporal split boundaries
TRAIN_CUTOFF = pd.Timestamp("2025-01-01", tz="UTC")
TEST_END     = pd.Timestamp("2025-01-31 23:59:59", tz="UTC")
# 2026 rows have fire_detected_binary=0 (FIRMS unconfirmed — unlabelled, not real non-fires)
LABEL_CUTOFF = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")


class DataLoadError(Exception):
    pass


def load_backfill(
    backfill_dir: str | Path,
    schema: FeatureSchema | None = None,
    strict: bool = True,
) -> pd.DataFrame:
    backfill_dir = Path(backfill_dir)
    if not backfill_dir.exists():
        raise DataLoadError(f"Backfill directory not found: {backfill_dir}")

    parquet_files = sorted(backfill_dir.glob("*.parquet"))
    if not parquet_files:
        raise DataLoadError(f"No parquet files in: {backfill_dir}")

    logger.info("Loading %d parquet files from %s", len(parquet_files), backfill_dir)
    dfs = []
    for pf in parquet_files:
        try:
            dfs.append(pd.read_parquet(pf))
        except Exception as e:
            if strict:
                raise DataLoadError(f"Failed to read {pf.name}: {e}") from e
            logger.warning("Skipping %s: %s", pf.name, e)

    if not dfs:
        raise DataLoadError("All parquet files failed to load")

    df = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    if schema is None:
        schema = load_schema()
    errors = validate_dataframe(df, schema)
    if errors:
        msg = "Schema validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
        if strict:
            raise DataLoadError(msg)
        logger.warning(msg)

    return df


def split_features_target(
    df: pd.DataFrame,
    schema: FeatureSchema | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if schema is None:
        schema = load_schema()

    feature_cols = [f.name for f in schema.features if f.name in df.columns]
    missing = [f.name for f in schema.required_features if f.name not in df.columns]
    if missing:
        raise DataLoadError(f"Missing required features: {missing}")

    X = df[feature_cols].copy()
    y = df[schema.target_name].copy()
    meta_cols = [c for c in schema.index_column_names if c in df.columns]
    metadata = df[meta_cols].copy()

    logger.info(
        "Split — X: %d features, y: %d labels (%.1f%% pos)",
        X.shape[1], len(y), 100 * y.mean(),
    )
    return X, y, metadata


def load_and_split(
    backfill_dir: str | Path,
    schema: FeatureSchema | None = None,
    strict: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = load_backfill(backfill_dir, schema=schema, strict=strict)
    return split_features_target(df, schema=schema)


def load_region_from_gcs(
    bucket: str,
    blob: str,
) -> pd.DataFrame:
    """Download a single region's historical CSV from GCS.

    Used by the per-region training pipeline (one call per region).

    Parameters
    ----------
    bucket : GCS bucket name (e.g. "wildfire-mlops-123").
    blob   : path within the bucket (e.g. "historical_data/california_historical.csv").
    """
    from google.cloud import storage

    client = storage.Client()
    logger.info("Downloading gs://%s/%s ...", bucket, blob)
    data = client.bucket(bucket).blob(blob).download_as_bytes()
    df   = pd.read_csv(io.BytesIO(data))
    logger.info("  → %d rows", len(df))

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    return df


def load_from_gcs(
    bucket: str,
    ca_blob: str = "historical_data/california_historical.csv",
    tx_blob: str = "historical_data/texas_historical.csv",
) -> pd.DataFrame:
    """Download CA and TX historical CSVs from GCS and return a combined DataFrame.

    Kept for backwards-compatibility. Training pipeline now calls
    load_region_from_gcs() once per region instead.
    """
    ca = load_region_from_gcs(bucket, ca_blob)
    tx = load_region_from_gcs(bucket, tx_blob)
    combined = pd.concat([ca, tx], ignore_index=True)
    logger.info("GCS load complete: %d total rows", len(combined))
    return combined


def temporal_split(
    df: pd.DataFrame,
    train_cutoff: pd.Timestamp = TRAIN_CUTOFF,
    test_end: pd.Timestamp = TEST_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train and test sets using a time boundary.

    Train: timestamp < train_cutoff  OR  timestamp > test_end
           (all data except the Jan 2025 test window — includes post-backfill rows)
    Test:  train_cutoff <= timestamp <= test_end  (Jan 2025 LA fires)

    A random split is NOT used — that would allow training on Jan 2025 fire data
    and testing on 2024 normal conditions, inverting causal ordering.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a 'timestamp' column (datetime, UTC-aware).
    train_cutoff : pd.Timestamp
        First timestamp to exclude from training (inclusive test start).
    test_end : pd.Timestamp
        Last timestamp to include in test set.

    Returns
    -------
    (train_df, test_df)
    """
    if "timestamp" not in df.columns:
        raise DataLoadError("DataFrame must have a 'timestamp' column for temporal split")

    ts = df["timestamp"]
    if not hasattr(ts.dtype, "tz") or ts.dtype.tz is None:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        ts = df["timestamp"]

    # Train: all labelled data except the Jan 2025 test window
    #   Jun–Dec 2024 + Feb–Dec 2025 (excludes 2026 — unlabelled FIRMS zeros)
    # Test:  Jan 2025 LA fires (held-out, never seen during tuning)
    train_df = df[
        ((ts < train_cutoff) | (ts > test_end)) & (ts <= LABEL_CUTOFF)
    ].copy()
    test_df  = df[(ts >= train_cutoff) & (ts <= test_end)].copy()

    logger.info(
        "Temporal split — train: %d rows (%s–%s excl. Jan 2025), test: %d rows (Jan 2025)",
        len(train_df), train_cutoff.date(), LABEL_CUTOFF.date(), len(test_df),
    )
    if len(train_df) == 0:
        raise DataLoadError(f"Training set is empty — check data range vs cutoff {train_cutoff}")
    if len(test_df) == 0:
        logger.warning("Test set is empty — data may not include Jan 2025 LA fire period")

    return train_df, test_df
