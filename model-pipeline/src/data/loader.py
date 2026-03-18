from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.schema import FeatureSchema, load_schema, validate_dataframe

logger = logging.getLogger(__name__)


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
