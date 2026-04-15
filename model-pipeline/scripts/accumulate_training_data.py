#!/usr/bin/env python3
"""
Accumulate Training Data from GCS
===================================
Combines historical CSVs with daily pipeline-generated parquets and writes
versioned snapshots to gs://{bucket}/training-data/.

Sources
-------
  gs://{bucket}/historical_data/california_historical.csv
  gs://{bucket}/historical_data/texas_historical.csv
  gs://{bucket}/data/processed/64km/region=*/year=*/month=*/features_*.parquet

Outputs
-------
  gs://{bucket}/training-data/combined_{YYYYMMDD}.parquet   — dated snapshot (permanent)
  gs://{bucket}/training-data/combined_latest.parquet        — always overwritten

Run manually before retraining:
  python model-pipeline/scripts/accumulate_training_data.py
  python model-pipeline/scripts/accumulate_training_data.py --bucket my-bucket --dry-run

Deduplication key: (grid_id, timestamp)
Sort order: timestamp ascending (required for temporal train/test split)
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("accumulate")

_DEFAULT_BUCKET = "wildfire-mlops-123"
_PROCESSED_PREFIX = "data/processed/64km"
_HISTORICAL_BLOBS = [
    "historical_data/california_historical.csv",
    "historical_data/texas_historical.csv",
]
_OUTPUT_PREFIX = "training-data"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _gcs_client():
    from google.cloud import storage
    return storage.Client()


def _load_historical_csv(client, bucket_name: str, blob_path: str) -> pd.DataFrame | None:
    """Download and parse a historical CSV from GCS."""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    if not blob.exists():
        logger.warning("Historical CSV not found: gs://%s/%s — skipping", bucket_name, blob_path)
        return None
    logger.info("Downloading gs://%s/%s ...", bucket_name, blob_path)
    data = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(data))
    logger.info("  → %d rows × %d cols", len(df), len(df.columns))
    return df


def _list_pipeline_parquets(client, bucket_name: str) -> list[str]:
    """List all features_*.parquet blobs under the processed 64km prefix."""
    bucket = client.bucket(bucket_name)
    blobs = [
        b.name for b in bucket.list_blobs(prefix=_PROCESSED_PREFIX)
        if b.name.endswith(".parquet") and "/features_" in b.name
    ]
    blobs.sort()
    logger.info("Found %d pipeline parquet(s) under gs://%s/%s/", len(blobs), bucket_name, _PROCESSED_PREFIX)
    return blobs


def _load_parquet_blob(client, bucket_name: str, blob_name: str) -> pd.DataFrame | None:
    """Download a GCS blob and read it as parquet."""
    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        client.bucket(bucket_name).blob(blob_name).download_to_filename(tmp.name)
        try:
            return pd.read_parquet(tmp.name)
        except Exception as e:
            logger.warning("  Skipping %s: %s", blob_name, e)
            return None


def _normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and UTC-normalize the timestamp column if present."""
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def _upload_parquet(client, bucket_name: str, blob_name: str, df: pd.DataFrame) -> None:
    """Write DataFrame to a temp file and upload to GCS."""
    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        df.to_parquet(tmp.name, index=False)
        client.bucket(bucket_name).blob(blob_name).upload_from_filename(
            tmp.name, content_type="application/octet-stream"
        )
    logger.info("Uploaded → gs://%s/%s  (%d rows)", bucket_name, blob_name, len(df))


# ── Main ──────────────────────────────────────────────────────────────────────

def accumulate(bucket_name: str, dry_run: bool = False) -> None:
    client = _gcs_client()

    frames: list[pd.DataFrame] = []

    # ── 1. Historical CSVs ────────────────────────────────────────────────────
    logger.info("=== Loading historical CSVs ===")
    for blob_path in _HISTORICAL_BLOBS:
        df = _load_historical_csv(client, bucket_name, blob_path)
        if df is not None:
            df = _normalize_timestamp(df)
            frames.append(df)

    # ── 2. Pipeline-generated parquets ────────────────────────────────────────
    logger.info("=== Loading pipeline parquets ===")
    parquet_blobs = _list_pipeline_parquets(client, bucket_name)
    for blob_name in parquet_blobs:
        df = _load_parquet_blob(client, bucket_name, blob_name)
        if df is not None and not df.empty:
            df = _normalize_timestamp(df)
            frames.append(df)
            logger.info("  Loaded %s  (%d rows)", blob_name.split("/")[-1], len(df))

    if not frames:
        logger.error("No data loaded — nothing to accumulate.")
        return

    # ── 3. Combine ────────────────────────────────────────────────────────────
    logger.info("=== Combining %d dataframe(s) ===", len(frames))
    combined = pd.concat(frames, ignore_index=True, sort=False)
    logger.info("Raw combined: %d rows × %d cols", len(combined), len(combined.columns))

    # ── 4. Filter cross-region bleed (rows missing core weather) ─────────────
    weather_sentinel = [c for c in ["temperature_2m", "relative_humidity_2m"] if c in combined.columns]
    if weather_sentinel:
        before = len(combined)
        combined = combined.dropna(subset=weather_sentinel, how="all").reset_index(drop=True)
        dropped = before - len(combined)
        if dropped:
            logger.info("Dropped %d cross-region rows (null weather)", dropped)

    # ── 5. Deduplicate on (grid_id, timestamp) ────────────────────────────────
    dedup_cols = [c for c in ["grid_id", "timestamp"] if c in combined.columns]
    if dedup_cols:
        before = len(combined)
        combined = combined.drop_duplicates(subset=dedup_cols, keep="last").reset_index(drop=True)
        dropped = before - len(combined)
        if dropped:
            logger.info("Dropped %d duplicate rows (same grid_id+timestamp)", dropped)

    # ── 6. Sort by timestamp ──────────────────────────────────────────────────
    if "timestamp" in combined.columns:
        combined = combined.sort_values("timestamp").reset_index(drop=True)

    # ── 7. Summary ────────────────────────────────────────────────────────────
    fire_col = "fire_detected_binary"
    fire_rows = int(combined[fire_col].sum()) if fire_col in combined.columns else "?"
    fire_pct  = 100.0 * fire_rows / len(combined) if isinstance(fire_rows, int) and len(combined) > 0 else "?"

    logger.info("=== Combined dataset summary ===")
    logger.info("  Rows      : %d", len(combined))
    logger.info("  Columns   : %d", len(combined.columns))
    if isinstance(fire_rows, int):
        logger.info("  Fire+ rows: %d (%.2f%%)", fire_rows, fire_pct)
    if "timestamp" in combined.columns:
        logger.info(
            "  Date range: %s → %s",
            combined["timestamp"].min(),
            combined["timestamp"].max(),
        )
    if "region" in combined.columns:
        for region, grp in combined.groupby("region"):
            fire_n = int(grp[fire_col].sum()) if fire_col in grp.columns else "?"
            logger.info("  [%s] %d rows, %s fire+ events", region, len(grp), fire_n)

    # ── 8. Upload ─────────────────────────────────────────────────────────────
    today = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    dated_blob  = f"{_OUTPUT_PREFIX}/combined_{today}.parquet"
    latest_blob = f"{_OUTPUT_PREFIX}/combined_latest.parquet"

    if dry_run:
        logger.info("[DRY RUN] Would upload:")
        logger.info("  gs://%s/%s", bucket_name, dated_blob)
        logger.info("  gs://%s/%s", bucket_name, latest_blob)
        return

    logger.info("=== Uploading ===")
    _upload_parquet(client, bucket_name, dated_blob,  combined)
    _upload_parquet(client, bucket_name, latest_blob, combined)

    logger.info("Done. Versioned snapshot: gs://%s/%s", bucket_name, dated_blob)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Accumulate historical + pipeline data into a versioned training parquet on GCS."
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("GCS_BUCKET_NAME", _DEFAULT_BUCKET),
        help=f"GCS bucket name (default: {_DEFAULT_BUCKET} or $GCS_BUCKET_NAME)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually uploading",
    )
    args = parser.parse_args()

    logger.info("Bucket: gs://%s", args.bucket)
    if args.dry_run:
        logger.info("DRY RUN — no files will be written")

    accumulate(bucket_name=args.bucket, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
