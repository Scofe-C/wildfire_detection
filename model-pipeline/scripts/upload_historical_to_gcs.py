"""
One-time script to upload historical training data (CA + TX) to GCS.

Usage:
    cd wildfire_detection
    GCS_BUCKET_NAME=your-bucket-name python -m model-pipeline.scripts.upload_historical_to_gcs

    # Or with explicit bucket:
    python model-pipeline/scripts/upload_historical_to_gcs.py --bucket your-bucket-name

Uploads:
    historical_data/california_historical.csv
    historical_data/texas_historical.csv
"""

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Local paths relative to this script's location
_SCRIPT_DIR   = Path(__file__).resolve().parent
_HISTORICAL_DIR = _SCRIPT_DIR.parent / "historical_data"

HISTORICAL_FILES = [
    "california_historical.csv",
    "texas_historical.csv",
]

# GCS destination folder (under bucket root)
GCS_FOLDER = "historical_data"


def upload_historical(bucket_name: str, dry_run: bool = False) -> None:
    if not dry_run:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)

    for filename in HISTORICAL_FILES:
        local_path = _HISTORICAL_DIR / filename
        if not local_path.exists():
            logger.warning(f"File not found, skipping: {local_path}")
            continue

        gcs_path = f"{GCS_FOLDER}/{filename}"
        size_mb  = local_path.stat().st_size / (1024 ** 2)

        if dry_run:
            logger.info(f"[DRY RUN] Would upload: {local_path} → gs://{bucket_name}/{gcs_path} ({size_mb:.1f} MB)")
            continue

        logger.info(f"Uploading {local_path} ({size_mb:.1f} MB) → gs://{bucket_name}/{gcs_path} ...")
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(str(local_path), content_type="text/csv")
        logger.info(f"Done: gs://{bucket_name}/{gcs_path}")

    if not dry_run:
        logger.info(f"All files uploaded to gs://{bucket_name}/{GCS_FOLDER}/")


def _get_bucket_name() -> str:
    bucket = os.environ.get("GCS_BUCKET_NAME")
    if not bucket:
        raise ValueError(
            "GCS_BUCKET_NAME environment variable not set. "
            "Run: export GCS_BUCKET_NAME=your-bucket-name"
        )
    return bucket


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload historical CSVs to GCS.")
    parser.add_argument("--bucket",  default=None, help="GCS bucket name (overrides GCS_BUCKET_NAME env var)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded without uploading")
    args = parser.parse_args()

    bucket_name = args.bucket or _get_bucket_name()
    logger.info(f"Target bucket: gs://{bucket_name}/{GCS_FOLDER}/")

    upload_historical(bucket_name=bucket_name, dry_run=args.dry_run)
