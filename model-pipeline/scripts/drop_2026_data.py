"""
Remove unlabelled 2026 rows from historical CSVs and push cleaned files to GCS.

2026 rows have fire_detected_binary=0 because FIRMS hasn't confirmed them yet —
they are not real non-fires. Including them tanks model precision.

Usage:
    python scripts/drop_2026_data.py --bucket wildfire-mlops-123
"""
import argparse
import io
import logging
from pathlib import Path

import pandas as pd
from google.cloud import storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HISTORICAL_DIR = Path(__file__).resolve().parents[1] / "historical_data"
REGIONS = ["california", "texas"]


def clean_and_push(bucket_name: str) -> None:
    client = storage.Client()
    bkt = client.bucket(bucket_name)

    for region in REGIONS:
        csv_path = HISTORICAL_DIR / f"{region}_historical.csv"
        if not csv_path.exists():
            logger.warning("%s not found locally — skipping", csv_path)
            continue

        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        before = len(df)
        df = df[df["timestamp"].dt.year < 2026].reset_index(drop=True)
        dropped = before - len(df)

        df.to_csv(csv_path, index=False)
        logger.info("[%s] Dropped %d 2026 rows → %d rows remaining, saved locally", region, dropped, len(df))

        blob_path = f"historical_data/{region}_historical.csv"
        buf = io.BytesIO(df.to_csv(index=False).encode())
        bkt.blob(blob_path).upload_from_file(buf, content_type="text/csv")
        logger.info("[%s] Pushed gs://%s/%s", region, bucket_name, blob_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="wildfire-mlops-123")
    args = parser.parse_args()
    clean_and_push(args.bucket)
