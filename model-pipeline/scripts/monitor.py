"""CLI entry point for model monitoring / drift detection.

Usage:
    python -m scripts.monitor --baseline-run-id <run_id> [--bucket <bucket>]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model drift monitoring check")
    parser.add_argument("--baseline-run-id", required=True, help="run_id of the training baseline to compare against")
    parser.add_argument("--bucket", default=None, help="GCS bucket name (default: GCS_BUCKET_NAME env var)")
    parser.add_argument("--run-id", default=None, help="Identifier for this monitoring run")
    args = parser.parse_args()

    from src.monitoring.monitor_runner import run_monitoring_check

    result = run_monitoring_check(
        run_id=args.run_id,
        gcs_bucket=args.bucket,
        baseline_run_id=args.baseline_run_id,
    )

    print(json.dumps(result, indent=2))

    verdict = result.get("verdict", "UNKNOWN")
    if verdict == "CRITICAL":
        logger.warning("CRITICAL drift detected — review Slack alert and retrain logs")
        sys.exit(2)
    elif verdict == "WARNING":
        logger.warning("WARNING: moderate drift detected")
        sys.exit(1)
    else:
        logger.info("No significant drift detected")
        sys.exit(0)


if __name__ == "__main__":
    main()