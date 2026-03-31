"""
One-Time Static Data Downloader
================================
Downloads SRTM tiles from OpenTopography and processes both SRTM and LANDFIRE
into per-H3-cell feature Parquet files ready for the pipeline.

Run this ONCE inside Docker before starting the pipeline:

    docker exec -it wildfire-airflow-scheduler-1 \
        python scripts/ingestion/download_static.py --resolution-km 64

Requirements:
    - OPENTOPO_API_KEY env var (free at https://opentopography.org/developers)
    - LANDFIRE rasters manually placed in data/static/landfire_raw/ (see below)

LANDFIRE (manual download — no public API):
    1. Go to https://landfire.gov/data/FullExtentDownloads
    2. Select: LF 2022 → CONUS → Fuel
    3. Download and extract these three ZIPs:
         LF2022_FBFM40_230_CONUS.zip  → LC22_F40_230.tif
         LF2022_CC_230_CONUS.zip      → LC22_CC_230.tif
         LF2022_EVT_230_CONUS.zip     → LC22_EVT_230.tif
    4. Place the extracted .tif files in:  data/static/landfire_raw/

SRTM (automatic — downloaded by this script via OpenTopography API):
    7 tiles covering CA + TX, placed in: data/static/srtm_raw/

CLI:
    python scripts/ingestion/download_static.py --resolution-km 64
    python scripts/ingestion/download_static.py --resolution-km 22 --force-rebuild
    python scripts/ingestion/download_static.py --resolution-km 64 --skip-dvc
    python scripts/ingestion/download_static.py --srtm-only   # skip LANDFIRE
    python scripts/ingestion/download_static.py --landfire-only
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SRTM tile definitions — 7 tiles covering CA + TX
# Tiles are split to stay under OpenTopography's 450,000 km² per-request limit
# ---------------------------------------------------------------------------
SRTM_TILES = [
    # California (3 tiles — north/south split)
    {"name": "srtm_california_south.tif", "south": 32.53, "north": 36.0,  "west": -124.48, "east": -114.13},
    {"name": "srtm_california_mid.tif",   "south": 36.0,  "north": 39.0,  "west": -124.48, "east": -114.13},
    {"name": "srtm_california_north.tif", "south": 39.0,  "north": 42.01, "west": -124.48, "east": -114.13},
    # Texas (4 tiles — 2×2 grid)
    {"name": "srtm_texas_nw.tif", "south": 31.17, "north": 36.50, "west": -106.65, "east": -100.08},
    {"name": "srtm_texas_ne.tif", "south": 31.17, "north": 36.50, "west": -100.08, "east":  -93.51},
    {"name": "srtm_texas_sw.tif", "south": 25.84, "north": 31.17, "west": -106.65, "east": -100.08},
    {"name": "srtm_texas_se.tif", "south": 25.84, "north": 31.17, "west": -100.08, "east":  -93.51},
]

OPENTOPO_URL = "https://portal.opentopography.org/API/globaldem"


# ---------------------------------------------------------------------------
# SRTM download
# ---------------------------------------------------------------------------

def download_srtm_tiles(
    output_dir: Path,
    force: bool = False,
    api_key: str | None = None,
) -> None:
    """Download 7 SRTM tiles from OpenTopography into output_dir/srtm_raw/."""
    if api_key is None:
        api_key = os.environ.get("OPENTOPO_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENTOPO_API_KEY not set. "
            "Get a free key at https://opentopography.org/developers "
            "and set it: export OPENTOPO_API_KEY=your_key"
        )

    tile_dir = output_dir / "srtm_raw"
    tile_dir.mkdir(parents=True, exist_ok=True)

    for tile in SRTM_TILES:
        dest = tile_dir / tile["name"]
        if dest.exists() and not force:
            logger.info("SRTM tile already exists, skipping: %s", dest.name)
            continue

        logger.info("Downloading SRTM tile: %s ...", tile["name"])
        params = {
            "demtype": "SRTMGL1",
            "south":   tile["south"],
            "north":   tile["north"],
            "west":    tile["west"],
            "east":    tile["east"],
            "outputFormat": "GTiff",
            "API_Key": api_key,
        }
        resp = requests.get(OPENTOPO_URL, params=params, stream=True, timeout=300)

        if resp.status_code != 200:
            raise RuntimeError(
                f"OpenTopography API error for {tile['name']}: "
                f"HTTP {resp.status_code} — {resp.text[:200]}"
            )

        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.info("  → %s (%.1f MB)", dest.name, size_mb)

    logger.info("All SRTM tiles downloaded to %s", tile_dir)


# ---------------------------------------------------------------------------
# LANDFIRE instructions
# ---------------------------------------------------------------------------

def print_landfire_instructions(output_dir: Path) -> None:
    raw_dir = output_dir / "landfire_raw"
    print(
        "\n"
        "============================================================\n"
        "  LANDFIRE — MANUAL DOWNLOAD REQUIRED\n"
        "============================================================\n"
        "LANDFIRE has no public download API.\n"
        "Follow these steps once:\n"
        "\n"
        "  1. Go to: https://landfire.gov/data/FullExtentDownloads\n"
        "  2. Select: LF 2022 → CONUS → Fuel\n"
        "  3. Download and extract these three ZIPs (~1-2 GB each):\n"
        "       LF2022_FBFM40_230_CONUS.zip  → any filename containing 'F40' or 'FBFM40'\n"
        "       LF2022_CC_230_CONUS.zip      → any filename containing '_CC_'\n"
        "       LF2022_EVT_230_CONUS.zip     → any filename containing 'EVT'\n"
        f"  4. Place the extracted .tif files in:\n"
        f"       {raw_dir}\n"
        "\n"
        "  Then re-run this script (SRTM won't re-download if already present).\n"
        "============================================================\n"
    )


# ---------------------------------------------------------------------------
# DVC tracking
# ---------------------------------------------------------------------------

def dvc_track(output_dir: Path, resolution_km: int) -> None:
    """Run dvc add on the two output Parquet files."""
    files = [
        output_dir / f"landfire_features_{resolution_km}km.parquet",
        output_dir / f"srtm_features_{resolution_km}km.parquet",
    ]
    for f in files:
        if not f.exists():
            logger.warning("DVC: skipping missing file %s", f)
            continue
        logger.info("dvc add %s", f)
        result = subprocess.run(
            ["dvc", "add", str(f)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.warning("dvc add failed for %s:\n%s", f, result.stderr)
        else:
            logger.info("DVC tracking: %s.dvc created", f.name)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run(
    resolution_km: int = 64,
    output_dir: str | Path = "data/static",
    force_rebuild: bool = False,
    skip_dvc: bool = False,
    srtm_only: bool = False,
    landfire_only: bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download SRTM tiles (automatic)
    if not landfire_only:
        logger.info("=== Step 1: Downloading SRTM tiles ===")
        download_srtm_tiles(output_dir, force=force_rebuild)

    # Step 2: Check LANDFIRE raw files exist (manual)
    landfire_raw_dir = output_dir / "landfire_raw"
    landfire_ready = (
        landfire_raw_dir.exists()
        and any(landfire_raw_dir.glob("*.tif"))
    )
    if not srtm_only and not landfire_ready:
        print_landfire_instructions(output_dir)
        if landfire_only:
            sys.exit(1)
        logger.warning("LANDFIRE rasters not found — will process SRTM only.")

    # Step 3: Process SRTM → srtm_features_{N}km.parquet
    if not landfire_only:
        logger.info("=== Step 2: Processing SRTM → H3 features ===")
        from scripts.ingestion.ingest_srtm import run as srtm_run
        srtm_out = srtm_run(
            resolution_km=resolution_km,
            output_dir=str(output_dir),
            force_rebuild=force_rebuild,
        )
        logger.info("SRTM features written: %s", srtm_out)

    # Step 4: Process LANDFIRE → landfire_features_{N}km.parquet
    if not srtm_only and landfire_ready:
        logger.info("=== Step 3: Processing LANDFIRE → H3 features ===")
        from scripts.ingestion.ingest_landfire import run as landfire_run
        landfire_out = landfire_run(
            resolution_km=resolution_km,
            output_dir=str(output_dir),
            force_rebuild=force_rebuild,
        )
        logger.info("LANDFIRE features written: %s", landfire_out)

    # Step 5: DVC tracking
    if not skip_dvc:
        logger.info("=== Step 4: DVC tracking ===")
        dvc_track(output_dir, resolution_km)

    logger.info(
        "=== Static data setup complete (resolution=%dkm) ===\n"
        "You can now start the pipeline:\n"
        "  docker compose up -d\n"
        "  # Open http://localhost:8080, trigger wildfire_data_pipeline",
        resolution_km,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "One-time static data downloader. "
            "Downloads SRTM tiles and processes LANDFIRE + SRTM into "
            "H3-cell feature Parquet files for the wildfire pipeline."
        )
    )
    p.add_argument(
        "--resolution-km", type=int, default=64,
        help="H3 grid resolution in km (default: 64)",
    )
    p.add_argument(
        "--output-dir", default="data/static",
        help="Directory for raw tiles and output Parquet files (default: data/static)",
    )
    p.add_argument(
        "--force-rebuild", action="store_true",
        help="Redownload tiles and recompute Parquet even if already present",
    )
    p.add_argument(
        "--skip-dvc", action="store_true",
        help="Skip dvc add step (useful if DVC remote is not configured yet)",
    )
    p.add_argument(
        "--srtm-only", action="store_true",
        help="Process SRTM only, skip LANDFIRE",
    )
    p.add_argument(
        "--landfire-only", action="store_true",
        help="Process LANDFIRE only (tiles must already be in data/static/landfire_raw/)",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run(
        resolution_km=args.resolution_km,
        output_dir=args.output_dir,
        force_rebuild=args.force_rebuild,
        skip_dvc=args.skip_dvc,
        srtm_only=args.srtm_only,
        landfire_only=args.landfire_only,
    )
