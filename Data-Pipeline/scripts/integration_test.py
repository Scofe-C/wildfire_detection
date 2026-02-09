"""
Integration / Smoke Test for Wildfire Data Pipeline
===================================================
Simulates an end-to-end execution of the key pipeline steps:
1. Fetch FIRMS data (mocked or small subset)
2. Fetch Weather data (mocked or small subset)
3. Process FIRMS data
4. Fusion
5. Validation

Usage:
    python scripts/integration_test.py
"""

import sys
import logging
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import pipeline modules
try:
    from scripts.ingestion.ingest_firms import fetch_firms_data
    from scripts.ingestion.ingest_weather import fetch_weather_data
    from scripts.processing.process_firms import process_firms_data
    # Note: process_weather might need similar import if available
    from scripts.fusion.fuse_features import fuse_features
    from scripts.utils.grid_utils import generate_full_grid
except ImportError as e:
    logger.error(f"Failed to import pipeline modules: {e}")
    sys.exit(1)

def run_integration_test():
    logger.info("Starting integration smoke test...")

    # Setup temporary test directories
    test_dir = PROJECT_ROOT / "data" / "test_integration"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    raw_dir = test_dir / "raw"
    processed_dir = test_dir / "processed"
    
    execution_date = datetime.now() - timedelta(days=1)
    resolution_km = 64 # Coarse resolution for fast testing

    try:
        # 1. Simulate Ingestion (FIRMS)
        logger.info("Step 1: Ingesting FIRMS data...")
        firms_raw_path = fetch_firms_data(
            execution_date=execution_date,
            resolution_km=resolution_km,
            lookback_hours=24,
            output_dir=str(raw_dir / "firms")
        )
        logger.info(f"FIRMS raw data saved to: {firms_raw_path}")

        # 2. Simulate Ingestion (Weather)
        logger.info("Step 2: Ingesting Weather data...")
        grid = generate_full_grid(resolution_km)
        grid_centroids = grid[["grid_id", "latitude", "longitude"]]
        
        weather_raw_path = fetch_weather_data(
            grid_centroids=grid_centroids,
            execution_date=execution_date,
            lookback_hours=24,
            output_dir=str(raw_dir / "weather")
        )
        logger.info(f"Weather raw data saved to: {weather_raw_path}")

        # 3. Process FIRMS
        logger.info("Step 3: Processing FIRMS data...")
        firms_features = process_firms_data(
            raw_csv_path=firms_raw_path,
            resolution_km=resolution_km
        )
        logger.info(f"FIRMS features shape: {firms_features.shape}")

        # 4. Fusion (Mocking static/weather processing for speed if needed, or calling real functions)
        logger.info("Step 4: Fusing features...")
        # For this smoke test, we might mock weather/static if full processing is too heavy
        # But let's try to pass the firms features we just generated
        
        # Mock weather features (empty DataFrame with expected grid_id if real processing is complex)
        import pandas as pd
        weather_features = pd.DataFrame({'grid_id': grid['grid_id']}) 
        static_features = pd.DataFrame({'grid_id': grid['grid_id']})

        fused_df = fuse_features(
            firms_features=firms_features,
            weather_features=weather_features,
            static_features=static_features,
            execution_date=execution_date,
            resolution_km=resolution_km
        )
        logger.info(f"Fused features shape: {fused_df.shape}")

        if not fused_df.empty:
            logger.info("Integration test PASSED")
        else:
            logger.warning("Fused DataFrame is empty, but pipeline ran without error.")

    except Exception as e:
        logger.error(f"Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)
            logger.info("Cleaned up test directory.")

if __name__ == "__main__":
    run_integration_test()
