"""
Weather Data Processing
=======================
Processes raw hourly weather data and computes derived features.
NO AGGREGATION - maintains hourly resolution.

Owner: Mohammed
Dependencies: pandas, numpy

Input: Raw hourly CSV from ingest_weather.py
Output: Hourly DataFrame with 8 base features + 4 derived features

Derived Features:
    1. fire_weather_index: Canadian FWI approximation
    2. days_since_precipitation: Days since last precip > 1mm
    3. cumulative_wind_run_24h: Total wind distance over last 24h (km)
    4. drought_proxy: Normalized drought index (0=wet, 1=extreme)
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def process_weather_data(
    raw_csv_path: str,
    execution_date: datetime,
    history_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Process raw hourly weather data - compute derived features only.
    NO AGGREGATION - maintains hourly resolution.

    Args:
        raw_csv_path: Path to raw weather CSV from ingest_weather.py
        execution_date: Pipeline execution timestamp
        history_dir: Directory containing past processed weather CSVs

    Returns:
        DataFrame with hourly records containing:
            Base features (8):
            - grid_id, timestamp
            - temperature_2m, relative_humidity_2m, wind_speed_10m
            - wind_direction_10m, precipitation, soil_moisture_3_to_9cm
            - vapor_pressure_deficit, data_quality_flag
            
            Derived features (4):
            - fire_weather_index (0-100+)
            - days_since_precipitation (integer days)
            - cumulative_wind_run_24h (km over last 24h)
            - drought_proxy (0-1, normalized)
    """
    logger.info(f"Processing weather data from {raw_csv_path}")
    
    # Load raw hourly data
    df = pd.read_csv(raw_csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["grid_id", "timestamp"]).reset_index(drop=True)
    
    logger.info(
        f"Loaded {len(df)} hourly records for {df['grid_id'].nunique()} grid cells"
    )
    
    # ========================================================================
    # DERIVED FEATURE 1: Fire Weather Index (per row)
    # ========================================================================
    logger.info("Computing Fire Weather Index...")
    df["fire_weather_index"] = df.apply(
        lambda row: _calculate_fire_weather_index(
            temp=row["temperature_2m"],
            rh=row["relative_humidity_2m"],
            wind_speed=row["wind_speed_10m"],
            precipitation=row["precipitation"]
        ),
        axis=1
    )
    
    # ========================================================================
    # DERIVED FEATURE 2: Days Since Precipitation (requires history)
    # ========================================================================
    logger.info("Computing days since precipitation...")
    history_df = _load_historical_precipitation(history_dir, execution_date)
    
    df["days_since_precipitation"] = df.apply(
        lambda row: _compute_days_since_precip_for_row(
            row, df, history_df
        ),
        axis=1
    )
    
    # ========================================================================
    # DERIVED FEATURE 3: Cumulative Wind Run (rolling 24h per grid)
    # ========================================================================
    logger.info("Computing cumulative wind run (24h rolling)...")
    df["cumulative_wind_run_24h"] = df.groupby("grid_id", group_keys=False).apply(
        _compute_rolling_wind_run
    ).reset_index(level=0, drop=True)
    
    # ========================================================================
    # DERIVED FEATURE 4: Drought Proxy (per row)
    # ========================================================================
    logger.info("Computing drought proxy...")
    df["drought_proxy"] = df.apply(
        lambda row: _compute_drought_proxy_for_row(row),
        axis=1
    )
    
    # Round numerical columns for cleaner output
    numeric_cols = [
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
        "wind_direction_10m", "precipitation", "soil_moisture_3_to_9cm",
        "vapor_pressure_deficit", "fire_weather_index", 
        "cumulative_wind_run_24h", "drought_proxy"
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].round(2)
    
    logger.info(
        f"Processing complete: {len(df)} hourly records with "
        f"{len(df.columns)} features"
    )
    
    return df


# ============================================================================
# DERIVED FEATURE FUNCTIONS
# ============================================================================

def _calculate_fire_weather_index(
    temp: Optional[float],
    rh: Optional[float],
    wind_speed: Optional[float],
    precipitation: Optional[float] = 0.0
) -> Optional[float]:
    """
    Calculate Fire Weather Index using simplified Canadian FWI System.
    
    Based on: Van Wagner, C.E. 1987. Development and structure of the 
    Canadian Forest Fire Weather Index System.
    
    Args:
        temp: Temperature in Celsius
        rh: Relative humidity (0-100%)
        wind_speed: Wind speed in km/h
        precipitation: Precipitation in mm (default 0)
        
    Returns:
        Fire Weather Index (0-100+, higher = more dangerous)
        None if required inputs are missing
    """
    if temp is None or rh is None or wind_speed is None:
        return None
    
    # Ensure valid ranges
    temp = max(-50, min(50, temp))
    rh = max(0, min(100, rh))
    wind_speed = max(0, wind_speed)
    precipitation = max(0, precipitation or 0)
    
    # Fine Fuel Moisture Code (FFMC) approximation
    if precipitation > 0.5:
        ffmc = 85 - (precipitation * 2)
    else:
        mo = 147.2 * (101 - rh) / (59.5 + rh)
        mo = max(0, min(150, mo))
        ffmc = 59.5 * (250 - mo) / (147.2 + mo)
    
    ffmc = max(0, min(101, ffmc))
    
    # Initial Spread Index (ISI)
    fw = np.exp(0.05039 * wind_speed)
    fm = 147.2 * (101 - ffmc) / (59.5 + ffmc)
    if fm <= 1:
        ff = 91.9 * np.exp(-0.1386 * fm) * (1 + (fm**5.31) / 49300000)
    else:
        ff = 91.9 * np.exp(-0.1386 * fm)
    
    isi = 0.208 * fw * ff
    
    # Buildup Index (BUI) approximation
    temp_factor = max(0, (temp + 10) / 50)
    dry_factor = max(0, (100 - rh) / 100)
    precip_factor = max(0, 1 - (precipitation / 10))
    
    bui = 80 * temp_factor * dry_factor * precip_factor
    bui = max(0, min(200, bui))
    
    # Fire Weather Index (FWI)
    if bui <= 80:
        fwi = 0.1 * isi * (0.626 * (bui**0.809) + 2)
    else:
        fwi = 0.1 * isi * (1000 / (25 + 108.64 / np.exp(0.023 * bui)))
    
    # Temperature adjustment
    if temp > 30:
        fwi *= (1 + (temp - 30) * 0.02)
    
    return round(min(100, fwi), 1)


def _compute_days_since_precip_for_row(
    row: pd.Series,
    current_df: pd.DataFrame,
    history_df: Optional[pd.DataFrame]
) -> int:
    """
    Compute days since last precipitation > 1mm for a single row.
    
    Logic:
        1. Check current hour - if precip > 1mm, return 0
        2. Look backwards in current data
        3. Look in historical data if needed
        4. Cap at 365 days if no recent rain found
    
    Args:
        row: Current row (single hour)
        current_df: Full current DataFrame
        history_df: Historical precipitation data
        
    Returns:
        Integer days since last precip > 1mm
    """
    grid_id = row["grid_id"]
    current_time = row["timestamp"]
    
    # Check current hour
    if row["precipitation"] > 1.0:
        return 0
    
    # Look backwards in current data
    grid_data = current_df[
        (current_df["grid_id"] == grid_id) &
        (current_df["timestamp"] < current_time)
    ].sort_values("timestamp", ascending=False)
    
    for _, past_row in grid_data.iterrows():
        if past_row["precipitation"] > 1.0:
            days_diff = (current_time - past_row["timestamp"]).total_seconds() / 86400
            return int(days_diff)
    
    # Look in historical data
    if history_df is not None and len(history_df) > 0:
        grid_history = history_df[history_df["grid_id"] == grid_id]
        
        if len(grid_history) > 0:
            significant_precip = grid_history[grid_history["precipitation"] > 1.0]
            
            if len(significant_precip) > 0:
                last_precip_time = significant_precip["timestamp"].max()
                days_diff = (current_time - last_precip_time).total_seconds() / 86400
                return int(days_diff)
    
    # No recent precipitation found - cap at 365 days
    return 365


def _compute_rolling_wind_run(group: pd.DataFrame) -> pd.Series:
    """
    Compute rolling 24-hour cumulative wind run for a single grid cell.
    
    Wind run = total distance wind would transport a particle.
    Each hour: distance = wind_speed (km/h) * 1 hour = wind_speed km
    
    Args:
        group: DataFrame for single grid_id, sorted by timestamp
        
    Returns:
        Series of cumulative wind run values (km over last 24h)
    """
    # Rolling sum over last 24 hours
    wind_run = group["wind_speed_10m"].rolling(
        window=24,
        min_periods=1  # Allow partial windows at start
    ).sum()
    
    return wind_run


def _compute_drought_proxy_for_row(row: pd.Series) -> float:
    """
    Compute drought proxy for a single hour.
    
    Formula: drought_proxy = 1 - (soil_moisture * precip_factor / temp_factor)
    
    Where:
        - soil_moisture: Normalized 0-1
        - precip_factor: Recent rain damping (0.5-1.0)
        - temp_factor: Temperature stress (1.0-2.0)
        
    Output: 0 = wet, 1 = extreme drought
    
    Args:
        row: Single row with soil_moisture, temperature, precipitation
        
    Returns:
        Drought proxy value (0-1)
    """
    # Normalize soil moisture (typical range: 0-0.5 m³/m³)
    soil_moisture = row["soil_moisture_3_to_9cm"]
    if pd.isna(soil_moisture):
        soil_moisture = 0.25  # Default to medium
    soil_norm = min(1.0, soil_moisture / 0.5)
    
    # Precipitation damping factor
    precip = row["precipitation"]
    if pd.isna(precip):
        precip = 0
    precip_factor = min(1.0, 0.5 + (precip / 10.0))
    
    # Temperature stress factor
    temp = row["temperature_2m"]
    if pd.isna(temp):
        temp = 15  # Default to mild
    temp_factor = min(2.0, 1.0 + max(0, (temp - 15) / 20.0))
    
    # Compute drought proxy
    drought = 1.0 - (soil_norm * precip_factor / temp_factor)
    
    # Clip to [0, 1]
    return max(0.0, min(1.0, drought))


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _load_historical_precipitation(
    history_dir: Optional[str],
    execution_date: datetime,
    lookback_days: int = 30
) -> Optional[pd.DataFrame]:
    """
    Load historical precipitation data from past pipeline outputs.
    
    Args:
        history_dir: Directory containing processed weather CSVs
        execution_date: Current execution date
        lookback_days: How many days to look back (default: 30)

    Returns:
        DataFrame with [grid_id, timestamp, precipitation] or None
    """
    if history_dir is None:
        history_dir = (
            Path(__file__).resolve().parent.parent.parent 
            / "data" / "processed" / "weather"
        )
    
    history_dir = Path(history_dir)
    
    if not history_dir.exists():
        logger.warning(f"History directory not found: {history_dir}")
        return None
    
    # Look for CSV files within lookback period
    start_date = execution_date - timedelta(days=lookback_days)
    
    historical_data = []
    
    for csv_file in sorted(history_dir.glob("weather_processed_*.csv")):
        # Extract date from filename: weather_processed_YYYYMMDD_HHMMSS.csv
        try:
            date_str = csv_file.stem.split("_")[2]
            file_date = datetime.strptime(date_str, "%Y%m%d")
            
            if start_date <= file_date <= execution_date:
                df = pd.read_csv(csv_file, usecols=["grid_id", "timestamp", "precipitation"])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                historical_data.append(df)
        except (ValueError, IndexError):
            continue
    
    if not historical_data:
        logger.warning("No historical precipitation data found")
        return None
    
    combined = pd.concat(historical_data, ignore_index=True)
    logger.info(f"Loaded {len(combined)} historical precipitation records")
    
    return combined


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Standalone test for weather processing.
    Assumes raw weather data already exists from ingest_weather.py
    """
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Find most recent raw weather file
    raw_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "weather"
    
    if not raw_dir.exists():
        print(f"❌ No raw weather data found in {raw_dir}")
        print("   Run ingest_weather.py first!")
        sys.exit(1)
    
    raw_files = sorted(raw_dir.glob("weather_raw_*.csv"))
    
    if not raw_files:
        print(f"❌ No raw weather CSV files found in {raw_dir}")
        sys.exit(1)
    
    latest_raw = raw_files[-1]
    print(f"\n🔥 Processing weather data from: {latest_raw}")
    
    # Extract execution date from filename
    date_str = latest_raw.stem.split("_")[2] + "_" + latest_raw.stem.split("_")[3]
    execution_date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
    
    try:
        processed_df = process_weather_data(
            raw_csv_path=str(latest_raw),
            execution_date=execution_date,
        )
        
        # Save processed data
        processed_dir = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "weather"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        date_str = execution_date.strftime("%Y%m%d_%H%M%S")
        output_path = processed_dir / f"weather_processed_{date_str}.csv"
        processed_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Success! Processed weather saved to: {output_path}")
        print(f"\n📊 Summary:")
        print(f"  Total records: {len(processed_df)}")
        print(f"  Grid cells: {processed_df['grid_id'].nunique()}")
        print(f"  Time range: {processed_df['timestamp'].min()} to {processed_df['timestamp'].max()}")
        print(f"\n📋 Columns ({len(processed_df.columns)}):")
        print(f"  {list(processed_df.columns)}")
        print(f"\n🔥 Sample data (first 5 rows):")
        print(processed_df.head())
        
        # Show derived feature statistics
        print(f"\n📈 Derived Feature Statistics:")
        derived_cols = ["fire_weather_index", "days_since_precipitation", 
                       "cumulative_wind_run_24h", "drought_proxy"]
        print(processed_df[derived_cols].describe())
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()