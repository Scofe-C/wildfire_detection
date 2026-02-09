"""
Weather Data Ingestion
======================
Fetches weather data from Open-Meteo API (primary) with NWS API fallback.
Handles rate limiting, coordinate snapping, and outputs raw JSON/CSV
to the staging area.

Owner: Person B
Dependencies: requests, pandas, numpy

Primary: Open-Meteo (https://open-meteo.com/en/docs)
Fallback: NWS API (https://api.weather.gov) — US only, no auth required

Key behaviors:
    - Batches grid cell centroids into Open-Meteo multi-coordinate requests
    - Falls back to NWS API if Open-Meteo fails for a grid cell
    - Rounds coordinates to 3 decimal places for consistency
    - Tags data quality: 0=Open-Meteo fresh, 1=forward-filled, 2=NWS fallback
"""

import io
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from scripts.utils.rate_limiter import RateLimiter, create_weather_limiter
from scripts.utils.schema_loader import get_registry

logger = logging.getLogger(__name__)

# Open-Meteo hourly variables that map to our schema weather features
OPEN_METEO_HOURLY_PARAMS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "soil_moisture_0_to_7cm",
    "vapor_pressure_deficit",
]

# Open-Meteo daily variables
OPEN_METEO_DAILY_PARAMS = [
    "fire_weather_index_max",
]

# Maximum coordinates per Open-Meteo request
# Open-Meteo supports multi-location in a single request
OPEN_METEO_MAX_LOCATIONS = 50


def fetch_weather_data(
    grid_centroids: pd.DataFrame,
    execution_date: datetime,
    lookback_hours: int = 24,
    output_dir: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Path:
    """Fetch weather data for all grid cell centroids.

    This is the main entry point called by the Airflow task.

    Args:
        grid_centroids: DataFrame with columns [grid_id, latitude, longitude].
        execution_date: Pipeline execution timestamp.
        lookback_hours: Hours of historical data to fetch.
        output_dir: Local directory to write raw output.
        config_path: Optional schema config override.

    Returns:
        Path to the output CSV file.
    """
    registry = get_registry(config_path)
    om_config = registry.get_source_config("open_meteo")
    limiter = create_weather_limiter(config_path)

    coord_precision = om_config.get("coordinate_precision", 3)

    # Round coordinates for consistency (prevents cache misses from float noise)
    grid_centroids = grid_centroids.copy()
    grid_centroids["latitude"] = grid_centroids["latitude"].round(coord_precision)
    grid_centroids["longitude"] = grid_centroids["longitude"].round(coord_precision)

    # Calculate time range
    end_date = execution_date
    start_date = end_date - timedelta(hours=lookback_hours)

    logger.info(
        f"Fetching weather for {len(grid_centroids)} grid cells, "
        f"{start_date.isoformat()} to {end_date.isoformat()}"
    )

    # Batch centroids into groups for multi-location requests
    batches = _create_coordinate_batches(grid_centroids, OPEN_METEO_MAX_LOCATIONS)

    all_weather = []
    failed_cells = []

    for batch_idx, batch in enumerate(batches):
        logger.info(
            f"  Weather batch {batch_idx + 1}/{len(batches)} "
            f"({len(batch)} locations)"
        )

        weather_df = _fetch_open_meteo_batch(
            batch=batch,
            start_date=start_date,
            end_date=end_date,
            base_url=om_config["base_url"],
            historical_url=om_config["historical_url"],
            limiter=limiter,
            timeout=om_config.get("timeout_seconds", 20),
            max_retries=om_config.get("max_retries", 3),
        )

        if weather_df is not None and len(weather_df) > 0:
            weather_df["data_quality_flag"] = 0  # Fresh Open-Meteo data
            all_weather.append(weather_df)
        else:
            # Open-Meteo failed for this batch — try NWS fallback
            logger.warning(
                f"  Open-Meteo failed for batch {batch_idx + 1}. "
                f"Attempting NWS fallback."
            )
            for _, cell in batch.iterrows():
                nws_data = _fetch_nws_fallback(
                    lat=cell["latitude"],
                    lon=cell["longitude"],
                    grid_id=cell["grid_id"],
                    config_path=config_path,
                )
                if nws_data is not None:
                    nws_data["data_quality_flag"] = 2  # NWS fallback
                    all_weather.append(nws_data)
                else:
                    failed_cells.append(cell["grid_id"])

    if failed_cells:
        logger.warning(
            f"{len(failed_cells)} cells failed both Open-Meteo and NWS. "
            f"These will be forward-filled from previous data."
        )

    # Combine results
    if all_weather:
        combined = pd.concat(all_weather, ignore_index=True)
    else:
        logger.error("All weather API requests failed.")
        combined = pd.DataFrame()

    # Write output
    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "weather"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = execution_date.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"weather_raw_{date_str}.csv"
    combined.to_csv(output_path, index=False)

    logger.info(
        f"Weather ingestion complete: {len(combined)} rows for "
        f"{combined['grid_id'].nunique() if 'grid_id' in combined.columns else 0} "
        f"cells → {output_path}"
    )
    return output_path


def _create_coordinate_batches(
    grid_centroids: pd.DataFrame, batch_size: int
) -> list[pd.DataFrame]:
    """Split grid centroids into batches for API requests."""
    return [
        grid_centroids.iloc[i : i + batch_size]
        for i in range(0, len(grid_centroids), batch_size)
    ]


def _fetch_open_meteo_batch(
    batch: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    base_url: str,
    historical_url: str,
    limiter: RateLimiter,
    timeout: int = 20,
    max_retries: int = 3,
) -> Optional[pd.DataFrame]:
    """Fetch weather data from Open-Meteo for a batch of locations.

    Open-Meteo supports multiple latitude/longitude pairs in a single request.

    Returns:
        DataFrame with columns: grid_id, timestamp, and all weather variables.
    """
    lats = ",".join(batch["latitude"].astype(str))
    lons = ",".join(batch["longitude"].astype(str))

    # Determine if we need historical or forecast API
    now = datetime.utcnow()
    if end_date < now - timedelta(days=5):
        url = historical_url
    else:
        url = base_url

    params = {
        "latitude": lats,
        "longitude": lons,
        "hourly": ",".join(OPEN_METEO_HOURLY_PARAMS),
        "daily": ",".join(OPEN_METEO_DAILY_PARAMS),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "UTC",
    }

    for attempt in range(max_retries):
        try:
            with limiter.acquire():
                response = requests.get(url, params=params, timeout=timeout)

            if response.status_code == 200:
                data = response.json()
                return _parse_open_meteo_response(data, batch)

            elif response.status_code == 429:
                delay = limiter.get_backoff_delay()
                logger.warning(f"Open-Meteo rate limited. Backing off {delay:.1f}s")
                limiter.record_failure()
                time.sleep(delay)

            else:
                logger.warning(
                    f"Open-Meteo HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
                limiter.record_failure()
                time.sleep(limiter.get_backoff_delay())

        except requests.exceptions.RequestException as e:
            logger.warning(f"Open-Meteo request error: {e}")
            limiter.record_failure()
            time.sleep(limiter.get_backoff_delay())

    return None


def _parse_open_meteo_response(
    data: dict, batch: pd.DataFrame
) -> pd.DataFrame:
    """Parse Open-Meteo JSON response into a flat DataFrame.

    Open-Meteo returns an array of results when multiple coordinates
    are requested. Each result contains hourly and daily arrays.
    """
    records = []

    # Handle single vs. multi-location response
    if isinstance(data.get("hourly"), dict):
        # Single location — wrap in list for uniform processing
        results = [data]
    else:
        # Multi-location — data is a list
        results = data if isinstance(data, list) else [data]

    grid_ids = batch["grid_id"].tolist()

    for idx, result in enumerate(results):
        if idx >= len(grid_ids):
            break

        grid_id = grid_ids[idx]
        hourly = result.get("hourly", {})
        daily = result.get("daily", {})

        timestamps = hourly.get("time", [])

        for t_idx, timestamp in enumerate(timestamps):
            record = {
                "grid_id": grid_id,
                "timestamp": timestamp,
            }

            # Extract hourly variables
            for param in OPEN_METEO_HOURLY_PARAMS:
                values = hourly.get(param, [])
                record[param] = values[t_idx] if t_idx < len(values) else None

            records.append(record)

        # Map daily fire_weather_index to the hourly timestamps
        # (apply the daily max to all hours of that day)
        fwi_values = daily.get("fire_weather_index_max", [])
        fwi_dates = daily.get("time", [])
        fwi_map = dict(zip(fwi_dates, fwi_values))

        for record in records:
            if record["grid_id"] == grid_id:
                date_str = record["timestamp"][:10]  # Extract YYYY-MM-DD
                record["fire_weather_index"] = fwi_map.get(date_str)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _fetch_nws_fallback(
    lat: float,
    lon: float,
    grid_id: str,
    config_path: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Fetch weather data from NWS API as fallback.

    NWS API is free, requires no key, but has a two-step process:
    1. Get the grid point metadata: /points/{lat},{lon}
    2. Get the forecast: /gridpoints/{office}/{gridX},{gridY}/forecast/hourly

    NWS provides forecast data only (not historical), so this is
    suitable only for recent/current conditions.

    Returns:
        DataFrame with weather variables for the grid cell, or None.
    """
    registry = get_registry(config_path)
    nws_config = registry.get_source_config("nws")
    base_url = nws_config["base_url"]
    user_agent = nws_config.get("user_agent", "WildfireMLOps/1.0")
    timeout = nws_config.get("timeout_seconds", 15)

    headers = {"User-Agent": user_agent, "Accept": "application/geo+json"}

    try:
        # Step 1: Resolve grid point
        points_url = f"{base_url}/points/{lat:.4f},{lon:.4f}"
        resp = requests.get(points_url, headers=headers, timeout=timeout)

        if resp.status_code != 200:
            logger.warning(f"NWS points lookup failed: HTTP {resp.status_code}")
            return None

        points_data = resp.json()
        properties = points_data.get("properties", {})
        forecast_url = properties.get("forecastHourly")

        if not forecast_url:
            logger.warning(f"NWS: No forecast URL for ({lat}, {lon})")
            return None

        # Step 2: Get hourly forecast
        resp = requests.get(forecast_url, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            logger.warning(f"NWS forecast failed: HTTP {resp.status_code}")
            return None

        forecast_data = resp.json()
        periods = forecast_data.get("properties", {}).get("periods", [])

        if not periods:
            return None

        # Parse NWS periods into our schema
        records = []
        for period in periods[:24]:  # Take up to 24 hours
            record = {
                "grid_id": grid_id,
                "timestamp": period.get("startTime"),
                "temperature_2m": _fahrenheit_to_celsius(
                    period.get("temperature")
                ) if period.get("temperatureUnit") == "F" else period.get("temperature"),
                "relative_humidity_2m": period.get("relativeHumidity", {}).get("value"),
                "wind_speed_10m": _parse_nws_wind_speed(period.get("windSpeed")),
                "wind_direction_10m": _parse_nws_wind_direction(
                    period.get("windDirection")
                ),
                "precipitation": None,  # NWS doesn't provide exact precip in forecast
                "soil_moisture_0_to_7cm": None,  # Not available from NWS
                "vpd": None,  # Not available from NWS
                "fire_weather_index": None,  # Not available from NWS
            }
            records.append(record)

        if not records:
            return None

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    except requests.exceptions.RequestException as e:
        logger.warning(f"NWS fallback failed for ({lat}, {lon}): {e}")
        return None


def _fahrenheit_to_celsius(f: Optional[float]) -> Optional[float]:
    """Convert Fahrenheit to Celsius."""
    if f is None:
        return None
    return round((f - 32) * 5 / 9, 1)


def _parse_nws_wind_speed(speed_str: Optional[str]) -> Optional[float]:
    """Parse NWS wind speed string (e.g., '15 mph') to km/h."""
    if speed_str is None:
        return None
    try:
        # NWS returns strings like "15 mph" or "10 to 20 mph"
        parts = speed_str.replace(" mph", "").split(" to ")
        if len(parts) == 2:
            avg_mph = (float(parts[0]) + float(parts[1])) / 2
        else:
            avg_mph = float(parts[0])
        return round(avg_mph * 1.60934, 1)  # mph to km/h
    except (ValueError, AttributeError):
        return None


def _parse_nws_wind_direction(direction: Optional[str]) -> Optional[float]:
    """Convert NWS cardinal direction to degrees."""
    if direction is None:
        return None
    direction_map = {
        "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
        "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
        "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
        "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
    }
    return direction_map.get(direction)
