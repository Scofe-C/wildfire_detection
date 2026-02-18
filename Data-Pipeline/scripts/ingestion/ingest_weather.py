"""
Weather Data Ingestion
======================
Fetches weather data from Open-Meteo API (primary) with NWS API fallback.

Phase 2 MVP notes:
- We do NOT request Open-Meteo daily variables (to avoid unsupported params).
- We still output a stable schema including fire_weather_index (as None).
- Always writes a CSV even if all API calls fail (empty but with headers).

Owner: Person B
Dependencies: requests, pandas, numpy
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from scripts.utils.rate_limiter import RateLimiter, create_weather_limiter
from scripts.utils.schema_loader import get_registry

logger = logging.getLogger(__name__)

# Open-Meteo hourly variables
OPEN_METEO_HOURLY_PARAMS: list[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "soil_moisture_0_to_7cm",
    "vapor_pressure_deficit",
]

# Phase2 MVP: Do NOT request daily vars (avoid unsupported "fire_weather_index_max")
OPEN_METEO_DAILY_PARAMS: list[str] = []

# Maximum coordinates per Open-Meteo request
OPEN_METEO_MAX_LOCATIONS = 50


from datetime import datetime, timezone
import pandas as pd

def _to_utc_aware(dt) -> datetime:
    """Force any Airflow/Pendulum/Proxy datetime into real UTC-aware datetime."""
    
    if dt is None:
        raise ValueError("Datetime cannot be None")

    # 🔥 FIX: unwrap Airflow Proxy objects
    if hasattr(dt, "__wrapped__"):
        dt = dt.__wrapped__

    # Convert to string first (safest way)
    dt = str(dt)

    ts = pd.to_datetime(dt, utc=True)

    return ts.to_pydatetime()




def _write_empty_weather_csv(output_dir: Path, execution_date: datetime, *, reason: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    execution_date = _to_utc_aware(execution_date)

    date_str = execution_date.strftime("%Y%m%d_%H%M%S")
    out = output_dir / f"weather_empty_{date_str}.csv"

    df = pd.DataFrame(columns=[
        "grid_id",
        "timestamp",
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "precipitation",
        "soil_moisture_0_to_7cm",
        "vpd",
        "fire_weather_index",
        "data_quality_flag",
    ])
    df.to_csv(out, index=False)
    logger.warning(f"{reason} — wrote empty weather CSV: {out}")
    return out


def fetch_weather_data(
    grid_centroids: pd.DataFrame,
    execution_date: datetime,
    lookback_hours: int = 24,
    output_dir: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Path:
    """Fetch weather data for all grid cell centroids.

    Returns:
        Path to raw weather CSV.

    Guarantees:
        - Always writes a CSV file (empty w/ headers if needed).
        - Output includes grid_id, timestamp, data_quality_flag columns.
    """
    registry = get_registry(config_path)
    om_config = registry.get_source_config("open_meteo")
    limiter = create_weather_limiter(config_path)

    # resolve output_dir early
    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "weather"
        )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # normalize execution_date to UTC-aware
    execution_date = _to_utc_aware(execution_date)

    # basic guards
    if grid_centroids is None or grid_centroids.empty:
        return _write_empty_weather_csv(out_dir, execution_date, reason="No grid centroids provided")

    required = {"grid_id", "latitude", "longitude"}
    if not required.issubset(set(grid_centroids.columns)):
        missing = sorted(required - set(grid_centroids.columns))
        return _write_empty_weather_csv(out_dir, execution_date, reason=f"grid_centroids missing columns: {missing}")

    coord_precision = om_config.get("coordinate_precision", 3)

    grid_centroids = grid_centroids.copy()
    grid_centroids["grid_id"] = grid_centroids["grid_id"].astype(str)
    grid_centroids["latitude"] = grid_centroids["latitude"].round(coord_precision)
    grid_centroids["longitude"] = grid_centroids["longitude"].round(coord_precision)

    # time range (UTC-aware)
    end_date = execution_date
    start_date = _to_utc_aware(end_date - timedelta(hours=lookback_hours))

    logger.info(
        f"Fetching weather for {len(grid_centroids)} grid cells, "
        f"{start_date.isoformat()} to {end_date.isoformat()}"
    )

    batches = _create_coordinate_batches(grid_centroids, OPEN_METEO_MAX_LOCATIONS)

    all_weather: list[pd.DataFrame] = []
    failed_cells: list[str] = []

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
            weather_df = weather_df.copy()
            weather_df["grid_id"] = weather_df["grid_id"].astype(str)
            weather_df["data_quality_flag"] = 0  # fresh Open-Meteo
            all_weather.append(weather_df)
        else:
            logger.warning(
                f"  Open-Meteo failed for batch {batch_idx + 1}. Attempting NWS fallback."
            )
            for _, cell in batch.iterrows():
                nws_df = _fetch_nws_fallback(
                    lat=float(cell["latitude"]),
                    lon=float(cell["longitude"]),
                    grid_id=str(cell["grid_id"]),
                    config_path=config_path,
                )
                if nws_df is not None and len(nws_df) > 0:
                    nws_df = nws_df.copy()
                    nws_df["grid_id"] = nws_df["grid_id"].astype(str)
                    nws_df["data_quality_flag"] = 2
                    all_weather.append(nws_df)
                else:
                    failed_cells.append(str(cell["grid_id"]))

    if failed_cells:
        logger.warning(
            f"{len(failed_cells)} cells failed both Open-Meteo and NWS. "
            f"These will be forward-filled from previous data (future step)."
        )

    combined = pd.concat(all_weather, ignore_index=True) if all_weather else pd.DataFrame()

    # guarantee required columns
    expected_cols = [
        "grid_id",
        "timestamp",
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "precipitation",
        "soil_moisture_0_to_7cm",
        "vpd",
        "fire_weather_index",
        "data_quality_flag",
    ]
    for c in expected_cols:
        if c not in combined.columns:
            combined[c] = None

    # normalize timestamp dtype if present
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")

    date_str = execution_date.strftime("%Y%m%d_%H%M%S")
    output_path = out_dir / f"weather_raw_{date_str}.csv"

    if combined.empty:
        return _write_empty_weather_csv(out_dir, execution_date, reason="All weather API requests failed")

    combined.to_csv(output_path, index=False)

    logger.info(
        f"Weather ingestion complete: {len(combined)} rows for "
        f"{combined['grid_id'].nunique() if 'grid_id' in combined.columns else 0} "
        f"cells → {output_path}"
    )
    return output_path


def _create_coordinate_batches(grid_centroids: pd.DataFrame, batch_size: int) -> list[pd.DataFrame]:
    return [
        grid_centroids.iloc[i: i + batch_size]
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
    """Fetch Open-Meteo for multiple locations in one request."""
    lats = ",".join(batch["latitude"].astype(str))
    lons = ",".join(batch["longitude"].astype(str))

    # HARD timezone enforcement (defensive)
    start_date = _to_utc_aware(start_date)
    end_date = _to_utc_aware(end_date)

    now = datetime.now(timezone.utc)

    # Ensure both sides are aware UTC
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    cutoff = now - timedelta(days=5)

    url = historical_url if end_date < cutoff else base_url


    params = {
        "latitude": lats,
        "longitude": lons,
        "hourly": ",".join(OPEN_METEO_HOURLY_PARAMS),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "UTC",
    }
    if OPEN_METEO_DAILY_PARAMS:
        params["daily"] = ",".join(OPEN_METEO_DAILY_PARAMS)

    for attempt in range(max_retries):
        try:
            with limiter.acquire():
                resp = requests.get(url, params=params, timeout=timeout)

            if resp.status_code == 200:
                data = resp.json()
                return _parse_open_meteo_response(data, batch)

            if resp.status_code == 429:
                delay = limiter.get_backoff_delay()
                logger.warning(f"Open-Meteo rate limited. Backing off {delay:.1f}s (attempt {attempt+1}/{max_retries})")
                limiter.record_failure()
                time.sleep(delay)
                continue

            logger.warning(f"Open-Meteo HTTP {resp.status_code}: {resp.text[:200]}")
            logger.warning(f"Open-Meteo request params: hourly={params.get('hourly')} daily={params.get('daily')}")
            limiter.record_failure()
            time.sleep(limiter.get_backoff_delay())

        except requests.exceptions.RequestException as e:
            logger.warning(f"Open-Meteo request error: {e} (attempt {attempt+1}/{max_retries})")
            limiter.record_failure()
            time.sleep(limiter.get_backoff_delay())

    return None


def _parse_open_meteo_response(data: dict, batch: pd.DataFrame) -> pd.DataFrame:
    """Parse Open-Meteo JSON into flat rows (hourly)."""
    records: list[dict] = []

    # Endpoint may return either a dict (single location) or list (multi-location)
    results = data if isinstance(data, list) else [data]

    # The order of returned results should align with order of input coordinates
    grid_ids = batch["grid_id"].astype(str).tolist()

    for idx, result in enumerate(results):
        if idx >= len(grid_ids):
            break

        grid_id = grid_ids[idx]
        hourly = result.get("hourly", {})
        if not isinstance(hourly, dict) or not hourly:
            continue

        timestamps = hourly.get("time", [])
        if not timestamps:
            continue

        for t_idx, ts in enumerate(timestamps):
            rec = {"grid_id": grid_id, "timestamp": ts}

            for param in OPEN_METEO_HOURLY_PARAMS:
                values = hourly.get(param, [])
                rec[param] = values[t_idx] if t_idx < len(values) else None

            # Phase2 MVP placeholder
            rec["fire_weather_index"] = None
            records.append(rec)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # rename vapor_pressure_deficit -> vpd for downstream
    if "vapor_pressure_deficit" in df.columns:
        df = df.rename(columns={"vapor_pressure_deficit": "vpd"})

    # ensure missing expected fields exist
    for c in [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "precipitation",
        "soil_moisture_0_to_7cm",
        "vpd",
        "fire_weather_index",
    ]:
        if c not in df.columns:
            df[c] = None

    return df


def _fetch_nws_fallback(
    lat: float,
    lon: float,
    grid_id: str,
    config_path: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Fallback to NWS forecastHourly."""
    registry = get_registry(config_path)
    nws_config = registry.get_source_config("nws")
    base_url = nws_config["base_url"]
    user_agent = nws_config.get("user_agent", "WildfireMLOps/1.0")
    timeout = nws_config.get("timeout_seconds", 15)

    headers = {"User-Agent": user_agent, "Accept": "application/geo+json"}

    try:
        points_url = f"{base_url}/points/{lat:.4f},{lon:.4f}"
        resp = requests.get(points_url, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            logger.warning(f"NWS points lookup failed: HTTP {resp.status_code}")
            return None

        points = resp.json()
        props = points.get("properties", {})
        forecast_url = props.get("forecastHourly")
        if not forecast_url:
            logger.warning(f"NWS: No forecast URL for ({lat}, {lon})")
            return None

        resp = requests.get(forecast_url, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            logger.warning(f"NWS forecast failed: HTTP {resp.status_code}")
            return None

        forecast = resp.json()
        periods = forecast.get("properties", {}).get("periods", [])
        if not periods:
            return None

        recs = []
        for p in periods[:24]:
            recs.append({
                "grid_id": grid_id,
                "timestamp": p.get("startTime"),
                "temperature_2m": _fahrenheit_to_celsius(p.get("temperature")) if p.get("temperatureUnit") == "F" else p.get("temperature"),
                "relative_humidity_2m": (p.get("relativeHumidity") or {}).get("value"),
                "wind_speed_10m": _parse_nws_wind_speed(p.get("windSpeed")),
                "wind_direction_10m": _parse_nws_wind_direction(p.get("windDirection")),
                "precipitation": None,
                "soil_moisture_0_to_7cm": None,
                "vpd": None,
                "fire_weather_index": None,
            })

        df = pd.DataFrame(recs)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df

    except requests.exceptions.RequestException as e:
        logger.warning(f"NWS fallback failed for ({lat}, {lon}): {e}")
        return None


def _fahrenheit_to_celsius(f: Optional[float]) -> Optional[float]:
    if f is None:
        return None
    try:
        return round((float(f) - 32) * 5 / 9, 1)
    except Exception:
        return None


def _parse_nws_wind_speed(speed_str: Optional[str]) -> Optional[float]:
    if speed_str is None:
        return None
    try:
        s = str(speed_str).replace(" mph", "")
        parts = s.split(" to ")
        if len(parts) == 2:
            avg_mph = (float(parts[0]) + float(parts[1])) / 2
        else:
            avg_mph = float(parts[0])
        return round(avg_mph * 1.60934, 1)
    except Exception:
        return None


def _parse_nws_wind_direction(direction: Optional[str]) -> Optional[float]:
    if direction is None:
        return None
    direction_map = {
        "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
        "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
        "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
        "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
    }
    return direction_map.get(str(direction).strip().upper())
