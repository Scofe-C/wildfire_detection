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
    trigger_source: str = "cron",
    fire_cells: Optional[list] = None,
    h3_ring_max: int = 5,
) -> Path:
    """Fetch weather data for all grid cell centroids.

    On emergency/active watchdog triggers, fetches HRRR (15-min cycle,
    3 km resolution) for focal cells around the confirmed fire, then merges
    with Open-Meteo for the remaining background grid cells.
    On cron triggers, uses Open-Meteo only (unchanged behaviour).

    Args:
        grid_centroids:  DataFrame with grid_id, latitude, longitude.
        execution_date:  Airflow execution_date (UTC).
        lookback_hours:  Weather lookback window. 24h for cron; 2h for watchdog.
        output_dir:      Output directory for raw CSV.
        config_path:     Optional schema config path override.
        trigger_source:  DAG trigger source. If 'watchdog_emergency' or
                         'watchdog_active', HRRR is attempted for focal cells.
        fire_cells:      H3 cell IDs confirmed by the watchdog (used to build
                         the focal grid for HRRR extraction).
        h3_ring_max:     Focal grid outer ring (passed from DAG params).

    Returns:
        Path to raw weather CSV.

    Guarantees:
        - Always writes a CSV file (empty w/ headers if needed).
        - Output includes grid_id, timestamp, data_quality_flag columns.
    """
    registry = get_registry(config_path)
    om_config = registry.get_source_config("open_meteo")
    limiter = create_weather_limiter(config_path)

    # ------------------------------------------------------------------
    # HRRR branch: emergency / active watchdog triggers only
    # ------------------------------------------------------------------
    is_watchdog = trigger_source in ("watchdog_emergency", "watchdog_active")

    if is_watchdog and fire_cells:
        hrrr_path = _try_hrrr_focal(
            grid_centroids=grid_centroids,
            fire_cells=fire_cells,
            h3_ring_max=h3_ring_max,
            execution_date=execution_date,
            output_dir=output_dir,
            config_path=config_path,
        )
        if hrrr_path is not None:
            # HRRR succeeded — merge with Open-Meteo for background cells
            return _merge_hrrr_with_background(
                hrrr_path=hrrr_path,
                grid_centroids=grid_centroids,
                execution_date=execution_date,
                lookback_hours=lookback_hours,
                output_dir=output_dir,
                om_config=om_config,
                limiter=limiter,
                config_path=config_path,
            )
        # HRRR failed — fall through to full Open-Meteo with narrowed window
        logger.warning(
            "HRRR fetch failed — falling back to Open-Meteo for all cells "
            f"(lookback_hours={lookback_hours})"
        )

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
            # Brief pause between batches to stay under rate limits
            time.sleep(0.3)
        else:
            logger.warning(
                f"  Open-Meteo failed for batch {batch_idx + 1}. Attempting NWS fallback."
            )
            for _, cell in batch.iterrows():
                lat = float(cell["latitude"])
                lon = float(cell["longitude"])
                # NWS only covers CONUS land — skip offshore / border points
                if lat < 24.5 or lat > 49.5 or lon < -125.0 or lon > -66.5:
                    failed_cells.append(str(cell["grid_id"]))
                    continue
                nws_df = _fetch_nws_fallback(
                    lat=lat,
                    lon=lon,
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

            # 4xx (except 429) are non-retryable — fail immediately
            if 400 <= resp.status_code < 500:
                logger.error(
                    f"Open-Meteo non-retryable error: HTTP {resp.status_code}: "
                    f"{resp.text[:200]}"
                )
                return None

            # 5xx — transient server error, retry with backoff
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


# ---------------------------------------------------------------------------
# HRRR integration helpers (called only on watchdog emergency/active triggers)
# ---------------------------------------------------------------------------

def _try_hrrr_focal(
    grid_centroids: pd.DataFrame,
    fire_cells: list,
    h3_ring_max: int,
    execution_date,
    output_dir: Optional[str],
    config_path: Optional[str],
) -> Optional[Path]:
    """Attempt HRRR fetch for the focal grid around confirmed fire cells.

    Returns:
        Path to HRRR CSV, or None if HRRR is unavailable / fails.
        None signals the caller to fall back to Open-Meteo.
    """
    try:
        from scripts.ingestion.ingest_hrrr import fetch_hrrr_for_focal_grid
        from scripts.utils.grid_utils import generate_fire_focal_grid
    except ImportError as e:
        logger.warning(f"HRRR dependencies not installed — skipping HRRR: {e}")
        return None

    try:
        focal_grid = generate_fire_focal_grid(
            fire_cell_ids=fire_cells,
            ring_min=0,          # include the fire cells themselves (ring 0)
            ring_max=h3_ring_max,
        )

        if focal_grid.empty:
            logger.warning("HRRR: focal grid is empty — skipping")
            return None

        logger.info(
            f"HRRR: generated focal grid with {len(focal_grid)} cells "
            f"({sum(focal_grid['cell_type'] == 'fire')} fire, "
            f"{sum(focal_grid['cell_type'] == 'detection_zone')} detection zone)"
        )

        return fetch_hrrr_for_focal_grid(
            focal_grid=focal_grid[["grid_id", "latitude", "longitude"]],
            execution_date=execution_date,
            output_dir=output_dir,
            config_path=config_path,
        )

    except Exception as e:
        logger.warning(f"HRRR focal fetch failed: {e}")
        return None


def _merge_hrrr_with_background(
    hrrr_path: Path,
    grid_centroids: pd.DataFrame,
    execution_date,
    lookback_hours: int,
    output_dir: Optional[str],
    om_config: dict,
    limiter,
    config_path: Optional[str],
) -> Path:
    """Merge HRRR focal data with Open-Meteo background data.

    Strategy:
      1. Read HRRR CSV — these are the focal (fire + detection zone) cells.
      2. Identify background cells: grid cells NOT covered by HRRR.
      3. Fetch Open-Meteo for background cells only (narrowed lookback window).
         Background cells = previous cron run data → data_quality_flag = 4
         if Open-Meteo also fails, they stay null (fuse_features forward-fills).
      4. Concatenate HRRR rows + Open-Meteo rows → write merged CSV.

    This gives the model:
      - flag=3 (HRRR, ~15 min fresh) for focal cells
      - flag=0 (Open-Meteo, ~1h fresh) or flag=4 (forward-fill) for background

    Returns:
        Path to merged CSV.
    """
    execution_date = _to_utc_aware(execution_date)

    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "weather"
        )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load HRRR focal data
    hrrr_df = pd.read_csv(hrrr_path)
    hrrr_cell_ids = set(hrrr_df["grid_id"].astype(str).tolist())

    # Identify background cells (not covered by HRRR)
    all_cell_ids = set(grid_centroids["grid_id"].astype(str).tolist())
    background_ids = all_cell_ids - hrrr_cell_ids

    logger.info(
        f"HRRR merge: {len(hrrr_cell_ids)} HRRR focal cells, "
        f"{len(background_ids)} background cells for Open-Meteo"
    )

    parts = [hrrr_df]

    if background_ids:
        background_centroids = grid_centroids[
            grid_centroids["grid_id"].astype(str).isin(background_ids)
        ].copy()

        # Fetch Open-Meteo for background cells
        # Batched using the same logic as the main Open-Meteo path
        batches = _create_coordinate_batches(background_centroids, OPEN_METEO_MAX_LOCATIONS)
        bg_weather = []

        for batch_idx, batch in enumerate(batches):
            weather_df = _fetch_open_meteo_batch(
                batch=batch,
                start_date=_to_utc_aware(execution_date - timedelta(hours=lookback_hours)),
                end_date=execution_date,
                base_url=om_config["base_url"],
                historical_url=om_config["historical_url"],
                limiter=limiter,
                timeout=om_config.get("timeout_seconds", 20),
                max_retries=om_config.get("max_retries", 3),
            )
            if weather_df is not None and not weather_df.empty:
                weather_df = weather_df.copy()
                weather_df["grid_id"] = weather_df["grid_id"].astype(str)
                weather_df["data_quality_flag"] = 0
                bg_weather.append(weather_df)
            else:
                logger.debug(f"Open-Meteo failed for background batch {batch_idx + 1}")

        if bg_weather:
            parts.append(pd.concat(bg_weather, ignore_index=True))

    merged = pd.concat(parts, ignore_index=True)

    # Ensure schema columns
    expected_cols = [
        "grid_id", "timestamp", "temperature_2m", "relative_humidity_2m",
        "wind_speed_10m", "wind_direction_10m", "precipitation",
        "soil_moisture_0_to_7cm", "vpd", "fire_weather_index", "data_quality_flag",
    ]
    for c in expected_cols:
        if c not in merged.columns:
            merged[c] = None

    date_str = execution_date.strftime("%Y%m%d_%H%M%S")
    output_path = out_dir / f"weather_raw_{date_str}.csv"
    merged.to_csv(output_path, index=False)

    logger.info(
        f"HRRR+OM merge complete: {len(hrrr_df)} HRRR rows + "
        f"{len(merged) - len(hrrr_df)} Open-Meteo rows → {output_path}"
    )
    return output_path