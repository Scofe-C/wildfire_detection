"""
Weather Data Ingestion (Optimized for 22km+ Regions)
=====================================================
Fetches weather data from Open-Meteo API (primary) with NWS API fallback.

Changes vs previous version:
  1. OPEN_METEO_MAX_LOCATIONS raised to 100 for higher throughput.
  2. _NWS_MAX_FALLBACK_CELLS raised to 50 (module-level constant).
  3. _BATCH_PAUSE_SECONDS reduced to 0.2s.
  4. NWS_HEADERS constant with compliant User-Agent (contact info included).
  5. _nws_is_reachable() uses DNS check (socket.gethostbyname) — simpler.
  6. _fetch_nws_fallback() uses NWS_HEADERS + 4-decimal rounding, no config_path.
  7. CONUS bbox relaxed to 24.0-50.0 lat / -125.0 to -66.0 lon.
  8. Rate limiter: wait_if_needed() + conditional record_request() on 200 only.
  9. Gulf water exclusion replaced by relaxed CONUS bbox check.

Owner: Person B
Dependencies: requests, pandas
"""

import logging
import socket
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from scripts.utils.datetime_utils import coerce_to_utc
from scripts.utils.rate_limiter import RateLimiter, create_weather_limiter
from scripts.utils.schema_loader import get_registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Open-Meteo variable lists
# ---------------------------------------------------------------------------

OPEN_METEO_HOURLY_PARAMS: list[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "soil_moisture_0_to_7cm",
    "vapor_pressure_deficit",
]

# Phase 2 MVP: daily vars disabled (avoid unsupported "fire_weather_index_max")
OPEN_METEO_DAILY_PARAMS: list[str] = []

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

OPEN_METEO_MAX_LOCATIONS = 100   # 100 locations/batch — 500 causes HTTP 414 (URL too long)

_NWS_MAX_FALLBACK_CELLS = 50     # Raised from 10 — more coverage on batch failure
_BATCH_PAUSE_SECONDS    = 1.5    # 1.5s inter-batch pause — stays under Open-Meteo burst limit

# NWS strictly requires a User-Agent with contact info per api.weather.gov TOS
NWS_HEADERS = {
    "User-Agent": "(Wildfire-MLOps-Pipeline, contact-admin@yourdomain.com)",
    "Accept":     "application/geo+json",
}

# Expected output schema columns (in order)
_SCHEMA_COLS = [
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


# ---------------------------------------------------------------------------
# Empty-CSV helper
# ---------------------------------------------------------------------------

def _write_empty_weather_csv(
    output_dir: Path, execution_date: datetime, *, reason: str
) -> Path:
    """Write an empty CSV with the canonical schema and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    execution_date = coerce_to_utc(execution_date)

    date_str = execution_date.strftime("%Y%m%d_%H%M%S")
    out = output_dir / f"weather_empty_{date_str}.csv"

    pd.DataFrame(columns=_SCHEMA_COLS).to_csv(out, index=False)
    logger.warning("%s — wrote empty weather CSV: %s", reason, out)
    return out


# ---------------------------------------------------------------------------
# Schema enforcement helper
# ---------------------------------------------------------------------------

def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing schema columns as None and normalise timestamp dtype."""
    for col in _SCHEMA_COLS:
        if col not in df.columns:
            df[col] = None
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df[_SCHEMA_COLS]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch_weather_data(
    grid_centroids: pd.DataFrame,
    execution_date: datetime,
    lookback_hours: int = 24,
    output_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    trigger_source: str = "cron",
    fire_cells: Optional[list] = None,
    h3_ring_max: int = 5,
    region: str = "unknown",
) -> Path:
    """Fetch weather data for all grid cell centroids.

    On emergency/active watchdog triggers, attempts HRRR (15-min cycle,
    3 km resolution) for the focal grid around confirmed fire cells, then
    fills *only the focal + detection-zone ring* with Open-Meteo if HRRR
    fails.  The full background region grid is intentionally NOT re-fetched
    on watchdog runs (fuse_features forward-fills from the last cron run).

    On cron triggers uses Open-Meteo for all region cells (unchanged).
    """
    logger.info(
        "fetch_weather_data called: trigger_source=%r  fire_cells=%r  "
        "lookback_hours=%d  len(grid_centroids)=%d",
        trigger_source,
        fire_cells,
        lookback_hours,
        len(grid_centroids) if grid_centroids is not None else 0,
    )

    registry = get_registry(config_path)
    om_config = registry.get_source_config("open_meteo")
    limiter = create_weather_limiter(config_path)

    is_watchdog = trigger_source in ("watchdog_emergency", "watchdog_active")

    if is_watchdog:
        if not fire_cells:
            logger.warning(
                "Watchdog trigger received but fire_cells is empty/None — "
                "falling back to cron-style full-grid Open-Meteo fetch."
            )
        else:
            logger.info(
                "Watchdog trigger: attempting HRRR for %d fire cells (h3_ring_max=%d)",
                len(fire_cells), h3_ring_max,
            )
            hrrr_path = _try_hrrr_focal(
                grid_centroids=grid_centroids,
                fire_cells=fire_cells,
                h3_ring_max=h3_ring_max,
                execution_date=execution_date,
                output_dir=output_dir,
                config_path=config_path,
            )
            if hrrr_path is not None:
                return _merge_hrrr_with_focal_background(
                    hrrr_path=hrrr_path,
                    fire_cells=fire_cells,
                    h3_ring_max=h3_ring_max,
                    grid_centroids=grid_centroids,
                    execution_date=execution_date,
                    lookback_hours=lookback_hours,
                    output_dir=output_dir,
                    om_config=om_config,
                    limiter=limiter,
                    config_path=config_path,
                )

            logger.warning(
                "HRRR fetch failed — fetching Open-Meteo for focal cells only "
                "(lookback_hours=%d).  Background cells will be forward-filled.",
                lookback_hours,
            )
            return _fetch_focal_open_meteo(
                fire_cells=fire_cells,
                h3_ring_max=h3_ring_max,
                grid_centroids=grid_centroids,
                execution_date=execution_date,
                lookback_hours=lookback_hours,
                output_dir=output_dir,
                om_config=om_config,
                limiter=limiter,
                config_path=config_path,
            )

    return _fetch_full_grid_open_meteo(
        grid_centroids=grid_centroids,
        execution_date=execution_date,
        lookback_hours=lookback_hours,
        output_dir=output_dir,
        om_config=om_config,
        limiter=limiter,
        config_path=config_path,
        region=region,
    )


# ---------------------------------------------------------------------------
# Cron path: full region grid
# ---------------------------------------------------------------------------

def _fetch_full_grid_open_meteo(
    grid_centroids: pd.DataFrame,
    execution_date: datetime,
    lookback_hours: int,
    output_dir: Optional[str],
    om_config: dict,
    limiter: RateLimiter,
    config_path: Optional[str],
    region: str = "unknown",
) -> Path:
    """Fetch Open-Meteo for every cell in the region (cron trigger)."""
    out_dir, execution_date = _resolve_output(output_dir, execution_date)

    if grid_centroids is None or grid_centroids.empty:
        return _write_empty_weather_csv(
            out_dir, execution_date, reason="No grid centroids provided"
        )

    grid_centroids = _normalise_centroids(grid_centroids, om_config)
    end_dt   = execution_date
    start_dt = coerce_to_utc(end_dt - timedelta(hours=lookback_hours))

    logger.info(
        "Cron fetch: %d cells  %s -> %s",
        len(grid_centroids), start_dt.isoformat(), end_dt.isoformat(),
    )

    all_rows, failed_cells = _batch_fetch_open_meteo(
        grid_centroids=grid_centroids,
        start_dt=start_dt,
        end_dt=end_dt,
        om_config=om_config,
        limiter=limiter,
        config_path=config_path,
        quality_flag=0,
    )

    if failed_cells:
        logger.warning(
            "%d cells failed both Open-Meteo and NWS — will be forward-filled downstream.",
            len(failed_cells),
        )

    return _write_combined(all_rows, out_dir, execution_date, label=f"weather_raw_{region}")


# ---------------------------------------------------------------------------
# Watchdog path: focal cells only via Open-Meteo (HRRR failed)
# ---------------------------------------------------------------------------

def _fetch_focal_open_meteo(
    fire_cells: list,
    h3_ring_max: int,
    grid_centroids: pd.DataFrame,
    execution_date: datetime,
    lookback_hours: int,
    output_dir: Optional[str],
    om_config: dict,
    limiter: RateLimiter,
    config_path: Optional[str],
) -> Path:
    """Fetch Open-Meteo for focal + detection-zone cells only."""
    out_dir, execution_date = _resolve_output(output_dir, execution_date)

    focal_centroids = _get_focal_centroids(fire_cells, h3_ring_max, grid_centroids, om_config)
    if focal_centroids.empty:
        return _write_empty_weather_csv(
            out_dir, execution_date, reason="Focal grid resolved to zero centroids"
        )

    end_dt   = execution_date
    start_dt = coerce_to_utc(end_dt - timedelta(hours=lookback_hours))

    logger.info(
        "Watchdog focal Open-Meteo fetch: %d focal cells  %s -> %s",
        len(focal_centroids), start_dt.isoformat(), end_dt.isoformat(),
    )

    all_rows, failed_cells = _batch_fetch_open_meteo(
        grid_centroids=focal_centroids,
        start_dt=start_dt,
        end_dt=end_dt,
        om_config=om_config,
        limiter=limiter,
        config_path=config_path,
        quality_flag=0,
    )

    if failed_cells:
        logger.warning("%d focal cells failed Open-Meteo + NWS fallback.", len(failed_cells))

    return _write_combined(all_rows, out_dir, execution_date, label="weather_raw_focal")


# ---------------------------------------------------------------------------
# Watchdog path: merge HRRR focal + Open-Meteo detection-zone background
# ---------------------------------------------------------------------------

def _merge_hrrr_with_focal_background(
    hrrr_path: Path,
    fire_cells: list,
    h3_ring_max: int,
    grid_centroids: pd.DataFrame,
    execution_date: datetime,
    lookback_hours: int,
    output_dir: Optional[str],
    om_config: dict,
    limiter: RateLimiter,
    config_path: Optional[str],
) -> Path:
    """Merge HRRR focal data with Open-Meteo for detection-zone-only background.

    data_quality_flag values:
        3 — HRRR (~15 min fresh, 3 km resolution)
        0 — Open-Meteo (~1 h fresh)
        4 — forward-filled from previous run (handled downstream)
    """
    out_dir, execution_date = _resolve_output(output_dir, execution_date)

    hrrr_df      = pd.read_csv(hrrr_path)
    hrrr_cell_ids = set(hrrr_df["grid_id"].astype(str))

    focal_centroids = _get_focal_centroids(fire_cells, h3_ring_max, grid_centroids, om_config)
    background_centroids = focal_centroids[
        ~focal_centroids["grid_id"].astype(str).isin(hrrr_cell_ids)
    ].copy()

    logger.info(
        "HRRR merge: %d HRRR focal cells, %d focal-background cells for "
        "Open-Meteo (full region background skipped — will be forward-filled)",
        len(hrrr_cell_ids), len(background_centroids),
    )

    parts = [hrrr_df]

    if not background_centroids.empty:
        end_dt   = execution_date
        start_dt = coerce_to_utc(end_dt - timedelta(hours=lookback_hours))
        bg_rows, _ = _batch_fetch_open_meteo(
            grid_centroids=background_centroids,
            start_dt=start_dt,
            end_dt=end_dt,
            om_config=om_config,
            limiter=limiter,
            config_path=config_path,
            quality_flag=0,
        )
        if bg_rows:
            parts.append(pd.concat(bg_rows, ignore_index=True))

    merged = pd.concat(parts, ignore_index=True)
    merged = _ensure_schema(merged)

    date_str    = execution_date.strftime("%Y%m%d_%H%M%S")
    output_path = out_dir / f"weather_raw_{date_str}.csv"
    merged.to_csv(output_path, index=False)

    logger.info(
        "HRRR+OM merge complete: %d HRRR rows + %d Open-Meteo rows -> %s",
        len(hrrr_df), len(merged) - len(hrrr_df), output_path,
    )
    return output_path


# ---------------------------------------------------------------------------
# NWS network availability probe
# ---------------------------------------------------------------------------

def _nws_is_reachable() -> bool:
    """DNS check for api.weather.gov.

    Uses gethostbyname (fast, ~50ms) to detect broken/missing external DNS
    before entering the per-cell NWS fallback loop.  A Docker container with
    no external network access will fail here instantly instead of hanging
    ~10s per cell.
    """
    try:
        socket.gethostbyname("api.weather.gov")
        return True
    except (socket.gaierror, socket.error):
        return False


# ---------------------------------------------------------------------------
# Core batched Open-Meteo fetch with NWS fallback
# ---------------------------------------------------------------------------

def _batch_fetch_open_meteo(
    grid_centroids: pd.DataFrame,
    start_dt: datetime,
    end_dt: datetime,
    om_config: dict,
    limiter: RateLimiter,
    config_path: Optional[str],
    quality_flag: int,
) -> tuple[list[pd.DataFrame], list[str]]:
    """Fetch Open-Meteo in batches with per-cell NWS fallback on failure.

    NWS fallback is skipped entirely if api.weather.gov is unreachable.
    Returns (list of DataFrames, list of failed grid_ids).
    """
    batches    = _create_coordinate_batches(grid_centroids, OPEN_METEO_MAX_LOCATIONS)
    all_rows:    list[pd.DataFrame] = []
    failed_cells: list[str]         = []

    nws_available         = _nws_is_reachable()
    nws_attempted_total   = 0

    if not nws_available:
        logger.warning(
            "NWS api.weather.gov is NOT reachable — NWS fallback disabled for "
            "this run. Failed cells will be forward-filled downstream."
        )

    for idx, batch in enumerate(batches):
        logger.info("  Processing batch %d/%d (%d cells)", idx + 1, len(batches), len(batch))

        # Reset backoff between batches — stale failure count from a previous
        # batch causes the *first* retry of the next batch to sleep 120s+.
        limiter._consecutive_failures = 0

        weather_df = _fetch_open_meteo_batch(
            batch=batch,
            start_date=start_dt,
            end_date=end_dt,
            base_url=om_config["base_url"],
            historical_url=om_config["historical_url"],
            limiter=limiter,
            timeout=om_config.get("timeout_seconds", 20),
            max_retries=om_config.get("max_retries", 3),
        )

        if weather_df is not None and not weather_df.empty:
            weather_df = weather_df.copy()
            weather_df["grid_id"]          = weather_df["grid_id"].astype(str)
            weather_df["data_quality_flag"] = quality_flag
            all_rows.append(weather_df)
            time.sleep(_BATCH_PAUSE_SECONDS)
            continue

        logger.warning("  Open-Meteo batch %d failed — checking NWS fallback.", idx + 1)

        # Cooldown after a fully-failed batch before trying the next one.
        # Open-Meteo free tier resets burst window after ~60s.
        time.sleep(10)

        if not nws_available:
            failed_cells.extend(batch["grid_id"].astype(str).tolist())
            continue

        # Per-cell NWS fallback — capped at _NWS_MAX_FALLBACK_CELLS total
        offshore_count = 0
        nws_attempted  = 0

        for _, cell in batch.iterrows():
            lat = float(cell["latitude"])
            lon = float(cell["longitude"])

            # Broad CONUS land check — NWS covers land only
            if not (24.0 <= lat <= 50.0 and -125.0 <= lon <= -66.0):
                failed_cells.append(str(cell["grid_id"]))
                offshore_count += 1
                continue

            if nws_attempted_total >= _NWS_MAX_FALLBACK_CELLS:
                failed_cells.append(str(cell["grid_id"]))
                continue

            nws_df = _fetch_nws_fallback(lat=lat, lon=lon, grid_id=str(cell["grid_id"]))
            if nws_df is not None and not nws_df.empty:
                nws_df = nws_df.copy()
                nws_df["grid_id"]          = nws_df["grid_id"].astype(str)
                nws_df["data_quality_flag"] = 2   # NWS fallback flag
                all_rows.append(nws_df)
                nws_attempted       += 1
                nws_attempted_total += 1
            else:
                failed_cells.append(str(cell["grid_id"]))

        skipped = len(batch) - offshore_count - nws_attempted
        if offshore_count or skipped:
            logger.info(
                "  Batch %d: %d offshore/water skipped, %d NWS attempted, "
                "%d deferred to forward-fill.",
                idx + 1, offshore_count, nws_attempted, skipped,
            )

    return all_rows, failed_cells


# ---------------------------------------------------------------------------
# Open-Meteo single-batch HTTP fetch
# ---------------------------------------------------------------------------

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
    """Fetch Open-Meteo for multiple locations in one request.

    Passes lat/lon as lists so requests encodes them as repeated keys:
        ?latitude=32.1&latitude=33.2&...
    instead of URL-encoded CSV which caused 400/429 on multi-location requests.

    Rate limiter: wait_if_needed() + record_request() on 200 only.
    429 calls record_failure() so backoff escalates: 10s -> 20s -> 40s.
    """
    lats: list[str] = batch["latitude"].astype(str).tolist()
    lons: list[str] = batch["longitude"].astype(str).tolist()

    start_date = coerce_to_utc(start_date)
    end_date   = coerce_to_utc(end_date)

    cutoff = datetime.now(timezone.utc) - timedelta(days=5)
    url    = historical_url if end_date < cutoff else base_url

    params: dict = {
        "latitude":   lats,
        "longitude":  lons,
        "hourly":     ",".join(OPEN_METEO_HOURLY_PARAMS),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date":   end_date.strftime("%Y-%m-%d"),
        "timezone":   "UTC",
    }
    if OPEN_METEO_DAILY_PARAMS:
        params["daily"] = ",".join(OPEN_METEO_DAILY_PARAMS)

    for attempt in range(max_retries):
        try:
            limiter.wait_if_needed()
            resp = requests.get(url, params=params, timeout=timeout)

            if resp.status_code == 200:
                limiter.record_request()   # reset failure counter on success only
                return _parse_open_meteo_response(resp.json(), batch)

            if resp.status_code == 429:
                limiter.record_failure()   # accumulates: 1->2->3 -> backoff 10s->20s->40s
                delay = limiter.get_backoff_delay()
                logger.warning(
                    "Open-Meteo rate limited. Backing off %.1fs (attempt %d/%d)",
                    delay, attempt + 1, max_retries,
                )
                time.sleep(delay)
                continue

            # 4xx (except 429) — non-retryable
            if 400 <= resp.status_code < 500:
                logger.error(
                    "Open-Meteo non-retryable error: HTTP %d | %s->%s | %d locations | %s",
                    resp.status_code,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    len(lats),
                    resp.text[:400],
                )
                return None

            # 5xx — transient, retry with backoff
            logger.warning(
                "Open-Meteo HTTP %d (attempt %d/%d): %s",
                resp.status_code, attempt + 1, max_retries, resp.text[:200],
            )
            limiter.record_failure()
            time.sleep(limiter.get_backoff_delay())

        except requests.exceptions.RequestException as exc:
            logger.warning(
                "Open-Meteo request error (attempt %d/%d): %s",
                attempt + 1, max_retries, exc,
            )
            limiter.record_failure()
            time.sleep(limiter.get_backoff_delay())

    return None


# ---------------------------------------------------------------------------
# Open-Meteo response parser
# ---------------------------------------------------------------------------

def _parse_open_meteo_response(data: dict, batch: pd.DataFrame) -> pd.DataFrame:
    """Parse Open-Meteo JSON into a flat hourly DataFrame."""
    records:  list[dict] = []
    results   = data if isinstance(data, list) else [data]
    grid_ids  = batch["grid_id"].astype(str).tolist()

    for idx, result in enumerate(results):
        if idx >= len(grid_ids):
            break

        grid_id   = grid_ids[idx]
        hourly    = result.get("hourly", {})
        if not isinstance(hourly, dict) or not hourly:
            continue

        timestamps = hourly.get("time", [])
        if not timestamps:
            continue

        for t_idx, ts in enumerate(timestamps):
            rec: dict = {"grid_id": grid_id, "timestamp": ts}
            for param in OPEN_METEO_HOURLY_PARAMS:
                values   = hourly.get(param, [])
                rec[param] = values[t_idx] if t_idx < len(values) else None
            rec["fire_weather_index"] = None   # Phase 2 MVP placeholder
            records.append(rec)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if "vapor_pressure_deficit" in df.columns:
        df = df.rename(columns={"vapor_pressure_deficit": "vpd"})

    for col in [
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
        "wind_direction_10m", "precipitation", "soil_moisture_0_to_7cm",
        "vpd", "fire_weather_index",
    ]:
        if col not in df.columns:
            df[col] = None

    return df


# ---------------------------------------------------------------------------
# NWS fallback
# ---------------------------------------------------------------------------

def _fetch_nws_fallback(
    lat: float,
    lon: float,
    grid_id: str,
) -> Optional[pd.DataFrame]:
    """Single-cell fallback to NWS forecastHourly endpoint.

    Uses NWS_HEADERS (compliant User-Agent) and 4-decimal coordinate rounding
    as required by api.weather.gov.  Two-step: points lookup -> hourly forecast.
    """
    base_url          = "https://api.weather.gov"
    lat_r, lon_r      = round(lat, 4), round(lon, 4)

    try:
        # Step 1: points lookup — get the grid office + forecast URL
        pts_url = f"{base_url}/points/{lat_r},{lon_r}"
        resp    = requests.get(pts_url, headers=NWS_HEADERS, timeout=10)
        if resp.status_code != 200:
            logger.debug("NWS points lookup HTTP %d for (%.4f, %.4f)", resp.status_code, lat_r, lon_r)
            return None

        forecast_url = resp.json().get("properties", {}).get("forecastHourly")
        if not forecast_url:
            logger.debug("NWS: no forecastHourly URL for (%.4f, %.4f)", lat_r, lon_r)
            return None

        # Step 2: fetch hourly forecast
        f_resp = requests.get(forecast_url, headers=NWS_HEADERS, timeout=10)
        if f_resp.status_code != 200:
            logger.debug("NWS forecast HTTP %d for (%.4f, %.4f)", f_resp.status_code, lat_r, lon_r)
            return None

        periods = f_resp.json().get("properties", {}).get("periods", [])
        if not periods:
            return None

        recs = []
        for p in periods[:24]:
            temp = p.get("temperature")
            unit = p.get("temperatureUnit")
            recs.append({
                "grid_id":               grid_id,
                "timestamp":             p.get("startTime"),
                "temperature_2m":        _fahrenheit_to_celsius(temp) if unit == "F" else temp,
                "relative_humidity_2m":  (p.get("relativeHumidity") or {}).get("value"),
                "wind_speed_10m":        _parse_nws_wind_speed(p.get("windSpeed")),
                "wind_direction_10m":    _parse_nws_wind_direction(p.get("windDirection")),
                "precipitation":         None,
                "soil_moisture_0_to_7cm": None,
                "vpd":                   None,
                "fire_weather_index":    None,
            })

        df = pd.DataFrame(recs)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df

    except requests.exceptions.Timeout:
        logger.debug("NWS fallback timed out for (%.4f, %.4f)", lat_r, lon_r)
        return None
    except requests.exceptions.RequestException as exc:
        logger.debug("NWS fallback failed for (%.4f, %.4f): %s", lat_r, lon_r, exc)
        return None


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

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
        s = str(speed_str).strip().lower()
        if s in ("calm", "0", ""):
            return 0.0
        s     = s.replace(" mph", "").replace("mph", "")
        parts = s.split(" to ")
        avg_mph = (
            (float(parts[0]) + float(parts[1])) / 2
            if len(parts) == 2
            else float(parts[0])
        )
        return round(avg_mph * 1.60934, 1)
    except Exception:
        return None


def _parse_nws_wind_direction(direction: Optional[str]) -> Optional[float]:
    if direction is None:
        return None
    direction_map = {
        "N": 0,   "NNE": 22.5, "NE": 45,  "ENE": 67.5,
        "E": 90,  "ESE": 112.5,"SE": 135, "SSE": 157.5,
        "S": 180, "SSW": 202.5,"SW": 225, "WSW": 247.5,
        "W": 270, "WNW": 292.5,"NW": 315, "NNW": 337.5,
        "CALM": 0,
    }
    return direction_map.get(str(direction).strip().upper())


# ---------------------------------------------------------------------------
# HRRR integration helpers (watchdog triggers only)
# ---------------------------------------------------------------------------

def _try_hrrr_focal(
    grid_centroids: pd.DataFrame,
    fire_cells: list,
    h3_ring_max: int,
    execution_date,
    output_dir: Optional[str],
    config_path: Optional[str],
) -> Optional[Path]:
    """Attempt HRRR fetch for the focal grid. Returns None on any failure."""
    try:
        from scripts.ingestion.ingest_hrrr import fetch_hrrr_for_focal_grid
        from scripts.utils.grid_utils import generate_fire_focal_grid
    except ImportError as exc:
        logger.warning("HRRR deps not installed — skipping HRRR: %s", exc)
        return None

    try:
        focal_grid = generate_fire_focal_grid(
            fire_cell_ids=fire_cells, ring_min=1, ring_max=h3_ring_max,
        )
        if focal_grid.empty:
            logger.warning("HRRR: focal grid resolved to zero cells — skipping")
            return None

        logger.info(
            "HRRR: focal grid has %d cells (%d fire, %d detection zone)",
            len(focal_grid),
            (focal_grid["cell_type"] == "fire").sum(),
            (focal_grid["cell_type"] == "detection_zone").sum(),
        )
        return fetch_hrrr_for_focal_grid(
            focal_grid=focal_grid[["grid_id", "latitude", "longitude"]],
            execution_date=execution_date,
            output_dir=output_dir,
            config_path=config_path,
        )
    except Exception as exc:
        logger.warning("HRRR focal fetch raised an exception: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _get_focal_centroids(
    fire_cells: list,
    h3_ring_max: int,
    grid_centroids: pd.DataFrame,
    om_config: dict,
) -> pd.DataFrame:
    """Return centroids for the focal grid (fire cells + detection-zone rings).

    Derives lat/lon directly from H3 cell IDs using h3.cell_to_latlng() /
    h3.h3_to_geo() — bypasses grid_centroids filtering which always produced
    empty results due to cell ID mismatch between bbox scan and ring expansion.
    """
    try:
        import h3
        from scripts.utils.grid_utils import generate_fire_focal_grid

        focal_grid = generate_fire_focal_grid(
            fire_cell_ids=fire_cells, ring_min=1, ring_max=h3_ring_max,
        )
        if focal_grid.empty:
            logger.warning(
                "_get_focal_centroids: empty focal grid for fire_cells=%s ring_max=%d",
                fire_cells, h3_ring_max,
            )
            return pd.DataFrame()

        if hasattr(h3, "cell_to_latlng"):
            _h3_to_geo = h3.cell_to_latlng   # H3 v4
        elif hasattr(h3, "h3_to_geo"):
            _h3_to_geo = h3.h3_to_geo        # H3 v3
        else:
            raise RuntimeError("h3 library has neither cell_to_latlng (v4) nor h3_to_geo (v3).")

        rows = []
        for cell_id in focal_grid["grid_id"].astype(str):
            try:
                lat, lon = _h3_to_geo(cell_id)
                rows.append({"grid_id": cell_id, "latitude": lat, "longitude": lon})
            except Exception as cell_exc:
                logger.warning("Could not resolve centroid for H3 cell %s: %s", cell_id, cell_exc)

        if not rows:
            return pd.DataFrame()

        result = pd.DataFrame(rows)
        logger.info(
            "_get_focal_centroids: resolved %d centroids from %d focal cells "
            "(fire_cells=%d, ring_max=%d)",
            len(result), len(focal_grid), len(fire_cells), h3_ring_max,
        )
        return _normalise_centroids(result, om_config)

    except Exception as exc:
        logger.warning("_get_focal_centroids failed (%s) — returning empty DataFrame", exc)
        return pd.DataFrame()


def _normalise_centroids(
    grid_centroids: pd.DataFrame, om_config: dict
) -> pd.DataFrame:
    """Validate required columns and round coordinates."""
    required = {"grid_id", "latitude", "longitude"}
    missing  = sorted(required - set(grid_centroids.columns))
    if missing:
        raise ValueError(f"grid_centroids missing columns: {missing}")

    precision = om_config.get("coordinate_precision", 3)
    df = grid_centroids.copy()
    df["grid_id"]   = df["grid_id"].astype(str)
    df["latitude"]  = df["latitude"].round(precision)
    df["longitude"] = df["longitude"].round(precision)
    return df


def _resolve_output(
    output_dir: Optional[str], execution_date
) -> tuple[Path, datetime]:
    """Resolve output directory path and normalise execution_date."""
    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "weather"
        )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, coerce_to_utc(execution_date)


def _create_coordinate_batches(
    grid_centroids: pd.DataFrame, batch_size: int
) -> list[pd.DataFrame]:
    return [
        grid_centroids.iloc[i : i + batch_size]
        for i in range(0, len(grid_centroids), batch_size)
    ]


def _write_combined(
    all_rows: list[pd.DataFrame],
    out_dir: Path,
    execution_date: datetime,
    label: str,
) -> Path:
    """Concatenate rows, enforce schema, write CSV, return path."""
    date_str    = execution_date.strftime("%Y%m%d_%H%M%S")
    output_path = out_dir / f"{label}_{date_str}.csv"

    if not all_rows:
        return _write_empty_weather_csv(
            out_dir, execution_date, reason="All weather API requests failed"
        )

    non_empty = [df for df in all_rows if not df.empty and not df.isna().all().all()]
    if not non_empty:
        return _write_empty_weather_csv(
            out_dir, execution_date, reason="All weather rows were empty after filtering"
        )

    combined = pd.concat(non_empty, ignore_index=True)
    combined = _ensure_schema(combined)
    combined.to_csv(output_path, index=False)

    logger.info(
        "Weather ingestion complete: %d rows for %d cells -> %s",
        len(combined),
        combined["grid_id"].nunique() if "grid_id" in combined.columns else 0,
        output_path,
    )
    return output_path
