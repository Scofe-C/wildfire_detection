"""
Inference CLI entry point — 6-hour fire risk scoring for all grid cells.

Runs every 6 hours (aligned with FIRMS satellite windows):
  1. Fetch Open-Meteo weather for all CA + TX grid centroids (rolling 24h window)
  2. Apply feature_engineering.full_pipeline() — same transforms as training
  3. Load Production model + threshold from MLflow Model Registry
  4. Score all ~55 grid cells (35 CA + 20 TX)
  5. Assign risk tiers (LOW/MEDIUM/HIGH/CRITICAL)
  6. Write partitioned parquet to GCS (queryable history)
  7. Overwrite GCS latest JSON (polled by fire watchdog + OBJ-3)
  8. Send Slack alert if any CRITICAL cells detected

Usage
-----
  python -m scripts.inference
  python -m scripts.inference --regions california texas --dry-run
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Risk tier thresholds
RISK_TIERS = [
    ("CRITICAL", 0.65),
    ("HIGH",     0.365),
    ("MEDIUM",   0.15),
    ("LOW",      0.0),
]


def assign_risk_tier(score: float) -> str:
    for tier, lower_bound in RISK_TIERS:
        if score >= lower_bound:
            return tier
    return "LOW"


def fetch_open_meteo_inference(grid_centroids: pd.DataFrame) -> pd.DataFrame:
    """Fetch 24h rolling weather for all grid centroids from Open-Meteo.

    Parameters
    ----------
    grid_centroids : DataFrame with 'grid_id', 'latitude', 'longitude', 'region'.

    Returns
    -------
    DataFrame with weather features aggregated over the last 24 hours.
    The output format must match the training feature schema exactly.
    """
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.5)
    om = openmeteo_requests.Client(session=retry_session)

    hourly_vars = [
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
        "wind_direction_10m", "precipitation", "soil_moisture_0_to_7cm",
        "vapour_pressure_deficit",
    ]

    all_rows = []
    for _, cell in grid_centroids.iterrows():
        try:
            response = om.weather_api(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": cell["latitude"],
                    "longitude": cell["longitude"],
                    "hourly": hourly_vars,
                    "past_hours": 24,
                    "forecast_hours": 1,
                    "timezone": "UTC",
                },
            )[0]

            hourly = response.Hourly()
            n = hourly.VariablesLength()
            var_map = {hourly_vars[i]: hourly.Variables(i).ValuesAsNumpy() for i in range(n)}

            # Aggregate over last 24 hours (same window as training)
            temp     = float(np.nanmean(var_map["temperature_2m"]))
            humidity = float(np.nanmean(var_map["relative_humidity_2m"]))
            wind_spd = float(np.nanmean(var_map["wind_speed_10m"]))
            wind_dir = float(np.nanmean(var_map["wind_direction_10m"]))  # median for circular
            precip   = float(np.nansum(var_map["precipitation"]))
            soil_mst = float(np.nanmean(var_map["soil_moisture_0_to_7cm"]))
            vpd      = float(np.nanmean(var_map["vapour_pressure_deficit"]))

            # Derived features
            fire_weather_index = _compute_fwi(temp, humidity, wind_spd, precip)
            cum_wind_run = float(np.nansum(var_map["wind_speed_10m"]))  # sum over 24h
            drought_proxy = _compute_drought_proxy(temp, humidity, precip, soil_mst)

            row = {
                "grid_id": cell["grid_id"],
                "region": cell["region"],
                "latitude": cell["latitude"],
                "longitude": cell["longitude"],
                "temperature_2m": temp,
                "relative_humidity_2m": humidity,
                "wind_speed_10m": wind_spd,
                "wind_direction_10m": wind_dir,
                "precipitation": precip,
                "soil_moisture_0_to_7cm": soil_mst,
                "vpd": vpd,
                "fire_weather_index": fire_weather_index,
                "cumulative_wind_run_24h": cum_wind_run,
                "drought_index_proxy": drought_proxy,
            }
            all_rows.append(row)

        except Exception as e:
            logger.warning("Failed to fetch weather for cell %s: %s", cell["grid_id"], e)

    return pd.DataFrame(all_rows)


def _compute_fwi(temp: float, humidity: float, wind_spd: float, precip: float) -> float:
    """Simplified Canadian Fire Weather Index proxy."""
    ffmc = max(0.0, 101.0 - humidity * 0.8 + temp * 0.2 - precip * 5.0)
    dmc  = max(0.0, temp * 0.5 - humidity * 0.1 + 1.0)
    isi  = wind_spd * ffmc / 100.0
    bui  = dmc * 1.2
    return float(isi * bui / 100.0)


def _compute_drought_proxy(
    temp: float, humidity: float, precip: float, soil_mst: float
) -> float:
    """Drought index proxy: higher = drier."""
    return float(max(0.0, (temp / 40.0) * (1.0 - humidity / 100.0) * (1.0 - soil_mst) * 100.0))


_STATIC_PARQUET = (
    Path(__file__).resolve().parents[2]
    / "Data-Pipeline" / "data" / "static" / "static_features_64km.parquet"
)

STATIC_FEATURE_COLS = [
    "grid_id", "aspect_degrees", "elevation_m", "slope_degrees",
    "dominant_fuel_fraction", "ndvi", "fuel_model_fbfm40", "vegetation_type",
]


def load_static_features() -> pd.DataFrame:
    """Load per-grid static terrain/vegetation features from the Data-Pipeline static parquet.

    These features (elevation, slope, aspect, fuel model, NDVI, etc.) are
    time-invariant and not provided by Open-Meteo.  Joined onto weather rows
    by grid_id before preprocessing.
    """
    if not _STATIC_PARQUET.exists():
        raise RuntimeError(
            f"Static features file not found: {_STATIC_PARQUET}. "
            "Ensure Data-Pipeline/data/static/static_features_64km.parquet exists."
        )
    df = pd.read_parquet(_STATIC_PARQUET)
    available = [c for c in STATIC_FEATURE_COLS if c in df.columns]
    return df[available].drop_duplicates(subset=["grid_id"]).reset_index(drop=True)


def load_grid_centroids(regions: list[str]) -> pd.DataFrame:
    """Load grid cell centroids for specified regions.

    Falls back to loading from the Data-Pipeline static grid if available.
    """
    try:
        from scripts.utils.grid_utils import generate_full_grid  # type: ignore
        grid = generate_full_grid(resolution_km=64)
        if regions:
            grid = grid[grid["region"].isin(regions)]
        return grid[["grid_id", "latitude", "longitude", "region"]]
    except ImportError:
        pass

    # Fallback: look for local static cache
    static_path = Path(__file__).resolve().parents[1] / "data" / "static" / "grid_centroids.parquet"
    if static_path.exists():
        df = pd.read_parquet(static_path)
        if regions:
            df = df[df["region"].isin(regions)]
        return df[["grid_id", "latitude", "longitude", "region"]]

    # Final fallback: extract unique centroids from historical CSV
    for fname in [f"{r}_historical.csv" for r in (regions or ["california", "texas"])]:
        csv_path = Path(__file__).resolve().parents[1] / "historical_data" / fname
        if csv_path.exists():
            df = pd.read_csv(csv_path, usecols=["grid_id", "latitude", "longitude", "region"])
            df = df.drop_duplicates(subset=["grid_id"])
            if regions:
                df = df[df["region"].isin(regions)]
            if not df.empty:
                logger.info("Grid centroids extracted from historical CSV (%d cells)", len(df))
                return df[["grid_id", "latitude", "longitude", "region"]].reset_index(drop=True)

    raise RuntimeError(
        "Cannot load grid centroids. "
        "Ensure Data-Pipeline is accessible, data/static/grid_centroids.parquet exists, "
        "or historical_data/{region}_historical.csv is present."
    )


def write_outputs(
    scored_df: pd.DataFrame,
    run_timestamp: datetime,
    bucket: str,
    model_version: str,
    threshold: float,
    dry_run: bool = False,
) -> None:
    """Write parquet (partitioned history) and JSON latest to GCS."""
    if dry_run:
        logger.info("[DRY RUN] Would write %d rows to GCS", len(scored_df))
        return

    from google.cloud import storage
    client = storage.Client()
    bkt = client.bucket(bucket)

    ts_str = run_timestamp.strftime("%Y%m%dT%H%MZ")
    year   = run_timestamp.year
    month  = f"{run_timestamp.month:02d}"

    for region in scored_df["region"].unique():
        region_df = scored_df[scored_df["region"] == region].copy()

        # ── Parquet (partitioned history) ──────────────────────────────────
        parquet_blob = (
            f"inference/region={region}/year={year}/month={month}/"
            f"inference_{ts_str}.parquet"
        )
        buf = io.BytesIO()
        region_df.to_parquet(buf, index=False)
        bkt.blob(parquet_blob).upload_from_string(buf.getvalue(), content_type="application/octet-stream")
        logger.info("Written gs://%s/%s (%d rows)", bucket, parquet_blob, len(region_df))

        # ── JSON latest (overwrite) ────────────────────────────────────────
        cells_list = region_df[[
            "grid_id", "latitude", "longitude",
            "fire_risk_score", "fire_risk_flag", "risk_tier",
        ]].to_dict(orient="records")

        summary = {
            "total_cells": len(region_df),
            "flagged_cells": int(region_df["fire_risk_flag"].sum()),
            "max_risk_score": float(region_df["fire_risk_score"].max()),
            "risk_tier_counts": region_df["risk_tier"].value_counts().to_dict(),
        }

        latest_payload = {
            "run_timestamp": run_timestamp.isoformat(),
            "model_version": model_version,
            "threshold": threshold,
            "region": region,
            "cells": cells_list,
            "summary": summary,
        }

        latest_blob = f"inference/latest/{region}_latest.json"
        bkt.blob(latest_blob).upload_from_string(
            json.dumps(latest_payload, indent=2),
            content_type="application/json",
        )
        logger.info("Updated gs://%s/%s", bucket, latest_blob)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 6-hour wildfire risk inference.")
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["california", "texas"],
        help="Regions to score (default: california texas)",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="GCS bucket (default: GCS_BUCKET_NAME env var or 'wildfire-mlops-123')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and score but do not write to GCS",
    )
    parser.add_argument(
        "--local-model-dir",
        default=None,
        help=(
            "Local dev mode: path to reports/local_run/ where train --local saved models. "
            "Skips Vertex AI model loading."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    bucket = args.bucket or os.environ.get("GCS_BUCKET_NAME") or "wildfire-mlops-123"
    run_timestamp = datetime.now(UTC)

    logger.info("Inference run starting — %s — regions: %s", run_timestamp.isoformat(), args.regions)

    # ── Load config + imports ──────────────────────────────────────────────────
    from pathlib import Path as _Path

    import lightgbm as _lgb
    import xgboost as _xgb
    import yaml as _yaml

    from src.preprocessing.feature_engineering import full_pipeline

    _cfg_path = _Path(__file__).resolve().parents[1] / "configs" / "model_config.yaml"
    with open(_cfg_path) as _f:
        _cfg = _yaml.safe_load(_f)

    # Vertex AI settings — only needed when not in local mode
    _project_id = ""
    _location   = "us-central1"
    VertexRegistry = None
    if not args.local_model_dir:
        from src.tracking.vertex_registry import VertexRegistry  # type: ignore[assignment]
        _vai = _cfg["tracking"]["vertex_ai"]
        _project_id = os.environ.get("GCP_PROJECT_ID", _vai.get("project_id", ""))
        _location   = _vai.get("location", "us-central1")

    all_scored: list = []
    all_critical: list = []

    # ── Per-region: load model → fetch weather → score ────────────────────────
    for region in args.regions:
        logger.info("--- Region: %s ---", region.upper())

        # Load region-specific Production model
        if args.local_model_dir:
            # Local dev: load from reports/local_run/latest_{region}.txt pointer
            import json as _json
            from pathlib import Path as _Path

            import xgboost as _xgb_load
            _pointer = _Path(args.local_model_dir) / f"latest_{region}.txt"
            if not _pointer.exists():
                logger.error("[%s] No local model pointer at %s — run train --local first", region, _pointer)
                continue
            _model_dir = _Path(_pointer.read_text().strip())
            _meta = _json.loads((_model_dir / "model_metadata.json").read_text())
            threshold = float(_meta["threshold"])
            medians = _meta["medians"]
            framework = _meta.get("framework", "xgboost")
            if framework == "xgboost":
                model = _xgb_load.XGBClassifier()
                model.load_model(str(_model_dir / "model.bst"))
            else:
                import lightgbm as _lgb_load
                model = _lgb_load.LGBMClassifier()
                model._Booster = _lgb_load.Booster(model_file=str(_model_dir / "model.txt"))
            logger.info("[%s] Loaded local model from %s, threshold=%.4f", region, _model_dir, threshold)
        else:
            registry = VertexRegistry(
                project_id=_project_id,
                location=_location,
                display_name=f"wildfire-ignition-{region}",
                gcs_bucket=bucket,
            )
            try:
                model, medians, threshold = registry.load_production()
                logger.info("[%s] Loaded Production model, threshold=%.4f", region, threshold)
            except Exception as e:
                logger.error("[%s] Failed to load model from Vertex AI: %s — skipping region", region, e)
                continue

        # Load grid centroids for this region
        grid = load_grid_centroids([region])
        if grid.empty:
            logger.warning("[%s] No grid cells found — skipping", region)
            continue
        logger.info("[%s] Grid: %d cells", region, len(grid))

        # Fetch rolling 24h weather
        weather_df = fetch_open_meteo_inference(grid)
        if weather_df.empty:
            logger.error("[%s] No weather data returned — skipping", region)
            continue

        weather_df = weather_df.merge(
            grid[["grid_id", "latitude", "longitude", "region"]],
            on="grid_id", how="left", suffixes=("", "_grid"),
        )

        # Join static terrain/vegetation features (elevation, slope, aspect, fuel model, NDVI)
        try:
            static_df = load_static_features()
            weather_df = weather_df.merge(static_df, on="grid_id", how="left")
            logger.info("[%s] Joined static features (%d cols)", region, len(static_df.columns) - 1)
        except Exception as e:
            logger.error("[%s] Failed to load static features: %s — skipping", region, e)
            continue

        # Preprocess — pass training medians for consistent imputation
        try:
            X, _ = full_pipeline(weather_df, model_type="xgb", is_inference=True,
                                  fit_medians=medians)
        except Exception as e:
            logger.error("[%s] Preprocessing failed: %s — skipping", region, e)
            continue

        # Score
        logger.info("[%s] Scoring %d cells ...", region, len(X))
        if isinstance(model, _xgb.Booster):
            import xgboost as xgb
            y_prob = model.predict(xgb.DMatrix(X))
        elif isinstance(model, _lgb.Booster):
            y_prob = model.predict(X)
        elif hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
        else:
            y_prob = model.predict(X)

        scored_df = weather_df[["grid_id", "latitude", "longitude", "region"]].copy()
        scored_df["timestamp"]      = run_timestamp
        scored_df["fire_risk_score"] = y_prob
        scored_df["fire_risk_flag"]  = (y_prob >= threshold).astype(int)
        scored_df["risk_tier"]       = [assign_risk_tier(s) for s in y_prob]
        scored_df["model_version"]   = "production"
        scored_df["threshold_used"]  = threshold

        n_flagged  = int(scored_df["fire_risk_flag"].sum())
        n_crit     = int((scored_df["risk_tier"] == "CRITICAL").sum())
        logger.info(
            "[%s] flagged: %d, CRITICAL: %d, max_score: %.4f",
            region, n_flagged, n_crit, float(scored_df["fire_risk_score"].max()),
        )

        write_outputs(scored_df, run_timestamp, bucket, "production", threshold,
                      dry_run=args.dry_run)
        all_scored.append(scored_df)
        if n_crit > 0:
            critical_cells = scored_df[scored_df["risk_tier"] == "CRITICAL"][
                ["grid_id", "region", "fire_risk_score"]
            ].to_dict(orient="records")
            all_critical.extend(critical_cells)

    if not all_scored:
        logger.error("No regions scored successfully — aborting")
        return

    n_critical = len(all_critical)

    # ── Slack alert for CRITICAL cells ────────────────────────────────────────
    if n_critical > 0:
        try:
            from src.notifications.alerter import SlackAlerter
            alerter = SlackAlerter()
            top = all_critical[0]
            alerter.alert_critical_fire_risk(
                region=top.get("region", "unknown"),
                grid_id=str(top.get("grid_id", "unknown")),
                probability=float(top.get("fire_risk_score", 0.0)),
            )
        except Exception as e:
            logger.warning("Slack alert failed (non-blocking): %s", e)

    logger.info("Inference run complete.")


if __name__ == "__main__":
    main()
