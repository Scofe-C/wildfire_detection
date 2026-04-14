"""Operator override re-run engine — OBJ-1 + OBJ-2 with corrected local observations.

Allows fire commanders to replace API-sourced weather/vegetation/FIRMS values
with their own on-the-ground measurements, supply advisory input, and re-score
all grid cells using the production model.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import re

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed override models (validated at API boundary)
# ---------------------------------------------------------------------------

# H3 index pattern: 15-16 hex characters (resolutions 0-15)
_H3_PATTERN = re.compile(r"^[0-9a-fA-F]{15,16}$")

# Valid FBFM40 fuel model codes (Anderson 13 + Scott & Burgan 40)
_VALID_FBFM40 = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,  # Anderson 13
    91, 92, 93, 98, 99,  # non-burnable
    101, 102, 103, 104, 105, 106, 107, 108, 109,  # GR
    121, 122, 123, 124,  # GS
    141, 142, 143, 144, 145, 146, 147, 148, 149,  # SH
    161, 162, 163, 164, 165,  # TU
    181, 182, 183, 184, 185, 186, 187, 188, 189,  # TL
    201, 202, 203, 204,  # SB
}

# Allowed file MIME types for uploads
ALLOWED_MIME_TYPES = {
    "image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp", "image/tiff",
    "application/pdf",
    "text/plain", "text/csv", "text/xml",
    "application/json", "application/geo+json",
    "application/vnd.google-earth.kml+xml",
    "application/octet-stream",  # fallback for unknown types
}


class WeatherOverrides(BaseModel):
    temperature_f: float | None = Field(default=None, ge=-80.0, le=160.0)
    wind_speed_mph: float | None = Field(default=None, ge=0.0, le=250.0)
    relative_humidity: float | None = Field(default=None, ge=0.0, le=100.0)
    soil_moisture: float | None = Field(default=None, ge=0.0, le=1.0)
    fire_weather_index: float | None = Field(default=None, ge=0.0, le=150.0)
    wind_direction_deg: float | None = Field(default=None, ge=0.0, le=360.0)
    precipitation_mm: float | None = Field(default=None, ge=0.0, le=500.0)
    vpd_kpa: float | None = Field(default=None, ge=0.0, le=20.0)


class VegetationOverrides(BaseModel):
    ndvi: float | None = Field(default=None, ge=-1.0, le=1.0)
    dominant_fuel_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    fuel_model_fbfm40: int | None = None
    vegetation_type: str | None = Field(default=None, max_length=100)

    @field_validator("fuel_model_fbfm40")
    @classmethod
    def validate_fbfm40(cls, v: int | None) -> int | None:
        if v is not None and v not in _VALID_FBFM40:
            raise ValueError(
                f"Invalid FBFM40 fuel model code: {v}. "
                f"Valid codes: Anderson 13 (1-13), Scott & Burgan (101-204)"
            )
        return v


class FIRMSOverride(BaseModel):
    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)
    frp: float = Field(ge=0.0, le=5000.0)
    confidence: Literal["low", "nominal", "high"] = "nominal"
    acq_datetime: str | None = None


class XGBoostCellOverride(BaseModel):
    h3_index: str
    probability: float = Field(ge=0.0, le=1.0)
    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)

    @field_validator("h3_index")
    @classmethod
    def validate_h3(cls, v: str) -> str:
        if not _H3_PATTERN.match(v):
            raise ValueError(
                f"Invalid H3 index '{v}'. Must be 15-16 hex characters."
            )
        return v


class OBJ2SimOverrides(BaseModel):
    spread_speed_kmh: float | None = Field(default=None, ge=0.0, le=50.0)
    spread_direction_deg: float | None = Field(default=None, ge=0.0, le=360.0)
    crown_fire_status: Literal["none", "passive_crown", "active_crown"] | None = None
    dead_fuel_moisture_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    foliar_moisture_content_pct: float | None = Field(default=None, ge=0.0, le=300.0)


class RerunOverrides(BaseModel):
    """All operator-overridable data categories for a rerun."""

    grid_id: str = Field(min_length=1, max_length=20)
    region: Literal["california", "texas"] = "california"
    weather: WeatherOverrides | None = None
    vegetation: VegetationOverrides | None = None
    firms_hotspots: list[FIRMSOverride] | None = Field(default=None, max_length=50)
    xgboost_cells: list[XGBoostCellOverride] | None = Field(default=None, max_length=20)
    obj2_simulation: OBJ2SimOverrides | None = None
    risk_level_override: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"] | None = None


# ---------------------------------------------------------------------------
# Operator field → pipeline column + unit conversion
# ---------------------------------------------------------------------------

# Operator field → pipeline column + optional unit conversion
_OVERRIDE_MAP: dict[str, dict[str, Any]] = {
    # Weather (original 5)
    "temperature_f": {
        "column": "temperature_2m",
        "convert": lambda v: (v - 32) * 5 / 9,
    },
    "wind_speed_mph": {
        "column": "wind_speed_10m",
        "convert": lambda v: v * 1.60934,
    },
    "relative_humidity": {
        "column": "relative_humidity_2m",
        "convert": lambda v: v,
    },
    "soil_moisture": {
        "column": "soil_moisture_0_to_7cm",
        "convert": lambda v: v,
    },
    "fire_weather_index": {
        "column": "fire_weather_index",
        "convert": lambda v: v,
    },
    # Weather (new)
    "wind_direction_deg": {
        "column": "wind_direction_10m",
        "convert": lambda v: v,
    },
    "precipitation_mm": {
        "column": "precipitation",
        "convert": lambda v: v,
    },
    "vpd_kpa": {
        "column": "vpd",
        "convert": lambda v: v,
    },
    # Vegetation
    "ndvi": {
        "column": "ndvi",
        "convert": lambda v: v,
    },
    "dominant_fuel_fraction": {
        "column": "dominant_fuel_fraction",
        "convert": lambda v: v,
    },
    "fuel_model_fbfm40": {
        "column": "fuel_model_fbfm40",
        "convert": lambda v: int(v),
    },
}


class RerunEngine:
    """Loads the production model once, then applies operator overrides and re-scores."""

    def __init__(self, model_path: str | Path, config: dict[str, Any]):
        """Load model + preprocessing state.

        Parameters
        ----------
        model_path:
            Path to a directory containing ``model.bst`` / ``model.txt`` and
            ``model_metadata.json`` (written by orchestrator local-save or GCS pull).
        config:
            Dict with keys: ``framework`` (xgboost|lightgbm), ``threshold``,
            ``medians`` (feature medians for imputation).
        """
        self._config = config
        self._threshold: float = float(config["threshold"])
        self._medians: dict[str, float] = config.get("medians", {})
        self._framework: str = config.get("framework", "xgboost")
        self._model = self._load_model(Path(model_path))

    def _load_model(self, model_dir: Path) -> Any:
        if self._framework == "xgboost":
            import xgboost as xgb  # type: ignore[import]
            m = xgb.XGBClassifier()
            m.load_model(str(model_dir / "model.bst"))
            return m
        else:
            import lightgbm as lgb  # type: ignore[import]
            return lgb.Booster(model_file=str(model_dir / "model.txt"))

    def apply_overrides(
        self,
        df: pd.DataFrame,
        grid_id: str,
        overrides: dict[str, float] | RerunOverrides,
    ) -> pd.DataFrame:
        """Replace operator-supplied values for a specific grid cell.

        Accepts either a flat ``dict[str, float]`` (backward compat) or a
        typed ``RerunOverrides`` model covering weather + vegetation.
        Non-overridden columns keep their original API-sourced values.
        All other rows (other grid cells) are left unchanged.
        """
        # Normalise to flat dict for the column-level loop
        if isinstance(overrides, RerunOverrides):
            flat = self._flatten_overrides(overrides)
        else:
            flat = overrides

        result = df.copy()
        mask = result["grid_id"] == grid_id
        if not mask.any():
            logger.warning("grid_id %s not found in data — no overrides applied", grid_id)
            return result

        for op_field, value in flat.items():
            if op_field not in _OVERRIDE_MAP:
                logger.debug("Unknown operator field %s — skipping", op_field)
                continue
            mapping = _OVERRIDE_MAP[op_field]
            converted = mapping["convert"](value)
            col = mapping["column"]
            if col in result.columns:
                result.loc[mask, col] = converted
                logger.debug("Override %s → %s = %.4f", op_field, col, converted)

        return result

    @staticmethod
    def _flatten_overrides(ovr: RerunOverrides) -> dict[str, float]:
        """Convert typed RerunOverrides into a flat field→value dict."""
        flat: dict[str, float] = {}
        if ovr.weather:
            for field_name in ("temperature_f", "wind_speed_mph", "relative_humidity",
                               "soil_moisture", "fire_weather_index", "wind_direction_deg",
                               "precipitation_mm", "vpd_kpa"):
                val = getattr(ovr.weather, field_name, None)
                if val is not None:
                    flat[field_name] = val
        if ovr.vegetation:
            for field_name in ("ndvi", "dominant_fuel_fraction", "fuel_model_fbfm40"):
                val = getattr(ovr.vegetation, field_name, None)
                if val is not None:
                    flat[field_name] = float(val)
        return flat

    def run_obj1(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess features and run OBJ-1 fire risk scoring.

        Returns
        -------
        predictions_df:
            Copy of df with ``fire_risk_score``, ``fire_risk_flag``, ``risk_tier`` columns.
        input_df:
            The preprocessed feature matrix passed to the model.
        """
        from src.preprocessing.feature_engineering import full_pipeline
        def assign_risk_tier(score: float) -> str:
            if score >= 0.65:
                return "CRITICAL"
            elif score >= 0.365:
                return "HIGH"
            elif score >= 0.15:
                return "MEDIUM"
            return "LOW"

        model_type = "xgb" if self._framework == "xgboost" else "lgbm"
        X, _ = full_pipeline(df, model_type=model_type, fit_medians=self._medians)

        if self._framework == "xgboost":
            probs = self._model.predict_proba(X)[:, 1]
        else:
            probs = self._model.predict(X)

        result = df.copy()
        result["fire_risk_score"] = probs
        result["fire_risk_flag"] = (probs >= self._threshold).astype(int)
        result["risk_tier"] = [assign_risk_tier(p) for p in probs]
        return result, X

    def run_obj2(
        self,
        df: pd.DataFrame,
        predictions: pd.DataFrame,
    ) -> dict[str, Any] | None:
        """Run Rothermel fire spread simulation on the highest-risk cell (OBJ-1 output).

        Uses PythonFireSpreadSimulator (Rothermel 1972) with the full fused
        feature DataFrame so the simulator can resolve neighbour terrain/fuel.
        Also applies CBH physical clamp to prevent false crown fire from
        imputed low values.

        Returns the simulation result dict, or None on failure.
        """
        try:
            from src.models.obj2_spread.fire_spread_simulator import PythonFireSpreadSimulator

            # Clamp CBH to prevent false crown fire (imputed median = 0.175m)
            if "canopy_base_height_m" in df.columns:
                df = df.copy()
                df["canopy_base_height_m"] = df["canopy_base_height_m"].clip(lower=2.0)

            # Pick ignition cell: highest OBJ-1 fire_risk_score
            highest_risk_idx = predictions["fire_risk_score"].idxmax()
            ign_id = str(df.loc[highest_risk_idx, "grid_id"])
            ign_prob = float(predictions.loc[highest_risk_idx, "fire_risk_score"])

            sim = PythonFireSpreadSimulator()
            result = sim.simulate(df, ign_id, ign_prob)
            logger.info(
                "OBJ-2 simulation: cell=%s prob=%.3f speed=%.4f km/h dir=%.1f° crown=%s",
                ign_id, ign_prob,
                result.get("spread_speed_kmh", 0),
                result.get("spread_direction_deg", 0),
                result.get("crown_fire_status", "?"),
            )
            return result
        except Exception as e:
            logger.warning("OBJ-2 simulation failed (non-blocking): %s", e)
            return None

    # -- Post-pipeline injection helpers ----------------------------------------

    @staticmethod
    def inject_firms_overrides(
        pipeline_result: dict[str, Any],
        firms_list: list[FIRMSOverride],
    ) -> None:
        """Merge operator-supplied FIRMS hotspots into pipeline_result (in-place)."""
        if not firms_list:
            return
        existing = pipeline_result.get("firms_hotspots") or []
        for f in firms_list:
            existing.append({
                "lat": f.lat,
                "lon": f.lon,
                "frp": f.frp,
                "confidence": f.confidence,
                "acq_datetime": f.acq_datetime or datetime.now(tz=UTC).isoformat(),
                "source": "operator_override",
            })
        pipeline_result["firms_hotspots"] = existing
        pipeline_result["firms_hotspot_count"] = len(existing)
        logger.info("Injected %d operator FIRMS hotspots (total %d)", len(firms_list), len(existing))

    @staticmethod
    def inject_xgboost_overrides(
        pipeline_result: dict[str, Any],
        cells: list[XGBoostCellOverride],
    ) -> None:
        """Replace or merge operator-supplied XGBoost cells into pipeline_result (in-place)."""
        if not cells:
            return
        existing = pipeline_result.get("xgboost_top_cells") or []
        override_indices = {c.h3_index for c in cells}
        # Keep non-overridden cells
        kept = [c for c in existing if c.get("h3_index") not in override_indices]
        for c in cells:
            kept.append({
                "h3_index": c.h3_index,
                "probability": c.probability,
                "lat": c.lat,
                "lon": c.lon,
                "source": "operator_override",
            })
        # Sort by probability descending
        kept.sort(key=lambda x: float(x.get("probability", 0)), reverse=True)
        pipeline_result["xgboost_top_cells"] = kept
        logger.info("Injected %d operator XGBoost cell overrides", len(cells))

    @staticmethod
    def apply_obj2_overrides(
        sim_result: dict[str, Any] | None,
        overrides: OBJ2SimOverrides,
    ) -> dict[str, Any] | None:
        """Patch OBJ-2 simulation output with operator-supplied values."""
        if sim_result is None:
            # Build a minimal sim dict from overrides alone
            sim_result = {"source": "operator_override"}
        for field_name in ("spread_speed_kmh", "spread_direction_deg",
                           "crown_fire_status", "dead_fuel_moisture_pct",
                           "foliar_moisture_content_pct"):
            val = getattr(overrides, field_name, None)
            if val is not None:
                sim_result[field_name] = val
                logger.debug("OBJ-2 override: %s = %s", field_name, val)
        return sim_result

    @staticmethod
    def apply_risk_override(
        pipeline_result: dict[str, Any],
        level: str,
    ) -> None:
        """Override the derived risk_level in pipeline_result (in-place)."""
        pipeline_result["risk_level"] = level.upper()
        logger.info("Risk level overridden to %s by operator", level.upper())

    # -- Build result ----------------------------------------------------------

    def build_result(
        self,
        predictions: pd.DataFrame,
        input_df: pd.DataFrame,
        obj2_sim: dict[str, Any] | None,
        firms: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a pipeline_result dict for the OBJ-3 report generator.

        Calls bridge.build_pipeline_result() with real OBJ-1/OBJ-2 outputs.
        """
        from src.pipeline.bridge import build_pipeline_result  # type: ignore[import]

        return build_pipeline_result(
            obj1_predictions=predictions,
            obj1_input=input_df,
            obj2_simulation=obj2_sim,
            firms_hotspots=firms,
        )