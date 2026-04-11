"""Operator override re-run engine — OBJ-1 + OBJ-2 with corrected local observations.

Allows fire commanders to replace API-sourced weather values with their own
on-the-ground measurements and re-score all grid cells using the production model.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Operator field → pipeline column + optional unit conversion
_OVERRIDE_MAP: dict[str, dict[str, Any]] = {
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
        overrides: dict[str, float],
    ) -> pd.DataFrame:
        """Replace operator-supplied values for a specific grid cell.

        Non-overridden columns keep their original API-sourced values.
        All other rows (other grid cells) are left unchanged.
        """
        result = df.copy()
        mask = result["grid_id"] == grid_id
        if not mask.any():
            logger.warning("grid_id %s not found in data — no overrides applied", grid_id)
            return result

        for op_field, value in overrides.items():
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
        from src.validation.model_selector import assign_risk_tier  # type: ignore[import]

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
        """Run fire spread simulation on the highest-risk cell.

        Returns the simulation result dict, or None if the OBJ-2 binary/module
        is unavailable in this environment.
        """
        highest_risk_idx = predictions["fire_risk_score"].idxmax()
        top_row = df.loc[highest_risk_idx]

        try:
            from src.models.obj2_propagator.propagator import (
                PropagatorSpread,  # type: ignore[import]
            )

            sim = PropagatorSpread()
            result = sim.run(
                grid_id=str(top_row.get("grid_id", "unknown")),
                wind_speed=float(top_row.get("wind_speed_10m", 0)),
                wind_direction=float(top_row.get("wind_direction_10m", 0)),
                temperature=float(top_row.get("temperature_2m", 20)),
                relative_humidity=float(top_row.get("relative_humidity_2m", 30)),
            )
            return result
        except ImportError:
            logger.info("OBJ-2 propagator not available — skipping spread simulation")
            return None
        except Exception as e:
            logger.warning("OBJ-2 simulation failed (non-blocking): %s", e)
            return None

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
            predictions_df=predictions,
            input_df=input_df,
            obj2_result=obj2_sim,
            firms_data=firms,
        )
