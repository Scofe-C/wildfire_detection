import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.models.base import BaseModel
from src.validation.metrics import compute_auc_pr, compute_f1, compute_fnr

logger = logging.getLogger(__name__)

class XGBoostFireRiskModel(BaseModel):
    def __init__(self):
        super().__init__(model_name="xgboost_pof", version="1.0.0")
        # We define eval_metric="logloss" to avoid deprecation warnings
        self._model = XGBClassifier(
            max_depth=6,
            n_estimators=100,
            random_state=42,
            eval_metric="logloss"
        )
        self.features = [
            "temperature_c", "relative_humidity", "wind_speed_m_s",
            "wind_direction_rad", "precipitation_mm",
            "lag_fire_detected", "lag_active_fire_count"
        ]

    def preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Derive the specific weather features needed by the Objective 1 formulation
        if raw ERA5 columns are provided.
        """
        df = X.copy()

        if "u10" in df.columns and "v10" in df.columns:
            df["wind_speed_m_s"] = np.sqrt(df["u10"]**2 + df["v10"]**2)
            df["wind_direction_rad"] = np.arctan2(df["v10"], df["u10"])

        if "t2m" in df.columns and "d2m" in df.columns:
            t_c = df["t2m"] - 273.15
            td_c = df["d2m"] - 273.15
            df["temperature_c"] = t_c
            df["dewpoint_c"] = td_c

            e_td = np.exp((17.625 * td_c) / (243.04 + td_c))
            e_t = np.exp((17.625 * t_c) / (243.04 + t_c))
            df["relative_humidity"] = (100.0 * (e_td / e_t)).clip(0, 100)

        if "tp" in df.columns:
            df["precipitation_mm"] = df["tp"] * 1000.0

        # Ensure all required target features exist (fill missing manually if orchestrated otherwise structure)
        for col in self.features:
            if col not in df.columns:
                logger.warning(f"Feature '{col}' missing from data! Filling with zeros.")
                df[col] = 0.0

        return df[self.features]

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Trains the XGBoost model natively on the preprocessed feature distribution.
        """
        logger.info("Preprocessing features for training...")
        X_processed = self.preprocess_features(X)

        # Calculate scale_pos_weight for imbalance
        scale_pos = (len(y) - y.sum()) / max(1, y.sum())
        logger.info(f"Setting scale_pos_weight to {scale_pos:.2f}")
        self._model.set_params(scale_pos_weight=scale_pos)

        logger.info("Fitting XGBoost model...")
        self._model.fit(X_processed, y)
        self._is_loaded = True

    def load_model(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._model.load_model(str(model_path))
        self._is_loaded = True
        logger.info(f"Loaded XGBoost model from {model_path}")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded or trained. Call load_model() or train() first.")

        X_processed = self.preprocess_features(X)

        preds = self._model.predict(X_processed)
        probs = self._model.predict_proba(X_processed)[:, 1]

        return pd.DataFrame({
            "prediction": preds,
            "probability": probs
        })

    def validate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        predictions = self.predict(X)
        y_pred = predictions["prediction"].values
        y_prob = predictions["probability"].values
        y_true = np.asarray(y)

        return {
            "auc_pr": compute_auc_pr(y_true, y_prob),
            "f1": compute_f1(y_true, y_pred),
            "fnr": compute_fnr(y_true, y_pred)
        }

    def explain(self, X: pd.DataFrame) -> dict[str, Any]:
        """Provides basic feature importance explanation."""
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded. Cannot explain.")

        importances = self._model.feature_importances_
        feature_importance = dict(zip(self.features, importances, strict=False))
        # Sort descending
        feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))

        return {
            "feature_importance": feature_importance
        }
