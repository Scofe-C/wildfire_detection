"""
LightGBM fire risk model — OBJ-1 ignition prediction (secondary model).

Based on notebook experimentation results (california.ipynb):
  - ROC-AUC: 0.9374, PR-AUC: 0.8837  (XGBoost wins at 0.9426 / 0.8927)
  - Decision threshold: 0.239 (≥90% recall on Jan 2025 LA fires test set)
  - is_unbalance=True handles class imbalance (LightGBM equivalent of scale_pos_weight)
  - Natively handles pandas category dtype — no OrdinalEncoder needed
  - RandomizedSearchCV with TimeSeriesSplit(5), n_iter=50, scoring=roc_auc
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.preprocessing.feature_engineering import FEATURES, full_pipeline
from src.validation.metrics import compute_auc_pr, compute_f1, compute_fnr

logger = logging.getLogger(__name__)

# Hyperparameter search space
_PARAM_DISTRIBUTIONS = {
    "num_leaves":         [20, 31, 50, 63, 100, 127],
    "n_estimators":       [100, 200, 300, 400, 500],
    "learning_rate":      [0.01, 0.05, 0.1, 0.2, 0.3],
    "min_child_samples":  [10, 20, 30, 50, 100],
    "subsample":          [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":   [0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha":          [0.0, 0.01, 0.1, 0.5, 1.0],
    "reg_lambda":         [0.0, 0.01, 0.1, 0.5, 1.0],
}


class LightGBMFireRiskModel(BaseModel):
    """LightGBM classifier for wildfire ignition risk.

    Usage
    -----
    model = LightGBMFireRiskModel()
    best_params = model.tune(X_train, y_train)
    model.fit(X_train, y_train, best_params)
    threshold = model.tune_threshold(y_test, model.predict_proba(X_test))
    shap_dict = model.explain(X_test.sample(500))
    """

    def __init__(self, version: str = "1.0.0"):
        super().__init__(model_name="lightgbm_ignition", version=version)
        self._best_params: dict = {}
        self._threshold: float = 0.239       # default from notebook; overridden by tune_threshold
        self._fit_medians: dict = {}

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def preprocess(
        self,
        df: pd.DataFrame,
        is_inference: bool = False,
    ) -> tuple[pd.DataFrame, dict]:
        """Apply full preprocessing pipeline for LightGBM (category dtype, no OrdinalEncoder)."""
        return full_pipeline(df, model_type="lgbm", is_inference=is_inference,
                             fit_medians=self._fit_medians if is_inference else None)

    # ── Hyperparameter tuning ─────────────────────────────────────────────────

    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_iter: int = 50,
        cv_splits: int = 5,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> dict:
        """RandomizedSearchCV with TimeSeriesSplit."""
        from lightgbm import LGBMClassifier
        from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

        logger.info("Tuning LightGBM — n_iter=%d, cv=%d ...", n_iter, cv_splits)

        base_model = LGBMClassifier(
            is_unbalance=True,
            random_state=random_state,
            n_jobs=1,
            verbose=-1,
        )

        tscv = TimeSeriesSplit(n_splits=cv_splits)
        search = RandomizedSearchCV(
            base_model,
            param_distributions=_PARAM_DISTRIBUTIONS,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=tscv,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=1,
        )
        search.fit(X_train, y_train)

        self._best_params = search.best_params_
        logger.info("Best params: %s  (CV ROC-AUC=%.4f)", self._best_params, search.best_score_)
        return self._best_params

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: dict | None = None,
    ) -> None:
        """Train LightGBM on preprocessed features."""
        from lightgbm import LGBMClassifier

        logger.info("Fitting LightGBM — %d train rows", len(X_train))

        final_params: dict = {
            "is_unbalance": True,
            "random_state": 42,
            "verbose": -1,
        }
        if params:
            final_params.update(params)

        self._model = LGBMClassifier(**final_params)
        self._model.fit(X_train, y_train)
        self._is_loaded = True
        logger.info("LightGBM fit complete — %d features", X_train.shape[1])

    # ── Required BaseModel interface ──────────────────────────────────────────

    def load_model(self, model_path: str | Path) -> None:
        from lightgbm import Booster, LGBMClassifier
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        # LightGBM saves the booster; wrap in sklearn estimator
        booster = Booster(model_file=str(model_path))
        self._model = LGBMClassifier()
        self._model._Booster = booster
        self._is_loaded = True
        logger.info("Loaded LightGBM model from %s", model_path)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with 'prediction' (0/1) and 'probability' columns."""
        if not self._is_loaded:
            raise RuntimeError("Model not trained or loaded — call fit() or load_model() first")
        y_prob = self.predict_proba(X)
        y_pred = (y_prob >= self._threshold).astype(int)
        return pd.DataFrame({"prediction": y_pred, "probability": y_prob})

    def validate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        preds = self.predict(X)
        y_pred = preds["prediction"].values
        y_prob = preds["probability"].values
        y_true = np.asarray(y)
        return {
            "auc_pr": compute_auc_pr(y_true, y_prob),
            "f1":     compute_f1(y_true, y_pred),
            "fnr":    compute_fnr(y_true, y_pred),
        }

    def explain(self, X_sample: pd.DataFrame) -> dict[str, Any]:
        """SHAP TreeExplainer on a sample of the test set."""
        if not self._is_loaded:
            raise RuntimeError("Model not trained or loaded")

        feat_names = list(X_sample.columns)
        native_importance = dict(zip(
            feat_names,
            self._model.feature_importances_,
            strict=False,
        ))
        native_importance = dict(sorted(native_importance.items(), key=lambda x: x[1], reverse=True))

        shap_mean_abs: dict = {}
        try:
            import shap
            explainer = shap.TreeExplainer(self._model)
            shap_values = explainer.shap_values(X_sample)
            # LightGBM returns list [class0, class1] for binary classification
            sv = shap_values[1] if isinstance(shap_values, list) else shap_values
            shap_mean_abs = {
                feat: float(np.abs(sv[:, i]).mean())
                for i, feat in enumerate(feat_names)
            }
            shap_mean_abs = dict(sorted(shap_mean_abs.items(), key=lambda x: x[1], reverse=True))
        except ImportError:
            logger.warning("shap not installed — returning native importance only")

        return {"feature_importance": native_importance, "shap_mean_abs": shap_mean_abs}

    # ── Additional methods ────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_loaded:
            raise RuntimeError("Model not trained or loaded")
        return self._model.predict_proba(X)[:, 1]

    def tune_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        target_recall: float = 0.90,
    ) -> float:
        """Find the highest threshold achieving >= target_recall (candidates[-1] fix)."""
        from sklearn.metrics import precision_recall_curve

        prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
        candidates = np.where(rec[:-1] >= target_recall)[0]
        if len(candidates) == 0:
            logger.warning(
                "No threshold achieves recall >= %.2f — using 0.5 as fallback", target_recall
            )
            self._threshold = 0.5
        else:
            idx = candidates[-1]
            self._threshold = float(thresholds[idx])

        logger.info(
            "Threshold tuned: %.4f (target recall=%.2f)", self._threshold, target_recall
        )
        return self._threshold

    def get_params(self) -> dict:
        params = {
            "model_type": "lightgbm",
            "is_unbalance": True,
            "threshold": self._threshold,
            "n_features": len(FEATURES),
            "features": ",".join(FEATURES),
        }
        params.update(self._best_params)
        return params
