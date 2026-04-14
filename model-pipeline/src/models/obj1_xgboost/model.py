"""
XGBoost fire risk model — OBJ-1 ignition prediction.

Based on notebook experimentation results (california.ipynb):
  - ROC-AUC: 0.9426, PR-AUC: 0.8927
  - Decision threshold: 0.365 (≥90% recall on Jan 2025 LA fires test set)
  - scale_pos_weight handles class imbalance
  - OrdinalEncoder required (XGBoost cannot handle pandas category dtype)
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

# Hyperparameter search space from notebook Cell 17
_PARAM_DISTRIBUTIONS = {
    "max_depth":          [3, 4, 5, 6, 7, 8],
    "n_estimators":       [100, 200, 300, 400, 500],
    "learning_rate":      [0.01, 0.05, 0.1, 0.2, 0.3],
    "subsample":          [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":   [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight":   [1, 3, 5, 7, 10],
    "gamma":              [0, 0.1, 0.2, 0.3, 0.5],
}


class XGBoostFireRiskModel(BaseModel):
    """XGBoost classifier for wildfire ignition risk.

    Usage
    -----
    model = XGBoostFireRiskModel()
    best_params = model.tune(X_train, y_train)
    model.fit(X_train, y_train, best_params)
    threshold = model.tune_threshold(y_test, model.predict_proba(X_test))
    shap_dict = model.explain(X_test.sample(500))
    """

    def __init__(self, version: str = "1.0.0"):
        super().__init__(model_name="xgboost_ignition", version=version)
        self._best_params: dict = {}
        self._threshold: float = 0.365       # default from notebook; overridden by tune_threshold
        self._scale_pos_weight: float = 1.0
        self._fit_medians: dict = {}          # stored from training for inference-time imputation

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def preprocess(
        self,
        df: pd.DataFrame,
        is_inference: bool = False,
    ) -> tuple[pd.DataFrame, dict]:
        """Apply full preprocessing pipeline for XGBoost (with OrdinalEncoder).

        Returns (X, medians_dict).  At inference, pass self._fit_medians.
        """
        return full_pipeline(df, model_type="xgb", is_inference=is_inference,
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
        """RandomizedSearchCV with TimeSeriesSplit.

        Parameters
        ----------
        X_train, y_train : preprocessed training features and labels.
        n_iter : number of random parameter combinations to try.
        cv_splits : number of TimeSeriesSplit folds.

        Returns
        -------
        best_params : dict of best hyperparameters found.
        """
        from scipy.stats import randint, uniform
        from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
        from xgboost import XGBClassifier

        logger.info("Tuning XGBoost — n_iter=%d, cv=%d ...", n_iter, cv_splits)

        scale_pos = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
        base_model = XGBClassifier(
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,  # parallelism via RandomizedSearchCV n_jobs
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
        """Train XGBoost on preprocessed features.

        Parameters
        ----------
        X_train : preprocessed feature DataFrame (output of self.preprocess()).
        y_train : binary labels.
        params : hyperparameters (from tune()) or None for defaults.
        """
        from xgboost import XGBClassifier

        self._scale_pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
        logger.info(
            "Fitting XGBoost — %d train rows, scale_pos_weight=%.2f",
            len(X_train), self._scale_pos_weight,
        )

        final_params = {
            "scale_pos_weight": self._scale_pos_weight,
            "eval_metric": "logloss",
            "random_state": 42,
        }
        if params:
            final_params.update(params)

        self._model = XGBClassifier(**final_params)
        self._model.fit(X_train, y_train)
        self._is_loaded = True
        logger.info("XGBoost fit complete — %d features", X_train.shape[1])

    # ── Required BaseModel interface ──────────────────────────────────────────

    def load_model(self, model_path: str | Path) -> None:
        from xgboost import XGBClassifier
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self._model = XGBClassifier()
        self._model.load_model(str(model_path))
        self._is_loaded = True
        logger.info("Loaded XGBoost model from %s", model_path)

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
        """SHAP TreeExplainer on a sample of the test set.

        Returns dict with 'feature_importance' (native XGBoost gain) and
        'shap_mean_abs' (mean absolute SHAP values per feature).
        """
        if not self._is_loaded:
            raise RuntimeError("Model not trained or loaded")

        native_importance = dict(zip(
            self._model.feature_names_in_ if hasattr(self._model, "feature_names_in_") else FEATURES,
            self._model.feature_importances_,
            strict=False,
        ))
        native_importance = dict(sorted(native_importance.items(), key=lambda x: x[1], reverse=True))

        shap_mean_abs: dict = {}
        try:
            import shap
            explainer = shap.TreeExplainer(self._model)
            shap_values = explainer.shap_values(X_sample)
            feat_names = (
                list(self._model.feature_names_in_)
                if hasattr(self._model, "feature_names_in_")
                else list(X_sample.columns)
            )
            shap_mean_abs = {
                feat: float(np.abs(shap_values[:, i]).mean())
                for i, feat in enumerate(feat_names)
            }
            shap_mean_abs = dict(sorted(shap_mean_abs.items(), key=lambda x: x[1], reverse=True))
        except ImportError:
            logger.warning("shap not installed — returning native importance only")

        return {"feature_importance": native_importance, "shap_mean_abs": shap_mean_abs}

    # ── Additional methods ────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw probability scores (not thresholded)."""
        if not self._is_loaded:
            raise RuntimeError("Model not trained or loaded")
        return self._model.predict_proba(X)[:, 1]

    def tune_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        target_precision: float = 0.70,
    ) -> float:
        """Find threshold that maximises F1 subject to precision >= target_precision."""
        from sklearn.metrics import precision_recall_curve

        prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
        candidates = np.where(prec[:-1] >= target_precision)[0]
        if len(candidates) == 0:
            logger.warning(
                "No threshold achieves precision >= %.2f — using max-F1 fallback", target_precision
            )
            idx = int(np.argmax(f1_scores))
        else:
            idx = int(candidates[np.argmax(f1_scores[candidates])])
        self._threshold = float(thresholds[idx])

        logger.info(
            "Threshold tuned: %.4f (precision=%.3f, recall=%.3f, target_precision=%.2f)",
            self._threshold, prec[idx], rec[idx], target_precision,
        )
        return self._threshold

    def get_params(self) -> dict:
        """Return all model parameters for MLflow logging."""
        params = {
            "model_type": "xgboost",
            "scale_pos_weight": self._scale_pos_weight,
            "threshold": self._threshold,
            "n_features": len(FEATURES),
            "features": ",".join(FEATURES),
        }
        params.update(self._best_params)
        return params
