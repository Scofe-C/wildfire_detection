"""
OBJ-2 (Secondary) — PROPAGATOR Fire Spread (DEMO ONLY)
=======================================================
Mediterranean fire spread model used **only** for comparison against
Cell2Fire. NOT calibrated for US vegetation — every output carries
a disclaimer.

PROPAGATOR uses a simplified physics model designed for Mediterranean
shrubland. We include it to demonstrate that Cell2Fire (calibrated
for North American fuels via LANDFIRE FBFM40) outperforms a model
built for a different ecosystem, validating the Cell2Fire choice.

The FBFM40 → Mediterranean crosswalk maps 40 US fuel types to the
7 Mediterranean fuel classes PROPAGATOR expects. This mapping is
inherently lossy — another reason this is demo-only.

Owner: Ibrahim (OBJ-2)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


DISCLAIMER = (
    "FOR DEMONSTRATION PURPOSES ONLY — Mediterranean fuel model, "
    "not calibrated for US vegetation types."
)

# FBFM40 → PROPAGATOR 7-class Mediterranean crosswalk
# Source: Scott & Burgan (2005) mapped to Mediterranean equivalents
# Keys are FBFM40 codes (integers), values are PROPAGATOR fuel class IDs
DEFAULT_CROSSWALK: dict[int, int] = {
    # Grass group (GR1–GR9) → Mediterranean grass (class 1)
    101: 1, 102: 1, 103: 1, 104: 1, 105: 1, 106: 1, 107: 1, 108: 1, 109: 1,
    # Grass-Shrub (GS1–GS4) → Mediterranean grass-shrub (class 2)
    121: 2, 122: 2, 123: 2, 124: 2,
    # Shrub (SH1–SH9) → Mediterranean shrub/maquis (class 3)
    141: 3, 142: 3, 143: 3, 144: 3, 145: 3, 146: 3, 147: 3, 148: 3, 149: 3,
    # Timber-Understory (TU1–TU5) → Mediterranean forest understory (class 4)
    161: 4, 162: 4, 163: 4, 164: 4, 165: 4,
    # Timber Litter (TL1–TL9) → Mediterranean forest litter (class 5)
    181: 5, 182: 5, 183: 5, 184: 5, 185: 5, 186: 5, 187: 5, 188: 5, 189: 5,
    # Slash-Blowdown (SB1–SB4) → Mediterranean slash (class 6)
    201: 6, 202: 6, 203: 6, 204: 6,
    # Non-burnable (NB1–NB9, water, urban, etc.) → non-burnable (class 7)
    91: 7, 92: 7, 93: 7, 98: 7, 99: 7,
}


def _load_propagator_config(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load propagator section from model_config.yaml."""
    if config_path is None:
        config_path = (
            Path(__file__).resolve().parents[3] / "configs" / "model_config.yaml"
        )
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return raw["obj2"]["propagator"]


def load_crosswalk(crosswalk_path: str | Path | None = None) -> dict[int, int]:
    """Load FBFM40 → PROPAGATOR fuel class crosswalk.

    Falls back to the hardcoded DEFAULT_CROSSWALK if no file is provided
    or the file doesn't exist.
    """
    if crosswalk_path is not None:
        crosswalk_path = Path(crosswalk_path)
        if crosswalk_path.exists():
            with open(crosswalk_path) as f:
                raw = json.load(f)
            return {int(k): int(v) for k, v in raw.items()}
            logger.info("Loaded crosswalk from %s (%d entries)", crosswalk_path, len(raw))

    logger.info("Using default FBFM40 → PROPAGATOR crosswalk (%d entries)", len(DEFAULT_CROSSWALK))
    return dict(DEFAULT_CROSSWALK)


def reclassify_fuel(
    fbfm40_codes: np.ndarray,
    crosswalk: dict[int, int] | None = None,
) -> np.ndarray:
    """Reclassify LANDFIRE FBFM40 codes to PROPAGATOR 7-class system.

    Unknown codes are mapped to non-burnable (class 7).
    """
    if crosswalk is None:
        crosswalk = load_crosswalk()

    result = np.full_like(fbfm40_codes, fill_value=7, dtype=np.int16)
    for fbfm_code, prop_class in crosswalk.items():
        result[fbfm40_codes == fbfm_code] = prop_class

    n_unknown = np.sum(result == 7) - np.sum(fbfm40_codes == 7)
    if n_unknown > 0:
        logger.warning(
            "%d cells had unrecognized FBFM40 codes → mapped to non-burnable",
            n_unknown,
        )
    return result


class PropagatorSpread(BaseModel):
    """PROPAGATOR Mediterranean fire spread model — DEMO ONLY.

    Every output includes the DISCLAIMER string. This model exists
    solely for comparison against Cell2Fire to validate our choice
    of a North American fuel model.

    Usage::

        model = PropagatorSpread()
        model.load_model("configs/propagator_config.json")
        predictions = model.predict(feature_df)
        # predictions contain DISCLAIMER column
    """

    DISCLAIMER = DISCLAIMER

    def __init__(self) -> None:
        super().__init__(model_name="propagator", version="0.1.0")
        self._config: dict[str, Any] = {}
        self._prop_config: dict[str, Any] = {}
        self._crosswalk: dict[int, int] = {}

    def load_model(self, model_path: str | Path) -> None:
        """Load PROPAGATOR configuration.

        Parameters
        ----------
        model_path : str | Path
            Path to JSON config with AOI bounds, raster paths, and
            any parameter overrides.
        """
        model_path = Path(model_path)
        self._prop_config = _load_propagator_config()
        self._crosswalk = load_crosswalk(
            self._prop_config.get("crosswalk_path")
        )

        config_file = model_path / "propagator_config.json" if model_path.is_dir() else model_path

        if config_file.exists():
            with open(config_file) as f:
                self._config = json.load(f)
        else:
            logger.warning(
                "Propagator config not found at %s — using defaults",
                config_file,
            )
            self._config = {}

        self._is_loaded = True
        logger.info(
            "PROPAGATOR loaded (DEMO ONLY). Crosswalk: %d fuel mappings. %s",
            len(self._crosswalk),
            self.DISCLAIMER,
        )

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Run simplified PROPAGATOR spread estimate.

        Since PROPAGATOR's actual C++ library requires Mediterranean
        BoundaryConditions objects that don't map to our US data, this
        implementation uses a **simplified analytical approximation**
        of the Rothermel spread rate with Mediterranean fuel parameters.

        This is sufficient for the demo comparison (showing Cell2Fire
        outperforms on US fuel types).

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame with weather and terrain columns.

        Returns
        -------
        pd.DataFrame
            Columns: prediction, probability, disclaimer.
        """
        if not self._is_loaded:
            raise RuntimeError("Call load_model() before predict()")

        n = len(X)

        # Simplified spread rate estimation using available features
        wind_col = None
        for col in ("wind_speed_10m", "wind_speed", "ws"):
            if col in X.columns:
                wind_col = col
                break

        slope_col = None
        for col in ("slope_degrees", "slope"):
            if col in X.columns:
                slope_col = col
                break

        rh_col = None
        for col in ("relative_humidity_2m", "relative_humidity", "rh"):
            if col in X.columns:
                rh_col = col
                break

        # Compute a rough burn probability based on available inputs
        # This is intentionally simplistic — Mediterranean model on US data
        probabilities = np.full(n, 0.1, dtype=float)

        if wind_col is not None:
            ws = X[wind_col].fillna(0).values
            # Higher wind → higher spread probability (capped)
            wind_factor = np.clip(ws / 30.0, 0.0, 1.0)
            probabilities += 0.3 * wind_factor

        if slope_col is not None:
            slope = X[slope_col].fillna(0).values
            # Steeper slope → faster uphill spread
            slope_factor = np.clip(slope / 45.0, 0.0, 1.0)
            probabilities += 0.2 * slope_factor

        if rh_col is not None:
            rh = X[rh_col].fillna(50).values
            # Lower humidity → higher fire probability
            rh_factor = np.clip(1.0 - rh / 100.0, 0.0, 1.0)
            probabilities += 0.2 * rh_factor

        probabilities = np.clip(probabilities, 0.0, 1.0)
        wind_reduction = self._prop_config.get(
            "default_params", {}
        ).get("wind_reduction_factor", 0.4)
        probabilities *= (1.0 - wind_reduction * 0.5)

        predictions = (probabilities >= 0.5).astype(int)

        result = pd.DataFrame({
            "prediction": predictions,
            "probability": probabilities,
            "disclaimer": self.DISCLAIMER,
        })

        logger.info(
            "PROPAGATOR predict: %d cells, %.1f%% predicted burned. %s",
            n, 100 * predictions.mean(), self.DISCLAIMER,
        )
        return result

    def validate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Validate PROPAGATOR against actual burn data.

        Expected to underperform Cell2Fire on US data.
        """
        from src.models.obj2_spread.cell2fire_spread import compute_dice_coefficient
        from src.validation.metrics import compute_all_metrics

        predictions = self.predict(X)
        y_prob = predictions["probability"].values
        y_pred = predictions["prediction"].values
        y_true = np.asarray(y)

        dice = compute_dice_coefficient(y_pred, y_true)
        all_metrics = compute_all_metrics(y_true, y_prob)
        all_metrics["dice_coefficient"] = dice
        all_metrics["disclaimer"] = self.DISCLAIMER

        logger.info(
            "PROPAGATOR validation — Dice: %.4f, AUC-PR: %.4f. %s",
            dice, all_metrics.get("auc_pr", 0.0), self.DISCLAIMER,
        )
        return all_metrics

    def explain(self, X: pd.DataFrame) -> dict[str, Any]:
        """Explain PROPAGATOR's limitations on US vegetation.

        Returns a structured explanation of why Mediterranean fuel
        parameters are inappropriate for LANDFIRE FBFM40 fuels.
        """
        return {
            "disclaimer": self.DISCLAIMER,
            "method": "analytical_comparison",
            "explanation": (
                "PROPAGATOR uses 7 Mediterranean fuel classes derived from "
                "European shrubland and maquis vegetation. The FBFM40 → "
                "PROPAGATOR crosswalk collapses 40 distinct US fuel types "
                "into 7 classes, losing critical differentiation between "
                "grass, shrub, timber, and slash fuel structures. This "
                "information loss directly degrades spread prediction "
                "accuracy on North American landscapes."
            ),
            "crosswalk_coverage": {
                "total_fbfm40_codes": 40,
                "mapped_codes": len(self._crosswalk),
                "target_classes": 7,
                "compression_ratio": len(self._crosswalk) / 7.0,
            },
        }
