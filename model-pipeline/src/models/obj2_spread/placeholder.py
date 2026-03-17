"""OBJ-2 — Fire spread simulation. PLACEHOLDER for teammates.
TDD Sections 6.1-6.4. Cell2Fire primary, PROPAGATOR secondary (demo only)."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd
from src.models.base import BaseModel


class Cell2FireSpread(BaseModel):
    def __init__(self):
        super().__init__(model_name="cell2fire", version="0.0.0")

    def load_model(self, model_path: str | Path) -> None:
        raise NotImplementedError("OBJ-2: Cell2Fire load_model")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("OBJ-2: Cell2Fire predict")

    def validate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        raise NotImplementedError("OBJ-2: Cell2Fire validate")

    def explain(self, X: pd.DataFrame) -> dict[str, Any]:
        raise NotImplementedError("OBJ-2: Cell2Fire explain")


class PropagatorSpread(BaseModel):
    DISCLAIMER = "DEMO ONLY — not calibrated for US vegetation"

    def __init__(self):
        super().__init__(model_name="propagator", version="0.0.0")

    def load_model(self, model_path: str | Path) -> None:
        raise NotImplementedError("OBJ-2: PROPAGATOR load_model")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("OBJ-2: PROPAGATOR predict")

    def validate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        raise NotImplementedError("OBJ-2: PROPAGATOR validate")

    def explain(self, X: pd.DataFrame) -> dict[str, Any]:
        raise NotImplementedError("OBJ-2: PROPAGATOR explain")
