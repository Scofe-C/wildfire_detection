"""OBJ-3 — Gemini 3.1 Flash-Lite disaster reporting. PLACEHOLDER for owner.
TDD Sections 7.1-7.3. Only use as structured summarizer, never open-ended."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd
from src.models.base import BaseModel


class GeminiDisasterReporter(BaseModel):
    DISCLAIMER = "AI-generated. Not for operational use without human review."

    def __init__(self):
        super().__init__(model_name="gemini_disaster_reporter", version="0.0.0")

    def load_model(self, model_path: str | Path) -> None:
        raise NotImplementedError("OBJ-3: load_model")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("OBJ-3: predict")

    def validate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        raise NotImplementedError("OBJ-3: validate")

    def explain(self, X: pd.DataFrame) -> dict[str, Any]:
        raise NotImplementedError("OBJ-3: explain")
