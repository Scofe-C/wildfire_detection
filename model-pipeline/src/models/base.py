from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class BaseModel(ABC):
    def __init__(self, model_name: str, version: str = "0.0.0"):
        self.model_name = model_name
        self.version = version
        self._model: Any = None
        self._is_loaded = False

    @abstractmethod
    def load_model(self, model_path: str | Path) -> None: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Must return DataFrame with columns: prediction (0/1), probability [0,1]."""

    @abstractmethod
    def validate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]: ...

    @abstractmethod
    def explain(self, X: pd.DataFrame) -> dict[str, Any]: ...

    def compute_artifact_hash(self, model_path: str | Path) -> str:
        model_path = Path(model_path)
        sha256 = hashlib.sha256()
        if model_path.is_file():
            sha256.update(model_path.read_bytes())
        elif model_path.is_dir():
            for f in sorted(model_path.rglob("*")):
                if f.is_file():
                    sha256.update(f.read_bytes())
        else:
            raise FileNotFoundError(f"Model path not found: {model_path}")
        return sha256.hexdigest()

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def __repr__(self) -> str:
        s = "loaded" if self._is_loaded else "not loaded"
        return f"<{self.__class__.__name__} {self.model_name} v{self.version} ({s})>"
