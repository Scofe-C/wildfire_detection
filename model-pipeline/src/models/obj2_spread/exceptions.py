"""
Cell2Fire exceptions and configuration loader.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class Cell2FireError(Exception):
    """Raised when the C++ binary fails or inputs are invalid."""


class Cell2FireNotInstalledError(Cell2FireError):
    """Raised when the Cell2Fire binary is not found on PATH."""


def load_obj2_config(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the obj2 section from model_config.yaml."""
    if config_path is None:
        config_path = (
            Path(__file__).resolve().parents[3] / "configs" / "model_config.yaml"
        )
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return raw["obj2"]
