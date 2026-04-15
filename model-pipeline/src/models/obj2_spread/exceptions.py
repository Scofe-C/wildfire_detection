"""
Configuration loader for OBJ-2 fire spread simulation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_obj2_config(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the obj2 section from model_config.yaml."""
    if config_path is None:
        config_path = (
            Path(__file__).resolve().parents[3] / "configs" / "model_config.yaml"
        )
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw["obj2"]
