from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: str
    source: str
    bounds: tuple[float, float] | None
    required: bool


@dataclass(frozen=True)
class FeatureSchema:
    version: str
    index_columns: list[dict[str, Any]]
    target: dict[str, Any]
    features: list[FeatureSpec]
    normalization_stats_path: str

    @property
    def required_features(self) -> list[FeatureSpec]:
        return [f for f in self.features if f.required]

    @property
    def optional_features(self) -> list[FeatureSpec]:
        return [f for f in self.features if not f.required]

    @property
    def all_feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    @property
    def required_feature_names(self) -> list[str]:
        return [f.name for f in self.required_features]

    @property
    def index_column_names(self) -> list[str]:
        return [c["name"] for c in self.index_columns]

    @property
    def target_name(self) -> str:
        return self.target["name"]


def load_schema(config_path: str | Path | None = None) -> FeatureSchema:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "feature_schema.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Feature schema not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    features = []
    for feat in raw.get("features", []):
        bounds = tuple(feat["bounds"]) if feat.get("bounds") else None
        features.append(FeatureSpec(
            name=feat["name"],
            dtype=feat["dtype"],
            source=feat.get("source", "unknown"),
            bounds=bounds,
            required=feat.get("required", True),
        ))

    return FeatureSchema(
        version=raw["version"],
        index_columns=raw.get("index_columns", []),
        target=raw.get("target", {}),
        features=features,
        normalization_stats_path=raw.get("normalization_stats_path", ""),
    )


def validate_dataframe(df: pd.DataFrame, schema: FeatureSchema) -> list[str]:
    errors: list[str] = []

    for col_spec in schema.index_columns:
        if col_spec["name"] not in df.columns:
            errors.append(f"Missing index column: {col_spec['name']}")

    if schema.target_name not in df.columns:
        errors.append(f"Missing target column: {schema.target_name}")

    for feat in schema.required_features:
        if feat.name not in df.columns:
            errors.append(f"Missing required feature: {feat.name} (source: {feat.source})")

    for feat in schema.features:
        if feat.name not in df.columns or feat.bounds is None:
            continue
        col = df[feat.name].dropna()
        if len(col) == 0:
            continue
        lo, hi = feat.bounds
        if col.min() < lo or col.max() > hi:
            errors.append(
                f"'{feat.name}' out of bounds: "
                f"expected [{lo}, {hi}], got [{col.min()}, {col.max()}]"
            )

    present_opt = [f.name for f in schema.optional_features if f.name in df.columns]
    missing_opt = [f.name for f in schema.optional_features if f.name not in df.columns]
    if present_opt:
        logger.info("Optional features present: %s", present_opt)
    if missing_opt:
        logger.info("Optional features not in data: %s", missing_opt)

    return errors
