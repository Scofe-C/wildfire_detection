"""
Feature engineering pipeline — single source of truth for all transforms.

Both model classes (XGBoost, LightGBM) and the 6-hour inference loop import
from this module.  Any change to transforms must be made here only, never
inside individual model files, to prevent training/inference skew.

Preprocessing order (must be preserved):
  1. drop_non_features      — remove metadata, leakage, and dropped columns
  2. impute_before_encoding — median-impute angular columns BEFORE sin/cos encoding
  3. apply_circular_encoding — wind_direction_10m + aspect_degrees → sin/cos pairs
  4. apply_log1p            — precipitation, vpd, fire_weather_index, soil_moisture
  5. apply_median_imputation — elevation_m, slope_degrees, ndvi, dominant_fuel_fraction
  6. apply_ordinal_encoding  — XGBoost only; LightGBM keeps category dtype
  7. validate_no_nulls       — assert; raises if any nulls remain
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Canonical feature set (20 features + target) ──────────────────────────────

FEATURES = [
    # Continuous weather
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "soil_moisture_0_to_7cm",
    "vpd",
    "fire_weather_index",
    # Angular (sin/cos encoded)
    "wind_direction_10m_sin",
    "wind_direction_10m_cos",
    "aspect_degrees_sin",
    "aspect_degrees_cos",
    # Static terrain
    "elevation_m",
    "slope_degrees",
    "dominant_fuel_fraction",
    "ndvi",
    # Geo
    "latitude",
    "longitude",
    # Static categorical
    "fuel_model_fbfm40",
    "vegetation_type",
    # Weather derived
    "cumulative_wind_run_24h",
    "drought_index_proxy",
]

TARGET = "fire_detected_binary"

# Columns log1p-transformed in notebook
LOG1P_COLS = ["precipitation", "vpd", "fire_weather_index", "soil_moisture_0_to_7cm"]

# Angular columns that must be imputed BEFORE circular encoding
ANGULAR_COLS = ["wind_direction_10m", "aspect_degrees"]

# Columns median-imputed after encoding
MEDIAN_IMPUTE_COLS = ["elevation_m", "slope_degrees", "ndvi", "dominant_fuel_fraction",
                      "soil_moisture_0_to_7cm"]

# Time-series derived cols — filled with ffill/bfill per grid_id (before grid_id is dropped)
TS_FILL_COLS = ["cumulative_wind_run_24h", "drought_index_proxy"]

# Categorical columns
CATEGORICAL_COLS = ["fuel_model_fbfm40", "vegetation_type"]

# Columns that must never appear at inference — raise if present
LEAKAGE_COLS = [
    "active_fire_count",
    "mean_frp",
    "median_frp",
    "max_confidence",
    "nearest_fire_distance_km",
]

# Metadata / constant columns always dropped
DROP_COLS = [
    "grid_id",
    "region",
    "data_quality_flag",
    "date",
    "resolution_km",
    "days_since_last_precipitation",
    "canopy_cover_pct",
    # NOTE: wind_direction_10m and aspect_degrees are NOT listed here —
    # they are dropped inside apply_circular_encoding() after sin/cos pairs are created.
]


# ── Step 1 ────────────────────────────────────────────────────────────────────

def drop_non_features(df: pd.DataFrame, is_inference: bool = False) -> pd.DataFrame:
    """Drop metadata, leakage, and ablated columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or partially processed input.
    is_inference : bool
        If True, raises ValueError when leakage columns are present (safety check).
        If False (training), leakage columns are silently dropped.
    """
    df = df.copy()

    leakage_present = [c for c in LEAKAGE_COLS if c in df.columns]
    if leakage_present:
        if is_inference:
            raise ValueError(
                f"Leakage columns present at inference — this must never happen: {leakage_present}"
            )
        logger.debug("Dropping leakage columns: %s", leakage_present)
        df = df.drop(columns=leakage_present)

    to_drop = [c for c in DROP_COLS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)

    # Also drop timestamp after it's no longer needed (caller splits first)
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    return df


# ── Step 2 ────────────────────────────────────────────────────────────────────

def impute_before_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Median-impute angular columns BEFORE circular encoding.

    Must happen before apply_circular_encoding — otherwise sin(NaN) = NaN
    propagates into the encoded features and imputation after encoding
    would require imputing two correlated columns separately.
    """
    df = df.copy()
    for col in ANGULAR_COLS:
        if col in df.columns and df[col].isna().any():
            med = df[col].median()
            n_null = df[col].isna().sum()
            df[col] = df[col].fillna(med)
            logger.debug("Imputed %d nulls in '%s' with median %.2f", n_null, col, med)
    return df


# ── Step 3 ────────────────────────────────────────────────────────────────────

def apply_circular_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Encode angular columns as sin/cos pairs and drop the originals.

    Raw degree values have a 0=360 discontinuity that confuses tree models
    (e.g. 359° and 1° are numerically far apart but directionally identical).
    """
    df = df.copy()
    for col in ANGULAR_COLS:
        if col in df.columns:
            rad = np.deg2rad(df[col])
            df[f"{col}_sin"] = np.sin(rad)
            df[f"{col}_cos"] = np.cos(rad)
            df = df.drop(columns=[col])
            logger.debug("Circular-encoded '%s' → %s_sin, %s_cos", col, col, col)
    return df


# ── Step 4 ────────────────────────────────────────────────────────────────────

def apply_log1p(df: pd.DataFrame) -> pd.DataFrame:
    """log1p-transform right-skewed weather columns.

    Columns: precipitation, vpd, fire_weather_index, soil_moisture_0_to_7cm.
    Clips negatives to 0 first (open-meteo can return tiny negative values
    for precipitation/soil moisture due to floating point).
    """
    df = df.copy()
    for col in LOG1P_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


# ── Step 5 ────────────────────────────────────────────────────────────────────

def apply_median_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Median-impute static terrain columns."""
    df = df.copy()
    for col in MEDIAN_IMPUTE_COLS:
        if col in df.columns and df[col].isna().any():
            med = df[col].median()
            n_null = df[col].isna().sum()
            df[col] = df[col].fillna(med)
            logger.debug("Imputed %d nulls in '%s' with median %.4f", n_null, col, med)
    return df


# ── Step 5b ───────────────────────────────────────────────────────────────────

def apply_categorical_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Fill categorical column nulls with mode — matches notebook Cell 5.

    Must run AFTER apply_median_imputation() and BEFORE encoding steps,
    so that OrdinalEncoder / cast_category_dtype never see NaN values.
    """
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                n_null = df[col].isna().sum()
                df[col] = df[col].fillna(mode_val[0])
                logger.debug("Mode-filled %d nulls in '%s' with %s", n_null, col, mode_val[0])
    return df


# ── Step 6 ────────────────────────────────────────────────────────────────────

def apply_ordinal_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """OrdinalEncoder for categorical columns — XGBoost path only.

    XGBoost cannot handle pandas category dtype (unlike LightGBM).
    Encodes to integer codes. Unknown categories at inference get -1.
    """
    from sklearn.preprocessing import OrdinalEncoder

    df = df.copy()
    cat_present = [c for c in CATEGORICAL_COLS if c in df.columns]
    if not cat_present:
        return df

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[cat_present] = enc.fit_transform(df[cat_present].astype(str))
    logger.debug("OrdinalEncoded: %s", cat_present)
    return df


def cast_category_dtype(
    df: pd.DataFrame,
    fit_categories: dict[str, list] | None = None,
) -> tuple[pd.DataFrame, dict[str, list]]:
    """Cast categorical columns to pandas category dtype — LightGBM path only.

    Notebook casts the FULL dataset before splitting so all splits share
    identical category definitions. We replicate this via fit_categories:
    - Training: derive categories from training df, return them
    - Test/inference: pass training fit_categories so test uses the same levels

    Without consistent category levels across train/test, LightGBM throws
    "train and valid dataset categorical_feature do not match".

    Returns
    -------
    (df_with_categories, categories_dict)
    """
    df = df.copy()
    out_cats: dict[str, list] = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            if fit_categories and col in fit_categories:
                cats = fit_categories[col]
            else:
                cats = sorted(df[col].dropna().astype(str).unique().tolist())
            out_cats[col] = cats
            df[col] = pd.Categorical(df[col].astype(str), categories=cats)
    return df, out_cats


# ── Step 7 ────────────────────────────────────────────────────────────────────

def validate_no_nulls(df: pd.DataFrame) -> None:
    """Raise ValueError if any nulls remain after preprocessing."""
    null_counts = df.isnull().sum()
    nulls = null_counts[null_counts > 0]
    if len(nulls) > 0:
        raise ValueError(
            f"Nulls remain after preprocessing — fix imputation:\n{nulls.to_string()}"
        )
    logger.debug("Null check passed — 0 nulls in %d columns", len(df.columns))


# ── Full pipeline ─────────────────────────────────────────────────────────────

def full_pipeline(
    df: pd.DataFrame,
    model_type: str = "xgb",
    is_inference: bool = False,
    fit_medians: dict | None = None,
    fit_categories: dict[str, list] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Apply the complete preprocessing pipeline in the correct order.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input (may include metadata columns, leakage columns, etc.)
    model_type : str
        "xgb" → applies OrdinalEncoder on categorical columns
        "lgbm" → casts categoricals to category dtype
    is_inference : bool
        If True, raises on leakage columns (safety check).
    fit_medians : dict | None
        Pre-computed medians from training data.  Pass these at inference
        to avoid computing medians on the (unlabeled) inference batch.
        If None, medians are computed from df (training mode).
    fit_categories : dict | None
        Pre-computed category levels from training data (LightGBM path only).
        Pass these when preprocessing test/inference data so category levels
        are identical to training — prevents LightGBM categorical mismatch.
        If None, categories are derived from df (training mode).

    Returns
    -------
    (X, state_dict)
        X          — preprocessed feature DataFrame ready for model
        state_dict — {"medians": ..., "categories": ...} — save and pass at test/inference time
    """
    # Step 0a: Fix sentinel values (-9999 from FIRMS/LANDFIRE → NaN)
    df = df.copy()
    df.replace(-9999, np.nan, inplace=True)
    if "canopy_cover_pct" in df.columns:
        df["canopy_cover_pct"] = df["canopy_cover_pct"].where(df["canopy_cover_pct"] >= 0, np.nan)

    # Step 0b: ffill/bfill time-series derived cols per grid_id BEFORE grid_id is dropped
    # Matches notebook Cell 4 Step 2 — median is wrong here because these are time-series features
    if "grid_id" in df.columns:
        ts_present = [c for c in TS_FILL_COLS if c in df.columns]
        if ts_present:
            df[ts_present] = df.groupby("grid_id")[ts_present].ffill().bfill()
    # Fallback: if grid_id already absent (inference batch), median-fill any remaining nulls
    for col in TS_FILL_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    df = drop_non_features(df, is_inference=is_inference)
    df = impute_before_encoding(df)
    df = apply_circular_encoding(df)
    df = apply_log1p(df)

    # Compute or apply medians for terrain imputation
    medians: dict = {}
    if fit_medians is not None:
        # Inference mode — use training medians
        for col, med in fit_medians.items():
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(med)
        medians = fit_medians
    else:
        # Training mode — compute from data
        for col in MEDIAN_IMPUTE_COLS:
            if col in df.columns:
                med = float(df[col].median())
                medians[col] = med
                if df[col].isna().any():
                    df[col] = df[col].fillna(med)

    # Mode-fill categoricals before encoding (matches notebook Cell 5)
    df = apply_categorical_imputation(df)

    out_categories: dict[str, list] = {}
    if model_type == "xgb":
        df = apply_ordinal_encoding(df)
    elif model_type == "lgbm":
        df, out_categories = cast_category_dtype(df, fit_categories=fit_categories)

    # Keep only canonical feature columns that exist in df
    feature_cols = [f for f in FEATURES if f in df.columns]
    missing_required = [
        f for f in FEATURES
        if f not in df.columns
        and f not in CATEGORICAL_COLS
    ]
    if missing_required:
        raise ValueError(f"Required features missing after preprocessing: {missing_required}")

    X = df[feature_cols].copy()

    # Final fallback for LOG1P_COLS: fill any remaining NaN with 0.
    # Covers inference gaps where the API returns no data (e.g. Open-Meteo soil moisture).
    # log1p(0) = 0, which correctly represents "no precipitation / no moisture".
    for col in LOG1P_COLS:
        if col in X.columns and X[col].isna().any():
            n = int(X[col].isna().sum())
            X[col] = X[col].fillna(0.0)
            logger.warning("Filled %d NaN in '%s' with 0 (API gap fallback)", n, col)

    validate_no_nulls(X)

    logger.info(
        "Preprocessing complete — %d rows × %d features (model_type=%s)",
        len(X), len(X.columns), model_type,
    )
    state = {"medians": medians, "categories": out_categories}
    return X, state


def extract_target(df: pd.DataFrame) -> pd.Series:
    """Extract the target column from a raw DataFrame."""
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in DataFrame")
    return df[TARGET].copy()
