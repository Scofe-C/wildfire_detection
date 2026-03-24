"""
OBJ-2 — Cell2Fire Fire Spread Simulation
=========================================
Replaces the placeholder stub with a working implementation.

Cell2Fire is a **physics-based** C++ simulator, not an ML model.
It reads DEM + fuel + weather inputs, runs Monte Carlo fire spread
simulations, and outputs burn probability grids.

This module wraps the C++ binary and exposes it through the same
BaseModel interface that OBJ-1 (XGBoost) and OBJ-3 (Gemini) use,
so the orchestrator can call it uniformly.

Mapping of BaseModel methods to Cell2Fire concepts:
    load_model()  → load simulation config (not weights)
    predict()     → run simulation → burn probability per cell
    validate()    → Dice coefficient vs CAL FIRE perimeters
    explain()     → parameter sensitivity analysis

Input requirements:
    - DEM:        GeoTIFF float32 (elevation in metres)
    - Fuel:       GeoTIFF int16  (LANDFIRE FBFM40 codes)
    - Weather:    CSV [datetime, ws, wd, tmp, rh]
    - Ignition:   (row, col) tuples in grid coordinates

Owner: Ibrahim (OBJ-2)
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


# ── Exceptions ───────────────────────────────────────────────────────────────

class Cell2FireError(Exception):
    """Raised when the C++ binary fails or inputs are invalid."""


class Cell2FireNotInstalledError(Cell2FireError):
    """Raised when the Cell2Fire binary is not found on PATH."""


# ── Helper: load OBJ-2 config ────────────────────────────────────────────────

def _load_obj2_config(
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


# ── Weather CSV formatter ────────────────────────────────────────────────────

def format_weather_csv(
    weather_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Convert pipeline weather DataFrame to Cell2Fire's expected CSV format.

    Cell2Fire expects columns: datetime, ws (wind speed m/s), wd (wind
    direction degrees), tmp (temperature C), rh (relative humidity %).

    The data pipeline produces columns like temperature_2m, wind_speed_10m,
    relative_humidity_2m, wind_direction_10m — this maps them.

    Parameters
    ----------
    weather_df : pd.DataFrame
        Weather data from the pipeline with timestamp and weather columns.
    output_path : str | Path
        Where to write the formatted CSV.

    Returns
    -------
    Path to the written CSV file.
    """
    output_path = Path(output_path)

    # Column mapping: pipeline name → Cell2Fire name
    col_map = {
        "wind_speed_10m": "ws",
        "wind_speed": "ws",
        "ws": "ws",
        "wind_direction_10m": "wd",
        "wind_direction": "wd",
        "wd": "wd",
        "temperature_2m": "tmp",
        "temperature": "tmp",
        "tmp": "tmp",
        "relative_humidity_2m": "rh",
        "relative_humidity": "rh",
        "rh": "rh",
    }

    out_df = pd.DataFrame()

    # Find timestamp column
    for ts_col in ("timestamp", "datetime", "time", "valid_time"):
        if ts_col in weather_df.columns:
            out_df["datetime"] = pd.to_datetime(weather_df[ts_col])
            break
    else:
        raise Cell2FireError(
            f"No timestamp column found. Available: {list(weather_df.columns)}"
        )

    # Map weather columns
    for src_col, tgt_col in col_map.items():
        if src_col in weather_df.columns and tgt_col not in out_df.columns:
            out_df[tgt_col] = weather_df[src_col].values

    missing = [c for c in ("ws", "wd", "tmp", "rh") if c not in out_df.columns]
    if missing:
        raise Cell2FireError(f"Missing weather columns after mapping: {missing}")

    out_df = out_df.sort_values("datetime").reset_index(drop=True)
    out_df.to_csv(output_path, index=False)
    logger.info("Weather CSV written: %d rows → %s", len(out_df), output_path)
    return output_path


# ── GeoTIFF clipping ────────────────────────────────────────────────────────

def clip_raster_to_aoi(
    raster_path: str | Path,
    bounds: tuple[float, float, float, float],
    output_path: str | Path,
    target_resolution: float | None = None,
) -> Path:
    """Clip a GeoTIFF to a bounding box, optionally resampling.

    Parameters
    ----------
    raster_path : path
        Source GeoTIFF.
    bounds : (west, south, east, north)
        Geographic bounding box in the raster's CRS.
    output_path : path
        Where to write the clipped raster.
    target_resolution : float | None
        If provided, resample to this cell size (in CRS units, typically metres).

    Returns
    -------
    Path to the clipped GeoTIFF.
    """
    import rasterio
    from rasterio.windows import from_bounds

    raster_path = Path(raster_path)
    output_path = Path(output_path)

    with rasterio.open(raster_path) as src:
        window = from_bounds(*bounds, transform=src.transform)

        # Read the windowed data
        data = src.read(1, window=window)
        win_transform = src.window_transform(window)

        profile = src.profile.copy()
        profile.update(
            height=data.shape[0],
            width=data.shape[1],
            transform=win_transform,
        )

        # Optional resampling
        if target_resolution is not None:
            from rasterio.enums import Resampling

            scale_x = abs(src.res[0]) / target_resolution
            scale_y = abs(src.res[1]) / target_resolution
            new_h = max(1, int(data.shape[0] * scale_y))
            new_w = max(1, int(data.shape[1] * scale_x))

            data = src.read(
                1,
                window=window,
                out_shape=(new_h, new_w),
                resampling=Resampling.bilinear,
            )
            from rasterio.transform import from_bounds as tfm_from_bounds

            profile.update(
                height=new_h,
                width=new_w,
                transform=tfm_from_bounds(*bounds, new_w, new_h),
            )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)

    logger.info(
        "Clipped raster %s → %s (%dx%d)",
        raster_path.name, output_path.name, data.shape[1], data.shape[0],
    )
    return output_path


# ── Burn probability parser ──────────────────────────────────────────────────

def parse_burn_probability(
    output_dir: Path,
    n_simulations: int,
) -> np.ndarray:
    """Read Cell2Fire output grids and compute burn probability.

    Cell2Fire writes one ASCII grid per simulation in the output/Grids/
    directory. Burn probability = fraction of simulations where each cell
    burned.

    Parameters
    ----------
    output_dir : Path
        Cell2Fire --output-folder path.
    n_simulations : int
        Number of simulations run (denominator for probability).

    Returns
    -------
    np.ndarray
        2D array of burn probabilities in [0, 1].
    """
    grids_dir = output_dir / "Grids" / "Grids1"
    if not grids_dir.exists():
        # Try alternative layout
        grids_dir = output_dir / "Grids"

    grid_files = sorted(grids_dir.glob("ForestGrid*.csv"))
    if not grid_files:
        raise Cell2FireError(f"No output grids found in {grids_dir}")

    burn_count = None
    for gf in grid_files:
        grid = np.loadtxt(gf, delimiter=",", skiprows=1)
        burned = (grid > 0).astype(float)
        if burn_count is None:
            burn_count = burned
        else:
            burn_count += burned

    burn_prob = burn_count / max(n_simulations, 1)
    logger.info(
        "Burn probability grid: %dx%d, mean=%.3f, max=%.3f",
        burn_prob.shape[1], burn_prob.shape[0],
        burn_prob.mean(), burn_prob.max(),
    )
    return burn_prob


# ── Burn probability → GeoJSON ───────────────────────────────────────────────

def burn_grid_to_geodataframe(
    burn_prob: np.ndarray,
    transform: Any,
    crs: str = "EPSG:4326",
    threshold: float = 0.1,
) -> Any:
    """Convert a burn probability grid to a GeoDataFrame of burned polygons.

    Parameters
    ----------
    burn_prob : np.ndarray
        2D burn probability array.
    transform : rasterio.Affine
        Georeferencing transform of the grid.
    crs : str
        Coordinate reference system.
    threshold : float
        Minimum burn probability to include (filters noise).

    Returns
    -------
    gpd.GeoDataFrame with columns: geometry, burn_probability.
    """
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape

    # Threshold the grid
    mask = burn_prob >= threshold
    burned_int = mask.astype(np.int16)

    records = []
    for geom_dict, value in shapes(burned_int, transform=transform):
        if value == 1:
            # Compute mean burn probability within this polygon
            # (simplified: use the threshold as lower bound)
            records.append({
                "geometry": shape(geom_dict),
                "burn_probability": float(burn_prob[mask].mean()),
            })

    if not records:
        logger.warning("No cells above burn threshold %.2f", threshold)
        return gpd.GeoDataFrame(
            columns=["geometry", "burn_probability"],
            geometry="geometry",
            crs=crs,
        )

    gdf = gpd.GeoDataFrame(records, crs=crs)
    logger.info("Burn GeoDataFrame: %d polygons", len(gdf))
    return gdf


# ── Dice coefficient ─────────────────────────────────────────────────────────

def compute_dice_coefficient(
    predicted_mask: np.ndarray,
    actual_mask: np.ndarray,
) -> float:
    """Dice coefficient between predicted and actual burn masks.

    Dice = 2 * |P ∩ A| / (|P| + |A|)
    Range: 0 (no overlap) to 1 (perfect overlap).

    Parameters
    ----------
    predicted_mask : np.ndarray
        Boolean or 0/1 array of predicted burned cells.
    actual_mask : np.ndarray
        Boolean or 0/1 array of actual burned cells.

    Returns
    -------
    float in [0, 1].
    """
    pred = np.asarray(predicted_mask, dtype=bool).ravel()
    actual = np.asarray(actual_mask, dtype=bool).ravel()

    if pred.shape != actual.shape:
        raise ValueError(
            f"Shape mismatch: predicted {pred.shape} vs actual {actual.shape}"
        )

    intersection = np.sum(pred & actual)
    total = np.sum(pred) + np.sum(actual)

    if total == 0:
        return 1.0  # both empty = perfect match

    return float(2.0 * intersection / total)


# =============================================================================
# Cell2FireSpread — BaseModel implementation
# =============================================================================

class Cell2FireSpread(BaseModel):
    """Cell2Fire C++ fire spread simulator wrapped as a BaseModel.

    Usage::

        model = Cell2FireSpread()
        model.load_model("configs/simulation_config.json")
        predictions = model.predict(feature_df)
        metrics = model.validate(feature_df, y_true)
        sensitivity = model.explain(feature_df)
    """

    def __init__(self) -> None:
        super().__init__(model_name="cell2fire", version="0.1.0")
        self._config: dict[str, Any] = {}
        self._obj2_config: dict[str, Any] = {}
        self._binary_path: str = ""
        self._sim_params: dict[str, Any] = {}
        self._last_burn_prob: np.ndarray | None = None
        self._last_transform: Any = None

    def load_model(self, model_path: str | Path) -> None:
        """Load simulation configuration.

        For Cell2Fire, 'model_path' points to either:
        - A JSON file with simulation-specific overrides (ignition points,
          AOI bounds, raster paths), OR
        - A directory containing the config JSON + pre-clipped rasters.

        The base Cell2Fire parameters come from model_config.yaml.

        Parameters
        ----------
        model_path : str | Path
            Path to simulation config JSON or directory containing one.
        """
        model_path = Path(model_path)
        self._obj2_config = _load_obj2_config()
        c2f_cfg = self._obj2_config["cell2fire"]

        self._binary_path = c2f_cfg.get("binary_path", "Cell2Fire")
        self._sim_params = dict(c2f_cfg.get("default_params", {}))

        # Load simulation-specific config
        if model_path.is_dir():
            config_file = model_path / "simulation_config.json"
            if not config_file.exists():
                raise FileNotFoundError(
                    f"No simulation_config.json in {model_path}"
                )
        elif model_path.suffix == ".json":
            config_file = model_path
        else:
            raise Cell2FireError(
                f"Expected .json file or directory, got: {model_path}"
            )

        with open(config_file) as f:
            self._config = json.load(f)

        # Override defaults with simulation-specific params
        if "params" in self._config:
            self._sim_params.update(self._config["params"])

        # Verify binary exists
        binary = shutil.which(self._binary_path)
        if binary is None:
            logger.warning(
                "Cell2Fire binary not found at '%s'. "
                "Simulation will fail unless installed before predict().",
                self._binary_path,
            )

        self._is_loaded = True
        logger.info(
            "Cell2Fire config loaded: %d simulations, %.1f hr periods, "
            "ignition points: %d",
            self._sim_params.get("n_simulations", 100),
            self._sim_params.get("fire_period_length_hr", 1.0),
            len(self._config.get("ignition_points", [])),
        )

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Run Cell2Fire simulation and return burn predictions per cell.

        The input DataFrame ``X`` must contain:
        - ``h3_index``: H3 cell IDs (used to define AOI bounds)
        - Weather columns (mapped to Cell2Fire format internally)

        If raster paths are provided in the simulation config, those are
        used directly. Otherwise, rasters are clipped from the paths
        specified in model_config.yaml.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame from the data pipeline.

        Returns
        -------
        pd.DataFrame
            Columns: ``prediction`` (0/1), ``probability`` [0, 1].
            One row per input cell in X.
        """
        if not self._is_loaded:
            raise Cell2FireError("Call load_model() before predict()")

        n_sims = self._sim_params.get("n_simulations", 100)
        period_hr = self._sim_params.get("fire_period_length_hr", 1.0)

        work_dir = Path(tempfile.mkdtemp(prefix="cell2fire_"))
        output_dir = work_dir / "output"
        output_dir.mkdir()

        try:
            # --- Prepare weather CSV ---
            weather_path = work_dir / "weather.csv"
            if "weather_csv" in self._config:
                shutil.copy(self._config["weather_csv"], weather_path)
            else:
                format_weather_csv(X, weather_path)

            # --- Prepare raster paths ---
            dem_path = self._resolve_raster("dem", work_dir)
            self._resolve_raster("fuel", work_dir)

            # --- Build ignition file ---
            ignition_points = self._config.get("ignition_points", [])
            ignition_file = work_dir / "ignitions.csv"
            with open(ignition_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Year", "Cell"])
                for pt in ignition_points:
                    if isinstance(pt, (list, tuple)) and len(pt) == 2:
                        # Convert (row, col) to cell number (1-indexed)
                        row, col = pt
                        # Need grid width from DEM
                        import rasterio
                        with rasterio.open(dem_path) as src:
                            grid_width = src.width
                        cell_num = row * grid_width + col + 1
                    else:
                        cell_num = int(pt)
                    writer.writerow([1, cell_num])

            # --- Build Cell2Fire command ---
            cmd = [
                self._binary_path,
                "--input-instance-folder", str(work_dir),
                "--output-folder", str(output_dir),
                "--nsims", str(n_sims),
                "--fire-period-length", str(period_hr),
                "--weather", str(weather_path),
                "--ignitions",
                "--IgnitionFile", str(ignition_file),
                "--sim", "S",
                "--finalGrid",
                "--output-messages",
            ]

            if self._sim_params.get("output_grid", True):
                cmd.append("--grids")

            logger.info("Running Cell2Fire: %d sims, %.1f hr periods", n_sims, period_hr)
            logger.debug("Command: %s", " ".join(cmd))

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour max
            )

            if proc.returncode != 0:
                raise Cell2FireError(
                    f"Cell2Fire exited with code {proc.returncode}.\n"
                    f"STDERR: {proc.stderr[:2000]}"
                )

            logger.info("Cell2Fire completed successfully")

            # --- Parse output ---
            burn_prob = parse_burn_probability(output_dir, n_sims)
            self._last_burn_prob = burn_prob

            # Store transform for later GeoJSON export
            import rasterio
            with rasterio.open(dem_path) as src:
                self._last_transform = src.transform

            # --- Map burn probability back to input cells ---
            result = self._map_burn_to_cells(X, burn_prob, dem_path)
            return result

        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(work_dir)
            except OSError as e:
                logger.warning("Failed to clean up %s: %s", work_dir, e)

    def validate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Validate spread predictions against actual fire perimeters.

        For Cell2Fire, validation means computing the Dice coefficient
        between the predicted burn area and actual burn area (from
        CAL FIRE FRAP perimeters or FIRMS detections).

        The ``y`` Series should contain binary labels: 1 = cell actually
        burned, 0 = did not burn. These come from intersecting historical
        fire perimeters with the H3 grid.

        Parameters
        ----------
        X : pd.DataFrame
            Same feature DataFrame used for predict().
        y : pd.Series
            Ground truth burn labels (0/1) per cell.

        Returns
        -------
        dict with: dice_coefficient, auc_pr, f1, fnr, n_samples, etc.
        """
        from src.validation.metrics import compute_all_metrics

        predictions = self.predict(X)
        y_prob = predictions["probability"].values
        y_pred = predictions["prediction"].values
        y_true = np.asarray(y)

        # Dice coefficient (primary metric for spread models)
        dice = compute_dice_coefficient(y_pred, y_true)

        # Standard metrics for pipeline compatibility
        all_metrics = compute_all_metrics(y_true, y_prob)
        all_metrics["dice_coefficient"] = dice

        # Check against threshold
        obj2_val = self._obj2_config.get("cell2fire", {}).get("validation", {})
        min_dice = obj2_val.get("minimum_dice", 0.50)
        all_metrics["dice_gate_passed"] = dice >= min_dice

        logger.info(
            "Cell2Fire validation — Dice: %.4f (threshold: %.2f), "
            "AUC-PR: %.4f, F1: %.4f",
            dice, min_dice,
            all_metrics.get("auc_pr", 0.0),
            all_metrics.get("f1", 0.0),
        )
        return all_metrics

    def explain(self, X: pd.DataFrame) -> dict[str, Any]:
        """Parameter sensitivity analysis for the Cell2Fire simulator.

        Since Cell2Fire is physics-based (no learned weights), SHAP doesn't
        apply. Instead, we run a parameter sweep and measure how each
        parameter affects the burn area, providing analogous insight into
        which inputs most influence the output.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame (used as the base scenario).

        Returns
        -------
        dict with:
            - parameter_sensitivity: {param: {value: burn_area_fraction}}
            - most_influential: name of the most influential parameter
            - sweep_results: full list of (params, burn_fraction) tuples
        """
        if not self._is_loaded:
            raise Cell2FireError("Call load_model() before explain()")

        sweep_space = (
            self._obj2_config.get("cell2fire", {})
            .get("sweep_space", {})
        )

        if not sweep_space:
            logger.warning("No sweep_space configured — returning empty sensitivity")
            return {"parameter_sensitivity": {}, "most_influential": None}

        sensitivity: dict[str, dict[str, float]] = {}
        all_results: list[dict[str, Any]] = []

        base_params = dict(self._sim_params)

        for param_name, values in sweep_space.items():
            sensitivity[param_name] = {}
            for val in values:
                # Override just this one parameter
                test_params = dict(base_params)
                test_params[param_name] = val

                # Store and run
                original_params = self._sim_params
                self._sim_params = test_params
                try:
                    preds = self.predict(X)
                    burn_frac = float(preds["prediction"].mean())
                except Exception as e:
                    logger.warning(
                        "Sweep %s=%s failed: %s", param_name, val, e
                    )
                    burn_frac = float("nan")
                finally:
                    self._sim_params = original_params

                sensitivity[param_name][str(val)] = burn_frac
                all_results.append({
                    "parameter": param_name,
                    "value": val,
                    "burn_area_fraction": burn_frac,
                })

        # Find most influential: parameter with largest range of burn_frac
        max_range = 0.0
        most_influential = None
        for param, vals in sensitivity.items():
            fracs = [v for v in vals.values() if not np.isnan(v)]
            if len(fracs) >= 2:
                param_range = max(fracs) - min(fracs)
                if param_range > max_range:
                    max_range = param_range
                    most_influential = param

        logger.info("Most influential parameter: %s (range: %.4f)", most_influential, max_range)

        return {
            "parameter_sensitivity": sensitivity,
            "most_influential": most_influential,
            "sweep_results": all_results,
        }

    # ── Private helpers ──────────────────────────────────────────────

    def _resolve_raster(
        self,
        raster_key: str,
        work_dir: Path,
    ) -> Path:
        """Resolve a raster path from config or clip from default location.

        Checks simulation config first (pre-clipped rasters), then falls
        back to clipping from the paths in model_config.yaml.
        """
        # Check simulation-specific config
        raster_paths = self._config.get("raster_paths", {})
        if raster_key in raster_paths:
            p = Path(raster_paths[raster_key])
            if p.exists():
                return p
            raise Cell2FireError(f"Configured raster not found: {p}")

        # Fall back to default raster locations + clip to AOI
        raster_inputs = (
            self._obj2_config.get("cell2fire", {})
            .get("raster_inputs", {})
        )
        default_dir = raster_inputs.get(raster_key)
        if default_dir is None:
            raise Cell2FireError(
                f"No raster path for '{raster_key}' in simulation config "
                f"or model_config.yaml"
            )

        # Find the actual GeoTIFF
        default_dir = Path(default_dir)
        tif_files = list(default_dir.glob("*.tif")) + list(default_dir.glob("*.tiff"))
        if not tif_files:
            raise Cell2FireError(f"No GeoTIFF files in {default_dir}")

        src_raster = tif_files[0]

        # Clip to AOI if bounds are specified
        bounds = self._config.get("aoi_bounds")
        if bounds:
            clipped = work_dir / f"{raster_key}_clipped.tif"
            target_res = self._sim_params.get("cell_size_m")
            return clip_raster_to_aoi(src_raster, tuple(bounds), clipped, target_res)

        return src_raster

    def _map_burn_to_cells(
        self,
        X: pd.DataFrame,
        burn_prob: np.ndarray,
        dem_path: str | Path,
    ) -> pd.DataFrame:
        """Map 2D burn probability grid back to input DataFrame rows.

        If X contains h3_index, converts each cell centroid to pixel
        coordinates in the DEM grid and samples the burn probability.
        Otherwise, flattens the grid and returns one row per pixel.
        """
        import rasterio

        # Use 0.5 as the burn/no-burn decision threshold
        burn_threshold = 0.5

        if "h3_index" in X.columns:
            import h3

            probabilities = []
            with rasterio.open(dem_path) as src:
                for cell_id in X["h3_index"]:
                    lat, lng = h3.cell_to_latlng(cell_id)
                    try:
                        row_idx, col_idx = src.index(lng, lat)
                        if (0 <= row_idx < burn_prob.shape[0]
                                and 0 <= col_idx < burn_prob.shape[1]):
                            prob = float(burn_prob[row_idx, col_idx])
                        else:
                            prob = 0.0
                    except Exception:
                        prob = 0.0
                    probabilities.append(prob)

            return pd.DataFrame({
                "prediction": [int(p >= burn_threshold) for p in probabilities],
                "probability": probabilities,
            })
        else:
            # Flatten grid — one row per pixel
            flat_prob = burn_prob.ravel()
            return pd.DataFrame({
                "prediction": (flat_prob >= burn_threshold).astype(int),
                "probability": flat_prob,
            })
