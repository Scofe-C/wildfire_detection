"""
Cell2FireSpread — BaseModel implementation for fire spread simulation.

This module contains only the Cell2FireSpread class.
Heavy utilities are in sibling modules:
    weather.py    — format_weather_csv
    raster.py     — clip_raster_to_aoi, parse_burn_probability
    evaluation.py — compute_dice_coefficient, compute_buffered_iou, find_best_threshold
    exceptions.py — Cell2FireError, Cell2FireNotInstalledError, load_obj2_config
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

from src.models.base import BaseModel

from .evaluation import compute_dice_coefficient, find_best_threshold
from .exceptions import Cell2FireError, load_obj2_config
from .raster import clip_raster_to_aoi, parse_burn_probability
from .weather import format_weather_csv

logger = logging.getLogger(__name__)


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

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def load_model(self, model_path: str | Path) -> None:
        """Load simulation configuration.

        Parameters
        ----------
        model_path : str | Path
            Path to simulation config JSON, or directory containing one.
        """
        model_path = Path(model_path)
        self._obj2_config = load_obj2_config()
        c2f_cfg = self._obj2_config["cell2fire"]

        self._binary_path = c2f_cfg.get("binary_path", "Cell2Fire")
        self._sim_params = dict(c2f_cfg.get("default_params", {}))

        if model_path.is_dir():
            config_file = model_path / "simulation_config.json"
            if not config_file.exists():
                raise FileNotFoundError(f"No simulation_config.json in {model_path}")
        elif model_path.suffix == ".json":
            config_file = model_path
        else:
            raise Cell2FireError(f"Expected .json file or directory, got: {model_path}")

        with open(config_file) as f:
            self._config = json.load(f)

        if "params" in self._config:
            self._sim_params.update(self._config["params"])

        if shutil.which(self._binary_path) is None:
            logger.warning(
                "Cell2Fire binary not found at '%s'. "
                "Simulation will fail unless installed before predict().",
                self._binary_path,
            )

        self._is_loaded = True
        logger.info(
            "Cell2Fire config loaded: %d simulations, %.1f hr periods, "
            "%d ignition points",
            self._sim_params.get("n_simulations", 100),
            self._sim_params.get("fire_period_length_hr", 1.0),
            len(self._config.get("ignition_points", [])),
        )

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Run Cell2Fire simulation and return burn predictions per cell.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame with h3_index and weather columns.

        Returns
        -------
        pd.DataFrame
            Columns: prediction (0/1), probability [0, 1].
        """
        if not self._is_loaded:
            raise Cell2FireError("Call load_model() before predict()")

        n_sims = self._sim_params.get("n_simulations", 100)
        period_hr = self._sim_params.get("fire_period_length_hr", 1.0)

        work_dir = Path(tempfile.mkdtemp(prefix="cell2fire_"))
        output_dir = work_dir / "output"
        # Resolve spain_lookup_table.csv relative to this file's package directory
        _pkg_data = Path(__file__).parent / "data" / "spain_lookup_table.csv"
        if not _pkg_data.exists():
            raise Cell2FireError(
                f"spain_lookup_table.csv not found at {_pkg_data}. "
                "Place it under model-pipeline/src/models/obj2_spread/data/"
            )
        shutil.copy(_pkg_data, work_dir / "spain_lookup_table.csv")
        output_dir.mkdir()

        try:
            # Prepare weather CSV
            weather_path = work_dir / "weather.csv"
            if "weather_csv" in self._config:
                shutil.copy(self._config["weather_csv"], weather_path)
            else:
                format_weather_csv(X, weather_path)

            # Resolve rasters
            dem_path = self._resolve_raster("dem", work_dir)
            self._resolve_raster("fuel", work_dir)

            # Build ignition file
            ignition_points = self._config.get("ignition_points", [])
            ignition_file = work_dir / "ignitions.csv"
            with open(ignition_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Year", "Cell"])
                for pt in ignition_points:
                    if isinstance(pt, (list, tuple)) and len(pt) == 2:
                        import rasterio
                        row, col = pt
                        with rasterio.open(dem_path) as src:
                            grid_width = src.width
                        cell_num = row * grid_width + col + 1
                    else:
                        cell_num = int(pt)
                    writer.writerow([1, cell_num])

            # Build command
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
                "--grids",
                "--output-messages",
            ]

            logger.info(
                "Running Cell2Fire: %d sims, %.1f hr periods", n_sims, period_hr
            )

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
            )

            if proc.returncode != 0:
                raise Cell2FireError(
                    f"Cell2Fire exited with code {proc.returncode}.\n"
                    f"STDERR: {proc.stderr[:2000]}"
                )

            logger.info("Cell2Fire completed successfully")

            # Parse output
            burn_prob = parse_burn_probability(output_dir, n_sims)
            self._last_burn_prob = burn_prob

            import rasterio
            with rasterio.open(dem_path) as src:
                self._last_transform = src.transform

            return self._map_burn_to_cells(X, burn_prob, dem_path)

        finally:
            try:
                shutil.rmtree(work_dir)
            except OSError as e:
                logger.warning("Failed to clean up %s: %s", work_dir, e)

    def validate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """Validate spread predictions against actual fire perimeters.

        Primary metric  : Buffered IoU (15% buffer, threshold sweep)
        Secondary metric: Dice coefficient (legacy)
        Supporting      : Directional accuracy, area ratio

        Gate passes if: buffered_iou >= 0.35 AND directional_accuracy
                        AND area_ratio in [0.70, 1.30]

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame with h3_index column.
        y : pd.Series
            Ground truth burn labels (0=no burn, 1=burned) per cell.

        Returns
        -------
        dict with buffered_iou, directional_accuracy, area_ratio,
                    gate_passed, dice_coefficient (legacy), auc_pr, f1
        """
        import geopandas as gpd
        import h3
        from shapely.geometry import Point
        from shapely.ops import unary_union

        from src.validation.metrics import compute_all_metrics

        predictions = self.predict(X)
        y_prob = predictions["probability"].values
        y_pred = predictions["prediction"].values
        y_true = np.asarray(y)

        # Legacy Dice
        dice = compute_dice_coefficient(y_pred, y_true)
        all_metrics = compute_all_metrics(y_true, y_prob)
        all_metrics["dice_coefficient"] = dice

        # Buffered evaluation
        try:
            burned_indices = np.where(y_true == 1)[0]

            if len(burned_indices) == 0:
                logger.warning("No burned cells in ground truth — skipping buffered eval")
                all_metrics.update({
                    "buffered_iou": 0.0,
                    "directional_accuracy": False,
                    "area_ratio": 0.0,
                    "gate_passed": False,
                })
            else:
                # Build actual perimeter GeoDataFrame from burned H3 cells
                if "h3_index" in X.columns:
                    from shapely.geometry import Polygon
                    burned_h3 = X.iloc[burned_indices]["h3_index"].values
                    polygons = []
                    for cell in burned_h3:
                        try:
                            boundary = h3.cell_to_boundary(cell)
                            polygons.append(
                                Polygon([(lon, lat) for lat, lon in boundary])
                            )
                        except Exception:
                            continue
                    actual_gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
                else:
                    lats = X.iloc[burned_indices]["latitude"].values
                    lons = X.iloc[burned_indices]["longitude"].values
                    actual_gdf = gpd.GeoDataFrame(
                        geometry=[
                            Point(lon, lat).buffer(0.005)
                            for lat, lon in zip(lats, lons, strict=False)
                        ],
                        crs="EPSG:4326",
                    )

                if self._last_burn_prob is not None and self._last_transform is not None:
                    logger.info("Running buffered threshold sweep...")
                    best = find_best_threshold(
                        self._last_burn_prob,
                        actual_gdf,
                        self._last_transform,
                    )
                    all_metrics.update(best)
                else:
                    # Cell-level fallback when no 2D grid available
                    logger.warning(
                        "No 2D burn probability grid — using cell-level fallback"
                    )
                    pred_burned = np.where(y_prob >= 0.10)[0]
                    if "h3_index" in X.columns:
                        from shapely.geometry import Polygon
                        pred_h3 = X.iloc[pred_burned]["h3_index"].values
                        pred_polygons = []
                        for cell in pred_h3:
                            try:
                                boundary = h3.cell_to_boundary(cell)
                                pred_polygons.append(
                                    Polygon([(lon, lat) for lat, lon in boundary])
                                )
                            except Exception:
                                continue
                        if pred_polygons:
                            import math
                            pred_union = unary_union(pred_polygons)
                            actual_union = unary_union(actual_gdf.geometry.values)
                            buf = 0.15 * math.sqrt(pred_union.area)
                            pred_buf = pred_union.buffer(buf)
                            inter = pred_buf.intersection(actual_union).area
                            union_area = pred_buf.union(actual_union).area
                            biou = inter / union_area if union_area > 0 else 0.0
                            ratio = (
                                pred_union.area / actual_union.area
                                if actual_union.area > 0 else 0.0
                            )
                            all_metrics.update({
                                "buffered_iou": round(biou, 4),
                                "area_ratio": round(ratio, 3),
                                "area_ratio_ok": 0.70 <= ratio <= 1.30,
                                "gate_passed": biou >= 0.35 and 0.70 <= ratio <= 1.30,
                            })

        except Exception as exc:
            logger.warning("Buffered evaluation failed: %s", exc, exc_info=True)
            all_metrics.update({"buffered_iou": 0.0, "gate_passed": False})

        logger.info(
            "Validation — Buffered IoU: %.4f | Dir: %s | Area ratio: %.3f | "
            "Gate: %s | Dice (legacy): %.4f",
            all_metrics.get("buffered_iou", 0.0),
            all_metrics.get("directional_accuracy", "N/A"),
            all_metrics.get("area_ratio", 0.0),
            "PASS" if all_metrics.get("gate_passed") else "FAIL",
            dice,
        )
        return all_metrics

    def explain(self, X: pd.DataFrame) -> dict[str, Any]:
        """Parameter sensitivity analysis (replaces SHAP for physics model).

        Varies one simulation parameter at a time and measures how
        burn area fraction changes — analogous to feature importance.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame (base scenario).

        Returns
        -------
        dict with parameter_sensitivity, most_influential, sweep_results.
        """
        if not self._is_loaded:
            raise Cell2FireError("Call load_model() before explain()")

        sweep_space = (
            self._obj2_config.get("cell2fire", {}).get("sweep_space", {})
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
                test_params = {**base_params, param_name: val}
                self._sim_params = test_params
                try:
                    preds = self.predict(X)
                    burn_frac = float(preds["prediction"].mean())
                except Exception as e:
                    logger.warning("Sweep %s=%s failed: %s", param_name, val, e)
                    burn_frac = float("nan")
                finally:
                    self._sim_params = base_params

                sensitivity[param_name][str(val)] = burn_frac
                all_results.append({
                    "parameter": param_name,
                    "value": val,
                    "burn_area_fraction": burn_frac,
                })

        # Most influential = largest range of burn_frac across values
        most_influential = None
        max_range = 0.0
        for param, vals in sensitivity.items():
            fracs = [v for v in vals.values() if not np.isnan(v)]
            if len(fracs) >= 2:
                param_range = max(fracs) - min(fracs)
                if param_range > max_range:
                    max_range = param_range
                    most_influential = param

        logger.info(
            "Most influential parameter: %s (range: %.4f)",
            most_influential, max_range,
        )
        return {
            "parameter_sensitivity": sensitivity,
            "most_influential": most_influential,
            "sweep_results": all_results,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_raster(self, raster_key: str, work_dir: Path) -> Path:
        """Resolve raster path from simulation config or model_config defaults."""
        raster_paths = self._config.get("raster_paths", {})
        if raster_key in raster_paths:
            p = Path(raster_paths[raster_key])
            if p.exists():
                return p
            raise Cell2FireError(f"Configured raster not found: {p}")

        raster_inputs = (
            self._obj2_config.get("cell2fire", {}).get("raster_inputs", {})
        )
        default_dir = raster_inputs.get(raster_key)
        if default_dir is None:
            raise Cell2FireError(
                f"No raster path for '{raster_key}' in simulation config "
                f"or model_config.yaml"
            )

        default_dir = Path(default_dir)
        tif_files = list(default_dir.glob("*.tif")) + list(default_dir.glob("*.tiff"))
        if not tif_files:
            raise Cell2FireError(f"No GeoTIFF files in {default_dir}")

        src_raster = tif_files[0]
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
        """Map 2D burn probability grid back to input DataFrame rows."""
        import h3
        import rasterio

        burn_threshold = 0.5

        if "h3_index" in X.columns:
            probabilities = []
            with rasterio.open(dem_path) as src:
                for cell_id in X["h3_index"]:
                    lat, lng = h3.cell_to_latlng(cell_id)
                    try:
                        row_idx, col_idx = src.index(lng, lat)
                        if (
                            0 <= row_idx < burn_prob.shape[0]
                            and 0 <= col_idx < burn_prob.shape[1]
                        ):
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
            flat_prob = burn_prob.ravel()
            return pd.DataFrame({
                "prediction": (flat_prob >= burn_threshold).astype(int),
                "probability": flat_prob,
            })
