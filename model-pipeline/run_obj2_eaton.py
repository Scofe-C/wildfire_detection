"""
End-to-end test for Cell2FireSpread on Eaton Fire Jan 7 2025.

Tests new features:
  - slope/aspect generation from DEM
  - wind gust substitution (Santa Ana events)
  - auto CRS reprojection of aoi_bounds
  - lat/lon ignition point auto-conversion

Run from model-pipeline root:
    cd <repo-root>/model-pipeline
    python run_obj2_eaton.py
"""
import sys
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from src.models.obj2_spread.cell2fire_spread import Cell2FireSpread

CONFIG = "configs/simulations/eaton_fire_config.json"


def make_weather_df() -> pd.DataFrame:
    """Simulated Santa Ana weather for Eaton Fire Jan 7 2025.

    NE offshore flow (Diablo-style), critically low RH, high gusts.
    wind_gusts_10m triggers gust substitution in weather.py where
    gust > mean wind speed — captures burst-driven spread.
    """
    hours = pd.date_range("2025-01-07 00:00", periods=24, freq="h")
    return pd.DataFrame({
        "timestamp":            hours,
        "wind_speed_10m":       [12.5, 13.2, 14.1, 15.3, 16.2, 17.5, 18.3, 19.1,
                                 20.2, 21.5, 22.3, 23.1, 22.5, 21.8, 21.2, 20.5,
                                 19.8, 18.9, 18.1, 17.2, 16.5, 15.8, 15.1, 14.5],
        "wind_direction_10m":   [45, 48, 50, 52, 50, 48, 45, 44,
                                 42, 40, 38, 36, 35, 36, 38, 40,
                                 42, 44, 45, 46, 47, 48, 49, 50],
        "wind_gusts_10m":       [22.0, 24.5, 26.0, 28.5, 31.0, 33.5, 35.0, 36.5,
                                 38.0, 40.5, 42.0, 43.5, 42.0, 40.5, 39.0, 38.0,
                                 36.5, 35.0, 33.5, 31.0, 29.5, 28.0, 26.5, 25.0],
        "temperature_2m":       [18.0, 17.5, 17.0, 16.5, 16.0, 15.5, 15.0, 15.5,
                                 16.5, 18.0, 19.5, 21.0, 22.5, 23.5, 24.0, 23.5,
                                 22.5, 21.0, 19.5, 18.5, 17.5, 17.0, 16.5, 16.0],
        "relative_humidity_2m": [8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 5.0,
                                 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0,
                                 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0],
    })


def main():
    log.info("=" * 60)
    log.info("Eaton Fire Simulation — Jan 7 2025")
    log.info("Testing: slope/aspect + wind gusts + CRS auto-reproject")
    log.info("=" * 60)

    # Step 1: Load model config
    log.info("Step 1: load_model(%s)", CONFIG)
    model = Cell2FireSpread()
    model.load_model(CONFIG)
    log.info("  ✅ loaded — binary: %s", model._binary_path)
    log.info("  ✅ sims: %d | period: %.1f hr",
             model._sim_params.get("n_simulations"),
             model._sim_params.get("fire_period_length_hr"))

    # Step 2: Run simulation
    log.info("Step 2: predict() — running %d simulations",
             model._sim_params.get("n_simulations", 50))
    weather_df = make_weather_df()

    # Log gust substitution info
    gust_exceeds = (weather_df["wind_gusts_10m"] > weather_df["wind_speed_10m"]).sum()
    log.info("  Wind gusts exceed mean ws in %d/%d time steps — gust substitution active",
             gust_exceeds, len(weather_df))

    predictions = model.predict(weather_df)
    burn_frac = 100 * predictions["prediction"].mean()
    log.info("  ✅ shape: %s | burn fraction: %.1f%%",
             predictions.shape, burn_frac)

    # Step 3: Results
    log.info("=" * 60)
    log.info("RESULTS")
    log.info("  Total cells:   %d", len(predictions))
    log.info("  Burned cells:  %d (%.1f%%)",
             predictions["prediction"].sum(), burn_frac)
    log.info("  Mean burn prob: %.3f", predictions["probability"].mean())
    log.info("  Max burn prob:  %.3f", predictions["probability"].max())
    log.info("=" * 60)

    print("\nSample output (first 10 rows):")
    print(predictions.head(10).to_string(index=False))

    return predictions


if __name__ == "__main__":
    main()      