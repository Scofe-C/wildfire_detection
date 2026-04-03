"""
End-to-end test for Cell2FireSpread on Dixie Fire 2021.
Run from model-pipeline root:
    cd <repo-root>/model-pipeline
    python run_obj2_dixie.py
"""
import sys, logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from src.models.obj2_spread.cell2fire_spread import Cell2FireSpread

CONFIG = "configs/simulations/dixie_fire_2021.json"

def make_weather_df():
    hours = pd.date_range("2021-07-13 10:00", periods=10, freq="h")
    return pd.DataFrame({
        "timestamp":               hours,
        "wind_speed_10m":          [15.0] * 10,
        "wind_direction_10m":      [225.0] * 10,
        "temperature_2m":          [35.0] * 10,
        "relative_humidity_2m":    [15.0] * 10,
    })

def main():
    log.info("Step 1: load_model()")
    model = Cell2FireSpread()
    model.load_model(CONFIG)
    log.info("  ✅ loaded — binary: %s", model._binary_path)

    log.info("Step 2: predict()")
    predictions = model.predict(make_weather_df())
    log.info("  ✅ shape: %s  burn fraction: %.1f%%",
             predictions.shape, 100 * predictions["prediction"].mean())

    print("\nSample output:")
    print(predictions.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
