import sys
import pandas as pd
from scripts.utils.grid_utils import generate_full_grid
sys.path.insert(0, "D:/NEU/wildfire_detection/Data-Pipeline")
# Check weather parquets
ca = pd.read_parquet(r"D:\NEU\wildfire_detection\Data-Pipeline\data\processed\weather\weather_features_california_latest.parquet")
tx = pd.read_parquet(r"D:\NEU\wildfire_detection\Data-Pipeline\data\processed\weather\weather_features_texas_latest.parquet")

print("=== CA weather ===")
print("rows:", len(ca))
print("null rates:\n", ca.isnull().mean().sort_values(ascending=False).to_string())

print("\n=== TX weather ===")
print("rows:", len(tx))
print("null rates:\n", tx.isnull().mean().sort_values(ascending=False).to_string())

# Check grid_id overlap
grid = generate_full_grid(64)
ca_master = set(grid[grid["region"] == "california"]["grid_id"])
tx_master = set(grid[grid["region"] == "texas"]["grid_id"])
ca_wx = set(ca["grid_id"].astype(str))
tx_wx = set(tx["grid_id"].astype(str))

print("\n=== grid_id overlap ===")
print(f"CA master: {len(ca_master)} cells, CA weather: {len(ca_wx)} cells, overlap: {len(ca_master & ca_wx)}")
print(f"TX master: {len(tx_master)} cells, TX weather: {len(tx_wx)} cells, overlap: {len(tx_master & tx_wx)}")
print("CA in master but NOT in weather:", ca_master - ca_wx)
