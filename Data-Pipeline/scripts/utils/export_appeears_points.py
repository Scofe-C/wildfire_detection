"""
Generate an AppEEARS Point Sample upload CSV from the H3 master grid.

Usage:
    python -m scripts.utils.export_appeears_points \
        --resolution-km 64 \
        --output data/static/appeears_upload.csv

Then in AppEEARS (appeears.earthdatacloud.nasa.gov):
  1. Start > Point Sample
  2. Upload the generated CSV
  3. Select: MOD13A2.061 → "1 km 16 days NDVI"
  4. Set date range (e.g. 2024-05-01 → 2024-08-31)
  5. Submit → wait for email → Download the results CSV
  6. Run:
       python -m scripts.ingestion.ingest_ndvi \
           --appeears-csv <downloaded_results.csv> \
           --resolution-km 64 \
           --output-dir data/static
"""

import argparse
import sys
from pathlib import Path

# Allow running as script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from scripts.utils.grid_utils import generate_full_grid


def export_appeears_csv(resolution_km: int = 64, output_path: str = "data/static/appeears_upload.csv") -> Path:
    """Write AppEEARS-compatible point sample CSV from the H3 master grid.

    AppEEARS expects columns: ID, Latitude, Longitude, Start, End
    We use grid_id as ID so the results map back to H3 cells directly.
    """
    grid = generate_full_grid(resolution_km)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "ID":        grid["grid_id"].astype(str),
        "Latitude":  grid["latitude"].round(6),
        "Longitude": grid["longitude"].round(6),
        "Start":     "01-01-2024",   # AppEEARS date format: MM-DD-YYYY
        "End":       "08-31-2024",
        "Category":  grid["region"],
    })

    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} points to {out}")
    print("\nNext steps:")
    print("  1. Go to https://appeears.earthdatacloud.nasa.gov/")
    print("  2. Sign in with your NASA Earthdata account")
    print("  3. Click 'Start' → 'Point Sample'")
    print("  4. Upload:", out.resolve())
    print("  5. Product: MOD13A2.061, Layer: '1 km 16 days NDVI'")
    print("  6. Date range: 01-01-2024 → 08-31-2024")
    print("  7. Submit, wait for email, download the results CSV")
    print("  8. Run:")
    print(f"       python -m scripts.ingestion.ingest_ndvi \\")
    print(f"           --appeears-csv <downloaded_results.csv> \\")
    print(f"           --resolution-km {resolution_km} \\")
    print(f"           --output-dir data/static")
    return out


def export_appeears_geojson(output_path: str = "data/static/appeears_area.geojson") -> Path:
    """Write a GeoJSON polygon covering CA + TX for AppEEARS Area Sample upload.

    AppEEARS Area Sample accepts GeoJSON and returns GeoTIFF files (no HDF4
    needed), which can be read directly by rasterio.

    The polygon is a simple bounding box union of both regions.
    """
    import json

    # CA bbox: W, S, E, N
    CA = (-124.48, 32.53, -114.13, 42.01)
    TX = (-106.65, 25.84, -93.51, 36.50)

    # Combined bounding box
    west  = min(CA[0], TX[0])
    south = min(CA[1], TX[1])
    east  = max(CA[2], TX[2])
    north = max(CA[3], TX[3])

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "california_texas_combined"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [west,  south],
                        [east,  south],
                        [east,  north],
                        [west,  north],
                        [west,  south],
                    ]],
                },
            }
        ],
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"Wrote area polygon to {out}")
    print("\nAppEEARS Area Sample steps:")
    print("  1. Go to https://appeears.earthdatacloud.nasa.gov/")
    print("  2. Start → Area Sample")
    print("  3. Upload:", out.resolve())
    print("  4. Product: MOD13A2.061, Layer: '1 km 16 days NDVI'")
    print("  5. Date range: 01-01-2024 → 08-31-2024")
    print("  6. Submit → wait for email → download the GeoTIFF zip")
    print("  7. Extract GeoTIFFs to: data/static/ndvi_raw/")
    print("  8. Run:")
    print("       python -m scripts.ingestion.ingest_ndvi \\")
    print("           --resolution-km 64 \\")
    print("           --output-dir data/static \\")
    print("           --force-rebuild")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export AppEEARS upload files (point CSV or area GeoJSON)"
    )
    parser.add_argument("--resolution-km", type=int, default=64)
    parser.add_argument("--output", default="data/static/appeears_upload.csv")
    parser.add_argument("--geojson", action="store_true",
                        help="Export area polygon GeoJSON instead of point CSV")
    parser.add_argument("--geojson-output", default="data/static/appeears_area.geojson")
    args = parser.parse_args()

    if args.geojson:
        export_appeears_geojson(args.geojson_output)
    else:
        export_appeears_csv(args.resolution_km, args.output)
