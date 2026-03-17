from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)

NRI_DOWNLOAD_URL = (
    "https://hazards.fema.gov/nri/Content/StaticDocuments/DataDownload/"
    "NRI_Shapefile_CensusTracts/NRI_Shapefile_CensusTracts.zip"
)


class NRILoadError(Exception):
    pass


def load_nri(
    cache_dir: str | Path = "data/static/fema_nri",
    shapefile_name: str = "NRI_Shapefile_CensusTracts.shp",
) -> gpd.GeoDataFrame:
    cache_dir = Path(cache_dir)
    shp_path = cache_dir / shapefile_name

    if not shp_path.exists():
        shp_files = list(cache_dir.glob("*.shp"))
        if shp_files:
            shp_path = shp_files[0]
        else:
            raise NRILoadError(
                f"FEMA NRI shapefile not found in {cache_dir}. "
                f"Download from: {NRI_DOWNLOAD_URL}"
            )

    nri = gpd.read_file(shp_path)
    if "SOVI_SCORE" not in nri.columns:
        raise NRILoadError("NRI shapefile missing SOVI_SCORE column")

    logger.info("NRI loaded: %d census tracts", len(nri))
    return nri


def compute_vulnerability_quartiles(
    nri: gpd.GeoDataFrame,
    score_column: str = "SOVI_SCORE",
) -> gpd.GeoDataFrame:
    labels = ["Low", "Medium", "High", "Very High"]
    nri = nri.copy()
    nri["nri_vulnerability_quartile"] = pd.qcut(
        nri[score_column].rank(method="first"), q=4, labels=labels,
    )
    for label in labels:
        logger.info("  %s: %d tracts", label, (nri["nri_vulnerability_quartile"] == label).sum())
    return nri


def spatial_join_predictions(
    predictions: pd.DataFrame,
    nri: gpd.GeoDataFrame,
    h3_col: str = "h3_index",
) -> gpd.GeoDataFrame:
    import h3
    from shapely.geometry import Point

    points = []
    for cell_id in predictions[h3_col]:
        lat, lng = h3.cell_to_latlng(cell_id)
        points.append(Point(lng, lat))

    pred_gdf = gpd.GeoDataFrame(predictions, geometry=points, crs="EPSG:4326")

    if nri.crs != pred_gdf.crs:
        nri = nri.to_crs(pred_gdf.crs)

    joined = gpd.sjoin_nearest(
        pred_gdf,
        nri[["geometry", "nri_vulnerability_quartile", "SOVI_SCORE"]],
        how="left",
    )

    unmatched = joined["nri_vulnerability_quartile"].isna().sum()
    if unmatched > 0:
        logger.warning("%d predictions unmatched to NRI tracts", unmatched)
        joined["nri_vulnerability_quartile"] = joined["nri_vulnerability_quartile"].fillna("Unknown")

    logger.info("Spatial join complete: %d rows", len(joined))
    return joined
