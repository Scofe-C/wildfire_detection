"""
Fire Detector — Four-Gate False Alarm Prevention
=================================================
Applies four sequential gates to raw GOES NRT detections before
escalating to the full pipeline. A detection must pass ALL gates.

Gates (applied in order):
  G1: Spatial corroboration  — ≥2 neighboring H3 cells also show fire
  G2: Temporal persistence   — detection in ≥2 consecutive GOES scans
  G3: Multi-source cross-ref — VIIRS confirmation OR high FRP bypass
  G4: Industrial exclusion   — not a known industrial heat source

Design principles:
  - Each gate failure writes an audit record to GCS
  - Gates are independent — a G2 failure doesn't re-run G1
  - All H3 operations use the compat wrappers from grid_utils
  - Industrial sources are loaded from GCS at call time (updateable without deploy)
"""

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# H3 compat helpers (inline copies to keep this module Cloud Function safe)
# Avoids importing grid_utils which pulls in heavy geospatial deps)
# ---------------------------------------------------------------------------

def _latlng_to_cell(lat: float, lon: float, res: int) -> str:
    """h3-version-safe lat/lon → cell conversion."""
    import h3
    if hasattr(h3, "geo_to_h3"):
        return h3.geo_to_h3(lat, lon, res)
    return h3.latlng_to_cell(lat, lon, res)


def _grid_disk(cell_id: str, k: int) -> set:
    """h3-version-safe k-ring / grid_disk."""
    import h3
    if hasattr(h3, "grid_disk"):
        return set(h3.grid_disk(cell_id, k))
    return set(h3.k_ring(cell_id, k))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class FireDetector:
    """Four-gate false alarm filter for GOES NRT detections.

    Usage:
        detector = FireDetector(config, state, gcs_state_module)
        result = detector.evaluate(detections, region)
    """

    def __init__(self, watchdog_config: dict, state: dict, gcs_state=None):
        """
        Args:
            watchdog_config: The 'watchdog' section from schema_config.yaml.
            state: Current watchdog state dict (from gcs_state.read_state()).
            gcs_state: The gcs_state module (injected to allow testing without GCS).
        """
        self.cfg = watchdog_config
        self.fa_cfg = watchdog_config.get("false_alarm", {})
        self.state = state
        self.gcs_state = gcs_state

        # H3 resolution for corroboration checks — use detection res (res 5)
        # We always corroborate at res 5 regardless of current pipeline resolution
        self._h3_res = 5

    def evaluate(
        self,
        detections: list[dict],
        region: str,
        previous_scan_detections: Optional[list[dict]] = None,
        industrial_sources: Optional[list[dict]] = None,
    ) -> dict:
        """Run all four gates on a set of detections.

        Args:
            detections: Current scan detections from ingest_goes.
            region: 'california' or 'texas'.
            previous_scan_detections: Detections from the immediately prior scan
                (needed for G2 temporal persistence).
            industrial_sources: List of industrial heat source dicts from GCS.

        Returns:
            {
                "confirmed": bool,
                "gate_failed": None | "G1" | "G2" | "G3" | "G4",
                "fire_cells": [h3_cell_id, ...],    # confirmed fire H3 cells
                "max_frp": float,
                "detection_summary": {...}
            }
        """
        result = {
            "confirmed": False,
            "gate_failed": None,
            "fire_cells": [],
            "max_frp": 0.0,
            "detection_summary": {
                "raw_count": len(detections),
                "region": region,
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
            },
        }

        if not detections:
            result["gate_failed"] = "no_detections"
            return result

        # Map detections to H3 cells at res 5
        candidate_cells = self._map_to_h3_cells(detections)
        result["detection_summary"]["candidate_cells"] = len(candidate_cells)

        # ------------------------------------------------------------------
        # G1: Spatial corroboration
        # ------------------------------------------------------------------
        g1_cells = self._gate1_spatial(candidate_cells)
        if not g1_cells:
            result["gate_failed"] = "G1"
            self._write_fa_record(result, detections, "G1_spatial_isolated")
            logger.info(f"[{region}] G1 FAILED: no spatially corroborated cells")
            return result
        logger.info(f"[{region}] G1 passed: {len(g1_cells)} spatially corroborated cells")

        # ------------------------------------------------------------------
        # G2: Temporal persistence
        # ------------------------------------------------------------------
        if previous_scan_detections is not None:
            g2_cells = self._gate2_temporal(g1_cells, previous_scan_detections)
            if not g2_cells:
                # Don't fail hard — update consecutive scan counter and wait
                consecutive = self.state.get("consecutive_fire_scans", 0)
                min_scans = self.fa_cfg.get("min_consecutive_scans", 2)
                if consecutive + 1 < min_scans:
                    result["gate_failed"] = "G2_pending"
                    logger.info(
                        f"[{region}] G2 pending: scan {consecutive + 1}/{min_scans} "
                        f"— waiting for next scan to confirm persistence"
                    )
                    return result
                g2_cells = g1_cells  # enough scans counted even without prior data match
            logger.info(f"[{region}] G2 passed: {len(g2_cells)} temporally persistent cells")
        else:
            # No prior scan data available — skip G2 if FRP is very high
            max_frp = max((d.get("frp") or 0 for d in detections), default=0)
            bypass_frp = self.fa_cfg.get("viirs_bypass_frp_mw", 50.0)
            if max_frp >= bypass_frp * 4:  # 4x bypass for G2 (200 MW+ = obvious fire)
                g2_cells = g1_cells
                logger.info(f"[{region}] G2 bypassed: FRP={max_frp:.0f} MW >> threshold")
            else:
                result["gate_failed"] = "G2_no_prior_scan"
                logger.info(f"[{region}] G2 skipped: no prior scan data for persistence check")
                return result

        # ------------------------------------------------------------------
        # G3: Multi-source cross-reference (VIIRS confirmation)
        # ------------------------------------------------------------------
        max_frp = max((d.get("frp") or 0 for d in detections), default=0)
        bypass_frp = self.fa_cfg.get("viirs_bypass_frp_mw", 50.0)

        if max_frp >= bypass_frp:
            logger.info(
                f"[{region}] G3 bypassed: FRP={max_frp:.0f} MW >= {bypass_frp} MW threshold"
            )
            g3_cells = g2_cells
        else:
            g3_cells = self._gate3_viirs(g2_cells, region)
            if not g3_cells:
                result["gate_failed"] = "G3"
                self._write_fa_record(result, detections, "G3_no_viirs_confirmation")
                logger.info(f"[{region}] G3 FAILED: no VIIRS confirmation and FRP < bypass threshold")
                return result
            logger.info(f"[{region}] G3 passed: VIIRS confirmed {len(g3_cells)} cells")

        # ------------------------------------------------------------------
        # G4: Industrial heat source exclusion
        # ------------------------------------------------------------------
        if industrial_sources:
            g4_cells = self._gate4_industrial(g3_cells, detections, industrial_sources)
            if not g4_cells:
                result["gate_failed"] = "G4"
                self._write_fa_record(result, detections, "G4_industrial_source")
                logger.info(f"[{region}] G4 FAILED: all detections match industrial sources")
                return result
            logger.info(
                f"[{region}] G4 passed: {len(g4_cells)} cells after industrial exclusion "
                f"({len(g3_cells) - len(g4_cells)} industrial cells removed)"
            )
        else:
            g4_cells = g3_cells
            logger.info(f"[{region}] G4 skipped: no industrial sources configured")

        # All gates passed
        result["confirmed"] = True
        result["fire_cells"] = list(g4_cells)
        result["max_frp"] = max_frp
        result["detection_summary"]["confirmed_cells"] = len(g4_cells)
        logger.info(
            f"[{region}] ✓ All gates passed: {len(g4_cells)} confirmed fire cells, "
            f"max FRP={max_frp:.0f} MW"
        )
        return result

    # ------------------------------------------------------------------
    # Gate implementations
    # ------------------------------------------------------------------

    def _map_to_h3_cells(self, detections: list[dict]) -> dict:
        """Map detections to H3 cells at res 5.

        Returns dict: {h3_cell_id: [detection, ...]} for all candidate cells.
        """
        cell_map = {}
        for det in detections:
            try:
                cell_id = _latlng_to_cell(det["lat"], det["lon"], self._h3_res)
                cell_map.setdefault(cell_id, []).append(det)
            except Exception as e:
                logger.debug(f"Could not map detection to H3 cell: {e}")
        return cell_map

    def _gate1_spatial(self, candidate_cells: dict) -> dict:
        """G1: Keep only cells that have ≥N neighbors also showing fire.

        A single isolated pixel is almost certainly noise (sun glint,
        industrial emission, sensor artifact). Wildfire always produces
        detections in a cluster of adjacent cells.
        """
        min_neighbors = self.fa_cfg.get("min_neighbor_detections", 2)
        confirmed = {}

        cell_ids = set(candidate_cells.keys())
        for cell_id in cell_ids:
            # Get ring-1 neighbors (immediate ring, ~5 km at res 5)
            neighbors = _grid_disk(cell_id, 1) - {cell_id}
            neighbor_fire_count = len(neighbors.intersection(cell_ids))
            if neighbor_fire_count >= min_neighbors:
                confirmed[cell_id] = candidate_cells[cell_id]

        return confirmed

    def _gate2_temporal(
        self,
        g1_cells: dict,
        previous_scan_detections: list[dict],
    ) -> dict:
        """G2: Keep only cells that also appeared in the previous scan.

        Maps previous scan detections to H3 cells and finds overlap.
        """
        prev_cells = set(self._map_to_h3_cells(previous_scan_detections).keys())
        confirmed = {
            cell_id: dets
            for cell_id, dets in g1_cells.items()
            if cell_id in prev_cells
        }
        return confirmed

    def _gate3_viirs(self, g2_cells: dict, region: str) -> dict:
        """G3: Attempt VIIRS confirmation for low-FRP candidates.

        Queries FIRMS VIIRS_SNPP_NRT and VIIRS_NOAA20_NRT for overlapping
        detections within the last `viirs_lookback_hours`.
        Returns g2_cells unchanged if VIIRS API is unavailable — fail-open
        (we don't want to suppress real fires because VIIRS hasn't had a pass).
        """
        from scripts.ingestion.ingest_goes import fetch_goes_nrt_detections
        import os

        lookback_hours = self.fa_cfg.get("viirs_lookback_hours", 3)
        api_key = os.environ.get("FIRMS_MAP_KEY")

        if not api_key or not g2_cells:
            return g2_cells   # fail-open if no API key

        # Compute bounding box from confirmed cell centroids
        import h3
        lats, lons = [], []
        for cell_id in g2_cells:
            try:
                if hasattr(h3, "h3_to_geo"):
                    lat, lon = h3.h3_to_geo(cell_id)
                else:
                    lat, lon = h3.cell_to_latlng(cell_id)
                lats.append(lat)
                lons.append(lon)
            except Exception:
                continue

        if not lats:
            return g2_cells

        # Add 0.5 degree buffer
        bbox = [min(lons) - 0.5, min(lats) - 0.5, max(lons) + 0.5, max(lats) + 0.5]

        viirs_sources = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"]
        viirs_cells = set()

        for source in viirs_sources:
            try:
                # M3 fix: pass source directly instead of mutating the module
                # global, which is not thread-safe under concurrent execution.
                dets = fetch_goes_nrt_detections(
                    bbox=bbox,
                    lookback_minutes=lookback_hours * 60,
                    min_frp_mw=1.0,  # Lower threshold for VIIRS confirmation
                    api_key=api_key,
                    max_retries=2,
                    source=source,
                )

                for det in dets:
                    try:
                        cell_id = _latlng_to_cell(det["lat"], det["lon"], self._h3_res)
                        viirs_cells.add(cell_id)
                        # Also add ring-1 neighbors (GOES pixel ≈ 2km, VIIRS pixel ≈ 375m)
                        viirs_cells.update(_grid_disk(cell_id, 1))
                    except Exception:
                        continue
            except Exception as e:
                logger.warning(f"G3: VIIRS {source} query failed: {e}")

        if not viirs_cells:
            # No VIIRS data available — fail-open, return g2_cells as confirmed
            logger.warning("G3: No VIIRS data available — failing open (fire may be real)")
            return g2_cells

        confirmed = {
            cell_id: dets
            for cell_id, dets in g2_cells.items()
            if cell_id in viirs_cells
        }
        return confirmed

    def _gate4_industrial(
        self,
        g3_cells: dict,
        detections: list[dict],
        industrial_sources: list[dict],
    ) -> dict:
        """G4: Remove cells that overlap known industrial heat sources.

        Industrial sources are loaded from GCS (updateable without redeployment).
        A cell is excluded only if its centroid is within the source's radius_km.
        """
        import h3

        excluded_cells = set()

        for cell_id in g3_cells:
            try:
                if hasattr(h3, "h3_to_geo"):
                    cell_lat, cell_lon = h3.h3_to_geo(cell_id)
                else:
                    cell_lat, cell_lon = h3.cell_to_latlng(cell_id)
            except Exception:
                continue

            for source in industrial_sources:
                try:
                    dist = _haversine_km(
                        cell_lat, cell_lon,
                        float(source["lat"]), float(source["lon"])
                    )
                    radius = float(source.get("radius_km", 2.0))
                    if dist <= radius:
                        excluded_cells.add(cell_id)
                        logger.info(
                            f"G4: Cell {cell_id} excluded — matches industrial source "
                            f"'{source.get('name', 'unknown')}' ({dist:.1f} km)"
                        )
                        break
                except Exception:
                    continue

        return {
            cell_id: dets
            for cell_id, dets in g3_cells.items()
            if cell_id not in excluded_cells
        }

    def _write_fa_record(
        self,
        result: dict,
        detections: list[dict],
        reason: str,
    ) -> None:
        """Write false alarm audit record if gcs_state is available."""
        if self.gcs_state is None:
            return
        try:
            self.gcs_state.write_false_alarm_record(
                detection_data={
                    "raw_detections": len(detections),
                    "summary": result.get("detection_summary", {}),
                },
                gate_failed=reason,
            )
        except Exception as e:
            logger.warning(f"Failed to write FA record: {e}")
