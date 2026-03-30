"""SMAP null handling — clean soil moisture grids at the ingestion boundary.

Architecture principle: Never propagate raw nulls downstream. Context builder
must only ever see clean values or explicit UNAVAILABLE markers.

Three statuses:
  - COMPLETE: all cells have valid values, pass through.
  - PARTIAL: some cells null → fill with regional mean, report coverage.
  - UNAVAILABLE: all cells null → return status dict, no crash.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def clean_smap_grid(raw_grid: dict[str, Any]) -> dict[str, Any]:
    """Clean a SMAP soil moisture grid, handling nulls at the boundary.

    Parameters
    ----------
    raw_grid:
        Raw SMAP grid dict with structure::

            {
                "cells": [
                    {"h3_index": "...", "soil_moisture_0_5cm": 0.23, "soil_moisture_0_100cm": 0.31},
                    {"h3_index": "...", "soil_moisture_0_5cm": None, "soil_moisture_0_100cm": None},
                    ...
                ],
                "timestamp": "2026-03-29T12:00:00Z",
                "source": "SMAP_L4"
            }

    Returns
    -------
    dict
        Cleaned grid with added ``smap_status`` metadata::

            {
                "cells": [...],          # cleaned cells (nulls filled or empty)
                "timestamp": "...",
                "source": "SMAP_L4",
                "smap_status": "COMPLETE" | "PARTIAL" | "UNAVAILABLE",
                "coverage_pct": 85.0,    # present if PARTIAL
                "null_cells": 3,         # present if PARTIAL
                "total_cells": 20,
                "regional_mean_0_5cm": 0.25,   # present if PARTIAL
                "regional_mean_0_100cm": 0.30, # present if PARTIAL
            }
    """
    cells = raw_grid.get("cells") or []
    total_cells = len(cells)

    if total_cells == 0:
        logger.warning("SMAP grid has no cells — marking UNAVAILABLE")
        return {
            **raw_grid,
            "cells": [],
            "smap_status": "UNAVAILABLE",
            "coverage_pct": 0.0,
            "null_cells": 0,
            "total_cells": 0,
        }

    # Count nulls and compute regional means from valid cells
    sm_fields = ["soil_moisture_0_5cm", "soil_moisture_0_100cm"]
    valid_values: dict[str, list[float]] = {f: [] for f in sm_fields}
    null_count = 0

    for cell in cells:
        is_null = all(cell.get(f) is None for f in sm_fields)
        if is_null:
            null_count += 1
        else:
            for f in sm_fields:
                val = cell.get(f)
                if val is not None:
                    valid_values[f].append(float(val))

    # All cells null → UNAVAILABLE
    if null_count == total_cells:
        logger.warning(
            "SMAP grid: all %d cells null — status UNAVAILABLE", total_cells,
        )
        return {
            **raw_grid,
            "cells": [],
            "smap_status": "UNAVAILABLE",
            "coverage_pct": 0.0,
            "null_cells": total_cells,
            "total_cells": total_cells,
        }

    # No nulls → COMPLETE
    if null_count == 0:
        return {
            **raw_grid,
            "smap_status": "COMPLETE",
            "coverage_pct": 100.0,
            "null_cells": 0,
            "total_cells": total_cells,
        }

    # Partial nulls → PARTIAL: fill with regional mean
    regional_means: dict[str, float] = {}
    for f in sm_fields:
        if valid_values[f]:
            regional_means[f] = sum(valid_values[f]) / len(valid_values[f])
        else:
            regional_means[f] = 0.0

    cleaned_cells = []
    for cell in cells:
        cleaned = dict(cell)
        for f in sm_fields:
            if cleaned.get(f) is None:
                cleaned[f] = round(regional_means[f], 4)
        cleaned_cells.append(cleaned)

    coverage_pct = round((total_cells - null_count) / total_cells * 100, 1)
    logger.info(
        "SMAP grid: %d/%d cells valid (%.1f%%) — filled %d nulls with regional mean",
        total_cells - null_count, total_cells, coverage_pct, null_count,
    )

    return {
        **raw_grid,
        "cells": cleaned_cells,
        "smap_status": "PARTIAL",
        "coverage_pct": coverage_pct,
        "null_cells": null_count,
        "total_cells": total_cells,
        "regional_mean_0_5cm": round(regional_means.get("soil_moisture_0_5cm", 0.0), 4),
        "regional_mean_0_100cm": round(regional_means.get("soil_moisture_0_100cm", 0.0), 4),
    }
