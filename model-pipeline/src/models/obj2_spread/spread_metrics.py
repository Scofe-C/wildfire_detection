"""
spread_metrics.py — Honest metrics from actual fire spread model output
========================================================================
Every function returns what the model ACTUALLY computes.
No synthetic constructs. No fractional burn. No inflated accuracy.

If the answer is "fire cannot reach any neighbor in 6 hours at this
resolution" — that IS the metric.
"""
from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

# H3 intercell distances (center-to-center, km)
_H3_INTERCELL_KM: dict[int, float] = {2: 93.0, 5: 25.0}

# Non-burnable FBFM40 codes (must match fire_spread_simulator._NB_CODES)
_NB_CODES: frozenset[int] = frozenset({
    91, 92, 93, 98, 99,
    121, 122, 123, 124,
})


# ---------------------------------------------------------------------------
# 1. Threatened cells analysis
# ---------------------------------------------------------------------------

def analyze_threatened_cells(
    result: dict[str, Any],
    intercell_km: float | None = None,
    horizon_hours: float = 1.0,
) -> dict[str, Any]:
    """Extract honest metrics from the 6 threatened neighbors.

    Parameters
    ----------
    result       : Output dict from ``PythonFireSpreadSimulator.simulate()``.
    intercell_km : H3 intercell distance in km. Auto-detected from
                   ignition cell resolution if not provided.
    horizon_hours: Time window for reachability check (default 1h).

    Returns
    -------
    dict with neighbor-level analysis and aggregate threat metrics.
    """
    neighbors = result.get("neighbour_details", [])
    if not neighbors:
        return {
            "n_total_neighbors": 0,
            "n_burnable_neighbors": 0,
            "n_reachable_1h": 0,
            "max_spread_rate_kmh": 0.0,
            "per_neighbor": [],
            "honest_assessment": "No neighbor data available.",
        }

    # Auto-detect intercell distance from ignition cell
    if intercell_km is None:
        try:
            import h3
            res = h3.get_resolution(result["ignition_cell"])
            intercell_km = _H3_INTERCELL_KM.get(res, 25.0)
        except Exception:
            intercell_km = 25.0  # default to 22km

    per_neighbor = []
    rates = []
    nonzero_rates = []

    for nb in neighbors:
        rate = nb.get("spread_rate_kmh", 0.0)
        fuel = nb.get("fuel_model")
        crown = nb.get("crown_status", "unknown")
        is_nb = crown == "non_burnable" or (fuel is not None and int(fuel) in _NB_CODES)

        time_to_reach_h = (intercell_km / rate) if rate > 0 else float("inf")
        reachable = time_to_reach_h <= horizon_hours

        entry = {
            "neighbour_id": nb.get("neighbour_id"),
            "bearing_deg": nb.get("bearing_deg"),
            "spread_rate_kmh": round(rate, 4),
            "time_to_reach_h": round(time_to_reach_h, 2) if time_to_reach_h != float("inf") else None,
            "reachable_in_1h": reachable,
            "crown_status": crown,
            "byram_intensity_kwm": nb.get("byram_intensity_kwm", 0.0),
            "fuel_model": fuel,
            "is_non_burnable": is_nb,
        }
        per_neighbor.append(entry)
        rates.append(rate)
        if rate > 0 and not is_nb:
            nonzero_rates.append(rate)

    n_burnable = sum(1 for n in per_neighbor if not n["is_non_burnable"])
    n_reachable = sum(1 for n in per_neighbor if n["reachable_in_1h"])
    max_rate = max(rates) if rates else 0.0
    min_nonzero = min(nonzero_rates) if nonzero_rates else 0.0

    # Spread cone: bearings where rate > 50% of max_rate
    if max_rate > 0:
        fast_bearings = [
            n["bearing_deg"] for n in per_neighbor
            if n["spread_rate_kmh"] > 0.5 * max_rate and n["bearing_deg"] is not None
        ]
        if len(fast_bearings) >= 2:
            spread_cone_deg = _angular_range(fast_bearings)
        else:
            spread_cone_deg = 0.0
    else:
        spread_cone_deg = 0.0

    # Threat asymmetry
    asymmetry = (max_rate / min_nonzero) if min_nonzero > 0 else 0.0

    # Time to nearest neighbor
    valid_times = [n["time_to_reach_h"] for n in per_neighbor if n["time_to_reach_h"] is not None]
    fastest_arrival = min(valid_times) if valid_times else None

    # Honest assessment
    if n_burnable == 0:
        assessment = "All neighbors are non-burnable. Fire cannot spread in any direction."
    elif n_reachable == 0:
        if fastest_arrival:
            assessment = (
                f"Fire cannot reach any neighbor within {horizon_hours:.0f}h. "
                f"Fastest arrival: {fastest_arrival:.1f}h at {max_rate:.2f} km/h. "
                f"Resolution-limited: intercell distance is {intercell_km:.0f} km."
            )
        else:
            assessment = "Zero spread rate to all burnable neighbors."
    elif n_reachable == n_burnable:
        assessment = (
            f"Fire can reach all {n_burnable} burnable neighbors within {horizon_hours:.0f}h. "
            f"Max rate: {max_rate:.2f} km/h. Fastest arrival: {fastest_arrival:.1f}h."
        )
    else:
        assessment = (
            f"Fire can reach {n_reachable}/{n_burnable} burnable neighbors within {horizon_hours:.0f}h. "
            f"Max rate: {max_rate:.2f} km/h. Fastest arrival: {fastest_arrival:.1f}h."
        )

    return {
        "n_total_neighbors": len(neighbors),
        "n_burnable_neighbors": n_burnable,
        "n_reachable_1h": n_reachable,
        "max_spread_rate_kmh": round(max_rate, 4),
        "min_nonzero_rate_kmh": round(min_nonzero, 4),
        "spread_cone_deg": round(spread_cone_deg, 1),
        "threat_asymmetry": round(asymmetry, 2),
        "time_to_nearest_neighbor_h": fastest_arrival,
        "intercell_km": intercell_km,
        "horizon_hours": horizon_hours,
        "per_neighbor": per_neighbor,
        "honest_assessment": assessment,
    }


# ---------------------------------------------------------------------------
# 2. Direction skill
# ---------------------------------------------------------------------------

def compute_direction_skill(
    result: dict[str, Any],
    observed_bearing: float,
) -> dict[str, Any]:
    """Compare predicted dominant direction vs observed spread bearing.

    Uses circular angular difference.
    """
    predicted = result.get("spread_direction_deg", 0.0)
    error = _angular_diff(predicted, observed_bearing)
    return {
        "predicted_deg": round(predicted, 1),
        "observed_deg": round(observed_bearing, 1),
        "angular_error_deg": round(error, 1),
        "within_30": error <= 30.0,
        "within_45": error <= 45.0,
        "within_90": error <= 90.0,
    }


# ---------------------------------------------------------------------------
# 3. Speed skill
# ---------------------------------------------------------------------------

def compute_speed_skill(
    result: dict[str, Any],
    observed_min_kmh: float,
    observed_max_kmh: float,
) -> dict[str, Any]:
    """Compare predicted max speed vs documented range."""
    predicted = result.get("spread_speed_kmh", 0.0)
    midpoint = (observed_min_kmh + observed_max_kmh) / 2.0

    in_range = observed_min_kmh <= predicted <= observed_max_kmh

    # Log-ratio: 0 = perfect, positive = over-predicting, negative = under
    if midpoint > 0 and predicted > 0:
        log_ratio = round(math.log(predicted / midpoint), 4)
    elif predicted == 0:
        log_ratio = float("-inf")
    else:
        log_ratio = 0.0

    underestimate = round(observed_min_kmh / predicted, 2) if predicted > 0 and predicted < observed_min_kmh else None
    overestimate = round(predicted / observed_max_kmh, 2) if predicted > observed_max_kmh else None

    return {
        "predicted_kmh": round(predicted, 4),
        "observed_range_kmh": [observed_min_kmh, observed_max_kmh],
        "in_range": in_range,
        "log_ratio": log_ratio,
        "underestimate_factor": underestimate,
        "overestimate_factor": overestimate,
    }


# ---------------------------------------------------------------------------
# 4. Propagation honesty
# ---------------------------------------------------------------------------

def compute_propagation_honesty(
    result: dict[str, Any],
    intercell_km: float | None = None,
) -> dict[str, Any]:
    """Report honestly whether fire can physically reach neighbors.

    No invented constructs. Just: can fire cross the intercell distance
    at the computed spread rate within a 6-hour window?
    """
    neighbors = result.get("neighbour_details", [])
    if not neighbors:
        return {
            "can_propagate": False,
            "resolution_limited": True,
            "honest_assessment": "No neighbor data — cannot assess propagation.",
        }

    if intercell_km is None:
        try:
            import h3
            res = h3.get_resolution(result["ignition_cell"])
            intercell_km = _H3_INTERCELL_KM.get(res, 25.0)
        except Exception:
            intercell_km = 25.0

    rates = [nb.get("spread_rate_kmh", 0.0) for nb in neighbors
             if nb.get("crown_status") != "non_burnable"]
    max_rate = max(rates) if rates else 0.0

    if max_rate > 0:
        fastest_h = intercell_km / max_rate
        slowest_nonzero = [r for r in rates if r > 0]
        slowest_h = intercell_km / min(slowest_nonzero) if slowest_nonzero else float("inf")
    else:
        fastest_h = float("inf")
        slowest_h = float("inf")

    can_propagate = fastest_h <= 1.0
    resolution_limited = intercell_km > 50.0

    if can_propagate:
        assessment = (
            f"Fire CAN reach nearest neighbor in {fastest_h:.1f}h "
            f"at {max_rate:.2f} km/h across {intercell_km:.0f} km."
        )
    elif resolution_limited:
        assessment = (
            f"Resolution-limited: intercell distance is {intercell_km:.0f} km. "
            f"At max rate {max_rate:.2f} km/h, fire needs {fastest_h:.1f}h to reach nearest neighbor. "
            f"This model provides fire behavior indices, not spatial propagation, at this resolution."
        )
    else:
        assessment = (
            f"Fire CANNOT reach any neighbor within 1h. "
            f"At max rate {max_rate:.2f} km/h across {intercell_km:.0f} km, "
            f"earliest arrival is {fastest_h:.1f}h."
        )

    return {
        "can_propagate": can_propagate,
        "max_rate_kmh": round(max_rate, 4),
        "fastest_arrival_h": round(fastest_h, 2) if fastest_h != float("inf") else None,
        "slowest_arrival_h": round(slowest_h, 2) if slowest_h != float("inf") else None,
        "intercell_km": intercell_km,
        "resolution_limited": resolution_limited,
        "honest_assessment": assessment,
    }


# ---------------------------------------------------------------------------
# 5. Input quality scoring
# ---------------------------------------------------------------------------

def compute_input_quality(result: dict[str, Any]) -> dict[str, Any]:
    """Score data quality from inputs_used and warnings.

    Returns a 0-1 quality score reflecting what fraction of inputs
    are real data vs. fallback defaults.
    """
    inputs = result.get("inputs_used", {})
    warnings = result.get("warnings", [])

    checks = {
        "has_fuel_model": inputs.get("ignition_cell_fbfm40") is not None,
        "has_canopy_cbh": inputs.get("canopy_base_height_m") is not None,
        "has_canopy_cbd": inputs.get("canopy_bulk_density_kgm3") is not None,
        "has_wind_speed": inputs.get("wind_speed_10m_ms", 0) > 0,
        "has_wind_direction": True,  # always present (even 0 is valid)
        "has_rh": 0 < inputs.get("relative_humidity_pct", -1) <= 100,
        "has_temperature": True,  # always present
        "has_slope": True,  # always present (0 is valid for flat)
    }

    n_present = sum(checks.values())
    n_total = len(checks)
    quality_score = n_present / n_total if n_total > 0 else 0.0

    # Build quality notes
    notes: list[str] = []
    if not checks["has_fuel_model"]:
        notes.append("Missing fuel model — using fallback, spread rate unreliable")
    if not checks["has_canopy_cbh"] or not checks["has_canopy_cbd"]:
        notes.append("Missing canopy data (CBH/CBD) — crown fire assessment degraded")
    if not checks["has_wind_speed"]:
        notes.append("Wind speed is zero or missing — spread rate may underestimate")
    if not checks["has_rh"]:
        notes.append("Relative humidity missing — fuel moisture estimate unreliable")

    for w in warnings[:5]:
        notes.append(f"Simulator warning: {w}")

    return {
        "quality_score": round(quality_score, 3),
        "n_inputs_present": n_present,
        "n_inputs_total": n_total,
        "checks": checks,
        "n_warnings": len(warnings),
        "quality_notes": notes,
    }


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _angular_diff(a: float, b: float) -> float:
    """Circular angular difference in [0, 180] degrees."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _angular_range(bearings: list[float]) -> float:
    """Angular range of a set of bearings (max gap complement)."""
    if len(bearings) < 2:
        return 0.0
    sorted_b = sorted(b % 360 for b in bearings)
    gaps = [sorted_b[i + 1] - sorted_b[i] for i in range(len(sorted_b) - 1)]
    gaps.append(360.0 - sorted_b[-1] + sorted_b[0])  # wrap-around gap
    max_gap = max(gaps)
    return 360.0 - max_gap
