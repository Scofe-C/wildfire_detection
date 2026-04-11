"""
Evaluation metrics for OBJ-2 fire spread simulator outputs.

Provides per-output evaluation functions for the 5 key simulator outputs:
    1. spread_direction_deg   — circular angular error
    2. spread_speed_kmh       — range containment + log-ratio
    3. dead_fuel_moisture_pct — range containment
    4. byram_intensity_kwm    — minimum threshold + fire behavior class
    5. crown_fire_status      — categorical match + severity error

Plus a combined physics gate that runs all 5 checks.

References
----------
- Beyki et al. (2025): Different Rothermel implementations produce different
  results as complexity increases — evaluation must use range-based thresholds.
- Alexander & Cruz (2019): Fire behavior class thresholds for intensity.

Owner: OBJ-2 (fire spread model)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Ground truth dataclass
# ---------------------------------------------------------------------------

@dataclass
class GroundTruth:
    """Documented fire behavior from CAL FIRE / NIFC / incident reports."""
    spread_direction_deg: float
    direction_tolerance_deg: float = 30.0
    spread_speed_kmh_min: float = 0.0
    spread_speed_kmh_max: float = 50.0
    dead_fuel_moisture_pct_min: float = 1.0
    dead_fuel_moisture_pct_max: float = 45.0
    byram_intensity_kwm_min: float = 0.0
    crown_fire_expected: list[str] = field(
        default_factory=lambda: ["surface", "passive_crown", "active_crown"]
    )
    source: str = ""


# ---------------------------------------------------------------------------
# Fire behavior intensity classes (Alexander & Cruz 2019)
# ---------------------------------------------------------------------------

_INTENSITY_CLASSES = [
    ("low",      0,    350),
    ("moderate", 350,  1750),
    ("high",     1750, 3500),
    ("extreme",  3500, float("inf")),
]


def _classify_intensity(kwm: float) -> str:
    """Classify Byram fireline intensity into fire behavior class."""
    for name, lo, hi in _INTENSITY_CLASSES:
        if lo <= kwm < hi:
            return name
    return "extreme"


# ---------------------------------------------------------------------------
# 1. Spread direction
# ---------------------------------------------------------------------------

def evaluate_direction(
    predicted_deg: float,
    expected_deg: float,
    tolerance_deg: float = 30.0,
) -> dict[str, Any]:
    """Evaluate spread direction using circular angular error.

    Parameters
    ----------
    predicted_deg : Predicted dominant spread bearing (0-360).
    expected_deg  : Documented fire spread bearing.
    tolerance_deg : Maximum acceptable angular error (default 30).

    Returns
    -------
    dict with angular_error_deg, passed, tolerance_used, detail.
    """
    diff = abs(predicted_deg - expected_deg) % 360
    if diff > 180:
        diff = 360 - diff
    passed = diff <= tolerance_deg
    return {
        "metric": "direction",
        "predicted": round(predicted_deg, 1),
        "expected": round(expected_deg, 1),
        "angular_error_deg": round(diff, 1),
        "tolerance_deg": tolerance_deg,
        "passed": passed,
        "detail": f"error={diff:.1f} deg (limit {tolerance_deg} deg)",
    }


# ---------------------------------------------------------------------------
# 2. Spread speed
# ---------------------------------------------------------------------------

def evaluate_speed(
    predicted_kmh: float,
    expected_min_kmh: float,
    expected_max_kmh: float,
) -> dict[str, Any]:
    """Evaluate spread speed using range containment + log-ratio.

    Log-ratio is used because spread speeds span orders of magnitude
    (0.5-25 km/h) and percentage error is more meaningful than absolute.

    Parameters
    ----------
    predicted_kmh    : Predicted maximum head fire speed.
    expected_min_kmh : Lower bound from incident timeline.
    expected_max_kmh : Upper bound from incident timeline.

    Returns
    -------
    dict with in_range, log_ratio, passed, detail.
    """
    in_range = expected_min_kmh <= predicted_kmh <= expected_max_kmh
    midpoint = (expected_min_kmh + expected_max_kmh) / 2.0
    if midpoint > 0 and predicted_kmh > 0:
        log_ratio = abs(math.log(predicted_kmh / midpoint))
    else:
        log_ratio = float("inf") if predicted_kmh <= 0 else 0.0

    return {
        "metric": "speed",
        "predicted_kmh": round(predicted_kmh, 4),
        "expected_range": [expected_min_kmh, expected_max_kmh],
        "in_range": in_range,
        "log_ratio": round(log_ratio, 4),
        "passed": in_range,
        "detail": (
            f"{predicted_kmh:.3f} km/h "
            f"{'within' if in_range else 'OUTSIDE'} "
            f"[{expected_min_kmh}, {expected_max_kmh}]"
        ),
    }


# ---------------------------------------------------------------------------
# 3. Dead fuel moisture
# ---------------------------------------------------------------------------

def evaluate_moisture(
    predicted_pct: float,
    expected_min_pct: float,
    expected_max_pct: float,
) -> dict[str, Any]:
    """Evaluate dead fuel moisture using range containment.

    The EMC model is deterministic given RH + temp, so the range check
    is the appropriate metric for a physics-based model.

    Returns
    -------
    dict with in_range, abs_error_pct, passed, detail.
    """
    in_range = expected_min_pct <= predicted_pct <= expected_max_pct
    if in_range:
        abs_error = 0.0
    else:
        abs_error = min(
            abs(predicted_pct - expected_min_pct),
            abs(predicted_pct - expected_max_pct),
        )

    return {
        "metric": "moisture",
        "predicted_pct": round(predicted_pct, 1),
        "expected_range": [expected_min_pct, expected_max_pct],
        "in_range": in_range,
        "abs_error_pct": round(abs_error, 1),
        "passed": in_range,
        "detail": (
            f"{predicted_pct:.1f}% "
            f"{'within' if in_range else 'OUTSIDE'} "
            f"[{expected_min_pct}, {expected_max_pct}]%"
        ),
    }


# ---------------------------------------------------------------------------
# 4. Byram fireline intensity
# ---------------------------------------------------------------------------

def evaluate_intensity(
    predicted_kwm: float,
    expected_min_kwm: float,
) -> dict[str, Any]:
    """Evaluate Byram fireline intensity using minimum threshold + class.

    Fire behavior classes (Alexander & Cruz 2019):
        low:      < 350 kW/m   (surface, creeping)
        moderate: 350-1750      (surface, vigorous)
        high:     1750-3500     (passive crown)
        extreme:  > 3500        (active crown)

    Returns
    -------
    dict with above_minimum, class_predicted, passed, detail.
    """
    above_min = predicted_kwm >= expected_min_kwm
    fire_class = _classify_intensity(predicted_kwm)

    return {
        "metric": "intensity",
        "predicted_kwm": round(predicted_kwm, 1),
        "expected_min_kwm": expected_min_kwm,
        "above_minimum": above_min,
        "class_predicted": fire_class,
        "passed": above_min,
        "detail": (
            f"{predicted_kwm:.1f} kW/m ({fire_class}) "
            f"{'above' if above_min else 'BELOW'} min {expected_min_kwm}"
        ),
    }


# ---------------------------------------------------------------------------
# 5. Crown fire status
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = {"surface": 0, "passive_crown": 1, "active_crown": 2}


def evaluate_crown_fire(
    predicted_status: str,
    expected_statuses: list[str],
) -> dict[str, Any]:
    """Evaluate crown fire classification using categorical match.

    Parameters
    ----------
    predicted_status  : Model output ("surface", "passive_crown", "active_crown").
    expected_statuses : Acceptable statuses from ground truth.

    Returns
    -------
    dict with match, severity_error, passed, detail.
    """
    match = predicted_status in expected_statuses
    pred_sev = _SEVERITY_ORDER.get(predicted_status, -1)
    # Find closest expected severity for error calculation
    expected_sevs = [_SEVERITY_ORDER.get(s, -1) for s in expected_statuses]
    severity_error = min(abs(pred_sev - es) for es in expected_sevs) if expected_sevs else 99

    return {
        "metric": "crown_fire",
        "predicted": predicted_status,
        "expected": expected_statuses,
        "match": match,
        "severity_error": severity_error,
        "passed": match,
        "detail": (
            f"{predicted_status} "
            f"{'in' if match else 'NOT in'} {expected_statuses}"
        ),
    }


# ---------------------------------------------------------------------------
# Combined physics gate
# ---------------------------------------------------------------------------

def compute_physics_gate(
    result: dict[str, Any],
    ground_truth: GroundTruth,
) -> dict[str, Any]:
    """Run all 5 per-output evaluators and combine into a single gate.

    Parameters
    ----------
    result       : Simulator output dict from PythonFireSpreadSimulator.simulate().
    ground_truth : Documented fire behavior.

    Returns
    -------
    dict with gate_passed (bool), per_output (dict of 5 results), summary (str).
    """
    direction = evaluate_direction(
        result["spread_direction_deg"],
        ground_truth.spread_direction_deg,
        ground_truth.direction_tolerance_deg,
    )
    speed = evaluate_speed(
        result["spread_speed_kmh"],
        ground_truth.spread_speed_kmh_min,
        ground_truth.spread_speed_kmh_max,
    )
    moisture = evaluate_moisture(
        result["dead_fuel_moisture_pct"],
        ground_truth.dead_fuel_moisture_pct_min,
        ground_truth.dead_fuel_moisture_pct_max,
    )
    intensity = evaluate_intensity(
        result["byram_intensity_kwm"],
        ground_truth.byram_intensity_kwm_min,
    )
    crown = evaluate_crown_fire(
        result["crown_fire_status"],
        ground_truth.crown_fire_expected,
    )

    per_output = {
        "direction": direction,
        "speed": speed,
        "moisture": moisture,
        "intensity": intensity,
        "crown_fire": crown,
    }

    n_passed = sum(1 for v in per_output.values() if v["passed"])
    gate_passed = all(v["passed"] for v in per_output.values())

    summary = f"{n_passed}/5 checks passed"
    if gate_passed:
        summary += " — GATE PASSED"
    else:
        failed = [k for k, v in per_output.items() if not v["passed"]]
        summary += f" — GATE FAILED (failed: {', '.join(failed)})"

    return {
        "gate_passed": gate_passed,
        "checks_passed": n_passed,
        "checks_total": 5,
        "per_output": per_output,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Real-time sanity checks (no ground truth needed)
# ---------------------------------------------------------------------------

def sanity_check_output(
    result: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate simulator output for physical self-consistency.

    These are STRICT checks — not just range checks. Each check verifies
    that outputs are internally consistent with each other and with physics.
    A model that passes all checks is not "production ready" but at least
    not producing contradictory garbage.

    Checks
    ------
    1.  no_nan          — no None/NaN in any of the 5 critical outputs
    2.  direction_valid — direction in [0, 360]
    3.  speed_nonzero_with_fire — if fire_detected, speed must be > 0.05 km/h
    4.  moisture_rh_consistent — DFMC must be plausible given RH input
    5.  intensity_speed_consistent — Byram I_B ∝ speed; large mismatch = model error
    6.  crown_intensity_consistent — passive/active crown requires I_B > 350 kW/m
    7.  direction_wind_consistent — spread direction must be within 90° of downwind
    8.  speed_upper_bound — speed cannot exceed physically observed max (50 km/h)

    Returns
    -------
    dict with all_passed (bool), n_passed, n_total, checks (list).
    """
    cfg = config or {}
    checks = []

    speed   = result.get("spread_speed_kmh", None)
    direc   = result.get("spread_direction_deg", None)
    moist   = result.get("dead_fuel_moisture_pct", None)
    intens  = result.get("byram_intensity_kwm", None)
    crown   = result.get("crown_fire_status", None)

    # inputs passed through by simulator (needed for cross-checks)
    wind_dir  = result.get("wind_spread_direction_deg", None)   # downwind bearing
    rh_input  = result.get("input_rh_pct", None)               # RH used by simulator

    # ── Check 1: no NaN in any critical field ────────────────────────────────
    critical_vals = {"speed": speed, "direction": direc,
                     "moisture": moist, "intensity": intens, "crown": crown}
    nan_fields = [k for k, v in critical_vals.items()
                  if v is None or (isinstance(v, float) and math.isnan(v))]
    checks.append({
        "check": "no_nan",
        "passed": len(nan_fields) == 0,
        "detail": (f"NaN/None in: {nan_fields}" if nan_fields
                   else "all 5 outputs present"),
    })
    if nan_fields:   # can't cross-check NaN values
        return _build_sanity_result(checks)

    # ── Check 2: direction in [0, 360] ───────────────────────────────────────
    checks.append({
        "check": "direction_valid",
        "passed": 0.0 <= direc <= 360.0,
        "detail": f"direction={direc:.1f}° (must be 0–360)",
    })

    # ── Check 3: speed > 0.05 km/h when fire detected ───────────────────────
    # A simulator that returns 0 spread on an active fire cell is broken.
    fire_detected = result.get("fire_detected", False)
    if fire_detected:
        checks.append({
            "check": "speed_nonzero_with_fire",
            "passed": speed >= 0.05,
            "detail": (f"speed={speed:.4f} km/h on active fire cell "
                       f"(must be >= 0.05 km/h when fire_detected=True)"),
        })
    else:
        checks.append({
            "check": "speed_nonzero_with_fire",
            "passed": True,
            "detail": f"speed={speed:.4f} km/h (no active fire — not checked)",
        })

    # ── Check 4: moisture–RH self-consistency ────────────────────────────────
    # Rothermel DFMC = f(RH, T). At RH=5% → DFMC ≈ 2–4%.  At RH=90% → DFMC ≈ 20–30%.
    # Tight rule: DFMC should not exceed 0.35×RH + 5 (empirical upper bound)
    # and should not be below 0.02×RH + 0.5 (empirical lower bound).
    if rh_input is not None and rh_input > 0:
        upper_bound = 0.35 * rh_input + 5.0
        lower_bound = max(0.02 * rh_input + 0.5, 1.0)
        moist_ok = lower_bound <= moist <= upper_bound
        checks.append({
            "check": "moisture_rh_consistent",
            "passed": moist_ok,
            "detail": (f"DFMC={moist:.1f}% with RH={rh_input:.1f}% — "
                       f"expected [{lower_bound:.1f}, {upper_bound:.1f}]%"),
        })
    else:
        checks.append({
            "check": "moisture_rh_consistent",
            "passed": 1.0 <= moist <= 45.0,
            "detail": (f"DFMC={moist:.1f}% (RH input unavailable — "
                       f"checking physical range [1, 45]%)"),
        })

    # ── Check 5: intensity–speed physical consistency ────────────────────────
    # Byram: I_B = H × w × R.  For typical CA fuels (w=0.3–1.5 kg/m², H=18000 kJ/kg):
    #   speed 0.1 km/h → I_B ≤  750 kW/m  (low intensity)
    #   speed 1.0 km/h → I_B ≤ 7500 kW/m  (moderate-high)
    # Rule: I_B / speed should be in [50, 50000] kW/m per km/h.
    # Outside this band = speed and intensity are contradictory.
    if speed > 0:
        ratio = intens / speed
        ratio_ok = 50.0 <= ratio <= 50000.0
        checks.append({
            "check": "intensity_speed_consistent",
            "passed": ratio_ok,
            "detail": (f"I_B/speed = {ratio:.0f} kW·h/m/km "
                       f"(I_B={intens:.1f} kW/m, speed={speed:.4f} km/h) "
                       f"— expected ratio [50, 50000]"),
        })
    else:
        # Zero speed → intensity must also be ~zero
        checks.append({
            "check": "intensity_speed_consistent",
            "passed": intens < 10.0,
            "detail": (f"speed=0 but I_B={intens:.1f} kW/m "
                       f"— intensity must be <10 kW/m when spread rate is zero"),
        })

    # ── Check 6: crown fire intensity threshold ───────────────────────────────
    # Van Wagner (1977): passive crown requires I_B > 350 kW/m.
    # Active crown requires I_B > ~1750 kW/m AND R_active threshold.
    # If model says crown fire but intensity is low → model is lying.
    crown_intensity_thresholds = {
        "surface":             (0,      3500),   # surface: intensity below crown threshold
        "passive_crown":       (350,   10000),   # passive: must exceed Van Wagner 350 kW/m
        "active_crown":        (1750, 100000),   # active: must exceed Scott & Reinhardt 1750 kW/m
        "non_burnable":        (0,       50),    # non-burnable: near-zero intensity
        "crown_data_missing":  (1750, 100000),   # honest flag: high intensity, no CBH/CBD data
    }
    if crown in crown_intensity_thresholds:
        lo, hi = crown_intensity_thresholds[crown]
        crown_ok = lo <= intens <= hi
        checks.append({
            "check": "crown_intensity_consistent",
            "passed": crown_ok,
            "detail": (f"crown={crown} with I_B={intens:.1f} kW/m "
                       f"— expected I_B in [{lo}, {hi}] kW/m for {crown}"),
        })
    else:
        checks.append({
            "check": "crown_intensity_consistent",
            "passed": False,
            "detail": f"crown='{crown}' is not a valid status",
        })

    # ── Check 7: spread direction–wind consistency ────────────────────────────
    # Fire spreads downwind. Spread direction should be within 90° of the
    # downwind bearing. Beyond 90° means fire is spreading into the wind — impossible.
    if wind_dir is not None:
        diff = abs(direc - wind_dir) % 360
        diff = min(diff, 360 - diff)   # circular distance
        direction_ok = diff <= 90.0
        checks.append({
            "check": "direction_wind_consistent",
            "passed": direction_ok,
            "detail": (f"spread={direc:.1f}°, downwind={wind_dir:.1f}°, "
                       f"angular diff={diff:.1f}° (must be ≤ 90°)"),
        })
    else:
        checks.append({
            "check": "direction_wind_consistent",
            "passed": True,
            "detail": "wind direction not in result — skipped",
        })

    # ── Check 8: speed physical upper bound ──────────────────────────────────
    # Fastest observed wildfire spread: ~28 km/h (Camp Fire 2018).
    # 50 km/h is an absolute model ceiling; anything above is a computation error.
    MAX_PHYSICAL_SPEED = 50.0
    checks.append({
        "check": "speed_upper_bound",
        "passed": speed <= MAX_PHYSICAL_SPEED,
        "detail": (f"speed={speed:.4f} km/h "
                   f"(physical max observed = {MAX_PHYSICAL_SPEED} km/h)"),
    })

    return _build_sanity_result(checks)


def _build_sanity_result(checks: list[dict]) -> dict:
    """Aggregate check list into final sanity result dict."""
    all_passed = all(c["passed"] for c in checks)
    return {
        "all_passed": all_passed,
        "checks_passed": sum(1 for c in checks if c["passed"]),
        "checks_total": len(checks),
        "checks": checks,
    }
