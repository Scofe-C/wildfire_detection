"""
PythonFireSpreadSimulator — physics-based pure-Python fire spread simulation.

Implements the full Rothermel (1972) surface fire spread model with:
    - Scott & Burgan (2005) FBFM40 fuel parameters  (RMRS-GTR-153)
    - Nelson/Simard dead fuel moisture estimation    (EMC piecewise model)
    - Byram (1959) fireline intensity                (I_B = H × w_c × R)
    - Van Wagner (1977) crown fire initiation        (I_0 critical intensity)
    - Scott & Reinhardt (2001) active crown fire     (CBD threshold)
    - Anderson (1983) elliptical fire shape           (L/B ratio)
    - Andrews (2012) 10 m → midflame wind reduction  (WAF = 0.4)

No external C++ binaries required. All intermediate calculations use imperial
units (BTU, lb, ft, min) exactly as in the original Rothermel paper. SI
conversion happens only at the input/output boundary.

Interface
---------
    from src.models.obj2_spread.fire_spread_simulator import PythonFireSpreadSimulator

    sim = PythonFireSpreadSimulator()
    result = sim.simulate(df, ignition_grid_id="822937fffffffff", ignition_prob=0.30)

    result["spread_direction_deg"]   # dominant fire-front bearing (0–360°)
    result["spread_speed_kmh"]       # max spread rate (km/h)
    result["dead_fuel_moisture_pct"] # estimated 1-hr DFMC (%)
    result["crown_fire_status"]      # "surface" | "passive_crown" | "active_crown"

References
----------
[1] Rothermel, R.C. 1972. A mathematical model for predicting fire spread
    in wildland fuels. USDA Forest Service Research Paper INT-115.
[2] Scott, J.H.; Burgan, R.E. 2005. Standard fire behavior fuel models.
    USDA Forest Service RMRS-GTR-153.
[3] Van Wagner, C.E. 1977. Conditions for the start and spread of crown fire.
    Canadian Journal of Forest Research 7(1): 23–34.
[4] Byram, G.M. 1959. Combustion of forest fuels. In: Forest Fire: Control
    and Use (K.P. Davis, ed.), pp. 61–89.
[5] Anderson, H.E. 1983. Predicting wind-driven wildland fire size and shape.
    USDA Forest Service Research Paper INT-305.
[6] Scott, J.H.; Reinhardt, E.D. 2001. Assessing crown fire potential.
    USDA Forest Service RMRS-GTR-29.
[7] Andrews, P.L. 2012. Modeling wind adjustment factor and midflame wind
    speed. USDA Forest Service RMRS-GTR-266.

Owner: OBJ-2 (fire spread model)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — FBFM40 Fuel Parameter Table  (Scott & Burgan 2005, RMRS-GTR-153)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _FuelParams:
    """Rothermel surface fuel parameters for a single FBFM40 fuel model.

    All values are in imperial units matching the original Rothermel (1972)
    paper exactly. Unit conversions from the Scott & Burgan tables:
        tons/acre → lb/ft²  :  ÷ 21.78
    """
    w0:    float   # total 1-hr dead fuel load (lb/ft²)
    delta: float   # fuel bed depth (ft)
    sigma: float   # characteristic surface-area-to-volume ratio (1/ft)
    Mx:    float   # dead fuel moisture of extinction (fraction, e.g. 0.20)
    ST:    float = 0.0555   # total mineral content (Rothermel eq. 35)
    Se:    float = 0.01     # effective mineral content (Rothermel eq. 31)


# Values from Scott & Burgan (2005) Table 1, RMRS-GTR-153.
# w0 = 1-hr dead fuel load converted: tons/acre ÷ 21.78 = lb/ft²
# sigma = characteristic SAV (1/ft) from table's "characteristic SAV" column
_FBFM40_PARAMS: dict[int, _FuelParams] = {
    # ── Grass group (GR1–GR9) ─────────────────────────────────────────────
    101: _FuelParams(w0=0.0046, delta=0.4,  sigma=2200, Mx=0.15),   # GR1
    102: _FuelParams(w0=0.0092, delta=1.0,  sigma=2000, Mx=0.15),   # GR2
    103: _FuelParams(w0=0.0138, delta=2.0,  sigma=1500, Mx=0.30),   # GR3
    104: _FuelParams(w0=0.0184, delta=2.0,  sigma=2000, Mx=0.15),   # GR4
    105: _FuelParams(w0=0.0184, delta=1.5,  sigma=1800, Mx=0.40),   # GR5
    106: _FuelParams(w0=0.0322, delta=1.5,  sigma=2200, Mx=0.40),   # GR6
    107: _FuelParams(w0=0.0506, delta=3.0,  sigma=2000, Mx=0.15),   # GR7
    108: _FuelParams(w0=0.0460, delta=4.0,  sigma=1500, Mx=0.30),   # GR8
    109: _FuelParams(w0=0.0920, delta=5.0,  sigma=1800, Mx=0.40),   # GR9
    # ── Grass-Shrub group (GS1–GS4) ───────────────────────────────────────
    121: _FuelParams(w0=0.0092, delta=0.9,  sigma=2000, Mx=0.15),   # GS1
    122: _FuelParams(w0=0.0230, delta=1.5,  sigma=2000, Mx=0.15),   # GS2
    123: _FuelParams(w0=0.0138, delta=1.8,  sigma=1800, Mx=0.40),   # GS3
    124: _FuelParams(w0=0.0920, delta=2.1,  sigma=1800, Mx=0.40),   # GS4
    # ── Shrub group (SH1–SH9) ─────────────────────────────────────────────
    141: _FuelParams(w0=0.0138, delta=1.0,  sigma=2000, Mx=0.15),   # SH1
    142: _FuelParams(w0=0.0644, delta=1.0,  sigma=2000, Mx=0.15),   # SH2
    143: _FuelParams(w0=0.0230, delta=2.4,  sigma=1600, Mx=0.40),   # SH3
    144: _FuelParams(w0=0.0322, delta=3.0,  sigma=2000, Mx=0.30),   # SH4
    145: _FuelParams(w0=0.0644, delta=6.0,  sigma=750,  Mx=0.15),   # SH5
    146: _FuelParams(w0=0.0460, delta=2.0,  sigma=1800, Mx=0.30),   # SH6
    147: _FuelParams(w0=0.0966, delta=6.0,  sigma=1750, Mx=0.15),   # SH7
    148: _FuelParams(w0=0.0460, delta=3.0,  sigma=1800, Mx=0.40),   # SH8
    149: _FuelParams(w0=0.0828, delta=4.4,  sigma=1800, Mx=0.40),   # SH9
    # ── Timber-Understory (TU1–TU5) ───────────────────────────────────────
    161: _FuelParams(w0=0.0138, delta=0.6,  sigma=1800, Mx=0.20),   # TU1
    162: _FuelParams(w0=0.0184, delta=1.0,  sigma=1800, Mx=0.30),   # TU2
    163: _FuelParams(w0=0.0230, delta=1.3,  sigma=1800, Mx=0.30),   # TU3
    164: _FuelParams(w0=0.0920, delta=0.5,  sigma=2200, Mx=0.12),   # TU4
    165: _FuelParams(w0=0.0644, delta=1.0,  sigma=1800, Mx=0.25),   # TU5
    # ── Timber-Litter (TL1–TL9) ───────────────────────────────────────────
    181: _FuelParams(w0=0.0460, delta=0.2,  sigma=2000, Mx=0.30),   # TL1
    182: _FuelParams(w0=0.0460, delta=0.2,  sigma=2000, Mx=0.25),   # TL2
    183: _FuelParams(w0=0.0230, delta=0.3,  sigma=2000, Mx=0.20),   # TL3
    184: _FuelParams(w0=0.0460, delta=0.4,  sigma=2000, Mx=0.25),   # TL4
    185: _FuelParams(w0=0.0460, delta=0.6,  sigma=2000, Mx=0.25),   # TL5
    186: _FuelParams(w0=0.0460, delta=0.3,  sigma=2000, Mx=0.25),   # TL6
    187: _FuelParams(w0=0.0138, delta=0.4,  sigma=2000, Mx=0.25),   # TL7
    188: _FuelParams(w0=0.0828, delta=0.3,  sigma=1800, Mx=0.35),   # TL8
    189: _FuelParams(w0=0.1150, delta=0.6,  sigma=1800, Mx=0.35),   # TL9
    # ── Slash-Blowdown (SB1–SB4) ──────────────────────────────────────────
    201: _FuelParams(w0=0.0690, delta=1.0,  sigma=2000, Mx=0.25),   # SB1
    202: _FuelParams(w0=0.0920, delta=1.0,  sigma=2000, Mx=0.25),   # SB2
    203: _FuelParams(w0=0.1380, delta=1.2,  sigma=2000, Mx=0.25),   # SB3
    204: _FuelParams(w0=0.1840, delta=2.7,  sigma=2000, Mx=0.25),   # SB4
}

_NB_CODES: frozenset[int] = frozenset({91, 92, 93, 98, 99})
_FBFM40_NODATA = -9999

# Fallback for unknown but presumably burnable codes
_FBFM40_DEFAULT = _FuelParams(w0=0.0184, delta=1.0, sigma=1800, Mx=0.20)

# Physical constants
_RHO_P = 32.0         # particle density (lb/ft³)  — Rothermel eq. 30
_HEAT_CONTENT = 8000.0 # heat content h (BTU/lb)    — standard for forest fuels

# Andrews (2012) 10 m open-terrain wind adjustment factor
_WIND_REDUCTION_10M_TO_MIDFLAME = 0.4

# ft/min → km/h conversion: 1 ft/min × 60 min/hr × 0.0003048 km/ft = 0.018288
_FTMIN_TO_KMH = 0.018288


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — Dead Fuel Moisture Content  (Nelson/Simard EMC piecewise model)
# ═══════════════════════════════════════════════════════════════════════════════

def _estimate_dfmc(
    rh_pct: float,
    temp_c: float,
    days_since_precip: float = 3.0,
) -> float:
    """Estimate 1-hour dead fuel moisture content (fraction 0–1).

    Uses the equilibrium moisture content (EMC) piecewise regression from
    Simard (1968) / Nelson (1984), as cited in the BehavePlus documentation.
    Temperature is converted internally to °F.

    Parameters
    ----------
    rh_pct           : Relative humidity (0–100 %).
    temp_c           : Air temperature (°C).
    days_since_precip: Days since last measurable precipitation.
                       Used as a fine-tuning drying modifier on top of EMC.

    Returns
    -------
    Mf : 1-hr dead fuel moisture content as a fraction (e.g. 0.08 = 8 %).
    """
    # Clamp inputs to physical ranges
    h = max(1.0, min(99.0, rh_pct))        # RH in percent
    T = temp_c * 9.0 / 5.0 + 32.0          # °C → °F

    # EMC piecewise formula (percent moisture)
    if h <= 10.0:
        EMC = 0.03229 + 0.281073 * h - 0.000578 * h * T
    elif h <= 50.0:
        EMC = 2.22749 + 0.160107 * h - 0.014784 * T
    else:
        EMC = 21.0606 + 0.005565 * h * h - 0.00035 * h * T - 0.483199 * h

    # 1-hr correction (quick-responding fuels track EMC closely)
    Mf_pct = 1.03 * EMC

    # Drying modifier: prolonged dry spell further reduces DFMC
    days = max(0.0, min(30.0, days_since_precip))
    dry_factor = max(0.5, 1.0 - 0.01 * days)
    Mf_pct *= dry_factor

    # Clamp to physically realistic range (1–45 %)
    Mf_pct = max(1.0, min(45.0, Mf_pct))

    return Mf_pct / 100.0


def _estimate_fmc(temp_c: float, vpd_kpa: float) -> float:
    """Estimate canopy foliar moisture content (FMC) from temperature and VPD.

    FMC drives the Van Wagner crown fire critical intensity:
      I_0 = (0.010 × CBH × (460 + 25.9 × FMC_pct))^1.5

    A lower FMC (drought-stressed canopy in summer) lowers I_0, making
    crown fire easier to initiate — physically correct for extreme events
    like the Creek Fire (Sep) and Carr Fire (Jul).

    Reference: Cruz, M.G. et al. (2005). Predicting crown fire behavior
    to support forest fire management decision-making.

    Parameters
    ----------
    temp_c  : Air temperature (°C).
    vpd_kpa : Vapour pressure deficit (kPa).

    Returns
    -------
    FMC as a fraction (e.g. 0.90 = 90 %).
    """
    # Base FMC for temperate conifers in neutral conditions (115%)
    fmc = 1.15

    # Temperature drying: above 20°C, canopy loses ~0.3% moisture per °C
    if temp_c > 20.0:
        fmc -= 0.003 * (temp_c - 20.0)

    # VPD drying: above 2 kPa, canopy loses ~5% per kPa (drought stress)
    if vpd_kpa > 2.0:
        fmc -= 0.05 * (vpd_kpa - 2.0)

    # Clamp to physical range [60%, 150%]
    return max(0.60, min(1.50, fmc))


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — Full Rothermel (1972) Rate-of-Spread  (all equations, imperial)
# ═══════════════════════════════════════════════════════════════════════════════

def _rothermel_surface_ros(
    fuel: _FuelParams,
    Mf: float,
    U_midflame_ftmin: float,
    phi_s: float,
) -> tuple[float, float]:
    """Compute Rothermel (1972) surface fire Rate-of-Spread.

    All calculations in imperial units exactly as in the original paper.

    Parameters
    ----------
    fuel                : Fuel model parameters (_FuelParams).
    Mf                  : 1-hr dead fuel moisture content (fraction, 0–1).
    U_midflame_ftmin    : Effective midflame wind speed toward the neighbour
                          bearing (ft/min). Must be ≥ 0.
    phi_s               : Slope coefficient φ_s for this bearing (dimensionless).

    Returns
    -------
    (R_ftmin, I_R)
        R_ftmin : Rate of spread (ft/min). 0.0 if fuel is too wet.
        I_R     : Reaction intensity (BTU/ft²/min). Needed by Byram intensity.
    """
    # ── Step 1: Bulk density and packing ratio (Rothermel eqs. 40, 74) ────
    if fuel.delta <= 0 or fuel.w0 <= 0:
        return 0.0, 0.0

    rho_b = fuel.w0 / fuel.delta           # bulk density (lb/ft³)
    beta = rho_b / _RHO_P                  # packing ratio
    beta_op = 3.348 * fuel.sigma ** (-0.8189)   # optimum packing ratio

    if beta <= 0 or beta_op <= 0:
        return 0.0, 0.0

    # ── Step 2: Moisture damping η_M (Rothermel eq. 29) ───────────────────
    if fuel.Mx <= 0:
        return 0.0, 0.0
    r = Mf / fuel.Mx                       # moisture ratio
    if r >= 1.0:
        return 0.0, 0.0                    # fuel too wet: no spread
    eta_M = 1.0 - 2.59 * r + 5.11 * r**2 - 3.52 * r**3
    eta_M = max(0.0, eta_M)

    # ── Step 3: Mineral damping η_S (Rothermel eq. 31) ────────────────────
    eta_S = 0.174 * fuel.Se ** (-0.19)

    # ── Step 4: Net fuel load w_n (Rothermel eq. 59) ──────────────────────
    w_n = fuel.w0 * (1.0 - fuel.ST)

    # ── Step 5: Reaction velocity Γ' (Rothermel eqs. 36–38) ──────────────
    sigma = fuel.sigma
    Gamma_max = sigma**1.5 / (495.0 + 0.0594 * sigma**1.5)
    A = 1.0 / (4.774 * sigma**0.1 - 7.27)
    ratio = beta / beta_op
    # Guard against ratio = 0 (would cause 0^A problems)
    if ratio <= 0:
        return 0.0, 0.0
    Gamma_prime = Gamma_max * (ratio ** A) * math.exp(A * (1.0 - ratio))

    # ── Step 6: Reaction intensity I_R (Rothermel eq. 52) ─────────────────
    I_R = Gamma_prime * w_n * _HEAT_CONTENT * eta_M * eta_S

    if I_R <= 0:
        return 0.0, 0.0

    # ── Step 7: Propagating flux ratio ξ (Rothermel eq. 42) ──────────────
    xi_denom = 192.0 + 0.2595 * sigma
    if xi_denom <= 0:
        return 0.0, I_R
    xi = math.exp((0.792 + 0.681 * sigma**0.5) * (beta + 0.1)) / xi_denom

    # ── Step 8: Effective heating number ε (Rothermel eq. 14) ─────────────
    epsilon = math.exp(-138.0 / sigma) if sigma > 0 else 0.0

    # ── Step 9: Heat of preignition Q_ig (Rothermel eq. 12) ──────────────
    Q_ig = 250.0 + 1116.0 * Mf

    # ── Step 10: Wind coefficient φ_w (Rothermel eqs. 47–49) ─────────────
    C = 7.47 * math.exp(-0.133 * sigma**0.55)
    B = 0.02526 * sigma**0.54
    E = 0.715 * math.exp(-3.59e-4 * sigma)

    U_safe = max(0.0, U_midflame_ftmin)
    # U is already in ft/min — Rothermel eq. 47 uses ft/min directly.
    phi_w = C * U_safe ** B * (beta / beta_op) ** (-E) if U_safe > 0 and beta_op > 0 else 0.0

    # ── Step 11: Assemble ROS R (Rothermel eq. 52) ────────────────────────
    denominator = rho_b * epsilon * Q_ig
    if denominator <= 0:
        return 0.0, I_R

    R_ftmin = (I_R * xi * (1.0 + phi_w + phi_s)) / denominator
    return max(0.0, R_ftmin), I_R


def _phi_slope(
    beta: float,
    slope_deg: float,
    aspect_deg: float,
    bearing_deg: float,
) -> float:
    """Rothermel slope coefficient φ_s projected toward a bearing.

    Formula (Rothermel eq. 51):
        φ_s = 5.275 × β^(−0.3) × tan²(θ) × alignment

    where alignment = max(0, cos(bearing − uphill_direction)).
    """
    if slope_deg <= 0 or beta <= 0:
        return 0.0

    uphill_dir = (aspect_deg + 180.0) % 360.0
    diff = _angular_diff(bearing_deg, uphill_dir)
    alignment = max(0.0, math.cos(math.radians(abs(diff))))

    if alignment <= 0:
        return 0.0

    # Cap slope at 80° to avoid tan(90°) = infinity
    theta = math.radians(min(slope_deg, 80.0))
    tan_theta = math.tan(theta)

    return 5.275 * beta ** (-0.3) * tan_theta**2 * alignment


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 — Byram (1959) Fireline Intensity
# ═══════════════════════════════════════════════════════════════════════════════

def _byram_intensity(
    fuel: _FuelParams,
    Mf: float,
    R_ftmin: float,
) -> float:
    """Byram (1959) fireline intensity I_B in kW/m.

    Formula:  I_B = H × w_c × R_mmin / 60

    Where:
        H    = 18,600 kJ/kg (low heat of combustion for forest fuels)
        w_c  = fuel consumed in flaming front (kg/m²)
        R    = rate of spread converted to m/min

    Parameters
    ----------
    fuel     : Fuel model parameters.
    Mf       : Dead fuel moisture fraction.
    R_ftmin  : Surface fire ROS (ft/min).

    Returns
    -------
    I_B : Fireline intensity (kW/m).
    """
    H = 18600.0  # kJ/kg

    # Available fuel (fraction that burns in the flaming front)
    if fuel.Mx <= 0:
        return 0.0
    burn_frac = max(0.0, 1.0 - Mf / fuel.Mx)
    w_c_lbft2 = fuel.w0 * (1.0 - fuel.ST) * burn_frac
    w_c_kgm2 = w_c_lbft2 * 4.882    # 1 lb/ft² = 4.882 kg/m²

    # Convert ROS: ft/min → m/min
    R_mmin = R_ftmin * 0.3048

    # I_B = H × w_c × R / 60 → kW/m
    I_B = H * w_c_kgm2 * R_mmin / 60.0
    return max(0.0, I_B)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5 — Van Wagner (1977) Crown Fire + Scott & Reinhardt (2001)
# ═══════════════════════════════════════════════════════════════════════════════

def _crown_fire_assessment(
    surface_R_ftmin: float,
    I_B_kwm: float,
    cbh_m: float | None,
    cbd_kgm3: float | None,
    FMC: float = 1.0,
) -> tuple[float, str]:
    """Crown fire initiation and spread assessment.

    Van Wagner (1977) checks whether surface fireline intensity exceeds
    the critical intensity for crown fire initiation. Scott & Reinhardt
    (2001) then classify the crown fire as passive or active.

    Parameters
    ----------
    surface_R_ftmin : Surface fire ROS (ft/min) from Rothermel.
    I_B_kwm         : Byram fireline intensity (kW/m).
    cbh_m           : Canopy base height (metres). None → no crown fire.
    cbd_kgm3        : Canopy bulk density (kg/m³). None → no crown fire.
    FMC             : Foliar moisture content (fraction; 1.0 = 100 %).

    Returns
    -------
    (total_R_ftmin, status)
        total_R_ftmin : Total ROS including crown fire effect (ft/min).
        status        : "surface" | "passive_crown" | "active_crown"
    """
    # If canopy data is missing → surface fire only
    if cbh_m is None or cbd_kgm3 is None:
        return surface_R_ftmin, "surface"
    if cbh_m <= 0 or cbd_kgm3 <= 0:
        return surface_R_ftmin, "surface"

    # ── Van Wagner (1977): critical fireline intensity ────────────────────
    # I_0 = (0.010 × CBH × (460 + 25.9 × FMC_pct))^1.5   (kW/m)
    FMC_pct = FMC * 100.0
    I_0 = (0.010 * cbh_m * (460.0 + 25.9 * FMC_pct)) ** 1.5

    if I_B_kwm < I_0:
        return surface_R_ftmin, "surface"

    # Crown fire initiated — classify as passive or active
    # Scott & Reinhardt (2001): critical mass flow rate for active crown fire
    # R_0_active = 3.0 / CBD   (m/min)
    R_mmin = surface_R_ftmin * 0.3048
    R_0_active = 3.0 / cbd_kgm3

    if R_mmin < R_0_active:
        # Passive (torching) — moderate ROS boost
        boost = 1.0 + 0.5 * (R_mmin / R_0_active)
        return surface_R_ftmin * boost, "passive_crown"
    else:
        # Active crown fire — R_active ≈ 3.34 × surface ROS
        # Capped at 800 ft/min (~14.6 km/h) — physical upper bound for
        # sustained active crown fire in North American conifer stands.
        active_R = min(surface_R_ftmin * 3.34, 800.0)
        return active_R, "active_crown"


# ═══════════════════════════════════════════════════════════════════════════════
# PART 6 — Anderson (1983) Elliptical Fire Shape
# ═══════════════════════════════════════════════════════════════════════════════

def _elliptical_ros(
    head_R_ftmin: float,
    U_midflame_mph: float,
    bearing_deg: float,
    wind_from_deg: float,
) -> float:
    """Anderson (1983) elliptical fire shape: directional ROS adjustment.

    Wildfire does NOT spread as a circle. It spreads as an ellipse:
      - Maximum (head) ROS in the downwind direction
      - Minimum (backing) ROS directly upwind
      - Intermediate (flank) ROS perpendicular to wind

    Length-to-breadth ratio (Anderson 1983, eq. 1):
        LB = 0.936 × exp(0.2566 × U) + 0.461 × exp(−0.1548 × U) − 0.397

    Parameters
    ----------
    head_R_ftmin   : Head fire ROS from Rothermel (ft/min).
    U_midflame_mph : Midflame wind speed (mph, scalar — NOT projected).
    bearing_deg    : Bearing to this specific neighbour (degrees).
    wind_from_deg  : Meteorological wind-from direction (degrees).

    Returns
    -------
    Directional ROS (ft/min) for the given bearing.
    """
    if head_R_ftmin <= 0:
        return 0.0

    # ── Length-to-breadth ratio ────────────────────────────────────────────
    U_eff = max(0.0, U_midflame_mph)
    LB = (0.936 * math.exp(0.2566 * U_eff)
          + 0.461 * math.exp(-0.1548 * U_eff)
          - 0.397)
    LB = max(1.0, min(LB, 8.0))

    # ── Head / Flank / Backing ROS ────────────────────────────────────────
    flank_R = head_R_ftmin / LB
    back_R = head_R_ftmin / (LB * LB)

    # ── Angular offset from head (downwind) direction ─────────────────────
    downwind_dir = (wind_from_deg + 180.0) % 360.0
    abs_diff = abs(_angular_diff(bearing_deg, downwind_dir))

    # ── Smooth interpolation (Prometheus/FARSITE-style) ───────────────────
    cos_a = math.cos(math.radians(abs_diff))
    sin_a = abs(math.sin(math.radians(abs_diff)))

    if abs_diff <= 90.0:
        # Forward half: head → flank
        ros = head_R_ftmin * cos_a + flank_R * sin_a
    else:
        # Backward half: flank → back
        cos_b = math.cos(math.radians(abs_diff - 90.0))
        sin_b = abs(math.sin(math.radians(abs_diff - 90.0)))
        ros = flank_R * cos_b + back_R * sin_b

    return max(0.0, ros)


# ═══════════════════════════════════════════════════════════════════════════════
# Geometric helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing (degrees, 0 = N, clockwise) from point 1 to point 2."""
    lat1r, lon1r, lat2r, lon2r = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2r - lon1r
    x = math.sin(dlon) * math.cos(lat2r)
    y = (math.cos(lat1r) * math.sin(lat2r)
         - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _angular_diff(a: float, b: float) -> float:
    """Smallest signed angular difference a − b, in [−180, 180] degrees."""
    return (a - b + 180) % 360 - 180


def _weighted_circular_mean(bearings: list[float], weights: list[float]) -> float:
    """Circular mean of bearings weighted by spread rates. Returns [0, 360)."""
    total_w = sum(weights)
    if total_w <= 0:
        return 0.0
    sin_sum = sum(w * math.sin(math.radians(b)) for b, w in zip(bearings, weights, strict=False))
    cos_sum = sum(w * math.cos(math.radians(b)) for b, w in zip(bearings, weights, strict=False))
    return (math.degrees(math.atan2(sin_sum, cos_sum)) + 360) % 360


# ═══════════════════════════════════════════════════════════════════════════════
# Fuel model lookup
# ═══════════════════════════════════════════════════════════════════════════════

def _get_fuel_params(fbfm40: float | None) -> _FuelParams | None:
    """Look up FBFM40 fuel parameters. Returns None for non-burnable/nodata."""
    if fbfm40 is None or (isinstance(fbfm40, float) and math.isnan(fbfm40)):
        return _FBFM40_DEFAULT
    code = int(fbfm40)
    if code == _FBFM40_NODATA:
        return None
    if code in _NB_CODES:
        return None
    return _FBFM40_PARAMS.get(code, _FBFM40_DEFAULT)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 7 — Main Simulator Class
# ═══════════════════════════════════════════════════════════════════════════════

class PythonFireSpreadSimulator:
    """Monte Carlo fire spread simulator.

    Public interface
    ----------------
    simulate_monte_carlo() — N=100 perturbed-weather runs → burn probabilities,
                             spread speed/direction, fire behavior summary.

    Internal helper
    ---------------
    _simulate_once()       — single deterministic run used internally by MC
                             and by propagation/sensitivity helpers.

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.read_parquet("fused_2026-03-31.parquet")
    >>> sim = PythonFireSpreadSimulator()
    >>> result = sim.simulate_monte_carlo(df, "822937fffffffff", ignition_prob=0.30)
    >>> print(result["spread_speed_kmh"])       # p90 spread speed
    >>> print(result["crown_fire_probability"]) # fraction of runs with crown fire
    """

    def __init__(
        self,
        wind_reduction_factor: float = _WIND_REDUCTION_10M_TO_MIDFLAME,
    ) -> None:
        self.wind_reduction_factor = wind_reduction_factor

    def _simulate_once(
        self,
        df: pd.DataFrame,
        ignition_grid_id: str,
        ignition_prob: float,
    ) -> dict[str, Any]:
        """Internal: single deterministic run used by MC and propagation helpers.

        Not part of the public API — call simulate_monte_carlo() instead.
        """
        ignition_prob = float(max(0.0, min(1.0, ignition_prob)))
        warnings: list[str] = []

        # ==================================================================
        # 1. Resolve ignition cell data
        # ==================================================================
        try:
            import h3 as h3lib
        except ImportError as exc:
            raise ImportError("h3 package is required: pip install h3") from exc

        ignition_rows = df[df["grid_id"] == ignition_grid_id]
        if ignition_rows.empty:
            raise ValueError(
                f"Ignition cell '{ignition_grid_id}' not found in df. "
                f"Available grid_ids: {df['grid_id'].tolist()[:10]}"
            )
        ign = ignition_rows.iloc[0]

        # Weather
        # wind_speed_10m is stored in km/h in the fused parquet (Open-Meteo output).
        # Rothermel equations need m/s → convert here before any further unit work.
        _wind_kmh = _safe_float(ign, "wind_speed_10m", 0.0, warnings)
        wind_speed_ms = _wind_kmh / 3.6   # km/h → m/s
        wind_dir = _safe_float(ign, "wind_direction_10m", 0.0, warnings)
        rh = _safe_float(ign, "relative_humidity_2m", 50.0, warnings)
        temp_c = _safe_float(ign, "temperature_2m", 20.0, warnings)
        days_precip = _safe_float(ign, "days_since_last_precipitation", 3.0, warnings)

        # Terrain
        ign_slope = _safe_float(ign, "slope_degrees", 0.0, warnings)
        ign_aspect = _safe_float(ign, "aspect_degrees", 0.0, warnings)
        ign_fuel_code = _safe_float(ign, "fuel_model_fbfm40", None, warnings)

        # Canopy structure
        ign_cbh = _safe_float(ign, "canopy_base_height_m", None, warnings)
        ign_cbd = _safe_float(ign, "canopy_bulk_density", None, warnings)
        vpd = _safe_float(ign, "vpd", 2.0, warnings) or 2.0

        # ==================================================================
        # 2. Compute dead fuel moisture (EMC model) + foliar moisture (FMC)
        # ==================================================================
        Mf = _estimate_dfmc(rh, temp_c, days_precip)
        # FMC drives crown fire critical intensity (Van Wagner 1977).
        # Drought-stressed summer canopies have lower FMC → lower I_0 →
        # crown fire initiates more easily (Creek, Carr, Thomas fires).
        FMC = _estimate_fmc(temp_c, vpd)

        # ==================================================================
        # 3. Wind conversion: 10 m → midflame
        # ==================================================================
        # Andrews (2012): U_midflame = U_10m × WAF
        U_midflame_mph = wind_speed_ms * 2.23694 * self.wind_reduction_factor
        U_midflame_ftmin = U_midflame_mph * 88.0

        # Reference directions
        wind_spread_dir = (wind_dir + 180.0) % 360.0
        slope_spread_dir = (ign_aspect + 180.0) % 360.0 if ign_slope > 0 else None

        # ==================================================================
        # 4. Find H3 ring-1 neighbours
        # ==================================================================
        try:
            neighbour_set = set(h3lib.grid_disk(ignition_grid_id, 1)) - {ignition_grid_id}
            neighbours = sorted(neighbour_set)
        except Exception as exc:
            raise ValueError(
                f"Failed to compute H3 neighbours for '{ignition_grid_id}': {exc}"
            ) from exc

        ign_lat, ign_lon = h3lib.cell_to_latlng(ignition_grid_id)

        # ==================================================================
        # 5. Compute spread rate to each neighbour (Rothermel + crown + ellipse)
        # ==================================================================
        neighbour_details: list[dict[str, Any]] = []
        spread_rates: list[float] = []
        bearings_list: list[float] = []
        max_I_B = 0.0
        max_crown_status = "surface"

        for nb_id in neighbours:
            nb_lat, nb_lon = h3lib.cell_to_latlng(nb_id)
            bear = _bearing(ign_lat, ign_lon, nb_lat, nb_lon)

            # ── Resolve neighbour properties ──────────────────────────────
            nb_rows = df[df["grid_id"] == nb_id]
            if not nb_rows.empty:
                nb = nb_rows.iloc[0]
                nb_slope = _safe_float(nb, "slope_degrees", ign_slope)
                nb_aspect = _safe_float(nb, "aspect_degrees", ign_aspect)
                nb_fuel_code = _safe_float(nb, "fuel_model_fbfm40", ign_fuel_code)
                nb_cbh = _safe_float(nb, "canopy_base_height_m", ign_cbh)
                nb_cbd = _safe_float(nb, "canopy_bulk_density", ign_cbd)
            else:
                nb_slope = ign_slope
                nb_aspect = ign_aspect
                nb_fuel_code = ign_fuel_code
                nb_cbh = ign_cbh
                nb_cbd = ign_cbd

            # ── Fuel lookup ───────────────────────────────────────────────
            fuel = _get_fuel_params(nb_fuel_code)
            if fuel is None:
                # Non-burnable → zero spread
                spread_rates.append(0.0)
                bearings_list.append(bear)
                neighbour_details.append({
                    "neighbour_id": nb_id,
                    "bearing_deg": round(bear, 1),
                    "spread_rate_kmh": 0.0,
                    "surface_ros_kmh": 0.0,
                    "crown_status": "non_burnable",
                    "byram_intensity_kwm": 0.0,
                    "in_dataset": not nb_rows.empty,
                })
                continue

            # ── Rothermel intermediate values ─────────────────────────────
            rho_b = fuel.w0 / fuel.delta if fuel.delta > 0 else 0
            beta = rho_b / _RHO_P if rho_b > 0 else 0

            # Slope coefficient projected to this bearing
            phi_s = _phi_slope(beta, nb_slope, nb_aspect, bear)

            # ── Head fire ROS (full wind, no slope — for crown fire check) ─
            head_R_ftmin, head_I_R = _rothermel_surface_ros(
                fuel, Mf, U_midflame_ftmin, 0.0
            )

            # ── Byram intensity (from head fire ROS) ──────────────────────
            I_B = _byram_intensity(fuel, Mf, head_R_ftmin)

            # ── Per-bearing directional ROS (wind projected + slope) ──────
            # Project wind speed toward this specific bearing direction and
            # combine with slope coefficient. This gives correct directional
            # ROS without double-counting (no need for max with ellipse).
            downwind_dir = (wind_dir + 180.0) % 360.0
            diff_rad = math.radians(abs(_angular_diff(bear, downwind_dir)))
            U_proj_ftmin = U_midflame_ftmin * max(0.0, math.cos(diff_rad))
            dir_surface_R_ftmin, _ = _rothermel_surface_ros(
                fuel, Mf, U_proj_ftmin, phi_s
            )

            # ── Elliptical shape ROS (head fire only, no slope) ──────────
            ellipse_R_ftmin = _elliptical_ros(
                head_R_ftmin, U_midflame_mph, bear, wind_dir
            )

            # Best surface estimate for this bearing: max of directional
            # (slope+projected wind) and elliptical (wind-only shape)
            surface_R_ftmin = max(dir_surface_R_ftmin, ellipse_R_ftmin)

            # ── Crown fire applied to per-bearing surface ROS ────────────
            # Crown fire cap must be applied AFTER combining wind+slope to
            # prevent uncapped slope_R from overriding the crown limit.
            # FMC estimated from temperature + VPD (lower in summer/drought).
            crown_R_ftmin, crown_status = _crown_fire_assessment(
                surface_R_ftmin, I_B, nb_cbh, nb_cbd, FMC=FMC
            )

            final_R_ftmin = crown_R_ftmin

            # ── Convert to km/h and scale by ignition probability ─────────
            surface_ros_kmh = head_R_ftmin * _FTMIN_TO_KMH
            ros_kmh = final_R_ftmin * _FTMIN_TO_KMH * ignition_prob

            spread_rates.append(ros_kmh)
            bearings_list.append(bear)

            if max_I_B < I_B:
                max_I_B = I_B
                max_crown_status = crown_status

            neighbour_details.append({
                "neighbour_id": nb_id,
                "bearing_deg": round(bear, 1),
                "spread_rate_kmh": round(ros_kmh, 4),
                "surface_ros_kmh": round(surface_ros_kmh, 4),
                "head_ros_ftmin": round(head_R_ftmin, 2),
                "ellipse_ros_ftmin": round(ellipse_R_ftmin, 2),
                "dir_surface_ros_ftmin": round(dir_surface_R_ftmin, 2),
                "crown_status": crown_status,
                "byram_intensity_kwm": round(I_B, 1),
                "phi_slope": round(phi_s, 4),
                "fuel_model": int(nb_fuel_code) if nb_fuel_code is not None else None,
                "in_dataset": not nb_rows.empty,
            })

        # ==================================================================
        # 6. Aggregate: dominant direction + maximum speed
        # ==================================================================
        dominant_direction = _weighted_circular_mean(bearings_list, spread_rates)
        max_spread_rate = max(spread_rates) if spread_rates else 0.0

        # Dominant factor
        wind_contribution = abs(_angular_diff(dominant_direction, wind_spread_dir))
        slope_contribution = (
            abs(_angular_diff(dominant_direction, slope_spread_dir))
            if slope_spread_dir is not None else 90.0
        )
        if wind_contribution < 30.0:
            dominant_factor = "wind"
        elif slope_contribution < 30.0 and ign_slope >= 5.0:
            dominant_factor = "slope"
        else:
            dominant_factor = "balanced"

        # ==================================================================
        # 7. Assemble result
        # ==================================================================
        result: dict[str, Any] = {
            "ignition_cell": ignition_grid_id,
            "ignition_probability": round(ignition_prob, 4),
            "spread_direction_deg": round(dominant_direction, 1),
            "spread_speed_kmh": round(max_spread_rate, 4),
            "wind_spread_direction_deg": round(wind_spread_dir, 1),
            "slope_spread_direction_deg": (
                round(slope_spread_dir, 1) if slope_spread_dir is not None else None
            ),
            "dominant_factor": dominant_factor,
            "crown_fire_status": max_crown_status,
            "byram_intensity_kwm": round(max_I_B, 1),
            "dead_fuel_moisture_pct": round(Mf * 100, 1),
            "foliar_moisture_content_pct": round(FMC * 100, 1),
            "inputs_used": {
                "wind_speed_10m_ms": round(wind_speed_ms, 2),
                "midflame_wind_mph": round(U_midflame_mph, 2),
                "wind_from_direction_deg": round(wind_dir, 1),
                "temperature_c": round(temp_c, 1),
                "relative_humidity_pct": round(rh, 1),
                "days_since_precip": round(days_precip, 1),
                "dead_fuel_moisture_fraction": round(Mf, 4),
                "ignition_cell_slope_deg": round(ign_slope, 1),
                "ignition_cell_aspect_deg": round(ign_aspect, 1),
                "ignition_cell_fbfm40": ign_fuel_code,
                "canopy_base_height_m": round(ign_cbh, 2) if ign_cbh is not None else None,
                "canopy_bulk_density_kgm3": round(ign_cbd, 4) if ign_cbd is not None else None,
                "wind_reduction_factor": self.wind_reduction_factor,
            },
            "neighbour_details": neighbour_details,
            "warnings": warnings,
        }

        logger.info(
            "FireSpread | cell=%s | prob=%.2f | dir=%.1f° | speed=%.4f km/h | "
            "DFMC=%.1f%% | crown=%s | I_B=%.1f kW/m | factor=%s",
            ignition_grid_id, ignition_prob, dominant_direction,
            max_spread_rate, Mf * 100, max_crown_status, max_I_B,
            dominant_factor,
        )
        return result

    # ──────────────────────────────────────────────────────────────────────────
    def simulate_monte_carlo(
        self,
        df: pd.DataFrame,
        ignition_grid_id: str,
        ignition_prob: float,
        n_simulations: int = 100,
        horizon_hours: float = 24.0,
        rng_seed: int | None = 42,
    ) -> dict[str, Any]:
        """Monte Carlo fire spread — N perturbed-weather runs → burn probabilities.

        Fully Python-based stochastic simulation with no external binary dependency.
        Each of the N runs uses the same Rothermel + Van Wagner + Anderson physics
        as ``simulate()``, but with weather inputs randomly perturbed around the
        observed values using physically motivated distributions.

        Perturbation distributions
        --------------------------
        Wind speed      : Log-normal  σ=0.25  (preserves non-negativity;
                          ~25% coefficient of variation — NOAA WFO guidance)
        Wind direction  : Wrapped normal  σ=25°  (directional variability)
        Relative humidity: Normal  σ=8 %  (synoptic uncertainty, clamped 1–99)
        Temperature     : Normal  σ=2.5°C  (clamped –20 to 60°C)
        Days since precip: Normal  σ=0.5 d  (clamped 0–30)
        Slope / fuel / canopy: NOT perturbed (static LANDFIRE layers)

        Burn probability definition
        ---------------------------
        A neighbour cell is counted as "burned" in run *i* if the simulated
        spread rate toward it satisfies:

            speed_i  >=  intercell_distance_km / horizon_hours

        i.e. fire physically reaches the cell within ``horizon_hours`` at that
        run's spread rate.  Default horizon = 24 h at 22 km (res-5) spacing
        → threshold ≈ 1.04 km/h.

        Parameters
        ----------
        df              : Fused parquet DataFrame with ``grid_id`` column.
        ignition_grid_id: H3 cell where fire starts.
        ignition_prob   : OBJ-1 ignition probability (scales ROS).
        n_simulations   : Number of Monte Carlo runs (default 100).
        horizon_hours   : Time window for burn-probability threshold (default 24 h).
        rng_seed        : Numpy RNG seed for reproducibility (None = random).

        Returns
        -------
        dict with keys:
            spread_speed_kmh_mean/p50/p90/p95/max/std
            dominant_direction_deg, direction_uncertainty_deg
            crown_fire_probability
            neighbor_burn_probabilities   — {cell_id: probability}  ← main output
            max_neighbor_burn_probability
            spread_speed_kmh_all          — list of N values (for histogram)
            direction_deg_all             — list of N values (for wind rose)
            crown_status_all              — list of N strings
            byram_intensity_kwm_all       — list of N values
            deterministic_result          — single-run result for comparison
            perturbation_config           — perturbation parameters used
        """
        try:
            import h3 as h3lib
        except ImportError as exc:
            raise ImportError("h3 package required: pip install h3") from exc

        rng = np.random.default_rng(rng_seed)
        ignition_prob = float(max(0.0, min(1.0, ignition_prob)))
        warnings_mc: list[str] = []

        # ── Step 1: Extract ignition cell once ────────────────────────────────
        ignition_rows = df[df["grid_id"] == ignition_grid_id]
        if ignition_rows.empty:
            raise ValueError(
                f"Ignition cell '{ignition_grid_id}' not found in df."
            )
        ign = ignition_rows.iloc[0]

        # Base weather (will be perturbed each run)
        base_wind_kmh  = _safe_float(ign, "wind_speed_10m", 0.0,  warnings_mc) or 0.0
        base_wind_dir  = _safe_float(ign, "wind_direction_10m", 0.0, warnings_mc) or 0.0
        base_rh        = _safe_float(ign, "relative_humidity_2m", 50.0, warnings_mc) or 50.0
        base_temp      = _safe_float(ign, "temperature_2m", 20.0, warnings_mc) or 20.0
        base_days      = _safe_float(ign, "days_since_last_precipitation", 3.0, warnings_mc) or 3.0
        base_vpd       = _safe_float(ign, "vpd", 2.0, warnings_mc) or 2.0

        # Static (no perturbation)
        ign_slope      = _safe_float(ign, "slope_degrees", 0.0, warnings_mc) or 0.0
        ign_aspect     = _safe_float(ign, "aspect_degrees", 0.0, warnings_mc) or 0.0
        ign_fuel_code  = _safe_float(ign, "fuel_model_fbfm40", None, warnings_mc)
        ign_cbh        = _safe_float(ign, "canopy_base_height_m", None, warnings_mc)
        ign_cbd        = _safe_float(ign, "canopy_bulk_density", None, warnings_mc)

        # ── Step 2: Discover neighbours + pre-cache static data once ──────────
        neighbour_set = set(h3lib.grid_disk(ignition_grid_id, 1)) - {ignition_grid_id}
        neighbours    = sorted(neighbour_set)
        ign_lat, ign_lon = h3lib.cell_to_latlng(ignition_grid_id)

        nb_static: list[dict[str, Any]] = []
        for nb_id in neighbours:
            nb_lat, nb_lon = h3lib.cell_to_latlng(nb_id)
            bear    = _bearing(ign_lat, ign_lon, nb_lat, nb_lon)
            nb_rows = df[df["grid_id"] == nb_id]
            if not nb_rows.empty:
                nb = nb_rows.iloc[0]
                nb_slope     = _safe_float(nb, "slope_degrees",          ign_slope)     or ign_slope
                nb_aspect    = _safe_float(nb, "aspect_degrees",         ign_aspect)    or ign_aspect
                nb_fuel_code = _safe_float(nb, "fuel_model_fbfm40",      ign_fuel_code)
                nb_cbh       = _safe_float(nb, "canopy_base_height_m",   ign_cbh)
                nb_cbd       = _safe_float(nb, "canopy_bulk_density",    ign_cbd)
            else:
                nb_slope, nb_aspect = ign_slope, ign_aspect
                nb_fuel_code, nb_cbh, nb_cbd = ign_fuel_code, ign_cbh, ign_cbd
            nb_static.append({
                "id": nb_id, "bearing": bear,
                "slope": nb_slope, "aspect": nb_aspect,
                "fuel_code": nb_fuel_code, "cbh": nb_cbh, "cbd": nb_cbd,
            })

        # H3 res-5 intercell distance ≈ 25 km; speed threshold to reach in horizon
        INTERCELL_KM    = 25.0
        speed_threshold = INTERCELL_KM / horizon_hours   # km/h

        # ── Step 3: Pre-generate all N perturbed weather samples ──────────────
        # Wind speed: log-normal — preserves non-negativity, models gustiness
        base_wind_ms = base_wind_kmh / 3.6
        wind_ms_samples = np.maximum(
            0.0,
            base_wind_ms * np.exp(rng.normal(0.0, 0.25, n_simulations))
        )
        # Wind direction: wrapped-normal ±25°
        wind_dir_samples = (base_wind_dir + rng.normal(0.0, 25.0, n_simulations)) % 360.0
        # RH: normal ±8 %, clamped [1, 99]
        rh_samples   = np.clip(rng.normal(base_rh,   8.0, n_simulations),  1.0,  99.0)
        # Temp: normal ±2.5°C, clamped [-20, 60]
        temp_samples = np.clip(rng.normal(base_temp, 2.5, n_simulations), -20.0, 60.0)
        # Days since precip: normal ±0.5 d, clamped [0, 30]
        days_samples = np.clip(rng.normal(base_days, 0.5, n_simulations),  0.0,  30.0)

        # ── Step 4: N simulation runs ─────────────────────────────────────────
        all_speeds:      list[float] = []
        all_directions:  list[float] = []
        all_crown:       list[str]   = []
        all_intensities: list[float] = []

        # Accumulate per-neighbour spread rates across all runs
        nb_spread_runs: dict[str, list[float]] = {nb["id"]: [] for nb in nb_static}

        for i in range(n_simulations):
            w_ms   = float(wind_ms_samples[i])
            w_dir  = float(wind_dir_samples[i])
            rh_i   = float(rh_samples[i])
            temp_i = float(temp_samples[i])
            days_i = float(days_samples[i])

            # Fuel moisture for this weather sample
            Mf_i  = _estimate_dfmc(rh_i, temp_i, days_i)
            FMC_i = _estimate_fmc(temp_i, base_vpd)   # VPD correlated w/ T/RH

            # Wind unit conversions
            U_mph_i    = w_ms * 2.23694 * self.wind_reduction_factor
            U_ftmin_i  = U_mph_i * 88.0

            run_max_speed = 0.0
            run_dirs:    list[float] = []
            run_weights: list[float] = []
            run_max_IB   = 0.0
            run_crown    = "surface"

            for nb in nb_static:
                bear     = nb["bearing"]
                fuel     = _get_fuel_params(nb["fuel_code"])
                if fuel is None:
                    nb_spread_runs[nb["id"]].append(0.0)
                    continue

                rho_b = fuel.w0 / fuel.delta if fuel.delta > 0 else 0.0
                beta  = rho_b / _RHO_P if rho_b > 0 else 0.0
                phi_s = _phi_slope(beta, nb["slope"], nb["aspect"], bear)

                # Head-fire ROS (full wind, no slope) — drives crown + Byram
                head_R, _ = _rothermel_surface_ros(fuel, Mf_i, U_ftmin_i, 0.0)
                I_B       = _byram_intensity(fuel, Mf_i, head_R)

                # Directional ROS (wind projected onto bearing + slope)
                downwind_i  = (w_dir + 180.0) % 360.0
                diff_rad    = math.radians(abs(_angular_diff(bear, downwind_i)))
                U_proj      = U_ftmin_i * max(0.0, math.cos(diff_rad))
                dir_R, _    = _rothermel_surface_ros(fuel, Mf_i, U_proj, phi_s)

                # Elliptical shape ROS (head-fire only, no slope)
                ell_R      = _elliptical_ros(head_R, U_mph_i, bear, w_dir)
                surface_R  = max(dir_R, ell_R)

                # Crown fire
                crown_R, crown_status = _crown_fire_assessment(
                    surface_R, I_B, nb["cbh"], nb["cbd"], FMC=FMC_i
                )

                ros_kmh = crown_R * _FTMIN_TO_KMH * ignition_prob
                nb_spread_runs[nb["id"]].append(ros_kmh)

                run_dirs.append(bear)
                run_weights.append(ros_kmh)
                if ros_kmh > run_max_speed:
                    run_max_speed = ros_kmh
                if I_B > run_max_IB:
                    run_max_IB = I_B
                    run_crown  = crown_status

            all_speeds.append(run_max_speed)
            all_intensities.append(run_max_IB)
            all_crown.append(run_crown)

            if sum(run_weights) > 0:
                all_directions.append(
                    _weighted_circular_mean(run_dirs, run_weights)
                )
            else:
                all_directions.append(0.0)

        # ── Step 5: Aggregate ─────────────────────────────────────────────────
        speeds_arr = np.array(all_speeds)

        # Burn probability: fraction of runs where fire reaches cell in horizon
        nb_burn_probs: dict[str, float] = {
            nb["id"]: float(np.mean(
                np.array(nb_spread_runs[nb["id"]]) >= speed_threshold
            ))
            for nb in nb_static
        }

        # Crown fire probability
        crown_prob = sum(1 for c in all_crown if c != "surface") / n_simulations

        # Ensemble dominant direction (circular mean weighted by speed)
        ens_dir = _weighted_circular_mean(
            all_directions,
            [max(s, 1e-9) for s in all_speeds],
        )

        # Circular standard deviation → direction uncertainty
        angles_rad = np.radians(all_directions)
        R_bar      = math.sqrt(np.mean(np.cos(angles_rad))**2 +
                               np.mean(np.sin(angles_rad))**2)
        dir_std    = math.degrees(math.sqrt(-2.0 * math.log(max(R_bar, 1e-9))))

        # ── Step 6: Compute fire behavior summary from MC ensemble ───────────
        # Crown fire status — majority vote
        crown_counts: dict[str, int] = {}
        for c in all_crown:
            crown_counts[c] = crown_counts.get(c, 0) + 1
        mc_crown_status = max(crown_counts, key=crown_counts.get) if crown_counts else "surface"

        # Byram intensity mean
        mc_byram = float(np.mean(all_intensities)) if all_intensities else 0.0

        # Dead fuel moisture from base weather (un-perturbed)
        mc_dfmc_pct = _estimate_dfmc(base_rh, base_temp, base_days) * 100.0
        mc_fmc_pct  = _estimate_fmc(base_temp, base_vpd) * 100.0

        # Dominant spread factor from ens_dir vs base wind direction
        base_downwind = (base_wind_dir + 180.0) % 360.0
        wind_align = abs(_angular_diff(ens_dir, base_downwind))
        if wind_align < 30.0:
            mc_dominant_factor = "wind"
        elif ign_slope >= 5.0:
            mc_dominant_factor = "slope"
        else:
            mc_dominant_factor = "balanced"

        mc_inputs = {
            "wind_speed_10m_ms":       round(base_wind_ms,  2),
            "midflame_wind_mph":       round(base_wind_ms * 2.23694 * self.wind_reduction_factor, 2),
            "wind_from_direction_deg": round(base_wind_dir, 1),
            "temperature_c":           round(base_temp,     1),
            "relative_humidity_pct":   round(base_rh,       1),
            "days_since_precip":       round(base_days,      1),
            "ignition_cell_slope_deg": round(ign_slope,     1),
        }

        logger.info(
            "MC FireSpread | cell=%s | n=%d | p50=%.4f km/h | p90=%.4f km/h | "
            "crown_prob=%.1f%% | max_burn_prob=%.2f | dir=%.1f°±%.1f°",
            ignition_grid_id, n_simulations,
            float(np.percentile(speeds_arr, 50)),
            float(np.percentile(speeds_arr, 90)),
            crown_prob * 100,
            max(nb_burn_probs.values()) if nb_burn_probs else 0.0,
            ens_dir, dir_std,
        )

        return {
            "model": "monte_carlo",
            "n_simulations": n_simulations,
            "horizon_hours": horizon_hours,
            "intercell_distance_km": INTERCELL_KM,
            "burn_speed_threshold_kmh": round(speed_threshold, 4),
            "ignition_cell": ignition_grid_id,
            "ignition_probability": round(ignition_prob, 4),
            # ── Fire behavior summary (MC ensemble) ───────────────────────
            "spread_direction_deg":       round(ens_dir,   1),
            "spread_speed_kmh":           round(float(np.percentile(speeds_arr, 90)), 4),
            "crown_fire_status":          mc_crown_status,
            "byram_intensity_kwm":        round(mc_byram,  1),
            "dead_fuel_moisture_pct":     round(mc_dfmc_pct, 1),
            "foliar_moisture_content_pct": round(mc_fmc_pct, 1),
            "dominant_factor":            mc_dominant_factor,
            "inputs_used":                mc_inputs,
            # ── Spread speed distribution ──────────────────────────────────
            "spread_speed_kmh_mean": round(float(np.mean(speeds_arr)),           4),
            "spread_speed_kmh_p50":  round(float(np.percentile(speeds_arr, 50)), 4),
            "spread_speed_kmh_p90":  round(float(np.percentile(speeds_arr, 90)), 4),
            "spread_speed_kmh_p95":  round(float(np.percentile(speeds_arr, 95)), 4),
            "spread_speed_kmh_max":  round(float(np.max(speeds_arr)),            4),
            "spread_speed_kmh_std":  round(float(np.std(speeds_arr)),            4),
            # ── Direction ─────────────────────────────────────────────────
            "dominant_direction_deg":    round(ens_dir,  1),
            "direction_uncertainty_deg": round(dir_std,  1),
            # ── Crown fire ────────────────────────────────────────────────
            "crown_fire_probability": round(crown_prob, 4),
            # ── Burn probabilities ────────────────────────────────────────
            "neighbor_burn_probabilities": {
                k: round(v, 4) for k, v in nb_burn_probs.items()
            },
            "max_neighbor_burn_probability": round(
                max(nb_burn_probs.values()) if nb_burn_probs else 0.0, 4
            ),
            # ── Full distributions (histogram / wind-rose plotting) ───────
            "spread_speed_kmh_all":    [round(s, 4) for s in all_speeds],
            "direction_deg_all":       [round(d, 1) for d in all_directions],
            "crown_status_all":        all_crown,
            "byram_intensity_kwm_all": [round(v, 1) for v in all_intensities],
            # ── Perturbation config ───────────────────────────────────────
            "perturbation_config": {
                "wind_speed_lognormal_sigma":      0.25,
                "wind_direction_normal_sigma_deg": 25.0,
                "rh_normal_sigma_pct":             8.0,
                "temperature_normal_sigma_c":      2.5,
                "days_precip_normal_sigma":        0.5,
                "rng_seed": rng_seed,
                "base_weather": {
                    "wind_speed_kmh":        round(base_wind_kmh, 2),
                    "wind_direction_deg":    round(base_wind_dir, 1),
                    "relative_humidity_pct": round(base_rh,       1),
                    "temperature_c":         round(base_temp,      1),
                    "days_since_precip":     round(base_days,      1),
                },
            },
            "warnings": warnings_mc,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Safe data extraction utility
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_float(
    row: pd.Series,
    col: str,
    default: float | None = None,
    warnings: list[str] | None = None,
) -> float | None:
    """Extract a float from a Series row. Returns default on missing/NaN."""
    if col not in row.index:
        if warnings is not None:
            warnings.append(f"Column '{col}' not found — using default {default}")
        return default
    val = row[col]
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        if warnings is not None:
            warnings.append(f"Column '{col}' has non-numeric value '{val}' — using default {default}")
        return default
