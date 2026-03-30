"""State machine — determines report type from ML pipeline outputs.

Pure logic module (except IncidentTracker persistence). Receives pipeline
result dict and returns the operational mode + optional emergency sub-state.

IncidentTracker provides temporal state transitions for EMERGENCY sub-states:
  ACTIVE_FIRE → INTERIM → POST_FIRE → FINAL
based on elapsed time since last hotspot detection.
"""

from __future__ import annotations

import enum
import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OperationalMode(enum.Enum):
    """Top-level system mode."""

    QUIET = "QUIET"          # Low risk, no FIRMS hotspots
    ACTIVE = "ACTIVE"        # Elevated risk, no confirmed fire
    EMERGENCY = "EMERGENCY"  # Active fire confirmed


class EmergencySubState(enum.Enum):
    """Sub-states within EMERGENCY mode."""

    ACTIVE_FIRE = "ACTIVE_FIRE"
    INTERIM = "INTERIM"
    POST_FIRE = "POST_FIRE"
    FINAL = "FINAL"


# ---------------------------------------------------------------------------
# Mode resolution
# ---------------------------------------------------------------------------

VALID_RISK_LEVELS = {"LOW", "MODERATE", "HIGH", "CRITICAL"}


def resolve_mode(
    pipeline_result: dict[str, Any],
) -> tuple[OperationalMode, EmergencySubState | None, bool]:
    """Determine operational mode from ML pipeline outputs.

    Uses a 9-cell decision matrix plus an ``is_deployable`` safety gate.

    Parameters
    ----------
    pipeline_result:
        Raw dict from the pipeline (matches ``mock_pipeline_result.json``).
        Must contain ``risk_level`` (str) and ``firms_hotspot_count`` (int).
        Optionally contains ``is_deployable`` (bool, default ``True``).

    Returns
    -------
    tuple of (OperationalMode, EmergencySubState | None, disagreement_flag: bool)
        The third element is ``True`` when satellite hotspots conflict with
        the ML risk assessment (LOW/MODERATE risk but FIRMS hotspots > 0).

    Raises
    ------
    ValueError
        If required fields are missing or ``risk_level`` is unrecognised.

    Verification matrix::

        | risk_level     | firms_hotspot_count | is_deployable | → mode      | → disagreement_flag |
        |----------------|---------------------|---------------|-------------|---------------------|
        | LOW/MODERATE   | 0                   | True          | QUIET       | False               |
        | HIGH/CRITICAL  | 0                   | True          | ACTIVE      | False               |
        | LOW/MODERATE   | >0                  | True          | ACTIVE      | True                |
        | HIGH/CRITICAL  | >0                  | True          | EMERGENCY   | False               |
        | ANY            | ANY                 | False         | QUIET       | False               |
    """
    if "risk_level" not in pipeline_result:
        raise ValueError("pipeline_result missing required field: 'risk_level'")
    if "firms_hotspot_count" not in pipeline_result:
        raise ValueError("pipeline_result missing required field: 'firms_hotspot_count'")

    risk = pipeline_result["risk_level"].upper()
    firms_count: int = int(pipeline_result["firms_hotspot_count"])

    # Validate risk_level against known values
    if risk not in VALID_RISK_LEVELS:
        raise ValueError(
            f"Unknown risk_level: {risk!r}. Expected one of {VALID_RISK_LEVELS}"
        )

    # Gate 1: Non-deployable model → always QUIET, no disagreement.
    # Default is True (fail-open for backward compat with older pipeline
    # results that don't include the field).
    is_deployable = pipeline_result.get("is_deployable", True)
    if is_deployable is False:
        return OperationalMode.QUIET, None, False

    # Gate 2: Route by 9-cell matrix
    if firms_count == 0:
        if risk in ("HIGH", "CRITICAL"):
            return OperationalMode.ACTIVE, None, False
        # LOW, MODERATE
        return OperationalMode.QUIET, None, False

    # firms_count > 0
    if risk in ("HIGH", "CRITICAL"):
        return OperationalMode.EMERGENCY, EmergencySubState.ACTIVE_FIRE, False
    # LOW, MODERATE + firms > 0 → MODEL DISAGREEMENT
    return OperationalMode.ACTIVE, None, True


def mode_to_report_type(
    mode: OperationalMode,
    sub_state: EmergencySubState | None,
) -> str:
    """Map operational mode to the report type string.

    Returns one of: ``"daily"``, ``"high_risk"``, ``"incident"``, ``"final"``.
    """
    if mode == OperationalMode.QUIET:
        return "daily"
    if mode == OperationalMode.ACTIVE:
        return "high_risk"
    # EMERGENCY
    if sub_state == EmergencySubState.FINAL:
        return "final"
    return "incident"


# ---------------------------------------------------------------------------
# Admin Toggle
# ---------------------------------------------------------------------------

class AdminToggle:
    """Controls whether the human input channel is open.

    Phase 1 implementation: persists to/from ``reporting_config.yaml``
    (the ``admin_toggle.current_state`` field).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._persistence = config.get("persistence", "local")
        self._config_path: Path | None = config.get("_config_path")
        self._state: bool = self._load_from_persistence()

    # -- public API --

    @property
    def is_on(self) -> bool:  # noqa: FBT003
        """Return current toggle state."""
        return self._state

    def enable(self, admin_id: str) -> None:
        """Enable the human input channel."""
        self._state = True
        self._write_to_persistence(True)
        logger.info("AdminToggle ENABLED by %s", admin_id)

    def disable(self, admin_id: str) -> None:
        """Disable the human input channel."""
        self._state = False
        self._write_to_persistence(False)
        logger.info("AdminToggle DISABLED by %s", admin_id)

    # -- persistence helpers --

    def _load_from_persistence(self) -> bool:
        """Read toggle state from persistence backend."""
        if self._persistence == "local":
            return self._load_local()
        # Firestore placeholder — Phase 3
        logger.warning("Firestore persistence not implemented; using default.")
        return self._config.get("default", True)

    def _load_local(self) -> bool:
        """Read from the config dict (which was loaded from YAML)."""
        return bool(self._config.get("current_state", self._config.get("default", True)))

    def _write_to_persistence(self, state: bool) -> None:
        """Write toggle state back to persistence backend."""
        if self._persistence == "local":
            self._write_local(state)
            return
        logger.warning("Firestore persistence not implemented; skipping write.")

    def _write_local(self, state: bool) -> None:
        """Write state back to the YAML config file on disk."""
        if self._config_path is None:
            logger.warning("No config path set — toggle state not persisted to disk.")
            return
        try:
            with open(self._config_path, encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            raw.setdefault("admin_toggle", {})["current_state"] = state
            with open(self._config_path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(raw, fh, default_flow_style=False, sort_keys=False)
        except Exception:
            logger.exception("Failed to persist admin toggle state to %s", self._config_path)


# ---------------------------------------------------------------------------
# Incident Tracker — temporal sub-state transitions for EMERGENCY mode
# ---------------------------------------------------------------------------

# Default thresholds for sub-state transitions (in hours).
# Tag: # TODO: calibrate from production data
_DEFAULT_INTERIM_HOURS = 2       # ACTIVE_FIRE → INTERIM after no hotspots for 2h
_DEFAULT_POST_FIRE_HOURS = 12    # INTERIM → POST_FIRE after no hotspots for 12h
_DEFAULT_FINAL_HOURS = 48        # POST_FIRE → FINAL after no hotspots for 48h


class IncidentTracker:
    """Tracks active fire incidents and manages EMERGENCY sub-state transitions.

    Persists state to a YAML file so that sub-state transitions survive
    process restarts. Each incident is identified by a unique ``incident_id``
    and tracks temporal signals (last hotspot, report count, peak FRP).

    State transitions are based on elapsed time since last hotspot::

        ACTIVE_FIRE ──(no hotspots for 2h)──→ INTERIM
        INTERIM     ──(no hotspots for 12h)─→ POST_FIRE
        POST_FIRE   ──(no hotspots for 48h)─→ FINAL
        FINAL       ──(manual close)────────→ (incident archived)

        Any state ──(new hotspots detected)──→ ACTIVE_FIRE  (re-escalation)

    Parameters
    ----------
    state_file:
        Path to the YAML file for persisting incident state.
    config:
        Optional config dict with threshold overrides under
        ``incident_tracker`` key.
    """

    def __init__(
        self,
        state_file: Path,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._state_file = Path(state_file)
        cfg = (config or {}).get("incident_tracker", {})
        self._interim_hours = cfg.get("interim_hours", _DEFAULT_INTERIM_HOURS)
        self._post_fire_hours = cfg.get("post_fire_hours", _DEFAULT_POST_FIRE_HOURS)
        self._final_hours = cfg.get("final_hours", _DEFAULT_FINAL_HOURS)
        self._incidents: dict[str, dict[str, Any]] = self._load()

    # -- Public API --

    @property
    def active_incidents(self) -> dict[str, dict[str, Any]]:
        """Return all non-archived incidents."""
        return {
            k: v for k, v in self._incidents.items()
            if v.get("archived") is not True
        }

    @property
    def current_incident_id(self) -> str | None:
        """Return the ID of the most recent non-archived incident, if any."""
        active = self.active_incidents
        if not active:
            return None
        # Most recent by started_at
        return max(active, key=lambda k: active[k].get("started_at", ""))

    def update(
        self,
        pipeline_result: dict[str, Any],
        now: datetime | None = None,
    ) -> tuple[str, EmergencySubState]:
        """Update incident state from the latest pipeline result.

        If FIRMS hotspots are present and no active incident exists, a new
        incident is created. If hotspots are present and an incident exists,
        it's re-escalated to ACTIVE_FIRE.

        Parameters
        ----------
        pipeline_result:
            Must contain ``firms_hotspot_count`` (int). Optionally
            ``firms_hotspots`` (list of dicts with ``frp`` field).
        now:
            Current timestamp (defaults to UTC now).

        Returns
        -------
        tuple of (incident_id, EmergencySubState)
        """
        now = now or datetime.now(tz=UTC)
        now_iso = now.isoformat()
        firms_count = int(pipeline_result.get("firms_hotspot_count", 0))

        # Find current active incident
        incident_id = self.current_incident_id
        incident: dict[str, Any] | None = None
        if incident_id:
            incident = self._incidents[incident_id]

        if firms_count > 0:
            # New or re-escalated fire
            peak_frp = self._extract_peak_frp(pipeline_result)
            if incident is None:
                # New incident
                incident_id = str(uuid.uuid4())[:12]
                incident = {
                    "incident_id": incident_id,
                    "started_at": now_iso,
                    "last_hotspot_at": now_iso,
                    "peak_frp": peak_frp,
                    "report_count": 0,
                    "sub_state": EmergencySubState.ACTIVE_FIRE.value,
                    "archived": False,
                }
                self._incidents[incident_id] = incident
                logger.info("New incident created: %s", incident_id)
            else:
                # Re-escalate if needed
                incident["last_hotspot_at"] = now_iso
                if peak_frp > incident.get("peak_frp", 0):
                    incident["peak_frp"] = peak_frp
                old_state = incident.get("sub_state")
                incident["sub_state"] = EmergencySubState.ACTIVE_FIRE.value
                if old_state != EmergencySubState.ACTIVE_FIRE.value:
                    logger.info(
                        "Incident %s re-escalated %s → ACTIVE_FIRE",
                        incident_id, old_state,
                    )

            self._save()
            return incident_id, EmergencySubState.ACTIVE_FIRE

        # No hotspots — transition based on elapsed time
        if incident is None:
            # No active incident and no hotspots — shouldn't be in EMERGENCY mode
            # but handle gracefully
            incident_id = str(uuid.uuid4())[:12]
            incident = {
                "incident_id": incident_id,
                "started_at": now_iso,
                "last_hotspot_at": now_iso,
                "peak_frp": 0.0,
                "report_count": 0,
                "sub_state": EmergencySubState.ACTIVE_FIRE.value,
                "archived": False,
            }
            self._incidents[incident_id] = incident
            self._save()
            return incident_id, EmergencySubState.ACTIVE_FIRE

        last_hotspot = datetime.fromisoformat(incident["last_hotspot_at"])
        hours_since = (now - last_hotspot).total_seconds() / 3600

        new_sub_state = self._compute_sub_state(hours_since)
        old_sub_state = incident.get("sub_state")

        if new_sub_state.value != old_sub_state:
            logger.info(
                "Incident %s transition: %s → %s (%.1fh since last hotspot)",
                incident_id, old_sub_state, new_sub_state.value, hours_since,
            )
            incident["sub_state"] = new_sub_state.value

        self._save()
        return incident_id, new_sub_state

    def increment_report_count(self, incident_id: str) -> None:
        """Increment the report count for an incident (call after report is saved)."""
        if incident_id in self._incidents:
            self._incidents[incident_id]["report_count"] = (
                self._incidents[incident_id].get("report_count", 0) + 1
            )
            self._save()

    def archive_incident(self, incident_id: str) -> None:
        """Mark an incident as archived (after FINAL report is produced)."""
        if incident_id in self._incidents:
            self._incidents[incident_id]["archived"] = True
            self._incidents[incident_id]["archived_at"] = datetime.now(tz=UTC).isoformat()
            self._save()
            logger.info("Incident %s archived", incident_id)

    # -- Private helpers --

    def _compute_sub_state(self, hours_since_last_hotspot: float) -> EmergencySubState:
        """Determine sub-state from elapsed hours since last hotspot."""
        if hours_since_last_hotspot >= self._final_hours:
            return EmergencySubState.FINAL
        if hours_since_last_hotspot >= self._post_fire_hours:
            return EmergencySubState.POST_FIRE
        if hours_since_last_hotspot >= self._interim_hours:
            return EmergencySubState.INTERIM
        return EmergencySubState.ACTIVE_FIRE

    @staticmethod
    def _extract_peak_frp(pipeline_result: dict[str, Any]) -> float:
        """Extract peak FRP from FIRMS hotspot list."""
        hotspots = pipeline_result.get("firms_hotspots") or []
        if not hotspots:
            return 0.0
        return max(float(h.get("frp", 0)) for h in hotspots)

    def _load(self) -> dict[str, dict[str, Any]]:
        """Load incident state from YAML file."""
        if not self._state_file.exists():
            return {}
        try:
            with open(self._state_file, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            return data.get("incidents", {})
        except Exception:
            logger.exception("Failed to load incident state from %s", self._state_file)
            return {}

    def _save(self) -> None:
        """Persist incident state to YAML file."""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._state_file, "w", encoding="utf-8") as fh:
                yaml.safe_dump(
                    {"incidents": self._incidents},
                    fh,
                    default_flow_style=False,
                    sort_keys=False,
                )
        except Exception:
            logger.exception("Failed to save incident state to %s", self._state_file)
