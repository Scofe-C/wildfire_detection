"""State machine — determines report type from ML pipeline outputs.

Pure logic module: no I/O, no LLM calls. Receives pipeline result dict
and returns the operational mode + optional emergency sub-state.
"""

from __future__ import annotations

import enum
import logging
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

def resolve_mode(
    pipeline_result: dict[str, Any],
) -> tuple[OperationalMode, EmergencySubState | None]:
    """Determine operational mode from ML pipeline outputs.

    Parameters
    ----------
    pipeline_result:
        Raw dict from the pipeline (matches ``mock_pipeline_result.json``).
        Must contain ``risk_level`` (str) and ``firms_hotspot_count`` (int).

    Returns
    -------
    tuple of (OperationalMode, EmergencySubState | None)

    Raises
    ------
    ValueError
        If required fields are missing from *pipeline_result*.
    """
    if "risk_level" not in pipeline_result:
        raise ValueError("pipeline_result missing required field: 'risk_level'")
    if "firms_hotspot_count" not in pipeline_result:
        raise ValueError("pipeline_result missing required field: 'firms_hotspot_count'")

    risk = pipeline_result["risk_level"].upper()
    firms_count: int = int(pipeline_result["firms_hotspot_count"])

    # EMERGENCY: critical risk OR any FIRMS hotspots
    if risk == "CRITICAL" or firms_count > 0:
        return OperationalMode.EMERGENCY, EmergencySubState.ACTIVE_FIRE

    # ACTIVE: high or moderate risk, no confirmed fire
    if risk in ("HIGH", "MODERATE"):
        return OperationalMode.ACTIVE, None

    # QUIET: low (or moderate with no hotspots — already handled above)
    return OperationalMode.QUIET, None


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
            with open(self._config_path) as fh:
                raw = yaml.safe_load(fh) or {}
            raw.setdefault("admin_toggle", {})["current_state"] = state
            with open(self._config_path, "w") as fh:
                yaml.safe_dump(raw, fh, default_flow_style=False, sort_keys=False)
        except Exception:
            logger.exception("Failed to persist admin toggle state to %s", self._config_path)
