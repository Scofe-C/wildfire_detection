"""
Emergency State Machine
=======================
Manages emergency activation, expansion tracking, and deactivation.
Called by the Cloud Function after fire_detector confirms a detection.

Emergency lifecycle:
  INACTIVE → ACTIVE   : FRP > 200 MW + ≥2 consecutive expanding scans
  ACTIVE → ACTIVE     : New cells added in latest scan (fire growing)
  ACTIVE → INACTIVE   : No new cells for 3 scans OR FRP < 50 MW for 2 scans

When ACTIVE:
  - Polling interval drops to 5 minutes
  - Full pipeline triggers every 30 minutes at 22km resolution
  - Slack alert sent with fire cell coordinates and FRP
  - Emergency log written to GCS
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def evaluate_emergency(
    state: dict,
    confirmed_cells: list[str],
    max_frp: float,
    watchdog_config: dict,
    gcs_state=None,
) -> dict:
    """Evaluate whether emergency should activate, continue, or deactivate.

    Args:
        state: Current watchdog state.
        confirmed_cells: H3 cell IDs confirmed by fire_detector.
        max_frp: Maximum FRP from current scan (MW).
        watchdog_config: The 'watchdog' section from schema_config.yaml.
        gcs_state: gcs_state module (for writing emergency log).

    Returns:
        Updated state dict with emergency fields set.
    """
    emg_cfg = watchdog_config.get("emergency", {})
    state = dict(state)  # don't mutate caller's state

    min_frp      = emg_cfg.get("min_frp_mw", 200.0)
    min_expand   = emg_cfg.get("min_expanding_scans", 2)
    deact_no_exp = emg_cfg.get("deactivate_no_expansion_scans", 3)
    deact_frp    = emg_cfg.get("deactivate_frp_mw", 50.0)
    deact_scans  = emg_cfg.get("deactivate_low_frp_scans", 2)

    current_mode = state.get("mode", "quiet")
    prev_cells = set(state.get("active_fire_cells", []))
    current_cells = set(confirmed_cells)
    new_cells = current_cells - prev_cells

    now = datetime.now(timezone.utc).isoformat()

    if current_mode == "emergency":
        # --- Deactivation checks ---
        if len(new_cells) == 0:
            state["consecutive_no_expansion_scans"] = state.get("consecutive_no_expansion_scans", 0) + 1
        else:
            state["consecutive_no_expansion_scans"] = 0

        if max_frp < deact_frp:
            state["consecutive_low_frp_scans"] = state.get("consecutive_low_frp_scans", 0) + 1
        else:
            state["consecutive_low_frp_scans"] = 0

        no_exp_scans = state.get("consecutive_no_expansion_scans", 0)
        low_frp_scans = state.get("consecutive_low_frp_scans", 0)

        if no_exp_scans >= deact_no_exp or low_frp_scans >= deact_scans:
            reason = (
                f"no expansion for {no_exp_scans} scans"
                if no_exp_scans >= deact_no_exp
                else f"FRP < {deact_frp} MW for {low_frp_scans} scans"
            )
            logger.info(f"Emergency DEACTIVATING: {reason}")
            state["mode"] = state.get("prior_mode") or "active"
            state["emergency_activated_at"] = None
            state["consecutive_no_expansion_scans"] = 0
            state["consecutive_low_frp_scans"] = 0
            state["active_fire_cells"] = list(current_cells)

            if gcs_state:
                gcs_state.write_emergency_log(
                    "deactivated",
                    {"reason": reason, "final_cells": len(current_cells), "frp": max_frp},
                )
            _send_slack_alert(
                f"🟢 Emergency DEACTIVATED — {reason}. Fire cells: {len(current_cells)}",
                watchdog_config,
            )
        else:
            # Still in emergency — update cells
            if new_cells:
                logger.info(
                    f"Emergency EXPANDING: +{len(new_cells)} new cells "
                    f"(total: {len(current_cells)}, FRP: {max_frp:.0f} MW)"
                )
                _send_slack_alert(
                    f"🔴 Fire EXPANDING: +{len(new_cells)} new cells "
                    f"(total {len(current_cells)}), FRP={max_frp:.0f} MW",
                    watchdog_config,
                )
            state["active_fire_cells"] = list(current_cells)

    else:
        # --- Activation check ---
        if max_frp >= min_frp and len(new_cells) > 0:
            state["consecutive_expanding_scans"] = state.get("consecutive_expanding_scans", 0) + 1
        else:
            state["consecutive_expanding_scans"] = 0

        if (
            max_frp >= min_frp
            and state.get("consecutive_expanding_scans", 0) >= min_expand
        ):
            logger.warning(
                f"Emergency ACTIVATING: FRP={max_frp:.0f} MW, "
                f"{len(new_cells)} new cells, "
                f"{state['consecutive_expanding_scans']} expanding scans"
            )
            state["prior_mode"] = current_mode
            state["mode"] = "emergency"
            state["emergency_activated_at"] = now
            state["active_fire_cells"] = list(current_cells)
            state["consecutive_no_expansion_scans"] = 0
            state["consecutive_low_frp_scans"] = 0

            if gcs_state:
                gcs_state.write_emergency_log(
                    "activated",
                    {
                        "frp_mw": max_frp,
                        "fire_cells": list(current_cells),
                        "new_cells": list(new_cells),
                        "expanding_scans": state["consecutive_expanding_scans"],
                    },
                )
            _send_slack_alert(
                f"🚨 EMERGENCY ACTIVATED — FRP={max_frp:.0f} MW, "
                f"{len(current_cells)} fire cells confirmed. "
                f"Pipeline escalating to 22km / 5-min polling.",
                watchdog_config,
            )
        else:
            # Normal confirmed fire — update cells, stay in current mode
            state["active_fire_cells"] = list(current_cells)

    return state


def _send_slack_alert(message: str, watchdog_config: dict) -> None:
    """Send Slack notification for emergency events."""
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logger.info("SLACK_WEBHOOK_URL not set — Slack alert skipped")
        logger.info(f"[Emergency alert would send]: {message}")
        return

    try:
        import requests
        resp = requests.post(
            webhook_url,
            json={"text": message},
            timeout=10,
        )
        if resp.status_code == 200:
            logger.info("Emergency Slack alert sent")
        else:
            logger.warning(f"Slack alert HTTP {resp.status_code}")
    except Exception as e:
        logger.warning(f"Failed to send emergency Slack alert: {e}")


def get_pipeline_params_for_mode(
    mode: str,
    fire_cells: list[str],
    watchdog_config: dict,
    region: str,
) -> dict:
    """Determine pipeline trigger parameters based on current mode.

    Returns the dict written to the GCS trigger file.
    """
    modes_cfg = watchdog_config.get("modes", {})
    mode_cfg = modes_cfg.get(mode, modes_cfg.get("quiet", {}))

    resolution_km = mode_cfg.get("resolution_km", 64)

    return {
        "trigger_source": f"watchdog_{mode}",
        "resolution_km": resolution_km,
        "regions": [region],
        "fire_cells": fire_cells,
        "mode": mode,
        "detection_range_km": watchdog_config.get("detection", {}).get("max_range_km", 25),
        "h3_ring_max": watchdog_config.get("detection", {}).get("h3_ring_max", 5),
    }
