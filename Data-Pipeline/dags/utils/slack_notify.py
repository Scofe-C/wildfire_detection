"""
Slack Notification Callbacks for Airflow
========================================
Provides:
  - notify_slack()           — basic on_failure_callback (original)
  - sla_on_failure_callback() — stateful SLA tracking via XCom; escalates
                                 after SLA_CONSECUTIVE_THRESHOLD consecutive
                                 failures of the same task.
  - sla_on_success_callback() — resets consecutive failure counter on success.

State storage: Airflow XCom (key: _sla_consecutive_fails_{task_id}).
"""

import os
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Number of consecutive failures before SLA breach escalation
SLA_CONSECUTIVE_THRESHOLD = 3


def notify_slack(context: Dict[str, Any]) -> None:
    """
    Airflow on_failure_callback.
    If SLACK_WEBHOOK_URL is not set, do nothing (safe for local/dev).
    """
    webhook = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if not webhook:
        logger.info("SLACK_WEBHOOK_URL not set; skipping Slack notification.")
        return

    dag_id = context.get("dag").dag_id if context.get("dag") else "unknown_dag"
    task_id = context.get("task_instance").task_id if context.get("task_instance") else "unknown_task"
    run_id = context.get("run_id", "unknown_run")
    log_url = ""
    try:
        ti = context.get("task_instance")
        if ti:
            log_url = ti.log_url
    except Exception:
        pass

    text = (
        f":x: *Airflow Task Failed*\n"
        f"*DAG*: `{dag_id}`\n"
        f"*Task*: `{task_id}`\n"
        f"*Run*: `{run_id}`\n"
        + (f"*Log*: {log_url}\n" if log_url else "")
    )

    _post_slack(webhook, text)


# ---------------------------------------------------------------------------
# SLA-aware callbacks
# ---------------------------------------------------------------------------

def _get_sla_xcom_key(task_id: str) -> str:
    """XCom key for tracking consecutive failures of a specific task."""
    return f"_sla_consecutive_fails_{task_id}"


def sla_on_failure_callback(context: Dict[str, Any]) -> None:
    """Stateful on_failure_callback with SLA breach escalation.

    Tracks consecutive failures per task via XCom.  After
    SLA_CONSECUTIVE_THRESHOLD consecutive failures, sends an escalated
    Slack message with a distinct ":rotating_light: SLA BREACH" prefix.
    """
    ti = context.get("task_instance")
    if ti is None:
        notify_slack(context)  # fallback to basic
        return

    task_id = ti.task_id
    dag_id = context.get("dag").dag_id if context.get("dag") else "unknown_dag"
    run_id = context.get("run_id", "unknown_run")
    xcom_key = _get_sla_xcom_key(task_id)

    # Read previous consecutive failure count from XCom
    prev_count = 0
    try:
        prev_val = ti.xcom_pull(key=xcom_key, task_ids=task_id)
        if prev_val is not None:
            prev_count = int(prev_val)
    except Exception:
        pass

    new_count = prev_count + 1
    ti.xcom_push(key=xcom_key, value=new_count)

    logger.warning(
        f"Task '{task_id}' failed — consecutive failure #{new_count} "
        f"(threshold: {SLA_CONSECUTIVE_THRESHOLD})"
    )

    webhook = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if not webhook:
        logger.info("SLACK_WEBHOOK_URL not set; skipping Slack notification.")
        return

    log_url = ""
    try:
        log_url = ti.log_url
    except Exception:
        pass

    if new_count >= SLA_CONSECUTIVE_THRESHOLD:
        # --- SLA breach escalation ---
        text = (
            f":rotating_light: *SLA BREACH — Incident Commander Alert*\n"
            f"*DAG*: `{dag_id}`\n"
            f"*Task*: `{task_id}`\n"
            f"*Consecutive Failures*: {new_count}\n"
            f"*Run*: `{run_id}`\n"
            f"*Action Required*: This task has failed {new_count} times in a row. "
            f"Manual investigation needed.\n"
            + (f"*Log*: {log_url}\n" if log_url else "")
        )
        logger.error(
            f"SLA BREACH: Task '{task_id}' has failed {new_count} consecutive times."
        )
    else:
        # --- Normal failure (1-2 failures, pre-breach) ---
        text = (
            f":x: *Airflow Task Failed* ({new_count}/{SLA_CONSECUTIVE_THRESHOLD} before SLA breach)\n"
            f"*DAG*: `{dag_id}`\n"
            f"*Task*: `{task_id}`\n"
            f"*Run*: `{run_id}`\n"
            + (f"*Log*: {log_url}\n" if log_url else "")
        )

    _post_slack(webhook, text)


def sla_on_success_callback(context: Dict[str, Any]) -> None:
    """Reset consecutive failure counter on success."""
    ti = context.get("task_instance")
    if ti is None:
        return

    task_id = ti.task_id
    xcom_key = _get_sla_xcom_key(task_id)

    try:
        prev_val = ti.xcom_pull(key=xcom_key, task_ids=task_id)
        if prev_val is not None and int(prev_val) > 0:
            logger.info(
                f"Task '{task_id}' succeeded — resetting consecutive failure "
                f"counter (was {prev_val})."
            )
            ti.xcom_push(key=xcom_key, value=0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _post_slack(webhook_url: str, text: str) -> None:
    """Fire-and-forget Slack webhook post."""
    try:
        import requests
        requests.post(webhook_url, json={"text": text}, timeout=10)
        logger.info("Sent Slack notification.")
    except Exception as e:
        logger.warning(f"Failed to send Slack notification: {e}")
