import os
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

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

    try:
        import requests
        requests.post(webhook, json={"text": text}, timeout=10)
        logger.info("Sent Slack failure notification.")
    except Exception as e:
        logger.warning(f"Failed to send Slack notification: {e}")
