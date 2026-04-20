"""
Cloud Function: dag-trigger
===========================
Triggered by Cloud Scheduler every 30 minutes.
POSTs directly to Airflow REST API to start a wildfire_data_pipeline DAG run.

Environment variables (set via --set-env-vars in deploy.sh):
  AIRFLOW_URL   — base URL of Airflow webserver, e.g. https://your-composer-url.com
                  or http://<public-ip>:8080 for a self-hosted instance
  AIRFLOW_USER  — Airflow basic-auth username (default: admin)
  AIRFLOW_PASS  — Airflow basic-auth password (default: admin)
"""

import json
import os
import urllib.error
import urllib.request
from base64 import b64encode
from datetime import datetime, timezone


DAG_ID = "wildfire_data_pipeline"


def trigger_wildfire_dag(request):  # noqa: ARG001
    airflow_url = os.environ["AIRFLOW_URL"].rstrip("/")
    user = os.environ.get("AIRFLOW_USER", "admin")
    password = os.environ.get("AIRFLOW_PASS", "admin")

    run_id = f"scheduled__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    payload = json.dumps(
        {"dag_run_id": run_id, "conf": {"trigger_source": "cloud_scheduler"}}
    ).encode()

    credentials = b64encode(f"{user}:{password}".encode()).decode()
    req = urllib.request.Request(
        f"{airflow_url}/api/v1/dags/{DAG_ID}/dagRuns",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Basic {credentials}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
            print(f"[dag-trigger] DAG run created: {body.get('dag_run_id')}")
            return (json.dumps({"status": "ok", "dag_run_id": body.get("dag_run_id")}), 200)
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        print(f"[dag-trigger] Airflow API error {e.code}: {error_body}")
        return (json.dumps({"status": "error", "code": e.code, "detail": error_body}), 502)
    except Exception as e:
        print(f"[dag-trigger] Unexpected error: {e}")
        return (json.dumps({"status": "error", "detail": str(e)}), 500)
