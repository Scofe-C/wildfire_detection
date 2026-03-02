"""
Tests for SLA-aware Slack notification callbacks.
Verifies consecutive failure tracking and escalation behavior.
"""

from unittest.mock import MagicMock, patch

import pytest

from dags.utils.slack_notify import (
    SLA_CONSECUTIVE_THRESHOLD,
    sla_on_failure_callback,
    sla_on_success_callback,
    _get_sla_xcom_key,
)


@pytest.fixture
def mock_context():
    """Create a mock Airflow context with a XCom-capable task instance."""
    ti = MagicMock()
    ti.task_id = "ingest_firms"
    ti.log_url = "http://localhost:8080/log"
    ti._xcom_store = {}

    def xcom_push(key, value):
        ti._xcom_store[key] = value

    def xcom_pull(key, task_ids=None):
        return ti._xcom_store.get(key)

    ti.xcom_push = MagicMock(side_effect=xcom_push)
    ti.xcom_pull = MagicMock(side_effect=xcom_pull)

    dag = MagicMock()
    dag.dag_id = "wildfire_data_pipeline"

    return {
        "task_instance": ti,
        "dag": dag,
        "run_id": "manual__2025-01-15T00:00:00",
    }


class TestSLATracking:
    """Tests for consecutive failure tracking and escalation."""

    @patch.dict("os.environ", {"SLACK_WEBHOOK_URL": ""})
    def test_first_failure_increments_counter(self, mock_context):
        """First failure should set counter to 1."""
        sla_on_failure_callback(mock_context)

        ti = mock_context["task_instance"]
        xcom_key = _get_sla_xcom_key("ingest_firms")
        assert ti._xcom_store[xcom_key] == 1

    @patch.dict("os.environ", {"SLACK_WEBHOOK_URL": ""})
    def test_second_failure_increments_counter(self, mock_context):
        """Second consecutive failure should set counter to 2."""
        ti = mock_context["task_instance"]
        xcom_key = _get_sla_xcom_key("ingest_firms")
        ti._xcom_store[xcom_key] = 1  # simulate prior failure

        sla_on_failure_callback(mock_context)
        assert ti._xcom_store[xcom_key] == 2

    @patch.dict("os.environ", {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/test"})
    @patch("dags.utils.slack_notify._post_slack")
    def test_escalation_fires_on_threshold(self, mock_post, mock_context):
        """SLA breach message should fire on reaching the threshold."""
        ti = mock_context["task_instance"]
        xcom_key = _get_sla_xcom_key("ingest_firms")
        ti._xcom_store[xcom_key] = SLA_CONSECUTIVE_THRESHOLD - 1

        sla_on_failure_callback(mock_context)

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        text = call_args[0][1]
        assert "SLA BREACH" in text
        assert "Incident Commander" in text

    @patch.dict("os.environ", {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/test"})
    @patch("dags.utils.slack_notify._post_slack")
    def test_no_escalation_before_threshold(self, mock_post, mock_context):
        """Normal failure message (not escalation) before reaching threshold."""
        sla_on_failure_callback(mock_context)

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        text = call_args[0][1]
        assert "SLA BREACH" not in text
        assert "Task Failed" in text

    @patch.dict("os.environ", {"SLACK_WEBHOOK_URL": ""})
    def test_success_resets_counter(self, mock_context):
        """Success callback should reset counter to 0."""
        ti = mock_context["task_instance"]
        xcom_key = _get_sla_xcom_key("ingest_firms")
        ti._xcom_store[xcom_key] = 2

        sla_on_success_callback(mock_context)
        assert ti._xcom_store[xcom_key] == 0

    @patch.dict("os.environ", {"SLACK_WEBHOOK_URL": ""})
    def test_success_with_no_prior_failures(self, mock_context):
        """Success when no prior failures should not raise."""
        sla_on_success_callback(mock_context)
        # Should complete without error
