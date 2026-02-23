"""
Tests for gcs_state GCS utility functions
==========================================
Covers write_false_alarm_record, write_emergency_log, read_industrial_sources.

gcs_state imports google.cloud.storage INSIDE each function body (lazy import),
so `storage` is NOT a module-level attribute. We must patch at the import site:
    patch("google.cloud.storage.Client", ...)
and also patch get_registry at its definition site.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_registry():
    reg = MagicMock()
    reg.config = {
        "watchdog": {
            "gcs_paths": {
                "false_alarms": "watchdog/false_alarms/",
                "emergency_log": "watchdog/emergency/",
                "industrial_sources": "watchdog/config/industrial_sources.json",
            }
        }
    }
    return reg


def _make_mock_storage(blob_exists=True, blob_data=None):
    mock_blob = MagicMock()
    mock_blob.exists.return_value = blob_exists
    if blob_data is not None:
        mock_blob.download_as_text.return_value = (
            json.dumps(blob_data) if not isinstance(blob_data, str) else blob_data
        )
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    return mock_client, mock_bucket, mock_blob


class TestWriteFalseAlarmRecord:

    def test_writes_blob_to_gcs(self):
        mock_client, mock_bucket, mock_blob = _make_mock_storage()
        mock_reg = _make_mock_registry()
        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch("scripts.utils.schema_loader.get_registry", return_value=mock_reg), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "test-bucket"}):
            from scripts.utils.gcs_state import write_false_alarm_record
            write_false_alarm_record(detection_data={"lat": 37.5}, gate_failed="G1_spatial")
        mock_blob.upload_from_string.assert_called_once()

    def test_uploaded_payload_contains_gate_failed(self):
        mock_client, mock_bucket, mock_blob = _make_mock_storage()
        mock_reg = _make_mock_registry()
        captured = {}
        mock_blob.upload_from_string.side_effect = lambda d, **kw: captured.update({"data": json.loads(d)})
        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch("scripts.utils.schema_loader.get_registry", return_value=mock_reg), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "test-bucket"}):
            from scripts.utils.gcs_state import write_false_alarm_record
            write_false_alarm_record(detection_data={}, gate_failed="G3_viirs")
        assert captured["data"]["gate_failed"] == "G3_viirs"

    def test_uploaded_payload_contains_record_id(self):
        mock_client, mock_bucket, mock_blob = _make_mock_storage()
        mock_reg = _make_mock_registry()
        captured = {}
        mock_blob.upload_from_string.side_effect = lambda d, **kw: captured.update({"data": json.loads(d)})
        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch("scripts.utils.schema_loader.get_registry", return_value=mock_reg), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "test-bucket"}):
            from scripts.utils.gcs_state import write_false_alarm_record
            write_false_alarm_record(detection_data={}, gate_failed="G4")
        assert "record_id" in captured["data"]
        assert len(captured["data"]["record_id"]) == 36

    def test_gcs_error_does_not_raise(self):
        mock_client, mock_bucket, mock_blob = _make_mock_storage()
        mock_blob.upload_from_string.side_effect = Exception("GCS unavailable")
        mock_reg = _make_mock_registry()
        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch("scripts.utils.schema_loader.get_registry", return_value=mock_reg), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "test-bucket"}):
            from scripts.utils.gcs_state import write_false_alarm_record
            write_false_alarm_record(detection_data={}, gate_failed="G1")

    def test_blob_path_starts_with_false_alarms_prefix(self):
        mock_client, mock_bucket, mock_blob = _make_mock_storage()
        mock_reg = _make_mock_registry()
        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch("scripts.utils.schema_loader.get_registry", return_value=mock_reg), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "test-bucket"}):
            from scripts.utils.gcs_state import write_false_alarm_record
            write_false_alarm_record(detection_data={}, gate_failed="G2")
        blob_path = mock_bucket.blob.call_args[0][0]
        assert blob_path.startswith("watchdog/false_alarms/")


class TestWriteEmergencyLog:

    def test_writes_blob_to_gcs(self):
        mock_client, mock_bucket, mock_blob = _make_mock_storage()
        mock_reg = _make_mock_registry()
        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch("scripts.utils.schema_loader.get_registry", return_value=mock_reg), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "test-bucket"}):
            from scripts.utils.gcs_state import write_emergency_log
            write_emergency_log(event="activated", details={"cells": 5})
        mock_blob.upload_from_string.assert_called_once()

    def test_uploaded_payload_contains_event(self):
        mock_client, mock_bucket, mock_blob = _make_mock_storage()
        mock_reg = _make_mock_registry()
        captured = {}
        mock_blob.upload_from_string.side_effect = lambda d, **kw: captured.update({"data": json.loads(d)})
        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch("scripts.utils.schema_loader.get_registry", return_value=mock_reg), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "test-bucket"}):
            from scripts.utils.gcs_state import write_emergency_log
            write_emergency_log(event="deactivated", details={"reason": "low FRP"})
        assert captured["data"]["event"] == "deactivated"
        assert captured["data"]["details"]["reason"] == "low FRP"

    def test_blob_path_starts_with_emergency_prefix(self):
        mock_client, mock_bucket, mock_blob = _make_mock_storage()
        mock_reg = _make_mock_registry()
        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch("scripts.utils.schema_loader.get_registry", return_value=mock_reg), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "test-bucket"}):
            from scripts.utils.gcs_state import write_emergency_log
            write_emergency_log(event="expanding", details={})
        blob_path = mock_bucket.blob.call_args[0][0]
        assert blob_path.startswith("watchdog/emergency/")

    def test_gcs_error_does_not_raise(self):
        mock_client, mock_bucket, mock_blob = _make_mock_storage()
        mock_blob.upload_from_string.side_effect = Exception("timeout")
        mock_reg = _make_mock_registry()
        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch("scripts.utils.schema_loader.get_registry", return_value=mock_reg), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "test-bucket"}):
            from scripts.utils.gcs_state import write_emergency_log
            write_emergency_log(event="test", details={})


class TestReadIndustrialSources:

    def test_returns_list_of_dicts(self):
        sources = [{"name": "Tesoro Refinery", "lat": 37.9, "lon": -122.0, "radius_km": 2.0}]
        mock_client, mock_bucket, mock_blob = _make_mock_storage(blob_exists=True, blob_data=sources)
        mock_reg = _make_mock_registry()
        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch("scripts.utils.schema_loader.get_registry", return_value=mock_reg), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "test-bucket"}):
            from scripts.utils.gcs_state import read_industrial_sources
            result = read_industrial_sources()
        assert isinstance(result, list)
        assert result[0]["name"] == "Tesoro Refinery"

    def test_returns_empty_list_when_file_absent(self):
        mock_client, mock_bucket, mock_blob = _make_mock_storage(blob_exists=False)
        mock_reg = _make_mock_registry()
        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch("scripts.utils.schema_loader.get_registry", return_value=mock_reg), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "test-bucket"}):
            from scripts.utils.gcs_state import read_industrial_sources
            result = read_industrial_sources()
        assert result == []

    def test_returns_empty_list_on_gcs_error(self):
        mock_client, mock_bucket, mock_blob = _make_mock_storage()
        mock_blob.download_as_text.side_effect = Exception("GCS error")
        mock_reg = _make_mock_registry()
        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch("scripts.utils.schema_loader.get_registry", return_value=mock_reg), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "test-bucket"}):
            from scripts.utils.gcs_state import read_industrial_sources
            result = read_industrial_sources()
        assert result == []

    def test_reads_from_correct_gcs_path(self):
        sources = [{"name": "Plant A", "lat": 34.0, "lon": -118.0, "radius_km": 1.0}]
        mock_client, mock_bucket, mock_blob = _make_mock_storage(blob_exists=True, blob_data=sources)
        mock_reg = _make_mock_registry()
        with patch("google.cloud.storage.Client", return_value=mock_client), \
             patch("scripts.utils.schema_loader.get_registry", return_value=mock_reg), \
             patch.dict(os.environ, {"GCS_BUCKET_NAME": "test-bucket"}):
            from scripts.utils.gcs_state import read_industrial_sources
            read_industrial_sources()
        blob_path = mock_bucket.blob.call_args[0][0]
        assert blob_path == "watchdog/config/industrial_sources.json"