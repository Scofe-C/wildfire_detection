"""
Tests for field telemetry placeholder module.
Validates schema acceptance/rejection and DataFrame conversion.
"""

import pytest
import pandas as pd

from scripts.ingestion.ingest_field_telemetry import (
    VALID_SOURCE_TYPES,
    validate_field_telemetry,
    field_telemetry_to_dataframe,
    batch_field_telemetry_to_dataframe,
)


@pytest.fixture
def valid_payload():
    """A complete, valid field telemetry payload."""
    return {
        "source_type": "drone",
        "priority": 1,
        "latitude": 34.0522,
        "longitude": -118.2437,
        "timestamp": "2025-07-15T14:30:00Z",
        "confidence": 95,
        "frp": 150.0,
        "report_text": None,
        "spatial_trust_radius_km": 5.0,
    }


@pytest.fixture
def minimal_payload():
    """Payload with only required fields."""
    return {
        "source_type": "firefighter",
        "priority": 1,
        "latitude": 36.7783,
        "longitude": -119.4179,
        "timestamp": "2025-07-15T16:00:00Z",
        "confidence": 80,
    }


class TestValidation:
    """Tests for validate_field_telemetry()."""

    def test_valid_payload_accepted(self, valid_payload):
        is_valid, issues = validate_field_telemetry(valid_payload)
        assert is_valid is True
        assert issues == []

    def test_minimal_payload_accepted(self, minimal_payload):
        is_valid, issues = validate_field_telemetry(minimal_payload)
        assert is_valid is True
        assert issues == []

    def test_missing_required_field_rejected(self, valid_payload):
        del valid_payload["latitude"]
        is_valid, issues = validate_field_telemetry(valid_payload)
        assert is_valid is False
        assert any("latitude" in i for i in issues)

    def test_invalid_source_type_rejected(self, valid_payload):
        valid_payload["source_type"] = "satellite"
        is_valid, issues = validate_field_telemetry(valid_payload)
        assert is_valid is False
        assert any("source_type" in i for i in issues)

    def test_all_valid_source_types(self):
        for src in VALID_SOURCE_TYPES:
            payload = {
                "source_type": src,
                "priority": 1,
                "latitude": 34.0,
                "longitude": -118.0,
                "timestamp": "2025-01-01T00:00:00Z",
                "confidence": 50,
            }
            is_valid, _ = validate_field_telemetry(payload)
            assert is_valid, f"source_type '{src}' should be valid"

    def test_out_of_range_confidence_rejected(self, valid_payload):
        valid_payload["confidence"] = 150
        is_valid, issues = validate_field_telemetry(valid_payload)
        assert is_valid is False
        assert any("confidence" in i for i in issues)

    def test_out_of_range_latitude_rejected(self, valid_payload):
        valid_payload["latitude"] = 100.0
        is_valid, issues = validate_field_telemetry(valid_payload)
        assert is_valid is False
        assert any("latitude" in i for i in issues)

    def test_non_dict_rejected(self):
        is_valid, issues = validate_field_telemetry("not a dict")
        assert is_valid is False
        assert any("dictionary" in i for i in issues)


class TestDataFrameConversion:
    """Tests for field_telemetry_to_dataframe()."""

    def test_valid_payload_returns_dataframe(self, valid_payload):
        df = field_telemetry_to_dataframe(valid_payload)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_required_columns_present(self, valid_payload):
        df = field_telemetry_to_dataframe(valid_payload)
        for col in ["latitude", "longitude", "timestamp", "fire_detected_binary",
                     "data_source_priority", "data_quality_flag"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_ground_truth_defaults(self, valid_payload):
        df = field_telemetry_to_dataframe(valid_payload)
        assert df["fire_detected_binary"].values[0] == 1
        assert df["data_source_priority"].values[0] == 1
        assert df["data_quality_flag"].values[0] == 0

    def test_optional_fields_get_defaults(self, minimal_payload):
        df = field_telemetry_to_dataframe(minimal_payload)
        assert "spatial_trust_radius_km" in df.columns

    def test_invalid_payload_raises(self):
        with pytest.raises(ValueError, match="Invalid field telemetry"):
            field_telemetry_to_dataframe({"source_type": "bad"})


class TestBatchConversion:
    """Tests for batch_field_telemetry_to_dataframe()."""

    def test_batch_multiple_valid(self, valid_payload, minimal_payload):
        df = batch_field_telemetry_to_dataframe([valid_payload, minimal_payload])
        assert len(df) == 2

    def test_batch_skips_invalid(self, valid_payload):
        invalid = {"source_type": "bad"}
        df = batch_field_telemetry_to_dataframe([valid_payload, invalid])
        assert len(df) == 1

    def test_batch_all_invalid(self):
        df = batch_field_telemetry_to_dataframe([{"bad": True}])
        assert len(df) == 0
