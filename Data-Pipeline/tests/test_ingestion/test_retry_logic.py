"""
Tests for HTTP retry logic.
Verifies Bug #3 fix: non-retryable 4xx errors (except 429) fail immediately.
"""

from unittest.mock import MagicMock, patch

import pytest

from scripts.ingestion.ingest_firms import _fetch_single_request


@pytest.fixture
def mock_limiter():
    """Create a mock RateLimiter."""
    limiter = MagicMock()
    limiter.acquire.return_value.__enter__ = MagicMock()
    limiter.acquire.return_value.__exit__ = MagicMock(return_value=False)
    limiter.get_backoff_delay.return_value = 0.01
    return limiter


class TestRetryLogic:
    """Tests for _fetch_single_request retry behavior."""

    @patch("scripts.ingestion.ingest_firms.requests.get")
    def test_403_no_retry(self, mock_get, mock_limiter):
        """HTTP 403 (auth failure) should fail immediately without retry."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"
        mock_get.return_value = mock_response

        result = _fetch_single_request(
            base_url="https://firms.test",
            api_key="test_key",
            sensor="VIIRS_SNPP_NRT",
            bbox_str="-124,32,-114,42",
            day_range=1,
            limiter=mock_limiter,
            max_retries=3,
        )

        assert result is None
        # Should only be called once — no retries
        assert mock_get.call_count == 1

    @patch("scripts.ingestion.ingest_firms.requests.get")
    def test_404_no_retry(self, mock_get, mock_limiter):
        """HTTP 404 (not found) should fail immediately without retry."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_get.return_value = mock_response

        result = _fetch_single_request(
            base_url="https://firms.test",
            api_key="test_key",
            sensor="VIIRS_SNPP_NRT",
            bbox_str="-124,32,-114,42",
            day_range=1,
            limiter=mock_limiter,
            max_retries=3,
        )

        assert result is None
        assert mock_get.call_count == 1

    @patch("scripts.ingestion.ingest_firms.requests.get")
    def test_500_retries(self, mock_get, mock_limiter):
        """HTTP 500 (server error) should retry up to max_retries."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response

        result = _fetch_single_request(
            base_url="https://firms.test",
            api_key="test_key",
            sensor="VIIRS_SNPP_NRT",
            bbox_str="-124,32,-114,42",
            day_range=1,
            limiter=mock_limiter,
            max_retries=3,
        )

        assert result is None
        assert mock_get.call_count == 3  # all retries exhausted

    @patch("scripts.ingestion.ingest_firms.requests.get")
    def test_429_retries(self, mock_get, mock_limiter):
        """HTTP 429 (rate limit) should retry with backoff."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate Limited"
        mock_get.return_value = mock_response

        result = _fetch_single_request(
            base_url="https://firms.test",
            api_key="test_key",
            sensor="VIIRS_SNPP_NRT",
            bbox_str="-124,32,-114,42",
            day_range=1,
            limiter=mock_limiter,
            max_retries=3,
        )

        assert result is None
        assert mock_get.call_count == 3  # all retries exhausted

    @patch("scripts.ingestion.ingest_firms.requests.get")
    def test_200_success(self, mock_get, mock_limiter):
        """HTTP 200 with valid CSV should return a DataFrame."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "latitude,longitude,frp,confidence\n34.0,-118.0,10.5,85\n"
        mock_get.return_value = mock_response

        result = _fetch_single_request(
            base_url="https://firms.test",
            api_key="test_key",
            sensor="VIIRS_SNPP_NRT",
            bbox_str="-124,32,-114,42",
            day_range=1,
            limiter=mock_limiter,
        )

        assert result is not None
        assert len(result) == 1
        assert mock_get.call_count == 1
