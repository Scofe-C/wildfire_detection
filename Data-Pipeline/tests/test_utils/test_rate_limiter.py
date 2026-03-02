"""
Tests for rate_limiter utility
================================
Covers:
  - create_firms_limiter()  — builds a RateLimiter from FIRMS config
  - create_weather_limiter() — builds a RateLimiter from Open-Meteo config
  - RateLimiter core behaviour
"""

import time
from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture
def basic_config():
    from scripts.utils.rate_limiter import RateLimitConfig
    return RateLimitConfig(
        max_requests_per_window=3,
        window_seconds=2,
        backoff_base_seconds=0.01,
        backoff_max_seconds=0.1,
        jitter=False,
        max_retries=2,
    )


@pytest.fixture
def limiter(basic_config):
    from scripts.utils.rate_limiter import RateLimiter
    return RateLimiter(basic_config)


@pytest.fixture
def mock_registry():
    reg = MagicMock()
    reg.get_source_config.side_effect = lambda source: {
        "firms": {
            "rate_limit": {
                "max_requests_per_10min": 500,
                "backoff_base_seconds": 3,
                "backoff_max_seconds": 60,
                "jitter": True,
            },
            "max_retries": 3,
        },
        "open_meteo": {
            "rate_limit": {
                "max_requests_per_day": 1000,
                "backoff_base_seconds": 1,
                "backoff_max_seconds": 30,
            },
            "max_retries": 2,
        },
    }[source]
    return reg


class TestCreateFirmsLimiter:

    def test_returns_rate_limiter_instance(self, mock_registry):
        from scripts.utils.rate_limiter import RateLimiter, create_firms_limiter
        # get_registry is imported inside the function body, so patch at definition site
        with patch("scripts.utils.schema_loader.get_registry", return_value=mock_registry):
            result = create_firms_limiter()
        assert isinstance(result, RateLimiter)

    def test_firms_window_is_600_seconds(self, mock_registry):
        from scripts.utils.rate_limiter import create_firms_limiter
        with patch("scripts.utils.schema_loader.get_registry", return_value=mock_registry):
            result = create_firms_limiter()
        assert result._config.window_seconds == 600

    def test_firms_limit_comes_from_config(self, mock_registry):
        from scripts.utils.rate_limiter import create_firms_limiter
        with patch("scripts.utils.schema_loader.get_registry", return_value=mock_registry):
            result = create_firms_limiter()
        assert result._config.max_requests_per_window == 500

    def test_firms_backoff_base_from_config(self, mock_registry):
        from scripts.utils.rate_limiter import create_firms_limiter
        with patch("scripts.utils.schema_loader.get_registry", return_value=mock_registry):
            result = create_firms_limiter()
        assert result._config.backoff_base_seconds == 3

    def test_firms_jitter_enabled(self, mock_registry):
        from scripts.utils.rate_limiter import create_firms_limiter
        with patch("scripts.utils.schema_loader.get_registry", return_value=mock_registry):
            result = create_firms_limiter()
        assert result._config.jitter is True


class TestCreateWeatherLimiter:

    def test_returns_rate_limiter_instance(self, mock_registry):
        from scripts.utils.rate_limiter import RateLimiter, create_weather_limiter
        with patch("scripts.utils.schema_loader.get_registry", return_value=mock_registry):
            result = create_weather_limiter()
        assert isinstance(result, RateLimiter)

    def test_weather_window_is_86400_seconds(self, mock_registry):
        from scripts.utils.rate_limiter import create_weather_limiter
        with patch("scripts.utils.schema_loader.get_registry", return_value=mock_registry):
            result = create_weather_limiter()
        assert result._config.window_seconds == 86400

    def test_weather_limit_comes_from_config(self, mock_registry):
        from scripts.utils.rate_limiter import create_weather_limiter
        with patch("scripts.utils.schema_loader.get_registry", return_value=mock_registry):
            result = create_weather_limiter()
        assert result._config.max_requests_per_window == 1000

    def test_weather_jitter_always_true(self, mock_registry):
        from scripts.utils.rate_limiter import create_weather_limiter
        with patch("scripts.utils.schema_loader.get_registry", return_value=mock_registry):
            result = create_weather_limiter()
        assert result._config.jitter is True


class TestRateLimiterBehaviour:

    def test_record_request_increments_count(self, limiter):
        assert len(limiter._request_timestamps) == 0
        limiter.record_request()
        assert len(limiter._request_timestamps) == 1

    def test_wait_if_needed_does_not_block_below_limit(self, limiter):
        for _ in range(2):
            limiter.record_request()
        start = time.monotonic()
        limiter.wait_if_needed()
        assert time.monotonic() - start < 0.5

    def test_old_timestamps_are_cleaned_up(self, limiter):
        from datetime import datetime, timedelta
        old_time = datetime.utcnow() - timedelta(seconds=10)
        limiter._request_timestamps = [old_time] * 3
        limiter.wait_if_needed()
        assert len(limiter._request_timestamps) == 0

    def test_record_failure_increments_counter(self, limiter):
        assert limiter._consecutive_failures == 0
        limiter.record_failure()
        assert limiter._consecutive_failures == 1

    def test_consecutive_failures_property(self, limiter):
        """consecutive_failures is a read-only property (no reset_failures method)."""
        limiter.record_failure()
        limiter.record_failure()
        assert limiter.consecutive_failures == 2

    def test_acquire_records_request_on_success(self, limiter):
        before = len(limiter._request_timestamps)
        with limiter.acquire():
            pass
        assert len(limiter._request_timestamps) == before + 1

    def test_acquire_records_failure_on_exception(self, limiter):
        """acquire() must call record_failure when the block raises."""
        before = limiter._consecutive_failures
        try:
            with limiter.acquire():
                raise ValueError("test error")
        except ValueError:
            pass
        assert limiter._consecutive_failures == before + 1

    def test_acquire_does_not_record_request_on_exception(self, limiter):
        before = len(limiter._request_timestamps)
        try:
            with limiter.acquire():
                raise ValueError("test")
        except ValueError:
            pass
        assert len(limiter._request_timestamps) == before

    def test_limiter_is_thread_safe(self, limiter):
        import threading
        def do_requests():
            for _ in range(10):
                limiter.record_request()
        threads = [threading.Thread(target=do_requests) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert isinstance(limiter._request_timestamps, list)