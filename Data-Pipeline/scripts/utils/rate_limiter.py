"""
Rate Limiter Utility
====================
Provides thread-safe rate limiting with exponential backoff and jitter
for all API-calling pipeline tasks.

Owner: Person A (FIRMS ingestion)
Consumers: Person A (FIRMS), Person B (Open-Meteo, NWS)
"""

import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting behavior."""
    max_requests_per_window: int = 5000
    window_seconds: int = 600  # 10 minutes
    backoff_base_seconds: float = 5.0
    backoff_max_seconds: float = 300.0
    jitter: bool = True
    max_retries: int = 3


class RateLimiter:
    """Thread-safe rate limiter with exponential backoff.

    Usage:
        limiter = RateLimiter(RateLimitConfig(max_requests_per_window=5000))

        with limiter.acquire():
            response = requests.get(url)

        # Or manually:
        limiter.wait_if_needed()
        response = requests.get(url)
        limiter.record_request()
    """

    def __init__(self, config: RateLimitConfig):
        self._config = config
        self._request_timestamps: list[datetime] = []
        self._lock = Lock()
        self._consecutive_failures = 0

    def wait_if_needed(self) -> None:
        """Block until a request can be made within rate limits."""
        with self._lock:
            self._clean_old_timestamps()

            if len(self._request_timestamps) >= self._config.max_requests_per_window:
                oldest = self._request_timestamps[0]
                wait_until = oldest + timedelta(seconds=self._config.window_seconds)
                wait_seconds = (wait_until - datetime.utcnow()).total_seconds()

                if wait_seconds > 0:
                    logger.warning(
                        f"Rate limit reached ({len(self._request_timestamps)} "
                        f"requests in window). Waiting {wait_seconds:.1f}s."
                    )
                    time.sleep(wait_seconds)
                    self._clean_old_timestamps()

    def record_request(self) -> None:
        """Record that a request was made (call after successful request)."""
        with self._lock:
            self._request_timestamps.append(datetime.utcnow())
            self._consecutive_failures = 0

    def record_failure(self) -> None:
        """Record a failed request and compute backoff delay."""
        with self._lock:
            self._consecutive_failures += 1

    def get_backoff_delay(self) -> float:
        """Calculate exponential backoff delay with optional jitter.

        Returns:
            Delay in seconds before retrying.
        """
        delay = min(
            self._config.backoff_base_seconds * (2 ** (self._consecutive_failures - 1)),
            self._config.backoff_max_seconds,
        )
        if self._config.jitter:
            delay = delay * (0.5 + random.random())
        return delay

    @property
    def requests_remaining(self) -> int:
        """Approximate number of requests remaining in current window."""
        with self._lock:
            self._clean_old_timestamps()
            return max(
                0,
                self._config.max_requests_per_window - len(self._request_timestamps),
            )

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def _clean_old_timestamps(self) -> None:
        """Remove timestamps outside the current rate limit window."""
        cutoff = datetime.utcnow() - timedelta(seconds=self._config.window_seconds)
        self._request_timestamps = [
            ts for ts in self._request_timestamps if ts > cutoff
        ]

    class _AcquireContext:
        """Context manager for rate-limited requests."""

        def __init__(self, limiter: "RateLimiter"):
            self._limiter = limiter

        def __enter__(self):
            self._limiter.wait_if_needed()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                self._limiter.record_request()
            else:
                self._limiter.record_failure()
            return False  # Don't suppress exceptions

    def acquire(self) -> "_AcquireContext":
        """Context manager that waits for rate limit and records the request.

        Usage:
            with limiter.acquire():
                response = requests.get(url)
        """
        return self._AcquireContext(self)


def create_firms_limiter(config_path: Optional[str] = None) -> RateLimiter:
    """Create a rate limiter configured for the FIRMS API."""
    from scripts.utils.schema_loader import get_registry

    registry = get_registry(config_path)
    firms_config = registry.get_source_config("firms")
    rl_config = firms_config.get("rate_limit", {})

    return RateLimiter(RateLimitConfig(
        max_requests_per_window=rl_config.get("max_requests_per_10min", 5000),
        window_seconds=600,
        backoff_base_seconds=rl_config.get("backoff_base_seconds", 5),
        backoff_max_seconds=rl_config.get("backoff_max_seconds", 300),
        jitter=rl_config.get("jitter", True),
        max_retries=firms_config.get("max_retries", 3),
    ))


def create_weather_limiter(config_path: Optional[str] = None) -> RateLimiter:
    """Create a rate limiter configured for Open-Meteo API."""
    from scripts.utils.schema_loader import get_registry

    registry = get_registry(config_path)
    om_config = registry.get_source_config("open_meteo")
    rl_config = om_config.get("rate_limit", {})

    return RateLimiter(RateLimitConfig(
        max_requests_per_window=rl_config.get("max_requests_per_day", 10000),
        window_seconds=86400,  # 24 hours
        backoff_base_seconds=rl_config.get("backoff_base_seconds", 2),
        backoff_max_seconds=rl_config.get("backoff_max_seconds", 120),
        jitter=True,
        max_retries=om_config.get("max_retries", 3),
    ))
