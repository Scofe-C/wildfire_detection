"""
Shared Datetime Utilities
=========================
Canonical datetime coercion for all pipeline components.

Airflow passes various datetime types (Pendulum, lazy proxy objects,
raw datetime, strings). This module normalises them to a standard
UTC-aware ``datetime.datetime``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)


def coerce_to_utc(dt) -> datetime:
    """Coerce any datetime-like to a UTC-aware :class:`datetime.datetime`.

    Handles:
      - Airflow lazy-proxy objects (``__wrapped__`` attribute)
      - Pendulum DateTime instances
      - ``pd.Timestamp``
      - Raw ``datetime.datetime`` (naïve or aware)
      - ISO-format strings

    Returns:
        A timezone-aware ``datetime`` in UTC.

    Raises:
        ValueError: If *dt* is ``None``.
    """
    if dt is None:
        raise ValueError("Datetime cannot be None")

    # Unwrap Airflow lazy-proxy objects (present in Airflow >= 2.3 task context)
    if hasattr(dt, "__wrapped__"):
        dt = dt.__wrapped__

    try:
        ts = pd.to_datetime(str(dt), utc=True)
        return ts.to_pydatetime()
    except Exception:
        # Fallback: try pd.Timestamp constructor
        try:
            ts = pd.Timestamp(dt)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            return ts.to_pydatetime()
        except Exception:
            pass

        # Last resort: assume datetime-like
        if isinstance(dt, datetime):
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt

        return datetime.now(timezone.utc)
