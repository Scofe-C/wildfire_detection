"""
Tests for the historical backfill script.
Verifies date range generation, resume logic, and output path structure.
"""

import pandas as pd
import pytest

from scripts.backfill.historical_backfill import (
    generate_backfill_dates,
    _output_path_for_window,
    run_backfill,
)


class TestGenerateBackfillDates:
    """Tests for generate_backfill_dates()."""

    def test_default_range(self):
        """Default range should span 2023-01-01 to 2025-01-31 at 6h intervals."""
        dates = generate_backfill_dates()
        assert len(dates) > 0
        assert dates[0] == pd.Timestamp("2023-01-01 00:00:00")
        assert dates[-1] <= pd.Timestamp("2025-01-31 23:59:59")

    def test_custom_range(self):
        """Custom range should produce the expected number of windows."""
        dates = generate_backfill_dates(
            start="2024-01-01",
            end="2024-01-02",
            freq_hours=6,
        )
        # 2024-01-01 00:00, 06:00, 12:00, 18:00, 2024-01-02 00:00 = 5
        assert len(dates) == 5

    def test_single_day(self):
        """Single day should produce 4 windows at 6h intervals."""
        dates = generate_backfill_dates(
            start="2024-06-15",
            end="2024-06-15 18:00",
            freq_hours=6,
        )
        assert len(dates) == 4  # 00, 06, 12, 18

    def test_custom_frequency(self):
        """12-hour frequency should produce half as many windows."""
        dates_6h = generate_backfill_dates(
            start="2024-01-01", end="2024-01-02", freq_hours=6,
        )
        dates_12h = generate_backfill_dates(
            start="2024-01-01", end="2024-01-02", freq_hours=12,
        )
        assert len(dates_12h) < len(dates_6h)

    def test_returns_timestamps(self):
        """All elements should be pd.Timestamp."""
        dates = generate_backfill_dates(
            start="2024-01-01", end="2024-01-01 06:00", freq_hours=6,
        )
        for d in dates:
            assert isinstance(d, pd.Timestamp)


class TestOutputPath:
    """Tests for _output_path_for_window()."""

    def test_path_structure(self, tmp_path):
        """Output path should have year/month partitioning."""
        ts = pd.Timestamp("2024-07-15 12:00")
        path = _output_path_for_window(ts, tmp_path, resolution_km=64)

        assert "64km" in str(path)
        assert "year=2024" in str(path)
        assert "month=07" in str(path)
        assert path.name == "features_2024-07-15_1200.parquet"

    def test_creates_directories(self, tmp_path):
        """Output path parent should be created automatically."""
        ts = pd.Timestamp("2023-03-20 06:00")
        path = _output_path_for_window(ts, tmp_path, resolution_km=64)

        # _output_path_for_window creates the dirs
        assert path.parent.exists()


class TestRunBackfillResume:
    """Tests for resume logic in run_backfill()."""

    def test_skips_existing_files(self, tmp_path):
        """Backfill should skip windows that already have output files."""
        # Pre-create one output file
        ts = pd.Timestamp("2024-01-01 00:00")
        existing_path = _output_path_for_window(ts, tmp_path, resolution_km=64)
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        existing_path.write_text("dummy")

        stats = run_backfill(
            start="2024-01-01",
            end="2024-01-01 00:00",
            freq_hours=6,
            output_dir=str(tmp_path),
            skip_existing=True,
        )

        assert stats["skipped"] >= 1
