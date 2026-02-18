"""
Tests for Seasonal Anomaly Detection
=====================================
Owner: Person D (Bohan)

Covers all required scenarios from Section 5.7 of the assignment guide,
plus tests for the new baseline storage (JSON per season) requirement
from Section 5.5.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class DummyRegistry:
    anomaly_config = {"monitored_features": ["temperature_2m", "wind_speed_10m"]}
    fire_season_months = [6, 7, 8, 9, 10, 11]

    def get_z_threshold(self, month):
        return 4.0 if month in self.fire_season_months else 3.5

    @property
    def fire_season_months(self):
        return [6, 7, 8, 9, 10, 11]

    @fire_season_months.setter
    def fire_season_months(self, v):
        pass


class DateFireSeason:
    """Execution date in fire season (August)."""
    month = 8


class DateOffSeason:
    """Execution date in off-season (February)."""
    month = 2


@pytest.fixture
def baseline_dir(tmp_path):
    return tmp_path / "baselines"


@pytest.fixture
def mature_baseline(baseline_dir):
    """Pre-seed a baseline with enough samples to activate detection."""
    from scripts.validation.detect_anomalies import _DEFAULT_BASELINE_DIR, MIN_SAMPLE_COUNT
    baseline_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "feature": "temperature_2m",
        "season": "fire_season",
        "mean": 25.0,
        "std": 5.0,
        "sample_count": MIN_SAMPLE_COUNT + 100,
        "last_updated": "2026-01-01T00:00:00",
    }
    path = baseline_dir / "baseline_temperature_2m_fire_season.json"
    path.write_text(json.dumps(data))
    return baseline_dir


# ---------------------------------------------------------------------------
# Core detection tests (assignment Section 5.7)
# ---------------------------------------------------------------------------

def test_detect_anomalies_returns_empty_on_normal_data(baseline_dir):
    """Normal data (within baseline range) must produce no anomalies."""
    from scripts.validation.detect_anomalies import detect_anomalies

    # Seed a baseline with mean=25, std=5
    baseline_dir.mkdir(parents=True)
    baseline_data = {
        "feature": "temperature_2m", "season": "fire_season",
        "mean": 25.0, "std": 5.0, "sample_count": 5000,
        "last_updated": "2026-01-01T00:00:00",
    }
    (baseline_dir / "baseline_temperature_2m_fire_season.json").write_text(
        json.dumps(baseline_data)
    )

    # Values all within 1-2 std of mean — no anomalies expected
    df = pd.DataFrame({"temperature_2m": [24.0, 25.0, 26.0, 23.5, 27.0] * 20})
    reg = DummyRegistry()

    anomalies = detect_anomalies(df, reg, DateFireSeason(), baseline_dir=baseline_dir)
    assert anomalies == [], f"Expected no anomalies, got: {anomalies}"


def test_detect_anomalies_flags_outliers(baseline_dir):
    """A single extreme outlier (z > 4.0) must be flagged."""
    from scripts.validation.detect_anomalies import detect_anomalies

    baseline_dir.mkdir(parents=True)
    baseline_data = {
        "feature": "temperature_2m", "season": "fire_season",
        "mean": 25.0, "std": 5.0, "sample_count": 5000,
        "last_updated": "2026-01-01T00:00:00",
    }
    (baseline_dir / "baseline_temperature_2m_fire_season.json").write_text(
        json.dumps(baseline_data)
    )

    # Mean=25, std=5 → z=4 threshold → need value > 25 + 4×5 = 45°C to trigger
    df = pd.DataFrame({"temperature_2m": [25.0] * 49 + [999.0]})
    reg = DummyRegistry()

    anomalies = detect_anomalies(df, reg, DateFireSeason(), baseline_dir=baseline_dir)
    assert len(anomalies) >= 1
    assert anomalies[0]["feature"] == "temperature_2m"
    assert anomalies[0]["outlier_count"] >= 1


def test_detect_anomalies_respects_fire_season_threshold(baseline_dir):
    """Z-score threshold must be 4.0 in August (fire season)."""
    from scripts.validation.detect_anomalies import detect_anomalies

    baseline_dir.mkdir(parents=True)
    # Baseline: mean=0, std=1 → z=4 threshold → outlier at value=4.01
    baseline_data = {
        "feature": "temperature_2m", "season": "fire_season",
        "mean": 0.0, "std": 1.0, "sample_count": 5000,
        "last_updated": "2026-01-01T00:00:00",
    }
    (baseline_dir / "baseline_temperature_2m_fire_season.json").write_text(
        json.dumps(baseline_data)
    )

    # Value at z=3.9: should NOT trigger fire season threshold (4.0)
    df_safe = pd.DataFrame({"temperature_2m": [0.0] * 49 + [3.9]})
    reg = DummyRegistry()
    result = detect_anomalies(df_safe, reg, DateFireSeason(), baseline_dir=baseline_dir)
    assert result == [], f"z=3.9 should not trigger fire season threshold 4.0, got: {result}"

    # Value at z=4.1: SHOULD trigger
    df_alert = pd.DataFrame({"temperature_2m": [0.0] * 49 + [4.1]})
    result = detect_anomalies(df_alert, reg, DateFireSeason(), baseline_dir=baseline_dir)
    assert len(result) >= 1, "z=4.1 should trigger fire season threshold 4.0"


def test_detect_anomalies_respects_off_season_threshold(baseline_dir):
    """Z-score threshold must be 3.5 in February (off-season)."""
    from scripts.validation.detect_anomalies import detect_anomalies

    baseline_dir.mkdir(parents=True)
    baseline_data = {
        "feature": "temperature_2m", "season": "off_season",
        "mean": 0.0, "std": 1.0, "sample_count": 5000,
        "last_updated": "2026-01-01T00:00:00",
    }
    (baseline_dir / "baseline_temperature_2m_off_season.json").write_text(
        json.dumps(baseline_data)
    )

    # z=3.4 should NOT trigger off-season threshold (3.5)
    df_safe = pd.DataFrame({"temperature_2m": [0.0] * 49 + [3.4]})
    reg = DummyRegistry()
    result = detect_anomalies(df_safe, reg, DateOffSeason(), baseline_dir=baseline_dir)
    assert result == [], f"z=3.4 should not trigger off-season threshold 3.5"

    # z=3.6 SHOULD trigger
    df_alert = pd.DataFrame({"temperature_2m": [0.0] * 49 + [3.6]})
    result = detect_anomalies(df_alert, reg, DateOffSeason(), baseline_dir=baseline_dir)
    assert len(result) >= 1, "z=3.6 should trigger off-season threshold 3.5"


def test_anomaly_detection_skipped_below_min_sample_count(baseline_dir):
    """Baseline with fewer than MIN_SAMPLE_COUNT samples skips detection."""
    from scripts.validation.detect_anomalies import detect_anomalies, MIN_SAMPLE_COUNT

    baseline_dir.mkdir(parents=True)
    # Deliberately small sample count
    baseline_data = {
        "feature": "temperature_2m", "season": "fire_season",
        "mean": 0.0, "std": 1.0, "sample_count": MIN_SAMPLE_COUNT - 1,
        "last_updated": "2026-01-01T00:00:00",
    }
    (baseline_dir / "baseline_temperature_2m_fire_season.json").write_text(
        json.dumps(baseline_data)
    )

    # Even with extreme outlier, detection is skipped (not enough history)
    df = pd.DataFrame({"temperature_2m": [0.0] * 49 + [999.0]})
    reg = DummyRegistry()
    result = detect_anomalies(df, reg, DateFireSeason(), baseline_dir=baseline_dir)
    assert result == [], "Anomaly detection must be skipped below MIN_SAMPLE_COUNT"


def test_anomaly_seeds_baseline_on_first_run(baseline_dir):
    """With no existing baseline, detect_anomalies must create baseline file."""
    from scripts.validation.detect_anomalies import detect_anomalies

    baseline_dir.mkdir(parents=True)
    df = pd.DataFrame({"temperature_2m": [25.0] * 50})
    reg = DummyRegistry()

    detect_anomalies(df, reg, DateFireSeason(), baseline_dir=baseline_dir)

    baseline_path = baseline_dir / "baseline_temperature_2m_fire_season.json"
    assert baseline_path.exists(), "Baseline file must be created on first run"

    with open(baseline_path) as f:
        saved = json.load(f)
    assert saved["feature"] == "temperature_2m"
    assert saved["season"] == "fire_season"
    assert saved["sample_count"] == 50
    assert "mean" in saved and "std" in saved


def test_baseline_welford_update_is_numerically_stable(baseline_dir):
    """Welford update over many batches should converge to correct mean/std."""
    from scripts.validation.detect_anomalies import detect_anomalies, load_baseline
    import numpy as np

    baseline_dir.mkdir(parents=True)
    reg = DummyRegistry()

    # Feed 10 batches of 100 samples each from N(30, 4)
    true_mean, true_std = 30.0, 4.0
    rng = np.random.default_rng(42)
    for _ in range(10):
        batch = rng.normal(true_mean, true_std, 100)
        df = pd.DataFrame({"temperature_2m": batch})
        detect_anomalies(
            df, reg, DateFireSeason(),
            baseline_dir=baseline_dir,
            update_baseline=True,
        )

    final = load_baseline("temperature_2m", "fire_season", baseline_dir)
    assert final is not None
    assert abs(final["mean"] - true_mean) < 1.0, (
        f"Welford mean {final['mean']:.2f} diverged from true mean {true_mean}"
    )
    assert abs(final["std"] - true_std) < 1.0, (
        f"Welford std {final['std']:.2f} diverged from true std {true_std}"
    )


def test_reset_baseline_clears_file(baseline_dir):
    """reset_baseline must delete the baseline file."""
    from scripts.validation.detect_anomalies import (
        detect_anomalies, reset_baseline, load_baseline
    )

    baseline_dir.mkdir(parents=True)
    df = pd.DataFrame({"temperature_2m": [25.0] * 100})
    reg = DummyRegistry()

    # Create a baseline
    detect_anomalies(df, reg, DateFireSeason(), baseline_dir=baseline_dir)
    assert load_baseline("temperature_2m", "fire_season", baseline_dir) is not None

    # Reset it
    reset_baseline("temperature_2m", "fire_season", baseline_dir=baseline_dir)
    assert load_baseline("temperature_2m", "fire_season", baseline_dir) is None


def test_season_uses_correct_baseline_file(baseline_dir):
    """Fire season and off-season must use separate baseline files."""
    from scripts.validation.detect_anomalies import detect_anomalies, load_baseline

    baseline_dir.mkdir(parents=True)
    reg = DummyRegistry()

    # Feed fire season data (high temps)
    df_summer = pd.DataFrame({"temperature_2m": [30.0] * 100})
    detect_anomalies(df_summer, reg, DateFireSeason(), baseline_dir=baseline_dir)

    # Feed off-season data (low temps)
    df_winter = pd.DataFrame({"temperature_2m": [10.0] * 100})
    detect_anomalies(df_winter, reg, DateOffSeason(), baseline_dir=baseline_dir)

    summer_baseline = load_baseline("temperature_2m", "fire_season", baseline_dir)
    winter_baseline = load_baseline("temperature_2m", "off_season", baseline_dir)

    assert summer_baseline is not None
    assert winter_baseline is not None
    assert summer_baseline["mean"] > winter_baseline["mean"], (
        "Summer baseline mean should be higher than winter baseline mean"
    )
