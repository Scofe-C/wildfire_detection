"""Unit tests for drift detection and performance monitoring."""
from __future__ import annotations

import numpy as np
import pytest

from src.monitoring.drift_detector import DriftDetector, _compute_psi
from src.monitoring.performance_monitor import PerformanceMonitor


# ── PSI computation ───────────────────────────────────────────────────────────

def test_psi_identical_distributions():
    arr = np.random.default_rng(0).uniform(0, 1, 500)
    psi = _compute_psi(arr, arr)
    assert psi < 0.01, f"Expected near-zero PSI for identical distributions, got {psi}"


def test_psi_very_different_distributions():
    baseline = np.zeros(500)
    current = np.ones(500)
    psi = _compute_psi(baseline, current)
    assert psi > 0.25, f"Expected high PSI for fully shifted distribution, got {psi}"


def test_psi_moderate_shift():
    rng = np.random.default_rng(42)
    baseline = rng.normal(0, 1, 500)
    current = rng.normal(0.5, 1, 500)  # moderate mean shift
    psi = _compute_psi(baseline, current)
    assert psi > 0.0, "PSI should be positive for shifted distribution"


# ── DriftDetector ─────────────────────────────────────────────────────────────

def test_drift_detector_no_drift():
    rng = np.random.default_rng(1)
    vals = rng.uniform(0, 100, 300).tolist()
    baseline_stats = {"samples": {"temperature_2m": vals}}
    current_data = {"temperature_2m": np.array(vals)}  # same distribution

    detector = DriftDetector()
    report = detector.detect(baseline_stats, current_data)
    assert report.verdict == "OK"
    assert len(report.feature_results) == 1


def test_drift_detector_critical_drift():
    rng = np.random.default_rng(2)
    baseline_vals = rng.normal(20, 5, 500).tolist()
    current_vals = rng.normal(50, 5, 500)  # large shift

    baseline_stats = {"samples": {"temperature_2m": baseline_vals}}
    current_data = {"temperature_2m": current_vals}

    detector = DriftDetector()
    report = detector.detect(baseline_stats, current_data)
    assert report.verdict == "CRITICAL"
    assert "temperature_2m" in report.drifted_features


def test_drift_detector_missing_feature_in_current():
    baseline_stats = {"samples": {"temperature_2m": [1.0, 2.0, 3.0] * 50}}
    current_data = {}  # missing feature

    detector = DriftDetector()
    report = detector.detect(baseline_stats, current_data)
    assert report.verdict == "OK"  # no features to compare → no drift flagged


def test_drift_detector_extra_feature_in_current_ignored():
    rng = np.random.default_rng(3)
    vals = rng.uniform(0, 1, 200).tolist()
    baseline_stats = {"samples": {"temperature_2m": vals}}
    current_data = {
        "temperature_2m": np.array(vals),
        "extra_feature": np.ones(200),  # not in baseline — should be ignored
    }
    detector = DriftDetector()
    report = detector.detect(baseline_stats, current_data)
    assert len(report.feature_results) == 1


# ── PerformanceMonitor ────────────────────────────────────────────────────────

def test_performance_monitor_no_drift():
    rng = np.random.default_rng(4)
    baseline_scores = rng.beta(1, 5, 1000)
    baseline_stats = {
        "mean": float(np.mean(baseline_scores)),
        "std": float(np.std(baseline_scores)),
        "critical_rate": float(np.mean(baseline_scores >= 0.75)),
    }
    current_scores = rng.beta(1, 5, 500)  # same distribution

    monitor = PerformanceMonitor()
    report = monitor.check(baseline_stats, current_scores)
    assert report.verdict == "OK"


def test_performance_monitor_critical_mean_shift():
    baseline_stats = {"mean": 0.1, "std": 0.05, "critical_rate": 0.02}
    current_scores = np.full(500, 0.9)  # massive shift

    monitor = PerformanceMonitor(mean_shift_threshold=0.1, critical_rate_multiplier=2.0)
    report = monitor.check(baseline_stats, current_scores)
    assert report.verdict == "CRITICAL"


def test_performance_monitor_critical_rate_explosion():
    baseline_stats = {"mean": 0.15, "std": 0.05, "critical_rate": 0.05}
    current_scores = np.concatenate([
        np.full(200, 0.8),   # 40% CRITICAL — 8x baseline rate
        np.full(300, 0.1),
    ])
    monitor = PerformanceMonitor(critical_rate_multiplier=2.0)
    report = monitor.check(baseline_stats, current_scores)
    assert report.critical_rate_ratio > 2.0
    assert report.verdict == "CRITICAL"