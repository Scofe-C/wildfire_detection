import numpy as np
import pytest

from src.validation.metrics import (
    compute_all_metrics,
    compute_auc_pr,
    compute_confusion_matrix,
    compute_fnr,
)


@pytest.fixture
def perfect():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.1, 0.9, 0.8, 0.95])
    return y_true, y_prob


@pytest.fixture
def imbalanced():
    rng = np.random.default_rng(42)
    n = 1000
    y_true = np.zeros(n, dtype=int)
    y_true[:15] = 1
    rng.shuffle(y_true)
    y_prob = rng.uniform(0.0, 0.3, size=n)
    y_prob[y_true == 1] = rng.uniform(0.4, 0.9, size=y_true.sum())
    return y_true, y_prob


class TestAUCPR:
    def test_perfect(self, perfect):
        assert compute_auc_pr(*perfect) > 0.95

    def test_random_low(self):
        rng = np.random.default_rng(42)
        assert compute_auc_pr(rng.integers(0, 2, 500), rng.uniform(0, 1, 500)) < 0.7


class TestFNR:
    def test_zero(self, perfect):
        y_true, y_prob = perfect
        assert compute_fnr(y_true, (y_prob >= 0.5).astype(int)) == 0.0

    def test_all_miss(self):
        assert compute_fnr(np.array([1, 1, 1]), np.array([0, 0, 0])) == 1.0

    def test_no_positives(self):
        assert compute_fnr(np.array([0, 0, 0]), np.array([0, 0, 0])) == 0.0


class TestConfusionMatrix:
    def test_keys(self, perfect):
        y_true, y_prob = perfect
        cm = compute_confusion_matrix(y_true, (y_prob >= 0.5).astype(int))
        assert set(cm.keys()) == {"true_negatives", "false_positives", "false_negatives", "true_positives"}

    def test_perfect_no_errors(self, perfect):
        y_true, y_prob = perfect
        cm = compute_confusion_matrix(y_true, (y_prob >= 0.5).astype(int))
        assert cm["false_positives"] == 0 and cm["false_negatives"] == 0


class TestAllMetrics:
    def test_required_keys(self, imbalanced):
        m = compute_all_metrics(*imbalanced)
        required = {"auc_pr", "f1", "fnr", "confusion_matrix", "positive_rate", "threshold", "n_samples", "accuracy"}
        assert required.issubset(m.keys())

    def test_positive_rate(self, imbalanced):
        m = compute_all_metrics(*imbalanced)
        assert abs(m["positive_rate"] - 0.015) < 0.01

    def test_latency_passed_through(self, perfect):
        m = compute_all_metrics(*perfect, inference_latency_ms=42.5)
        assert m["inference_latency_ms"] == 42.5
