import numpy as np
import pytest
from pathlib import Path

from src.validation.visualizations import (
    plot_precision_recall_curve, plot_confusion_matrix,
    plot_model_comparison, generate_all_visualizations,
)


@pytest.fixture
def preds():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.uniform(0, 1, size=200)
    y_prob[y_true == 1] += 0.3
    y_prob = np.clip(y_prob, 0, 1)
    return y_true, y_prob


class TestPRCurve:
    def test_creates_png(self, tmp_path, preds):
        p = plot_precision_recall_curve(*preds, tmp_path / "pr.png")
        assert p.exists() and p.stat().st_size > 1000


class TestConfusionMatrix:
    def test_creates_png(self, tmp_path, preds):
        y_true, y_prob = preds
        p = plot_confusion_matrix(y_true, (y_prob >= 0.5).astype(int), tmp_path / "cm.png")
        assert p.exists()


class TestModelComparison:
    def test_creates_png(self, tmp_path):
        m = {
            "XGBoost": {"auc_pr": 0.82, "f1": 0.75},
            "FWI": {"auc_pr": 0.68, "f1": 0.61},
        }
        p = plot_model_comparison(m, tmp_path / "cmp.png")
        assert p.exists()

    def test_empty(self, tmp_path):
        plot_model_comparison({}, tmp_path / "empty.png")


class TestGenerateAll:
    def test_all_paths(self, tmp_path, preds):
        comparison = {"A": {"auc_pr": 0.78}, "B": {"auc_pr": 0.65}}
        paths = generate_all_visualizations(*preds, 0.5, comparison, tmp_path)
        assert "pr_curve" in paths and "confusion_matrix" in paths and "model_comparison" in paths
        assert all(Path(p).exists() for p in paths.values())
