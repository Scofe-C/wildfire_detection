from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, auc, precision_recall_curve

logger = logging.getLogger(__name__)

_RC = {"figure.figsize": (8, 6), "font.size": 12, "axes.titlesize": 14}


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str | Path,
    model_name: str = "XGBoost PoF",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = auc(recall, precision)

    with plt.rc_context(_RC):
        fig, ax = plt.subplots()
        ax.plot(recall, precision, lw=2, label=f"{model_name} (AUC-PR={auc_pr:.3f})")
        ax.fill_between(recall, precision, alpha=0.15)
        ax.axhline(y=y_true.mean(), color="gray", ls="--", label=f"No-skill ({y_true.mean():.3f})")
        ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve",
               xlim=[0, 1], ylim=[0, 1.05])
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info("PR curve saved: %s", output_path)
    return output_path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
    model_name: str = "XGBoost PoF",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(_RC):
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred,
            display_labels=["No Fire", "Fire"],
            cmap="Blues", ax=ax, colorbar=False,
        )
        ax.set_title(f"Confusion Matrix — {model_name}")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info("Confusion matrix saved: %s", output_path)
    return output_path


def plot_model_comparison(
    metrics: dict[str, dict[str, float]],
    output_path: str | Path,
    metric_names: list[str] | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_names = list(metrics.keys())
    if not model_names:
        logger.warning("No models to compare")
        return output_path

    if metric_names is None:
        all_keys = [set(m.keys()) for m in metrics.values()]
        metric_names = sorted(set.intersection(*all_keys)) if all_keys else []
    if not metric_names:
        logger.warning("No common metrics to compare")
        return output_path

    x = np.arange(len(metric_names))
    width = 0.8 / len(model_names)

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, name in enumerate(model_names):
            vals = [metrics[name].get(m, 0) for m in metric_names]
            offset = (i - len(model_names) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=name, alpha=0.85)
            for bar, v in zip(bars, vals, strict=False):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set(xlabel="Metric", ylabel="Score", title="Model Comparison", ylim=[0, 1.1])
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", " ").title() for m in metric_names])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info("Comparison chart saved: %s", output_path)
    return output_path


def generate_all_visualizations(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    comparison_metrics: dict[str, dict[str, float]] | None,
    output_dir: str | Path,
    model_name: str = "XGBoost PoF",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    y_pred = (y_prob >= threshold).astype(int)
    paths: dict[str, Path] = {}

    paths["pr_curve"] = plot_precision_recall_curve(
        y_true, y_prob, output_dir / "precision_recall_curve.png", model_name)
    paths["confusion_matrix"] = plot_confusion_matrix(
        y_true, y_pred, output_dir / "confusion_matrix.png", model_name)
    if comparison_metrics:
        paths["model_comparison"] = plot_model_comparison(
            comparison_metrics, output_dir / "model_comparison.png")

    return paths
