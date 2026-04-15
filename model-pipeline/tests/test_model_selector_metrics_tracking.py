"""Unit tests for model_selector, metrics (gaps), vertex_sync, mlflow_logger.

All external dependencies (mlflow, google-cloud-aiplatform, yaml config) are mocked.
No real config files, no real API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ===========================================================================
# Helpers shared across test classes
# ===========================================================================

def _perfect_binary(n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Perfect classifier: y_prob == y_true."""
    y_true = np.array([0] * (n // 2) + [1] * (n // 2))
    y_prob = y_true.astype(float)
    return y_true, y_prob


def _random_binary(n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=n)
    y_prob = rng.random(size=n)
    return y_true, y_prob


# ===========================================================================
# metrics.py — gaps (lines 71-72, 83-90): measure_inference_latency
# and auc_roc branch in compute_all_metrics
# ===========================================================================

class TestMeasureInferenceLatency:
    def test_returns_float(self) -> None:
        from src.validation.metrics import measure_inference_latency
        X = pd.DataFrame({"a": [1, 2, 3]})
        latency = measure_inference_latency(lambda df: df * 2, X, n_runs=3)
        assert isinstance(latency, float)
        assert latency >= 0.0

    def test_calls_predict_fn_n_runs_times(self) -> None:
        from src.validation.metrics import measure_inference_latency
        X = pd.DataFrame({"a": [1]})
        call_count = 0

        def count_fn(df: pd.DataFrame) -> pd.DataFrame:
            nonlocal call_count
            call_count += 1
            return df

        measure_inference_latency(count_fn, X, n_runs=5)
        assert call_count == 5

    def test_single_run(self) -> None:
        from src.validation.metrics import measure_inference_latency
        X = pd.DataFrame({"a": [1]})
        latency = measure_inference_latency(lambda df: df, X, n_runs=1)
        assert latency >= 0.0


class TestComputeAllMetricsAucRoc:
    def test_auc_roc_present_for_binary(self) -> None:
        from src.validation.metrics import compute_all_metrics
        y_true, y_prob = _random_binary()
        metrics = compute_all_metrics(y_true, y_prob)
        assert "auc_roc" in metrics
        assert metrics["auc_roc"] is not None
        assert 0.0 <= metrics["auc_roc"] <= 1.0

    def test_auc_roc_none_or_nan_when_single_class(self) -> None:
        import math

        from src.validation.metrics import compute_all_metrics
        # Only one class → roc_auc_score raises ValueError or returns nan
        y_true = np.zeros(50)
        y_prob = np.random.default_rng(0).random(50)
        metrics = compute_all_metrics(y_true, y_prob)
        v = metrics["auc_roc"]
        assert v is None or (isinstance(v, float) and math.isnan(v))

    def test_latency_injected_when_provided(self) -> None:
        from src.validation.metrics import compute_all_metrics
        y_true, y_prob = _random_binary()
        metrics = compute_all_metrics(y_true, y_prob, inference_latency_ms=42.5)
        assert metrics["inference_latency_ms"] == 42.5

    def test_latency_absent_when_not_provided(self) -> None:
        from src.validation.metrics import compute_all_metrics
        y_true, y_prob = _random_binary()
        metrics = compute_all_metrics(y_true, y_prob)
        assert "inference_latency_ms" not in metrics


# ===========================================================================
# model_selector.py
# ===========================================================================

MOCK_CONFIG = {
    "validation": {
        "decision_threshold": 0.5,
        "auc_pr_threshold": 0.75,
    }
}


class TestValidationResultDataclass:
    def _make_result(self, **kwargs: Any):
        from src.validation.model_selector import ValidationResult
        defaults = dict(
            model_name="xgboost_pof",
            version="1.0.0",
            metrics={"auc_pr": 0.82},
            passed_validation=True,
            passed_bias_gate=True,
            bias_report_path=None,
            visualization_paths={},
        )
        defaults.update(kwargs)
        return ValidationResult(**defaults)

    def test_is_deployable_true(self) -> None:
        r = self._make_result(passed_validation=True, passed_bias_gate=True)
        assert r.is_deployable is True

    def test_is_deployable_false_when_validation_fails(self) -> None:
        r = self._make_result(passed_validation=False, passed_bias_gate=True)
        assert r.is_deployable is False

    def test_is_deployable_false_when_bias_gate_fails(self) -> None:
        r = self._make_result(passed_validation=True, passed_bias_gate=False)
        assert r.is_deployable is False

    def test_is_deployable_false_when_bias_gate_none(self) -> None:
        r = self._make_result(passed_validation=True, passed_bias_gate=None)
        assert r.is_deployable is False


class TestValidateModel:
    def test_passes_with_high_auc_pr(self) -> None:
        from src.validation.model_selector import validate_model
        y_true, y_prob = _perfect_binary()
        with patch("src.validation.model_selector._load_config", return_value=MOCK_CONFIG):
            metrics, passed = validate_model(y_true, y_prob)
        assert passed is True
        assert metrics["auc_pr"] > 0.75

    def test_fails_with_low_auc_pr(self) -> None:
        from src.validation.model_selector import validate_model
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=100)
        # Inverted probs → very low AUC-PR
        y_prob = 1.0 - y_true.astype(float)
        with patch("src.validation.model_selector._load_config", return_value=MOCK_CONFIG):
            metrics, passed = validate_model(y_true, y_prob)
        assert passed is False

    def test_returns_metrics_dict(self) -> None:
        from src.validation.model_selector import validate_model
        y_true, y_prob = _random_binary()
        with patch("src.validation.model_selector._load_config", return_value=MOCK_CONFIG):
            metrics, _ = validate_model(y_true, y_prob)
        assert "auc_pr" in metrics
        assert "f1" in metrics
        assert "fnr" in metrics


class TestSaveValidationReport:
    def _make_result(self):
        from src.validation.model_selector import ValidationResult
        return ValidationResult(
            model_name="xgboost_pof",
            version="1.0.0",
            metrics={"auc_pr": 0.82, "f1": 0.78},
            passed_validation=True,
            passed_bias_gate=True,
            bias_report_path="/tmp/bias.json",
            visualization_paths={"pr_curve": "/tmp/pr.png"},
        )

    def test_creates_report_file(self, tmp_path: Path) -> None:
        from src.validation.model_selector import save_validation_report
        result = self._make_result()
        report_path = save_validation_report(result, tmp_path)
        assert report_path.exists()
        assert report_path.name == "validation_report.json"

    def test_report_content(self, tmp_path: Path) -> None:
        from src.validation.model_selector import save_validation_report
        result = self._make_result()
        report_path = save_validation_report(result, tmp_path)
        data = json.loads(report_path.read_text())
        assert data["model_name"] == "xgboost_pof"
        assert data["passed_validation"] is True
        assert data["is_deployable"] is True
        assert data["metrics"]["auc_pr"] == 0.82
        assert data["bias_report_path"] == "/tmp/bias.json"

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        from src.validation.model_selector import save_validation_report
        result = self._make_result()
        deep_dir = tmp_path / "reports" / "validation"
        save_validation_report(result, deep_dir)
        assert deep_dir.exists()


# ===========================================================================
# vertex_sync.py
# ===========================================================================

MOCK_VERTEX_CONFIG = {
    "tracking": {
        "vertex_ai": {
            "project_id": "test-project",
            "location": "us-central1",
            "experiment_name": "test-experiment",
        }
    }
}


class TestVertexAISync:
    def _make_sync(self) -> Any:
        from src.tracking.vertex_sync import VertexAISync
        with patch("src.tracking.vertex_sync._load_vertex_config",
                   return_value=MOCK_VERTEX_CONFIG["tracking"]["vertex_ai"]):
            return VertexAISync(
                project_id="test-project",
                location="us-central1",
                experiment_name="test-experiment",
            )

    def test_init_sets_project_and_location(self) -> None:
        sync = self._make_sync()
        assert sync._project_id == "test-project"
        assert sync._location == "us-central1"
        assert sync._experiment_name == "test-experiment"
        assert sync._initialized is False

    def test_sync_run_calls_vertex_ai(self) -> None:
        sync = self._make_sync()

        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)

        mock_aiplatform = MagicMock()
        mock_aiplatform.start_run.return_value = mock_run

        with patch.dict("sys.modules", {"google.cloud.aiplatform": mock_aiplatform,
                                        "google.cloud": MagicMock(aiplatform=mock_aiplatform)}):
            sync._aiplatform = mock_aiplatform
            sync._initialized = True
            run_id = sync.sync_run(
                run_id="run-001",
                metrics={"auc_pr": 0.82},
                params={"model": "xgboost"},
            )

        assert run_id == "run-001"
        mock_run.log_metrics.assert_called_once_with({"auc_pr": 0.82})
        mock_run.log_params.assert_called_once_with({"model": "xgboost"})

    def test_sync_run_filters_non_numeric_metrics(self) -> None:
        sync = self._make_sync()

        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)

        mock_aiplatform = MagicMock()
        mock_aiplatform.start_run.return_value = mock_run

        with patch.dict("sys.modules", {"google.cloud.aiplatform": mock_aiplatform,
                                        "google.cloud": MagicMock(aiplatform=mock_aiplatform)}):
            sync._aiplatform = mock_aiplatform
            sync._initialized = True
            sync.sync_run(
                run_id="run-001",
                metrics={"auc_pr": 0.82, "label": "high"},  # "label" is str, should be filtered
                params={},
            )

        logged = mock_run.log_metrics.call_args[0][0]
        assert "auc_pr" in logged
        assert "label" not in logged

    def test_sync_rollback_prefixes_run_id(self) -> None:
        sync = self._make_sync()

        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)

        mock_aiplatform = MagicMock()
        mock_aiplatform.start_run.return_value = mock_run

        with patch.dict("sys.modules", {"google.cloud.aiplatform": mock_aiplatform,
                                        "google.cloud": MagicMock(aiplatform=mock_aiplatform)}):
            sync._aiplatform = mock_aiplatform
            sync._initialized = True
            sync.sync_rollback_event(
                run_id="run-001",
                reason_code="auc_pr_regression",
                delta_auc_pr=-0.05,
            )

        # start_run should be called with "rollback-run-001"
        mock_aiplatform.start_run.assert_called_with(run="rollback-run-001")


# ===========================================================================
# mlflow_logger.py
# ===========================================================================

MOCK_MLFLOW_CONFIG = {
    "tracking": {
        "mlflow": {
            "tracking_uri": "http://localhost:5000",
            "experiment_name": "test-experiment",
        }
    }
}


class TestMLflowLogger:
    def _make_logger(self) -> tuple[Any, MagicMock]:
        """Returns (MLflowLogger, mock_mlflow)."""
        from src.tracking.mlflow_logger import MLflowLogger
        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "run-abc-123"
        mock_mlflow.start_run.return_value = mock_run

        with patch("src.tracking.mlflow_logger._load_tracking_config",
                   return_value=MOCK_MLFLOW_CONFIG["tracking"]["mlflow"]),              patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            logger = MLflowLogger()
            logger._mlflow = mock_mlflow

        return logger, mock_mlflow

    def test_start_run_returns_run_id(self) -> None:
        logger, mock_mlflow = self._make_logger()
        run_id = logger.start_run(run_name="test-run")
        assert run_id == "run-abc-123"
        mock_mlflow.start_run.assert_called_once_with(run_name="test-run", tags=None)

    def test_end_run(self) -> None:
        logger, mock_mlflow = self._make_logger()
        logger.end_run()
        mock_mlflow.end_run.assert_called_once_with(status="FINISHED")

    def test_log_metrics_skips_none(self) -> None:
        logger, mock_mlflow = self._make_logger()
        logger.log_metrics({"auc_pr": 0.82, "f1": None, "fnr": 0.1})
        calls = [c[0] for c in mock_mlflow.log_metric.call_args_list]
        keys = [c[0] for c in calls]
        assert "auc_pr" in keys
        assert "fnr" in keys
        assert "f1" not in keys  # None filtered out

    def test_log_params_converts_to_str(self) -> None:
        logger, mock_mlflow = self._make_logger()
        logger.log_params({"n_estimators": 100, "model": "xgboost"})
        mock_mlflow.log_params.assert_called_once_with(
            {"n_estimators": "100", "model": "xgboost"}
        )

    def test_log_model_hash(self) -> None:
        logger, mock_mlflow = self._make_logger()
        logger.log_model_hash("abc123")
        mock_mlflow.log_param.assert_called_once_with("model_artifact_sha256", "abc123")

    def test_log_bias_gate_result(self) -> None:
        logger, mock_mlflow = self._make_logger()
        bias_report = {
            "gate_result": "PASS",
            "overall_fnr": 0.12,
            "slices": {
                "nri": {
                    "disparity": 0.03,
                    "per_group_fnr": {"high_risk": 0.10, "low_risk": 0.13},
                },
            },
        }
        logger.log_bias_gate_result(bias_report)
        mock_mlflow.log_param.assert_called_once_with("bias_gate_result", "PASS")
        metric_calls = {c[0][0]: c[0][1] for c in mock_mlflow.log_metric.call_args_list}
        assert metric_calls["bias_overall_fnr"] == 0.12
        assert metric_calls["bias_disparity_nri"] == 0.03
        assert metric_calls["bias_fnr_nri_high_risk"] == 0.10

    def test_log_validation_result(self) -> None:
        logger, mock_mlflow = self._make_logger()
        metrics = {"auc_pr": 0.82, "f1": 0.78, "confusion_matrix": {}}
        logger.log_validation_result(metrics, passed=True)
        mock_mlflow.log_param.assert_called_once_with("validation_passed", "True")
        metric_keys = [c[0][0] for c in mock_mlflow.log_metric.call_args_list]
        assert "auc_pr" in metric_keys
        assert "f1" in metric_keys
        assert "confusion_matrix" not in metric_keys  # dict, not numeric

    def test_log_visualization_skips_missing_files(self, tmp_path: Path) -> None:
        logger, mock_mlflow = self._make_logger()
        existing = tmp_path / "pr_curve.png"
        existing.write_bytes(b"fake-png")
        missing = tmp_path / "nonexistent.png"

        logger.log_visualization({"pr_curve": existing, "missing": missing})
        assert mock_mlflow.log_artifact.call_count == 1
        mock_mlflow.log_artifact.assert_called_once_with(
            str(existing), "visualizations"
        )


class TestComputeInputStatistics:
    def test_returns_stats_for_numeric_cols(self) -> None:
        from src.tracking.mlflow_logger import compute_input_statistics
        X = pd.DataFrame({"temp": [1.0, 2.0, 3.0], "rh": [10.0, 20.0, 30.0]})
        stats = compute_input_statistics(X)
        assert "temp" in stats
        assert "rh" in stats
        assert set(stats["temp"].keys()) == {"mean", "std", "min", "max"}
        assert stats["temp"]["mean"] == pytest.approx(2.0)

    def test_skips_non_numeric_cols(self) -> None:
        from src.tracking.mlflow_logger import compute_input_statistics
        X = pd.DataFrame({"temp": [1.0, 2.0], "label": ["a", "b"]})
        stats = compute_input_statistics(X)
        assert "temp" in stats
        assert "label" not in stats

    def test_skips_all_nan_cols(self) -> None:
        from src.tracking.mlflow_logger import compute_input_statistics
        X = pd.DataFrame({"temp": [float("nan"), float("nan")], "rh": [10.0, 20.0]})
        stats = compute_input_statistics(X)
        assert "temp" not in stats
        assert "rh" in stats

    def test_empty_dataframe_returns_empty(self) -> None:
        from src.tracking.mlflow_logger import compute_input_statistics
        stats = compute_input_statistics(pd.DataFrame())
        assert stats == {}
