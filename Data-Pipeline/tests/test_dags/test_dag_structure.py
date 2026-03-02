# tests/test_dags/test_dag_structure.py
"""
DAG structure tests — verify task graph wiring without executing any tasks.
These run at parse time, so they're fast and catch broken dependencies early.
"""
import platform
import sys

import pytest


@pytest.fixture(scope="module")
def wildfire_dag():
    """Import and return the wildfire_data_pipeline DAG object.

    Skips on Windows because Airflow depends on ``fcntl`` (Unix-only).
    These tests run in Docker / GitHub Actions CI on Linux.
    """
    if platform.system() == "Windows":
        pytest.skip(
            "Airflow requires fcntl (Unix-only); "
            "run DAG structure tests in Docker or CI"
        )
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[2]))
    import dags.wildfire_dag as dag_module
    return dag_module.dag


class TestDAGStructure:

    def test_dag_loads_without_error(self, wildfire_dag):
        assert wildfire_dag is not None
        assert wildfire_dag.dag_id == "wildfire_data_pipeline"

    def test_expected_tasks_present(self, wildfire_dag):
        task_ids = {t.task_id for t in wildfire_dag.tasks}
        required = {
            "check_static_cache",
            "fuse_features",
            "validate_schema",
            "detect_anomalies",
            "export_to_parquet",
            "export_spatial",
            "version_with_dvc",
        }
        assert required.issubset(task_ids), (
            f"Missing tasks: {required - task_ids}"
        )

    def test_regional_taskgroups_present(self, wildfire_dag):
        task_ids = {t.task_id for t in wildfire_dag.tasks}
        assert any("california" in tid for tid in task_ids)
        assert any("texas" in tid for tid in task_ids)

    def test_fuse_has_none_failed_trigger_rule(self, wildfire_dag):
        """fuse_features must use none_failed to handle ShortCircuit skip correctly."""
        fuse_task = wildfire_dag.get_task("fuse_features")
        assert fuse_task.trigger_rule == "none_failed", (
            "fuse_features trigger_rule must be 'none_failed' to handle "
            "check_static_cache ShortCircuit skip gracefully"
        )

    def test_max_active_runs_is_1(self, wildfire_dag):
        """Prevents concurrent DVC lock conflicts."""
        assert wildfire_dag.max_active_runs == 1

    def test_default_params_include_required_keys(self, wildfire_dag):
        required_params = {
            "resolution_km", "trigger_source", "fire_cells",
            "weather_lookback_hours", "h3_ring_max",
        }
        dag_params = set(wildfire_dag.params.keys())
        assert required_params.issubset(dag_params), (
            f"Missing DAG params: {required_params - dag_params}"
        )

    def test_export_tasks_depend_on_detect_anomalies(self, wildfire_dag):
        """Both export tasks must be downstream of detect_anomalies."""
        export_parquet = wildfire_dag.get_task("export_to_parquet")
        export_spatial = wildfire_dag.get_task("export_spatial")
        detect = wildfire_dag.get_task("detect_anomalies")

        upstream_of_parquet = {t.task_id for t in export_parquet.upstream_list}
        upstream_of_spatial = {t.task_id for t in export_spatial.upstream_list}

        assert "detect_anomalies" in upstream_of_parquet
        assert "detect_anomalies" in upstream_of_spatial

    def test_version_dvc_is_terminal(self, wildfire_dag):
        """version_with_dvc must have no downstream tasks."""
        dvc_task = wildfire_dag.get_task("version_with_dvc")
        assert len(dvc_task.downstream_list) == 0


class TestXComKeyConsistency:
    """XCom keys pushed by one task must match what the downstream task pulls."""

    def test_static_path_key_is_consistent(self):
        """check_static_cache and load_static_layers must push under the same key."""
        import platform
        if platform.system() == "Windows":
            pytest.skip("Airflow requires fcntl (Unix-only); run in Docker or CI")
        import inspect
        import dags.wildfire_dag as dag_module

        check_src = inspect.getsource(dag_module.task_check_static_cache)
        load_src  = inspect.getsource(dag_module.task_load_static_layers)
        fuse_src  = inspect.getsource(dag_module.task_fuse_features)

        # All must use the same key string
        assert 'key="static_features_path"' in check_src
        assert 'key="static_features_path"' in load_src
        assert '"static_features_path"' in fuse_src

    def test_fuse_pulls_from_both_static_task_ids(self):
        """Bug 1 fix: fuse_features must pull from both task_ids explicitly."""
        import platform
        if platform.system() == "Windows":
            pytest.skip("Airflow requires fcntl (Unix-only); run in Docker or CI")
        import inspect
        import dags.wildfire_dag as dag_module

        fuse_src = inspect.getsource(dag_module.task_fuse_features)
        assert 'task_ids="check_static_cache"' in fuse_src
        assert 'task_ids="load_static_layers"' in fuse_src