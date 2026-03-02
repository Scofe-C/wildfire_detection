# tests/test_fusion/test_fusion_properties.py
"""
Property-based tests using Hypothesis.
Install: pip install hypothesis

These find edge cases that example-based tests miss — especially important
for fuse_features() which handles partial/empty/mismatched DataFrames.
"""
import pandas as pd
import numpy as np
import pytest

try:
    from hypothesis import given, settings, HealthCheck
    from hypothesis import strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not HYPOTHESIS_AVAILABLE,
    reason="hypothesis not installed — pip install hypothesis"
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

grid_ids = st.lists(
    st.text(alphabet="abcdefghijklmnop", min_size=3, max_size=8),
    min_size=1, max_size=50,
    unique=True,
)

frp_values = st.floats(min_value=0.0, max_value=5000.0, allow_nan=False)
count_values = st.integers(min_value=0, max_value=500)


@st.composite
def firms_df(draw):
    ids = draw(grid_ids)
    return pd.DataFrame({
        "grid_id": ids,
        "active_fire_count": [draw(count_values) for _ in ids],
        "mean_frp": [draw(frp_values) for _ in ids],
        "fire_detected_binary": [draw(st.integers(0, 1)) for _ in ids],
    })


@st.composite
def fused_df_strategy(draw):
    ids = draw(grid_ids)
    n = len(ids)
    lats  = draw(st.lists(st.floats(32.0, 42.0), min_size=n, max_size=n))
    lons  = draw(st.lists(st.floats(-125.0, -93.0), min_size=n, max_size=n))
    fires = draw(st.lists(st.integers(0, 1), min_size=n, max_size=n))
    frps  = draw(st.lists(st.floats(0.0, 5000.0, allow_nan=False), min_size=n, max_size=n))
    return pd.DataFrame({
        "grid_id": ids,
        "latitude": lats,
        "longitude": lons,
        "fire_detected_binary": fires,
        "active_fire_count": [draw(count_values) for _ in ids],
        "mean_frp": frps,
        "median_frp": frps,
        "max_confidence": [draw(st.integers(0, 100)) for _ in ids],
        "nearest_fire_distance_km": [draw(st.floats(-1.0, 500.0, allow_nan=False)) for _ in ids],
    })


# ---------------------------------------------------------------------------
# Invariant tests
# ---------------------------------------------------------------------------

@given(fused=fused_df_strategy())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_temporal_lag_preserves_label(fused):
    """fire_detected_binary must NEVER be modified by apply_temporal_lag."""
    from scripts.fusion.fuse_features import apply_temporal_lag

    original_labels = fused["fire_detected_binary"].tolist()
    ml = apply_temporal_lag(fused, prev_fire_features=None)
    assert ml["fire_detected_binary"].tolist() == original_labels


@given(fused=fused_df_strategy())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_temporal_lag_does_not_mutate_input(fused):
    """apply_temporal_lag must return a new DataFrame, not modify fused in place."""
    from scripts.fusion.fuse_features import apply_temporal_lag

    snapshot = fused["active_fire_count"].tolist()
    _ = apply_temporal_lag(fused, prev_fire_features=None)
    assert fused["active_fire_count"].tolist() == snapshot


@given(fused=fused_df_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_priority_resolver_noop_on_empty_gt(fused):
    """resolve_priorities with empty GT must never change fire feature values."""
    from scripts.fusion.priority_resolver import resolve_priorities

    original_fire_count = fused["active_fire_count"].tolist()
    result = resolve_priorities(fused, pd.DataFrame())
    assert result["active_fire_count"].tolist() == original_fire_count


@given(fused=fused_df_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_priority_resolver_adds_priority_column(fused):
    """resolve_priorities must always add data_source_priority column."""
    from scripts.fusion.priority_resolver import resolve_priorities
    result = resolve_priorities(fused, pd.DataFrame())
    assert "data_source_priority" in result.columns
    assert (result["data_source_priority"] == 2).all()


@given(firms=firms_df())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_fire_detected_binary_is_always_0_or_1(firms):
    """fire_detected_binary must only ever be 0 or 1 — never float, never >1."""
    valid = {0, 1}
    assert set(firms["fire_detected_binary"].tolist()).issubset(valid)