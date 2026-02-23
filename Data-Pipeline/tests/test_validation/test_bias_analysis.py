"""
Tests for Data Bias Analysis
==============================

Covers:
  - _add_derived_columns: fuel tier mapping, season labelling, quality tier
  - _kl_divergence_approx: identical distributions, maximally different, edge cases
  - _compute_slice_stats: correct mean/std/null_rate per feature
  - _run_categorical_slices: correct grouping, KL flagging, fire rate disparity
  - run_bias_analysis: end-to-end with all four slicing dimensions
  - _synthesize_findings: finding and mitigation generation
  - Empty / degenerate DataFrame guards
  - CLI output path writing
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class DummyRegistry:
    """Minimal registry stub — mirrors the interface used in detect_anomalies tests."""
    fire_season_months = [6, 7, 8, 9, 10, 11]

    def get_z_threshold(self, month: int) -> float:
        return 4.0 if month in self.fire_season_months else 3.5


@pytest.fixture
def registry():
    return DummyRegistry()


@pytest.fixture
def base_df() -> pd.DataFrame:
    """A minimal valid fused DataFrame with all four slicing dimensions present."""
    rng = np.random.default_rng(42)
    n = 200

    return pd.DataFrame({
        "grid_id": [f"cell_{i}" for i in range(n)],
        "region": (["california"] * 120) + (["texas"] * 80),
        "timestamp_utc": (
            [pd.Timestamp("2026-08-01 00:00", tz="UTC")] * 100 +  # fire season CA
            [pd.Timestamp("2026-02-01 00:00", tz="UTC")] * 20 +   # off-season CA
            [pd.Timestamp("2026-08-01 00:00", tz="UTC")] * 50 +   # fire season TX
            [pd.Timestamp("2026-02-01 00:00", tz="UTC")] * 30     # off-season TX
        ),
        "fuel_model_fbfm40": (
            [101] * 50 +   # grass
            [141] * 50 +   # shrub
            [181] * 50 +   # timber litter
            [91]  * 30 +   # non-burnable
            [999] * 20     # unknown
        ),
        "data_quality_flag": (
            [0] * 120 +   # tier A — good quality
            [4] * 80      # tier B — degraded
        ),
        "temperature_2m":       rng.normal(28.0, 5.0, n).tolist(),
        "relative_humidity_2m": rng.uniform(10.0, 80.0, n).tolist(),
        "wind_speed_10m":       rng.exponential(8.0, n).tolist(),
        "active_fire_count":    rng.integers(0, 10, n).tolist(),
        "mean_frp":             rng.exponential(50.0, n).tolist(),
        "fire_detected_binary": ([1] * 20 + [0] * 180),
        "precipitation":        rng.exponential(2.0, n).tolist(),
    })


@pytest.fixture
def ca_tx_bias_df() -> pd.DataFrame:
    """DataFrame where Texas has a significantly different temperature distribution."""
    rng = np.random.default_rng(7)
    ca = pd.DataFrame({
        "region": ["california"] * 100,
        "temperature_2m": rng.normal(28.0, 3.0, 100),
        "fire_detected_binary": [0] * 90 + [1] * 10,
        "data_quality_flag": [0] * 100,
        "timestamp_utc": [pd.Timestamp("2026-08-01", tz="UTC")] * 100,
    })
    tx = pd.DataFrame({
        "region": ["texas"] * 100,
        # Mean shifted by 15°C — should exceed KL divergence threshold
        "temperature_2m": rng.normal(43.0, 3.0, 100),
        "fire_detected_binary": [0] * 50 + [1] * 50,  # much higher fire rate
        "data_quality_flag": [4] * 100,  # all degraded
        "timestamp_utc": [pd.Timestamp("2026-08-01", tz="UTC")] * 100,
    })
    return pd.concat([ca, tx], ignore_index=True)


# ---------------------------------------------------------------------------
# _add_derived_columns
# ---------------------------------------------------------------------------

class TestAddDerivedColumns:

    def test_fuel_model_tier_grass_code(self, registry):
        """FBFM40 codes 101-109 must map to 'grass'."""
        from scripts.validation.bias_analysis import _add_derived_columns
        df = pd.DataFrame({"fuel_model_fbfm40": [101, 105, 109]})
        out = _add_derived_columns(df, registry)
        assert "fuel_model_tier" in out.columns
        assert (out["fuel_model_tier"] == "grass").all()

    def test_fuel_model_tier_non_burnable_code(self, registry):
        """FBFM40 codes 91-99 must map to 'non_burnable'."""
        from scripts.validation.bias_analysis import _add_derived_columns
        df = pd.DataFrame({"fuel_model_fbfm40": [91, 92, 98, 99]})
        out = _add_derived_columns(df, registry)
        assert (out["fuel_model_tier"] == "non_burnable").all()

    def test_fuel_model_tier_unknown_code(self, registry):
        """Unrecognised FBFM40 codes must produce 'unknown'."""
        from scripts.validation.bias_analysis import _add_derived_columns
        df = pd.DataFrame({"fuel_model_fbfm40": [0, 999, 500]})
        out = _add_derived_columns(df, registry)
        assert (out["fuel_model_tier"] == "unknown").all()

    def test_fuel_model_tier_absent_column(self, registry):
        """If fuel_model_fbfm40 is missing, no fuel_model_tier column is added."""
        from scripts.validation.bias_analysis import _add_derived_columns
        df = pd.DataFrame({"temperature_2m": [25.0]})
        out = _add_derived_columns(df, registry)
        assert "fuel_model_tier" not in out.columns

    def test_fire_season_label_august(self, registry):
        """August timestamps must produce 'fire_season' label."""
        from scripts.validation.bias_analysis import _add_derived_columns
        df = pd.DataFrame({
            "timestamp_utc": [pd.Timestamp("2026-08-15", tz="UTC")] * 5
        })
        out = _add_derived_columns(df, registry)
        assert "fire_season_label" in out.columns
        assert (out["fire_season_label"] == "fire_season").all()

    def test_fire_season_label_february(self, registry):
        """February timestamps must produce 'off_season' label."""
        from scripts.validation.bias_analysis import _add_derived_columns
        df = pd.DataFrame({
            "timestamp_utc": [pd.Timestamp("2026-02-15", tz="UTC")] * 5
        })
        out = _add_derived_columns(df, registry)
        assert (out["fire_season_label"] == "off_season").all()

    def test_quality_tier_a(self, registry):
        """data_quality_flag 0-2 must map to 'tier_a'."""
        from scripts.validation.bias_analysis import _add_derived_columns
        df = pd.DataFrame({"data_quality_flag": [0, 1, 2]})
        out = _add_derived_columns(df, registry)
        assert "quality_tier" in out.columns
        assert (out["quality_tier"] == "tier_a").all()

    def test_quality_tier_b(self, registry):
        """data_quality_flag 3-5 must map to 'tier_b'."""
        from scripts.validation.bias_analysis import _add_derived_columns
        df = pd.DataFrame({"data_quality_flag": [3, 4, 5]})
        out = _add_derived_columns(df, registry)
        assert (out["quality_tier"] == "tier_b").all()

    def test_quality_tier_null_flag(self, registry):
        """Null data_quality_flag must produce None quality_tier (not crash)."""
        from scripts.validation.bias_analysis import _add_derived_columns
        df = pd.DataFrame({"data_quality_flag": [0, None, 4]})
        out = _add_derived_columns(df, registry)
        assert out["quality_tier"].iloc[0] == "tier_a"
        import pandas as _pd
        assert _pd.isna(out["quality_tier"].iloc[1])
        assert out["quality_tier"].iloc[2] == "tier_b"

    def test_original_df_not_mutated(self, registry):
        """_add_derived_columns must not modify the caller's DataFrame."""
        from scripts.validation.bias_analysis import _add_derived_columns
        df = pd.DataFrame({
            "fuel_model_fbfm40": [101],
            "data_quality_flag": [0],
            "timestamp_utc": [pd.Timestamp("2026-08-01", tz="UTC")],
        })
        original_cols = set(df.columns)
        _add_derived_columns(df, registry)
        assert set(df.columns) == original_cols, "Input DataFrame was mutated"


# ---------------------------------------------------------------------------
# _kl_divergence_approx
# ---------------------------------------------------------------------------

class TestKLDivergenceApprox:

    def test_identical_distributions_near_zero(self):
        """KL(P || P) must be approximately 0."""
        from scripts.validation.bias_analysis import _kl_divergence_approx
        rng = np.random.default_rng(42)
        samples = pd.Series(rng.normal(0, 1, 500))
        kl = _kl_divergence_approx(samples, samples)
        assert kl < 0.05, f"KL of identical distributions should be ~0, got {kl}"

    def test_non_overlapping_distributions_high_kl(self):
        """Distributions with no overlap must produce KL above bias threshold."""
        from scripts.validation.bias_analysis import (
            _kl_divergence_approx,
            KL_DIVERGENCE_THRESHOLD,
        )
        rng = np.random.default_rng(42)
        p = pd.Series(rng.normal(0, 1, 500))
        q = pd.Series(rng.normal(20, 1, 500))
        kl = _kl_divergence_approx(p, q)
        assert kl > KL_DIVERGENCE_THRESHOLD, (
            f"Non-overlapping distributions should have KL > {KL_DIVERGENCE_THRESHOLD}, got {kl}"
        )

    def test_constant_feature_returns_zero(self):
        """Constant feature (zero variance) must return 0 without error."""
        from scripts.validation.bias_analysis import _kl_divergence_approx
        p = pd.Series([5.0] * 100)
        q = pd.Series([5.0] * 100)
        assert _kl_divergence_approx(p, q) == 0.0

    def test_too_few_samples_returns_zero(self):
        """Fewer than 5 samples in either series must return 0 without error."""
        from scripts.validation.bias_analysis import _kl_divergence_approx
        small = pd.Series([1.0, 2.0, 3.0])
        large = pd.Series(np.random.default_rng(1).normal(0, 1, 200))
        assert _kl_divergence_approx(small, large) == 0.0
        assert _kl_divergence_approx(large, small) == 0.0

    def test_nan_values_are_dropped(self):
        """NaN values in either series must be silently dropped, not raise."""
        from scripts.validation.bias_analysis import _kl_divergence_approx
        rng = np.random.default_rng(42)
        p = pd.Series(np.append(rng.normal(0, 1, 100), [np.nan] * 20))
        q = pd.Series(rng.normal(0, 1, 120))
        kl = _kl_divergence_approx(p, q)
        assert kl >= 0.0

    def test_kl_is_non_negative(self):
        """KL divergence must always be non-negative."""
        from scripts.validation.bias_analysis import _kl_divergence_approx
        rng = np.random.default_rng(99)
        for _ in range(10):
            p = pd.Series(rng.normal(rng.uniform(-5, 5), rng.uniform(0.5, 3), 100))
            q = pd.Series(rng.normal(rng.uniform(-5, 5), rng.uniform(0.5, 3), 200))
            assert _kl_divergence_approx(p, q) >= 0.0


# ---------------------------------------------------------------------------
# _compute_slice_stats
# ---------------------------------------------------------------------------

class TestComputeSliceStats:

    def test_mean_std_correct(self):
        """Mean and std in the output must match pandas values."""
        from scripts.validation.bias_analysis import _compute_slice_stats
        df = pd.DataFrame({"temperature_2m": [10.0, 20.0, 30.0, 40.0, 50.0]})
        stats = _compute_slice_stats(df, ["temperature_2m"], label="test")
        feat = stats["feature_stats"]["temperature_2m"]
        assert abs(feat["mean"] - 30.0) < 0.01
        assert feat["n"] == 5

    def test_null_rate_computed(self):
        """Null rate must reflect the fraction of NaN values in the column."""
        from scripts.validation.bias_analysis import _compute_slice_stats
        df = pd.DataFrame({"wind_speed_10m": [5.0, None, 8.0, None, 6.0]})
        stats = _compute_slice_stats(df, ["wind_speed_10m"], label="test")
        feat = stats["feature_stats"]["wind_speed_10m"]
        assert abs(feat["null_rate"] - 0.4) < 0.01

    def test_all_null_column_skipped(self):
        """An all-null column must not appear in feature_stats (no crash)."""
        from scripts.validation.bias_analysis import _compute_slice_stats
        df = pd.DataFrame({"temperature_2m": [None, None, None]})
        stats = _compute_slice_stats(df, ["temperature_2m"], label="test")
        # Either absent or handled gracefully
        assert "temperature_2m" not in stats.get("feature_stats", {})

    def test_missing_column_skipped(self):
        """Columns listed but absent from df must be silently skipped."""
        from scripts.validation.bias_analysis import _compute_slice_stats
        df = pd.DataFrame({"temperature_2m": [25.0, 26.0]})
        stats = _compute_slice_stats(df, ["temperature_2m", "nonexistent"], label="test")
        assert "nonexistent" not in stats["feature_stats"]
        assert "temperature_2m" in stats["feature_stats"]

    def test_row_count_matches_input(self):
        """row_count in the result must equal len(df)."""
        from scripts.validation.bias_analysis import _compute_slice_stats
        df = pd.DataFrame({"temperature_2m": range(77)})
        stats = _compute_slice_stats(df, ["temperature_2m"], label="test")
        assert stats["row_count"] == 77


# ---------------------------------------------------------------------------
# _run_categorical_slices
# ---------------------------------------------------------------------------

class TestRunCategoricalSlices:

    def test_slice_groups_correct_row_count(self, registry, base_df):
        """Each slice entry's row_count must match the filtered DataFrame size."""
        from scripts.validation.bias_analysis import (
            _add_derived_columns,
            _compute_slice_stats,
            _run_categorical_slices,
            NUMERIC_FEATURES_OF_INTEREST,
        )
        df = _add_derived_columns(base_df, registry)
        present_numeric = [c for c in NUMERIC_FEATURES_OF_INTEREST if c in df.columns]
        overall = _compute_slice_stats(df, present_numeric, label="overall")

        slices = _run_categorical_slices(df, "region", "geographic_region", present_numeric, overall)
        ca = next(s for s in slices if s["slice_value"] == "california")
        tx = next(s for s in slices if s["slice_value"] == "texas")

        assert ca["row_count"] == 120
        assert tx["row_count"] == 80
        assert ca["row_count"] + tx["row_count"] == len(df)

    def test_biased_slice_flagged(self, registry, ca_tx_bias_df):
        """A slice with a 15°C mean shift must be flagged as biased."""
        from scripts.validation.bias_analysis import (
            _add_derived_columns,
            _compute_slice_stats,
            _run_categorical_slices,
        )
        df = _add_derived_columns(ca_tx_bias_df, registry)
        overall = _compute_slice_stats(df, ["temperature_2m"], label="overall")

        slices = _run_categorical_slices(
            df, "region", "geographic_region", ["temperature_2m"], overall
        )
        tx_slice = next((s for s in slices if s["slice_value"] == "texas"), None)
        assert tx_slice is not None, "Texas slice must be present"
        assert tx_slice["has_bias"] is True, (
            "A 15°C mean shift should produce KL > threshold and flag the slice"
        )
        assert "temperature_2m" in tx_slice["biased_features"]

    def test_fire_rate_disparity_flagged(self, registry, ca_tx_bias_df):
        """A slice with 40-pp fire rate difference must be flagged as fire_rate_biased."""
        from scripts.validation.bias_analysis import (
            _add_derived_columns,
            _compute_slice_stats,
            _run_categorical_slices,
        )
        df = _add_derived_columns(ca_tx_bias_df, registry)
        # CA: 10/100 = 0.10 fire rate, TX: 50/100 = 0.50 fire rate → 40pp disparity
        overall = _compute_slice_stats(df, ["fire_detected_binary"], label="overall")

        slices = _run_categorical_slices(
            df, "region", "geographic_region", ["fire_detected_binary"], overall
        )
        tx_slice = next(s for s in slices if s["slice_value"] == "texas")
        assert tx_slice["fire_rate_biased"] is True

    def test_small_slice_skipped(self, registry):
        """Slices with fewer than 10 rows must be omitted from results."""
        from scripts.validation.bias_analysis import (
            _add_derived_columns,
            _compute_slice_stats,
            _run_categorical_slices,
        )
        # Make texas appear only 3 times — below the minimum
        df = pd.DataFrame({
            "region": ["california"] * 100 + ["texas"] * 3,
            "temperature_2m": list(np.random.default_rng(0).normal(25, 3, 103)),
        })
        overall = _compute_slice_stats(df, ["temperature_2m"], label="overall")
        slices = _run_categorical_slices(df, "region", "geo", ["temperature_2m"], overall)
        slice_values = [s["slice_value"] for s in slices]
        assert "texas" not in slice_values, "Small slices (<10 rows) must be omitted"

    def test_kl_divergences_present_for_numeric_features(self, registry, base_df):
        """kl_divergences dict must contain an entry for each present numeric feature."""
        from scripts.validation.bias_analysis import (
            _add_derived_columns,
            _compute_slice_stats,
            _run_categorical_slices,
            NUMERIC_FEATURES_OF_INTEREST,
        )
        df = _add_derived_columns(base_df, registry)
        present_numeric = [c for c in NUMERIC_FEATURES_OF_INTEREST if c in df.columns]
        overall = _compute_slice_stats(df, present_numeric, label="overall")

        slices = _run_categorical_slices(df, "region", "geo", present_numeric, overall)
        for s in slices:
            for feat in present_numeric:
                assert feat in s["kl_divergences"], (
                    f"kl_divergences missing '{feat}' in slice {s['slice_value']}"
                )

    def test_pct_of_total_sums_to_100(self, registry, base_df):
        """pct_of_total across all region slices must sum to approximately 100."""
        from scripts.validation.bias_analysis import (
            _add_derived_columns,
            _compute_slice_stats,
            _run_categorical_slices,
            NUMERIC_FEATURES_OF_INTEREST,
        )
        df = _add_derived_columns(base_df, registry)
        present_numeric = [c for c in NUMERIC_FEATURES_OF_INTEREST if c in df.columns]
        overall = _compute_slice_stats(df, present_numeric, label="overall")

        slices = _run_categorical_slices(df, "region", "geo", present_numeric, overall)
        total_pct = sum(s["pct_of_total"] for s in slices)
        assert abs(total_pct - 100.0) < 1.0, f"pct_of_total sum = {total_pct}, expected ~100"


# ---------------------------------------------------------------------------
# run_bias_analysis (end-to-end)
# ---------------------------------------------------------------------------

class TestRunBiasAnalysis:

    def test_returns_required_keys(self, registry, base_df):
        """Report dict must contain all required top-level keys."""
        from scripts.validation.bias_analysis import run_bias_analysis
        report = run_bias_analysis(base_df, registry)
        for key in ("generated_at", "row_count", "overall_stats", "slices", "findings", "mitigations"):
            assert key in report, f"Missing key '{key}' in bias report"

    def test_row_count_matches_input(self, registry, base_df):
        """row_count in the report must match the input DataFrame length."""
        from scripts.validation.bias_analysis import run_bias_analysis
        report = run_bias_analysis(base_df, registry)
        assert report["row_count"] == len(base_df)

    def test_all_four_slice_dimensions_present(self, registry, base_df):
        """All four slicing dimensions must appear in the slices list."""
        from scripts.validation.bias_analysis import run_bias_analysis
        report = run_bias_analysis(base_df, registry)
        dimensions = {s["slice_dimension"] for s in report["slices"]}
        assert "geographic_region" in dimensions, "geographic_region slices missing"
        assert "fuel_model_tier" in dimensions, "fuel_model_tier slices missing"
        assert "fire_season" in dimensions, "fire_season slices missing"
        assert "data_quality_tier" in dimensions, "data_quality_tier slices missing"

    def test_findings_non_empty(self, registry, base_df):
        """findings list must always contain at least one entry."""
        from scripts.validation.bias_analysis import run_bias_analysis
        report = run_bias_analysis(base_df, registry)
        assert len(report["findings"]) >= 1

    def test_biased_df_produces_mitigations(self, registry, ca_tx_bias_df):
        """A strongly biased DataFrame must produce at least one mitigation."""
        from scripts.validation.bias_analysis import run_bias_analysis
        # Add required columns for all four slicers
        ca_tx_bias_df["fuel_model_fbfm40"] = [101] * len(ca_tx_bias_df)
        report = run_bias_analysis(ca_tx_bias_df, registry)
        assert len(report["mitigations"]) >= 1, (
            "Strongly biased input must produce at least one mitigation recommendation"
        )

    def test_empty_dataframe_returns_gracefully(self, registry):
        """An empty DataFrame must not raise and must return a valid (minimal) report."""
        from scripts.validation.bias_analysis import run_bias_analysis
        report = run_bias_analysis(pd.DataFrame(), registry)
        assert report["row_count"] == 0
        assert isinstance(report["findings"], list)
        assert len(report["findings"]) >= 1

    def test_report_written_to_file(self, registry, base_df, tmp_path):
        """When output_path is given, the JSON report must be written to disk."""
        from scripts.validation.bias_analysis import run_bias_analysis
        output = tmp_path / "bias_report.json"
        run_bias_analysis(base_df, registry, output_path=output)
        assert output.exists(), "Report file must be created at output_path"

        with open(output, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["row_count"] == len(base_df)

    def test_report_json_is_serialisable(self, registry, base_df, tmp_path):
        """All values in the report must be JSON-serialisable (no numpy scalars)."""
        from scripts.validation.bias_analysis import run_bias_analysis
        output = tmp_path / "bias_report.json"
        run_bias_analysis(base_df, registry, output_path=output)
        # If the file loads without error, all values serialised correctly
        with open(output, encoding="utf-8") as f:
            loaded = json.load(f)
        assert isinstance(loaded, dict)

    def test_slices_have_expected_structure(self, registry, base_df):
        """Each slice dict must contain the required keys."""
        from scripts.validation.bias_analysis import run_bias_analysis
        required_slice_keys = {
            "slice_dimension", "slice_value", "row_count", "pct_of_total",
            "fire_rate", "fire_rate_disparity", "fire_rate_biased",
            "null_rates", "kl_divergences", "biased_features", "has_bias",
        }
        report = run_bias_analysis(base_df, registry)
        for s in report["slices"]:
            missing = required_slice_keys - set(s.keys())
            assert not missing, f"Slice {s.get('slice_value')} missing keys: {missing}"

    def test_no_region_column_still_produces_report(self, registry):
        """If 'region' column is absent, the report must still run (with a warning)."""
        from scripts.validation.bias_analysis import run_bias_analysis
        df = pd.DataFrame({
            "temperature_2m": np.random.default_rng(0).normal(25, 5, 50),
            "data_quality_flag": [0] * 50,
            "fire_detected_binary": [0] * 45 + [1] * 5,
            "fuel_model_fbfm40": [101] * 50,
            "timestamp_utc": [pd.Timestamp("2026-08-01", tz="UTC")] * 50,
        })
        report = run_bias_analysis(df, registry)
        # geographic_region slice should be absent — others may still be present
        dims = {s["slice_dimension"] for s in report["slices"]}
        assert "geographic_region" not in dims


# ---------------------------------------------------------------------------
# _synthesize_findings
# ---------------------------------------------------------------------------

class TestSynthesizeFindings:

    def test_no_biased_slices_produces_clean_finding(self):
        """When no slices are biased, findings must report clean status."""
        from scripts.validation.bias_analysis import _synthesize_findings
        clean_slices = [
            {"slice_dimension": "geographic_region", "slice_value": "california",
             "has_bias": False, "biased_features": [], "fire_rate_biased": False,
             "fire_rate": 0.05, "fire_rate_disparity": 0.01, "kl_divergences": {}},
        ]
        findings, mitigations = _synthesize_findings(clean_slices, {})
        assert any("No significant bias" in f for f in findings)
        assert mitigations == []

    def test_geographic_bias_produces_mitigation(self):
        """Biased geographic_region slice must produce the region mitigation text."""
        from scripts.validation.bias_analysis import _synthesize_findings
        biased = [
            {"slice_dimension": "geographic_region", "slice_value": "texas",
             "has_bias": True, "biased_features": ["temperature_2m"],
             "fire_rate_biased": False, "fire_rate": 0.05, "fire_rate_disparity": 0.01,
             "kl_divergences": {"temperature_2m": 0.5},
             "row_count": 100, "pct_of_total": 50.0},
        ]
        findings, mitigations = _synthesize_findings(biased, {})
        assert any("geographic" in m.lower() or "region" in m.lower() for m in mitigations)

    def test_fire_season_bias_produces_mitigation(self):
        """Biased fire_season slice must produce the season-specific mitigation."""
        from scripts.validation.bias_analysis import _synthesize_findings
        biased = [
            {"slice_dimension": "fire_season", "slice_value": "off_season",
             "has_bias": True, "biased_features": ["active_fire_count"],
             "fire_rate_biased": True, "fire_rate": 0.001, "fire_rate_disparity": 0.10,
             "kl_divergences": {"active_fire_count": 0.25},
             "row_count": 80, "pct_of_total": 40.0},
        ]
        findings, mitigations = _synthesize_findings(biased, {})
        assert any("season" in m.lower() for m in mitigations)

    def test_data_quality_bias_produces_mitigation(self):
        """Biased data_quality_tier slice must produce the quality tier mitigation."""
        from scripts.validation.bias_analysis import _synthesize_findings
        biased = [
            {"slice_dimension": "data_quality_tier", "slice_value": "tier_b",
             "has_bias": True, "biased_features": ["wind_speed_10m"],
             "fire_rate_biased": False, "fire_rate": 0.04, "fire_rate_disparity": 0.02,
             "kl_divergences": {"wind_speed_10m": 0.15},
             "row_count": 80, "pct_of_total": 40.0},
        ]
        findings, mitigations = _synthesize_findings(biased, {})
        assert any("quality" in m.lower() or "tier" in m.lower() for m in mitigations)

    def test_fire_rate_disparity_appears_in_findings(self):
        """A fire_rate_biased slice must produce a finding mentioning the disparity."""
        from scripts.validation.bias_analysis import _synthesize_findings
        biased = [
            {"slice_dimension": "geographic_region", "slice_value": "texas",
             "has_bias": True, "biased_features": [],
             "fire_rate_biased": True, "fire_rate": 0.45, "fire_rate_disparity": 0.35,
             "kl_divergences": {},
             "row_count": 100, "pct_of_total": 50.0},
        ]
        findings, _ = _synthesize_findings(biased, {})
        # At least one finding must mention disparity or fire rate
        assert any("disparity" in f.lower() or "fire" in f.lower() for f in findings)

    def test_each_biased_dimension_gets_one_mitigation(self):
        """Two biased slices of the same dimension must produce only one mitigation entry."""
        from scripts.validation.bias_analysis import _synthesize_findings
        biased = [
            {"slice_dimension": "geographic_region", "slice_value": "california",
             "has_bias": True, "biased_features": ["temperature_2m"],
             "fire_rate_biased": False, "fire_rate": 0.05, "fire_rate_disparity": 0.01,
             "kl_divergences": {"temperature_2m": 0.3},
             "row_count": 120, "pct_of_total": 60.0},
            {"slice_dimension": "geographic_region", "slice_value": "texas",
             "has_bias": True, "biased_features": ["temperature_2m"],
             "fire_rate_biased": False, "fire_rate": 0.07, "fire_rate_disparity": 0.02,
             "kl_divergences": {"temperature_2m": 0.4},
             "row_count": 80, "pct_of_total": 40.0},
        ]
        _, mitigations = _synthesize_findings(biased, {})
        geo_mitigations = [m for m in mitigations if "region" in m.lower() or "geographic" in m.lower()]
        assert len(geo_mitigations) == 1, (
            "Same slice dimension appearing twice should produce only one mitigation"
        )