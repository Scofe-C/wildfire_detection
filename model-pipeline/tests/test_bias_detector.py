import numpy as np

from src.bias.detector import false_negative_rate, run_bias_gate


def _make_predictions(biased: bool, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = 200
    y_true, y_pred, groups = [], [], []
    for group in ["Low", "Medium", "High", "Very High"]:
        yt = rng.integers(0, 2, size=n)
        yp = yt.copy()
        if biased and group == "Very High":
            pos_idx = np.where(yt == 1)[0]
            flip = int(len(pos_idx) * 0.4)
            if flip > 0:
                yp[rng.choice(pos_idx, size=flip, replace=False)] = 0
        else:
            flip_idx = rng.choice(n, size=n // (20 if not biased else 10), replace=False)
            yp[flip_idx] = 1 - yp[flip_idx]
        y_true.extend(yt)
        y_pred.extend(yp)
        groups.extend([group] * n)
    return np.array(y_true), np.array(y_pred), np.array(groups)


def _write_config(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("bias_gate:\n  max_disparity: 0.05\n")
    return p


class TestFNRFunction:
    def test_perfect(self):
        assert false_negative_rate(np.array([0, 1, 1, 0]), np.array([0, 1, 1, 0])) == 0.0

    def test_all_miss(self):
        assert false_negative_rate(np.array([1, 1, 1]), np.array([0, 0, 0])) == 1.0


class TestBiasGate:
    def test_fair_pass(self, tmp_path):
        y_t, y_p, g = _make_predictions(biased=False)
        report, passed = run_bias_gate(y_t, y_p, g, _write_config(tmp_path))
        assert passed is True
        assert report["gate_result"] == "PASS"

    def test_biased_fail(self, tmp_path):
        y_t, y_p, g = _make_predictions(biased=True)
        report, passed = run_bias_gate(y_t, y_p, g, _write_config(tmp_path))
        assert passed is False
        assert report["gate_result"] == "FAIL"

    def test_report_has_per_group(self, tmp_path):
        y_t, y_p, g = _make_predictions(biased=False)
        report, _ = run_bias_gate(y_t, y_p, g, _write_config(tmp_path))
        assert "Very High" in report["per_group_fnr"]
        assert "Low" in report["per_group_fnr"]
