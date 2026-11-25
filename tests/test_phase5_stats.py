"""Unit tests for Phase 5 additions: dampened class weights, McNemar, bootstrap CIs, and threshold calibration."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from trustskin.data import compute_class_weights
from trustskin.selective import evaluate_at_threshold, select_threshold_for_coverage

# Load helper functions from scripts/07_difficult_case_eval.py
_spec = importlib.util.spec_from_file_location(
    "difficult_case_eval",
    Path(__file__).resolve().parent.parent / "scripts" / "07_difficult_case_eval.py",
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_compute_class_weights_power_dampening() -> None:
    labels = [0] * 100 + [1] * 10 + [2] * 1
    w_lin = compute_class_weights(labels, num_classes=3, power=1.0).numpy()
    w_sqrt = compute_class_weights(labels, num_classes=3, power=0.5).numpy()

    # Both normalized to mean 1.0
    assert np.isclose(w_lin.mean(), 1.0, atol=1e-5)
    assert np.isclose(w_sqrt.mean(), 1.0, atol=1e-5)

    # Square-root weights have strictly smaller max/min ratio than linear weights
    ratio_lin = w_lin.max() / w_lin.min()
    ratio_sqrt = w_sqrt.max() / w_sqrt.min()
    assert ratio_sqrt < ratio_lin
    assert np.isclose(ratio_sqrt, np.sqrt(ratio_lin), rtol=1e-4)


def test_exact_mcnemar_test_known_cases() -> None:
    # Identical models -> p = 1.0
    a = np.array([1, 0, 1, 1, 0])
    res_same = _mod.exact_mcnemar_test(a, a)
    assert res_same["net_gain"] == 0
    assert np.isclose(res_same["p_value"], 1.0)

    # Model B fixes 20 errors and regresses 0 -> p = 2 * 0.5^20 < 1e-5
    base = np.zeros(20, dtype=int)
    better = np.ones(20, dtype=int)
    res_better = _mod.exact_mcnemar_test(base, better)
    assert res_better["n_fixed_by_model"] == 20
    assert res_better["n_regressed_by_model"] == 0
    assert res_better["p_value"] < 1e-5


def test_bootstrap_metrics_ci_contains_point() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 7, size=200)
    y_pred = y_true.copy()
    y_pred[:40] = (y_pred[:40] + 1) % 7
    conf = np.where(y_pred == y_true, 0.85, 0.40)

    cis = _mod.bootstrap_metrics(y_true, y_pred, conf, n_resamples=200, seed=42)
    for key in ["accuracy", "balanced_accuracy", "macro_f1", "aurc"]:
        pt, lo, hi = cis[key]
        assert lo <= pt <= hi


def test_matching_val_test_threshold_coverage() -> None:
    rng = np.random.default_rng(42)
    val_scores = rng.beta(5, 2, size=500)
    test_scores = rng.beta(5, 2, size=2000)
    preds = np.zeros(2000, dtype=int)
    targets = np.zeros(2000, dtype=int)

    for target_cov in [0.9, 0.8, 0.7]:
        th = select_threshold_for_coverage(val_scores, target_cov, higher_is_more_confident=True)
        res = evaluate_at_threshold(preds, targets, test_scores, th, higher_is_more_confident=True)
        assert abs(res["coverage"] - target_cov) < 0.04
