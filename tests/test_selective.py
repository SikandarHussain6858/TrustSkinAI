"""Unit tests for selective classification, coverage-risk curve, and AURC."""

from __future__ import annotations

import numpy as np

from trustskin.selective import (
    compute_aurc,
    compute_risk_coverage_curve,
    evaluate_at_threshold,
    select_threshold_for_coverage,
)


def test_risk_coverage_perfect_ranking() -> None:
    # High confidence predictions are correct; low confidence predictions are wrong
    preds = np.array([0, 1, 2, 0])
    targets = np.array([0, 1, 0, 1])
    scores = np.array([0.95, 0.85, 0.40, 0.20])

    curve = compute_risk_coverage_curve(preds, targets, scores, higher_is_more_confident=True)
    assert np.allclose(curve["coverages"], [0.25, 0.5, 0.75, 1.0])
    assert np.allclose(curve["risks"], [0.0, 0.0, 1 / 3, 0.5])

    aurc = compute_aurc(preds, targets, scores, higher_is_more_confident=True)
    assert np.isclose(aurc, np.mean([0.0, 0.0, 1 / 3, 0.5]))


def test_evaluate_at_threshold_and_selection() -> None:
    preds = np.array([0, 1, 2, 0])
    targets = np.array([0, 1, 0, 1])
    scores = np.array([0.95, 0.85, 0.40, 0.20])

    th = select_threshold_for_coverage(scores, target_coverage=0.5, higher_is_more_confident=True)
    res = evaluate_at_threshold(preds, targets, scores, threshold=th, higher_is_more_confident=True)
    assert res["coverage"] == 0.5
    assert res["selective_risk"] == 0.0
    assert res["accepted_count"] == 2
    assert res["referred_count"] == 2
