"""Selective prediction (abstention) and risk-coverage evaluation."""

from __future__ import annotations

import numpy as np


def evaluate_at_threshold(
    preds: np.ndarray,
    targets: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    higher_is_more_confident: bool = True,
) -> dict[str, float]:
    """Evaluate coverage, selective risk (error rate), and selective accuracy at one threshold."""
    accepted = scores >= threshold if higher_is_more_confident else scores <= threshold
    n_total = len(targets)
    n_accepted = int(np.sum(accepted))
    coverage = n_accepted / float(max(n_total, 1))

    if n_accepted == 0:
        return {
            "threshold": float(threshold),
            "coverage": 0.0,
            "selective_risk": 0.0,
            "selective_accuracy": 1.0,
            "accepted_count": 0,
            "referred_count": n_total,
        }

    selective_acc = float(np.mean(preds[accepted] == targets[accepted]))
    return {
        "threshold": float(threshold),
        "coverage": float(coverage),
        "selective_risk": float(1.0 - selective_acc),
        "selective_accuracy": selective_acc,
        "accepted_count": n_accepted,
        "referred_count": n_total - n_accepted,
    }


def compute_risk_coverage_curve(
    preds: np.ndarray,
    targets: np.ndarray,
    scores: np.ndarray,
    higher_is_more_confident: bool = True,
) -> dict[str, np.ndarray]:
    """Compute exact sorted Risk-Coverage curve by ordering samples from most to least confident."""
    order = np.argsort(-scores) if higher_is_more_confident else np.argsort(scores)
    sorted_errors = (preds[order] != targets[order]).astype(np.float64)

    n = len(sorted_errors)
    counts = np.arange(1, n + 1, dtype=np.float64)
    coverages = counts / float(n)
    cumulative_errors = np.cumsum(sorted_errors)
    risks = cumulative_errors / counts
    sorted_thresholds = scores[order]

    return {
        "coverages": coverages,
        "risks": risks,
        "accuracies": 1.0 - risks,
        "thresholds": sorted_thresholds,
    }


def compute_aurc(
    preds: np.ndarray,
    targets: np.ndarray,
    scores: np.ndarray,
    higher_is_more_confident: bool = True,
) -> float:
    """Compute Area Under the Risk-Coverage curve (AURC); lower is better."""
    curve = compute_risk_coverage_curve(
        preds=preds,
        targets=targets,
        scores=scores,
        higher_is_more_confident=higher_is_more_confident,
    )
    # Discrete Riemann / trapezoidal integral over coverage in (0, 1]
    return float(np.mean(curve["risks"]))


def select_threshold_for_coverage(
    val_scores: np.ndarray,
    target_coverage: float,
    higher_is_more_confident: bool = True,
) -> float:
    """Choose a decision threshold strictly on validation scores for a target coverage."""
    cov = float(np.clip(target_coverage, 0.0, 1.0))
    if higher_is_more_confident:
        return float(np.quantile(val_scores, 1.0 - cov))
    return float(np.quantile(val_scores, cov))
