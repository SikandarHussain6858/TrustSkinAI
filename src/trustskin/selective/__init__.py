"""Selective classification, coverage-risk curves, and AURC."""

from .selective_risk import (
    compute_aurc,
    compute_risk_coverage_curve,
    evaluate_at_threshold,
    select_threshold_for_coverage,
)

__all__ = [
    "compute_aurc",
    "compute_risk_coverage_curve",
    "evaluate_at_threshold",
    "select_threshold_for_coverage",
]
