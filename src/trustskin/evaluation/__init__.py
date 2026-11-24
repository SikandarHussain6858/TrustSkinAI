"""Classification evaluation metrics and publication-quality plotting utilities."""

from .metrics import compute_classification_metrics
from .plots import (
    plot_confusion_matrix,
    plot_reliability_diagram,
    plot_risk_coverage_comparison,
    plot_uncertainty_histograms,
)

__all__ = [
    "compute_classification_metrics",
    "plot_confusion_matrix",
    "plot_reliability_diagram",
    "plot_risk_coverage_comparison",
    "plot_uncertainty_histograms",
]
