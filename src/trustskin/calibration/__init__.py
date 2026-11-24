"""Post-hoc confidence calibration (Temperature Scaling) and calibration metrics."""

from .temperature_scaling import (
    TemperatureScaler,
    compute_brier_score,
    compute_ece,
    reliability_bins,
)

__all__ = [
    "TemperatureScaler",
    "compute_brier_score",
    "compute_ece",
    "reliability_bins",
]
