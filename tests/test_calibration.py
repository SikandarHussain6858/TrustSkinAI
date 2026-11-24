"""Unit tests for Temperature Scaling, ECE, and Brier Score."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from trustskin.calibration import TemperatureScaler, compute_brier_score, compute_ece


def test_ece_and_brier_perfect_predictions() -> None:
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    targets = np.array([0, 1, 2])
    assert np.isclose(compute_ece(probs, targets, n_bins=10), 0.0, atol=1e-7)
    assert np.isclose(compute_brier_score(probs, targets), 0.0, atol=1e-7)


def test_temperature_scaler_improves_overconfident_nll() -> None:
    torch.manual_seed(0)
    # Create overconfident logits (scaled by 5.0)
    raw_logits = torch.randn(200, 7) * 5.0
    targets = torch.randint(0, 7, (200,))

    nll_before = F.cross_entropy(raw_logits, targets).item()
    scaler = TemperatureScaler()
    learned_t = scaler.fit(raw_logits, targets, lr=0.05, max_iter=50)
    nll_after = F.cross_entropy(scaler(raw_logits), targets).item()

    assert learned_t > 1.0
    assert nll_after <= nll_before + 1e-5
    probs = scaler.predict_probs(raw_logits)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
