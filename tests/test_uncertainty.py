"""Unit tests for uncertainty estimation metrics and MC Dropout aggregation."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from trustskin.models import UncertaintyCNN
from trustskin.uncertainty import mc_dropout_predict, mutual_information, predictive_entropy


def test_predictive_entropy_bounds() -> None:
    certain = np.array([[1.0, 0.0, 0.0]])
    uniform = np.array([[1 / 3, 1 / 3, 1 / 3]])

    h_certain = predictive_entropy(certain)[0]
    h_uniform = predictive_entropy(uniform)[0]

    assert np.isclose(h_certain, 0.0, atol=1e-6)
    assert np.isclose(h_uniform, np.log(3.0), atol=1e-6)


def test_mutual_information_zero_when_identical_passes() -> None:
    probs = np.array([[[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]]])
    mc_probs = np.repeat(probs, repeats=5, axis=0)
    mi = mutual_information(mc_probs)
    assert np.allclose(mi, 0.0, atol=1e-7)


def test_mc_dropout_predict_shapes() -> None:
    torch.manual_seed(42)
    x = torch.randn(6, 3, 224, 224)
    y = torch.randint(0, 7, (6,))
    loader = DataLoader(TensorDataset(x, y), batch_size=3)

    model = UncertaintyCNN(in_channels=3, num_classes=7, dropout_p=0.25)
    out = mc_dropout_predict(model, loader, device=torch.device("cpu"), mc_samples=4)

    assert out["mean_probs"].shape == (6, 7)
    assert out["preds"].shape == (6,)
    assert out["entropy"].shape == (6,)
    assert out["mutual_information"].shape == (6,)
    assert np.allclose(out["mean_probs"].sum(axis=1), 1.0, atol=1e-5)
