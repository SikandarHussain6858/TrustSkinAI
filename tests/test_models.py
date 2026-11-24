"""Unit tests for BaselineCNN, UncertaintyCNN, and config loading."""

from __future__ import annotations

import torch

from trustskin.models import BaselineCNN, UncertaintyCNN, build_model
from trustskin.models.uncertainty_cnn import enable_mc_dropout
from trustskin.utils import load_config


def test_baseline_cnn_forward_shape() -> None:
    model = BaselineCNN(in_channels=3, num_classes=7)
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (4, 7)


def test_uncertainty_cnn_mc_dropout_stochasticity() -> None:
    torch.manual_seed(0)
    model = UncertaintyCNN(in_channels=3, num_classes=7, dropout_p=0.3)
    x = torch.randn(2, 3, 224, 224)

    enable_mc_dropout(model)
    out1 = model(x)
    out2 = model(x)

    assert out1.shape == (2, 7)
    assert not torch.allclose(out1, out2), "MC Dropout passes should be stochastic."


def test_configs_load_and_build_models() -> None:
    base_cfg = load_config("configs/baseline_cnn.yaml")
    imp_cfg = load_config("configs/improved_mc_cnn.yaml")

    assert base_cfg["experiment"]["seed"] == 42
    assert base_cfg["dataset"]["augment"] is False
    assert imp_cfg["dataset"]["augment"] is True

    m1 = build_model(base_cfg["model"])
    m2 = build_model(imp_cfg["model"])
    assert isinstance(m1, BaselineCNN)
    assert isinstance(m2, UncertaintyCNN)
