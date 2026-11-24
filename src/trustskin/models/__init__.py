"""Model factory and architectures for TrustSkin AI."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .baseline_cnn import BaselineCNN
from .uncertainty_cnn import ResNet18MC, UncertaintyCNN


def build_model(model_cfg: dict[str, Any]) -> nn.Module:
    """Instantiate a model from config."""
    model_type = model_cfg.get("type", "baseline_cnn").lower()
    in_channels = int(model_cfg.get("in_channels", 3))
    num_classes = int(model_cfg.get("num_classes", 7))

    if model_type == "baseline_cnn":
        return BaselineCNN(in_channels=in_channels, num_classes=num_classes)
    if model_type == "uncertainty_cnn":
        return UncertaintyCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout_p=float(model_cfg.get("dropout_p", 0.25)),
        )
    if model_type == "resnet18_mc":
        return ResNet18MC(
            num_classes=num_classes,
            dropout_p=float(model_cfg.get("dropout_p", 0.3)),
            pretrained=bool(model_cfg.get("pretrained", True)),
        )
    raise ValueError(f"Unknown model type: {model_type}")


__all__ = ["BaselineCNN", "UncertaintyCNN", "ResNet18MC", "build_model"]
