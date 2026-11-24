"""CS320 Common Suggested Baseline CNN trained from random initialization.

Architecture (exact specification from CS320_Fall2026_Term_Project_Final.pdf):
- Input: 224 x 224 x 3
- Block 1: Conv 3x3 (32 filters, padding 1) -> ReLU -> 2x2 MaxPool
- Block 2: Conv 3x3 (64 filters, padding 1) -> ReLU -> 2x2 MaxPool
- Block 3: Conv 3x3 (128 filters, padding 1) -> ReLU -> 2x2 MaxPool
- Classifier: Global Average Pooling -> Linear(128, 7)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """Reference 3-block CNN baseline for 7-class dermoscopic classification."""

    def __init__(self, in_channels: int = 3, num_classes: int = 7) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
