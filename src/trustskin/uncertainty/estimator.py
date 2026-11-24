"""Deterministic and Monte Carlo Dropout uncertainty estimation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from trustskin.models.uncertainty_cnn import enable_mc_dropout


def compute_error_detection_auroc(
    preds: np.ndarray,
    targets: np.ndarray,
    uncertainty_scores: np.ndarray,
) -> float:
    """Measure how well an uncertainty score separates misclassifications (1) from correct predictions (0)."""
    errors = (np.asarray(preds) != np.asarray(targets)).astype(np.int64)
    if len(np.unique(errors)) < 2:
        return float("nan")
    return float(roc_auc_score(errors, np.asarray(uncertainty_scores, dtype=np.float64)))


def predictive_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute Shannon predictive entropy H(p) = -sum_c p_c log(p_c) in nats."""
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def mutual_information(mc_probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute epistemic uncertainty (BALD / mutual information) from MC samples.

    Args:
        mc_probs: Array of shape (T, N, C) with T stochastic forward passes.
    """
    mean_probs = np.mean(mc_probs, axis=0)
    total_entropy = predictive_entropy(mean_probs, eps=eps)
    expected_entropy = np.mean(predictive_entropy(mc_probs, eps=eps), axis=0)
    return np.maximum(total_entropy - expected_entropy, 0.0)


@torch.no_grad()
def collect_logits_and_targets(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run deterministic eval forward pass over a DataLoader and return (logits, targets)."""
    model.eval()
    logits_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits_list.append(model(images).cpu())
        targets_list.append(targets.cpu())
    return torch.cat(logits_list, dim=0), torch.cat(targets_list, dim=0)


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mc_samples: int = 20,
    temperature: float = 1.0,
) -> dict[str, np.ndarray]:
    """Run T stochastic MC Dropout passes over a DataLoader.

    Returns mean probabilities, predictions, max confidence, predictive entropy,
    mutual information, predictive variance, and ground-truth targets.
    """
    if mc_samples < 1:
        raise ValueError("mc_samples must be >= 1.")

    enable_mc_dropout(model)
    all_targets: list[np.ndarray] = []
    all_mc_probs: list[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        batch_passes = []
        for _ in range(mc_samples):
            logits = model(images) / float(temperature)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            batch_passes.append(probs)
        # Shape: (T, B, C)
        all_mc_probs.append(np.stack(batch_passes, axis=0))
        all_targets.append(targets.numpy())

    mc_probs = np.concatenate(all_mc_probs, axis=1)  # (T, N, C)
    mean_probs = np.mean(mc_probs, axis=0)  # (N, C)
    preds = np.argmax(mean_probs, axis=-1)
    confidence = np.max(mean_probs, axis=-1)
    entropy = predictive_entropy(mean_probs)
    mi = mutual_information(mc_probs)
    variance = np.mean(np.var(mc_probs, axis=0), axis=-1)
    targets_arr = np.concatenate(all_targets, axis=0)

    return {
        "mean_probs": mean_probs,
        "preds": preds,
        "confidence": confidence,
        "entropy": entropy,
        "mutual_information": mi,
        "variance": variance,
        "targets": targets_arr,
    }
