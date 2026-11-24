"""Post-hoc Temperature Scaling (Guo et al., 2017) and calibration metrics (ECE, Brier Score)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """Learns a single positive scalar temperature T on validation logits via NLL minimization."""

    def __init__(self, init_temperature: float = 1.0) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.tensor([float(np.log(max(init_temperature, 1e-4)))], dtype=torch.float32)
        )

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temperature).item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.exp(self.log_temperature).clamp(min=1e-4, max=100.0)
        return logits / temp

    def fit(
        self,
        val_logits: torch.Tensor,
        val_targets: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """Fit temperature strictly on validation split logits and targets."""
        logits = val_logits.detach().float()
        targets = val_targets.detach().long()
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter)

        def _closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), targets)
            loss.backward()
            return loss

        optimizer.step(_closure)
        return self.temperature

    @torch.no_grad()
    def predict_probs(self, logits: torch.Tensor) -> np.ndarray:
        """Apply learned temperature scaling and return calibrated softmax probabilities."""
        scaled = self.forward(logits.float())
        return F.softmax(scaled, dim=-1).cpu().numpy()


def reliability_bins(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> dict[str, np.ndarray]:
    """Compute per-bin confidence, accuracy, and sample counts for reliability diagrams."""
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == targets).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = np.zeros(n_bins, dtype=np.float64)
    bin_confs = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == 0:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences > low) & (confidences <= high)
        count = int(np.sum(mask))
        bin_counts[i] = count
        if count > 0:
            bin_accs[i] = float(np.mean(accuracies[mask]))
            bin_confs[i] = float(np.mean(confidences[mask]))

    return {
        "bin_edges": bin_edges,
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
    }


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE) over equal-width confidence bins."""
    bins = reliability_bins(probs, targets, n_bins=n_bins)
    total = int(np.sum(bins["bin_counts"]))
    if total == 0:
        return 0.0
    weights = bins["bin_counts"] / float(total)
    return float(np.sum(weights * np.abs(bins["bin_accuracies"] - bins["bin_confidences"])))


def compute_brier_score(probs: np.ndarray, targets: np.ndarray) -> float:
    """Compute multi-class Brier Score: mean squared error between probabilities and one-hot targets."""
    num_classes = probs.shape[-1]
    one_hot = np.eye(num_classes, dtype=np.float64)[targets.astype(np.int64)]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))
