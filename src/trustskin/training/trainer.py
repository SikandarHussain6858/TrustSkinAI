"""Reproducible training and validation loop for TrustSkin AI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from trustskin.evaluation.metrics import compute_classification_metrics


class FocalLoss(nn.Module):
    """Multi-class Focal Loss for hard/imbalanced skin lesion classes."""

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def build_criterion(
    loss_name: str,
    class_weights: torch.Tensor | None = None,
) -> nn.Module:
    """Create loss criterion from config name."""
    name = loss_name.lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    if name == "focal":
        return FocalLoss(weight=class_weights, gamma=2.0)
    raise ValueError(f"Unsupported loss function: {loss_name}")


class Trainer:
    """Handles model training, validation selection, early stopping, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        selection_metric: str = "macro_f1",
        early_stopping_patience: int = 0,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.selection_metric = selection_metric
        self.early_stopping_patience = early_stopping_patience

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, Any]:
        self.model.train(mode=train)
        total_loss = 0.0
        total_samples = 0
        all_targets: list[np.ndarray] = []
        all_preds: list[np.ndarray] = []

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for images, targets in tqdm(loader, leave=False, desc="Train" if train else "Val"):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                if train:
                    self.optimizer.zero_grad()

                logits = self.model(images)
                loss = self.criterion(logits, targets)

                if train:
                    loss.backward()
                    self.optimizer.step()

                batch_size = targets.size(0)
                total_loss += float(loss.item()) * batch_size
                total_samples += batch_size
                all_targets.append(targets.detach().cpu().numpy())
                all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())

        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)
        metrics = compute_classification_metrics(y_true, y_pred)
        metrics["loss"] = total_loss / max(total_samples, 1)
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        checkpoint_path: str | Path,
        history_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Train model for `epochs`, saving the best checkpoint based on validation metric."""
        ckpt_path = Path(checkpoint_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        best_score = -float("inf") if self.selection_metric != "val_loss" else float("inf")
        epochs_no_improve = 0
        history: list[dict[str, Any]] = []

        for epoch in range(1, epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False)

            if self.scheduler is not None:
                self.scheduler.step()

            record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            }
            history.append(record)

            current = (
                val_metrics["loss"]
                if self.selection_metric == "val_loss"
                else val_metrics[self.selection_metric]
            )
            improved = (
                current < best_score
                if self.selection_metric == "val_loss"
                else current > best_score
            )

            if improved:
                best_score = current
                epochs_no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "val_metrics": val_metrics,
                    },
                    ckpt_path,
                )
            else:
                epochs_no_improve += 1

            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['macro_f1']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                f"Macro-F1: {val_metrics['macro_f1']:.4f}"
            )

            if (
                self.early_stopping_patience > 0
                and epochs_no_improve >= self.early_stopping_patience
            ):
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        if history_path is not None:
            h_path = Path(history_path)
            h_path.parent.mkdir(parents=True, exist_ok=True)
            with h_path.open("w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        return history
