"""Step 3: Train improved uncertainty-aware model (regularization, imbalance handling, MC Dropout)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from trustskin.data import compute_class_weights, create_dataloaders, get_split_labels
from trustskin.models import build_model
from trustskin.training import Trainer, build_criterion
from trustskin.utils import get_device, load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train improved MC Dropout model.")
    parser.add_argument("--config", type=str, default="configs/improved_mc_cnn.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for epochs.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device = get_device(cfg["experiment"]["device"])

    loaders, _ = create_dataloaders(cfg)
    model = build_model(cfg["model"])

    train_cfg = cfg["training"]
    class_weights = None
    if train_cfg["loss"].lower() in {"weighted_ce", "focal"}:
        train_labels = get_split_labels(
            data_root=cfg["paths"]["data_root"],
            dataset_id=cfg["dataset"]["dataset_id"],
            task_name=cfg["dataset"]["task_name"],
            split="train",
        )
        class_weights = compute_class_weights(train_labels, cfg["dataset"]["num_classes"])

    criterion = build_criterion(train_cfg["loss"], class_weights=class_weights)

    opt_name = train_cfg.get("optimizer", "adamw").lower()
    lr = float(train_cfg["lr"])
    wd = float(train_cfg.get("weight_decay", 1e-4))
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    epochs = args.epochs if args.epochs is not None else int(train_cfg["epochs"])
    scheduler = None
    if train_cfg.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    exp_name = cfg["experiment"]["name"]
    ckpt_path = Path(cfg["paths"]["checkpoints_dir"]) / f"{exp_name}_best.pt"
    history_path = Path(cfg["paths"]["outputs_dir"]) / "logs" / f"{exp_name}_history.json"

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        selection_metric=train_cfg.get("selection_metric", "macro_f1"),
        early_stopping_patience=int(train_cfg.get("early_stopping_patience", 0)),
    )

    trainer.fit(
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        epochs=epochs,
        checkpoint_path=ckpt_path,
        history_path=history_path,
    )
    print(f"Training complete. Best checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
