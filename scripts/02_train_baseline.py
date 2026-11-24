"""Step 2: Train and evaluate the CS320 Common Suggested Reference Baseline CNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from trustskin.calibration import compute_brier_score, compute_ece
from trustskin.data import create_dataloaders
from trustskin.evaluation import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_reliability_diagram,
)
from trustskin.models import build_model
from trustskin.selective import compute_aurc
from trustskin.training import Trainer, build_criterion
from trustskin.uncertainty import (
    collect_logits_and_targets,
    compute_error_detection_auroc,
    predictive_entropy,
)
from trustskin.utils import get_device, load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CS320 Baseline CNN.")
    parser.add_argument("--config", type=str, default="configs/baseline_cnn.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for epochs.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device = get_device(cfg["experiment"]["device"])

    loaders, class_names = create_dataloaders(cfg)
    model = build_model(cfg["model"])
    criterion = build_criterion(cfg["training"]["loss"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )

    exp_name = cfg["experiment"]["name"]
    ckpt_path = Path(cfg["paths"]["checkpoints_dir"]) / f"{exp_name}_best.pt"
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    history_path = outputs_dir / "logs" / f"{exp_name}_history.json"

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        selection_metric=cfg["training"].get("selection_metric", "val_loss"),
        early_stopping_patience=int(cfg["training"].get("early_stopping_patience", 0)),
    )

    epochs = args.epochs if args.epochs is not None else int(cfg["training"]["epochs"])
    trainer.fit(
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        epochs=epochs,
        checkpoint_path=ckpt_path,
        history_path=history_path,
    )

    # Evaluate best checkpoint once on strictly held-out test split
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_logits, test_targets = collect_logits_and_targets(model, loaders["test"], device)
    test_probs = F.softmax(test_logits, dim=-1).numpy()
    y_true = test_targets.numpy()
    y_pred = np.argmax(test_probs, axis=-1)
    confidences = np.max(test_probs, axis=-1)
    entropies = predictive_entropy(test_probs)

    cls_metrics = compute_classification_metrics(y_true, y_pred, class_names=class_names)
    cls_metrics["ece"] = compute_ece(test_probs, y_true, n_bins=cfg["calibration"]["n_bins"])
    cls_metrics["brier_score"] = compute_brier_score(test_probs, y_true)
    cls_metrics["mean_entropy"] = float(np.mean(entropies))
    cls_metrics["aurc_confidence"] = compute_aurc(y_pred, y_true, confidences, True)
    cls_metrics["error_auroc_confidence"] = compute_error_detection_auroc(y_pred, y_true, 1.0 - confidences)
    cls_metrics["error_auroc_entropy"] = compute_error_detection_auroc(y_pred, y_true, entropies)

    tables_dir = outputs_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    with (tables_dir / f"{exp_name}_test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(cls_metrics, f, indent=2)

    # Export per-sample test predictions for cross-model comparison & error analysis
    pred_df = pd.DataFrame(
        {
            "sample_idx": np.arange(len(y_true)),
            "true_label": y_true,
            "pred_label": y_pred,
            "correct": (y_pred == y_true).astype(int),
            "conf_uncal": confidences,
            "conf_cal": confidences,
            "entropy": entropies,
            "mutual_information": np.zeros_like(entropies),
        }
    )
    pred_df.to_csv(tables_dir / f"{exp_name}_test_predictions.csv", index=False)

    plot_confusion_matrix(
        np.array(cls_metrics["confusion_matrix"]),
        class_names,
        outputs_dir / "figures" / f"{exp_name}_confusion_matrix.png",
        title="Baseline CNN — Test Confusion Matrix",
    )
    plot_reliability_diagram(
        test_probs,
        y_true,
        outputs_dir / "figures" / f"{exp_name}_reliability.png",
        title="Baseline CNN — Uncalibrated Reliability Diagram",
        n_bins=cfg["calibration"]["n_bins"],
    )

    print(
        f"Baseline Test Results | Acc: {cls_metrics['accuracy']:.4f} | "
        f"Balanced Acc: {cls_metrics['balanced_accuracy']:.4f} | "
        f"Macro F1: {cls_metrics['macro_f1']:.4f} | "
        f"ECE: {cls_metrics['ece']:.4f} | AURC: {cls_metrics['aurc_confidence']:.4f} | "
        f"Err-AUROC(Ent): {cls_metrics['error_auroc_entropy']:.4f}"
    )


if __name__ == "__main__":
    main()
