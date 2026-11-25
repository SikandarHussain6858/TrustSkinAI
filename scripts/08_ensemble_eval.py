"""Step 8: Deep Ensemble vs. MC Dropout vs. Deterministic Softmax Uncertainty Comparison.

Loads M independently trained seed checkpoints (e.g. seeds 42, 43, 44), fits Temperature Scaling
on the validation split for each member, and compares three uncertainty methods on identical test data:
1. Single-model Deterministic Softmax (calibrated)
2. Single-model MC Dropout (T=25, calibrated)
3. Deep Ensemble (average of M calibrated members)

Outputs:
- outputs/tables/uncertainty_method_comparison.csv
- outputs/tables/deep_ensemble_test_predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from trustskin.calibration import TemperatureScaler, compute_brier_score, compute_ece
from trustskin.data import create_dataloaders
from trustskin.evaluation import compute_classification_metrics
from trustskin.models import build_model
from trustskin.selective import compute_aurc
from trustskin.uncertainty import (
    collect_logits_and_targets,
    compute_error_detection_auroc,
    mc_dropout_predict,
    mutual_information,
    predictive_entropy,
)
from trustskin.utils import get_device, load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Deep Ensemble vs MC Dropout vs Deterministic Softmax.")
    parser.add_argument("--config", type=str, required=True, help="Config shared by the ensemble checkpoints.")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="List of 2+ seed checkpoint paths.")
    parser.add_argument("--mc-samples", type=int, default=25)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device = get_device(cfg["experiment"]["device"])
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    tables_dir = outputs_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    n_bins = int(cfg["calibration"]["n_bins"])

    loaders, _ = create_dataloaders(cfg)
    model_cfg = dict(cfg["model"])
    if model_cfg.get("type") == "resnet18_mc":
        model_cfg["pretrained"] = False

    member_test_probs: list[np.ndarray] = []
    member_temps: list[float] = []
    y_true: np.ndarray | None = None
    first_model = None
    first_temp = 1.0

    for idx, ckpt_path in enumerate(args.checkpoints):
        model = build_model(model_cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        val_logits, val_targets = collect_logits_and_targets(model, loaders["val"], device)
        scaler = TemperatureScaler()
        t_val = scaler.fit(
            val_logits,
            val_targets,
            lr=float(cfg["calibration"]["lr"]),
            max_iter=int(cfg["calibration"]["max_iter"]),
        )
        test_logits, test_targets = collect_logits_and_targets(model, loaders["test"], device)
        probs = scaler.predict_probs(test_logits)

        member_test_probs.append(probs)
        member_temps.append(t_val)
        y_true = test_targets.numpy()

        if idx == 0:
            first_model = model
            first_temp = t_val

    assert y_true is not None and first_model is not None

    # 1. Single-model Deterministic Softmax (seed 0, calibrated)
    det_probs = member_test_probs[0]

    # 2. Single-model MC Dropout (seed 0, T=mc_samples, calibrated)
    mc_out = mc_dropout_predict(
        first_model, loaders["test"], device=device, mc_samples=args.mc_samples, temperature=first_temp
    )

    # 3. Deep Ensemble (M calibrated members)
    ens_stack = np.stack(member_test_probs, axis=0)  # (M, N, C)
    ens_probs = np.mean(ens_stack, axis=0)
    ens_mi = mutual_information(ens_stack)

    def _row(method_name: str, probs: np.ndarray, mi_arr: np.ndarray) -> dict:
        preds = np.argmax(probs, axis=-1)
        conf = np.max(probs, axis=-1)
        ent = predictive_entropy(probs)
        cls = compute_classification_metrics(y_true, preds)
        return {
            "method": method_name,
            "accuracy": round(cls["accuracy"], 4),
            "balanced_accuracy": round(cls["balanced_accuracy"], 4),
            "macro_f1": round(cls["macro_f1"], 4),
            "ece": round(compute_ece(probs, y_true, n_bins=n_bins), 4),
            "brier": round(compute_brier_score(probs, y_true), 4),
            "aurc_conf": round(compute_aurc(preds, y_true, conf, True), 4),
            "aurc_ent": round(compute_aurc(preds, y_true, ent, False), 4),
            "error_auroc_conf": round(compute_error_detection_auroc(preds, y_true, 1.0 - conf), 4),
            "error_auroc_ent": round(compute_error_detection_auroc(preds, y_true, ent), 4),
            "error_auroc_mi": round(compute_error_detection_auroc(preds, y_true, mi_arr), 4),
        }

    rows = [
        _row("Deterministic Softmax (1 model, TS)", det_probs, np.zeros(len(y_true))),
        _row(f"MC Dropout (1 model, T={args.mc_samples}, TS)", mc_out["mean_probs"], mc_out["mutual_information"]),
        _row(f"Deep Ensemble ({len(args.checkpoints)} seeds, TS)", ens_probs, ens_mi),
    ]
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(tables_dir / "uncertainty_method_comparison.csv", index=False)

    # Export ensemble per-sample predictions for difficult-case & risk-coverage evaluation
    ens_preds = np.argmax(ens_probs, axis=-1)
    ens_df = pd.DataFrame(
        {
            "sample_idx": np.arange(len(y_true)),
            "true_label": y_true,
            "pred_label": ens_preds,
            "correct": (ens_preds == y_true).astype(int),
            "conf_uncal": np.max(ens_probs, axis=-1),
            "conf_cal": np.max(ens_probs, axis=-1),
            "entropy": predictive_entropy(ens_probs),
            "mutual_information": ens_mi,
        }
    )
    ens_df.to_csv(tables_dir / "deep_ensemble_test_predictions.csv", index=False)

    print("=== Uncertainty Method Comparison ===")
    print(comp_df.to_string(index=False))


if __name__ == "__main__":
    main()
