"""Step 4: Fit Temperature Scaling on Val, run MC Dropout (or deterministic) uncertainty & Selective Classification on Test.

Fixes the validation/test score-matching bug: when MC Dropout is enabled (`mc_samples > 1`),
selective confidence thresholds are selected on MC-averaged calibrated validation confidences
so validation and test confidence distributions match.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from trustskin.calibration import TemperatureScaler, compute_brier_score, compute_ece
from trustskin.data import create_dataloaders
from trustskin.evaluation import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_reliability_diagram,
    plot_risk_coverage_comparison,
    plot_uncertainty_histograms,
)
from trustskin.models import build_model
from trustskin.selective import (
    compute_aurc,
    compute_risk_coverage_curve,
    evaluate_at_threshold,
    select_threshold_for_coverage,
)
from trustskin.uncertainty import (
    collect_logits_and_targets,
    compute_error_detection_auroc,
    mc_dropout_predict,
    predictive_entropy,
)
from trustskin.utils import get_device, load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate model on Val and evaluate uncertainty + selective risk on Test."
    )
    parser.add_argument("--config", type=str, default="configs/improved_mc_cnn.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mc-samples", type=int, default=None, help="Override MC passes (use 1 for deterministic).")
    parser.add_argument("--exp-name", type=str, default=None, help="Optional override for output experiment name.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device = get_device(cfg["experiment"]["device"])
    exp_name = args.exp_name or cfg["experiment"]["name"]
    outputs_dir = Path(cfg["paths"]["outputs_dir"])

    loaders, class_names = create_dataloaders(cfg)
    model_cfg = dict(cfg["model"])
    if model_cfg.get("type") == "resnet18_mc":
        model_cfg["pretrained"] = False
    model = build_model(model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # 1. Fit Temperature Scaling strictly on Validation split
    val_logits, val_targets = collect_logits_and_targets(model, loaders["val"], device)
    scaler = TemperatureScaler()
    learned_t = scaler.fit(
        val_logits,
        val_targets,
        lr=float(cfg["calibration"]["lr"]),
        max_iter=int(cfg["calibration"]["max_iter"]),
    )
    print(f"[{exp_name}] Learned Temperature on Val split: T = {learned_t:.4f}")

    # 2. Deterministic Test evaluation (I1: uncalibrated, I2: + Temperature Scaling)
    test_logits, test_targets = collect_logits_and_targets(model, loaders["test"], device)
    y_true = test_targets.numpy()
    probs_uncal = F.softmax(test_logits, dim=-1).numpy()
    probs_cal = scaler.predict_probs(test_logits)

    # 3. MC Dropout Test evaluation (I3: MC uncalibrated, I4: MC + Temperature Scaling)
    mc_samples = args.mc_samples if args.mc_samples is not None else int(cfg["uncertainty"]["mc_samples"])
    if mc_samples > 1:
        mc_uncal = mc_dropout_predict(
            model, loaders["test"], device=device, mc_samples=mc_samples, temperature=1.0
        )
        mc_out = mc_dropout_predict(
            model, loaders["test"], device=device, mc_samples=mc_samples, temperature=learned_t
        )
        # Match validation score type to test score type (MC-averaged calibrated confidence)
        val_mc = mc_dropout_predict(
            model, loaders["val"], device=device, mc_samples=mc_samples, temperature=learned_t
        )
        val_conf_cal = val_mc["confidence"]
    else:
        mc_uncal = None
        mc_out = {
            "mean_probs": probs_cal,
            "preds": np.argmax(probs_cal, axis=-1),
            "confidence": np.max(probs_cal, axis=-1),
            "entropy": predictive_entropy(probs_cal),
            "mutual_information": np.zeros(len(y_true), dtype=np.float64),
            "variance": np.zeros(len(y_true), dtype=np.float64),
            "targets": y_true,
        }
        val_probs_cal = scaler.predict_probs(val_logits)
        val_conf_cal = np.max(val_probs_cal, axis=-1)

    n_bins = int(cfg["calibration"]["n_bins"])
    cls_metrics = compute_classification_metrics(y_true, mc_out["preds"], class_names=class_names)

    # Select coverage thresholds strictly on Validation matched confidence scores
    coverage_table = []
    for cov in cfg["selective"]["target_coverages"]:
        th = select_threshold_for_coverage(val_conf_cal, cov, higher_is_more_confident=True)
        row = evaluate_at_threshold(
            preds=mc_out["preds"],
            targets=y_true,
            scores=mc_out["confidence"],
            threshold=th,
            higher_is_more_confident=True,
        )
        row["target_coverage"] = cov
        coverage_table.append(row)

    # Inference-stage ablation summary (I1 -> I4 on identical trained weights)
    def _stage_summary(probs_arr: np.ndarray) -> dict[str, float]:
        p = np.argmax(probs_arr, axis=-1)
        c = np.max(probs_arr, axis=-1)
        ent = predictive_entropy(probs_arr)
        m = compute_classification_metrics(y_true, p)
        return {
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1": m["macro_f1"],
            "ece": compute_ece(probs_arr, y_true, n_bins=n_bins),
            "brier": compute_brier_score(probs_arr, y_true),
            "aurc_conf": compute_aurc(p, y_true, c, True),
            "error_auroc_ent": compute_error_detection_auroc(p, y_true, ent),
        }

    inference_ablation = {
        "I1_deterministic_uncalibrated": _stage_summary(probs_uncal),
        "I2_deterministic_temp_scaled": _stage_summary(probs_cal),
    }
    if mc_uncal is not None:
        inference_ablation["I3_mc_uncalibrated"] = _stage_summary(mc_uncal["mean_probs"])
        inference_ablation["I4_mc_temp_scaled"] = _stage_summary(mc_out["mean_probs"])

    results = {
        "experiment": exp_name,
        "learned_temperature": learned_t,
        "mc_samples": mc_samples,
        "classification": cls_metrics,
        "calibration": {
            "ece_uncalibrated": compute_ece(probs_uncal, y_true, n_bins=n_bins),
            "ece_calibrated": compute_ece(probs_cal, y_true, n_bins=n_bins),
            "ece_mc_calibrated": compute_ece(mc_out["mean_probs"], y_true, n_bins=n_bins),
            "brier_uncalibrated": compute_brier_score(probs_uncal, y_true),
            "brier_calibrated": compute_brier_score(probs_cal, y_true),
            "brier_mc_calibrated": compute_brier_score(mc_out["mean_probs"], y_true),
        },
        "uncertainty": {
            "error_auroc_uncal_conf": compute_error_detection_auroc(
                np.argmax(probs_uncal, axis=-1), y_true, 1.0 - np.max(probs_uncal, axis=-1)
            ),
            "error_auroc_mc_conf": compute_error_detection_auroc(
                mc_out["preds"], y_true, 1.0 - mc_out["confidence"]
            ),
            "error_auroc_mc_entropy": compute_error_detection_auroc(
                mc_out["preds"], y_true, mc_out["entropy"]
            ),
            "error_auroc_mc_mi": compute_error_detection_auroc(
                mc_out["preds"], y_true, mc_out["mutual_information"]
            ),
        },
        "selective": {
            "aurc_uncal_conf": compute_aurc(
                np.argmax(probs_uncal, axis=-1), y_true, np.max(probs_uncal, axis=-1), True
            ),
            "aurc_mc_conf": compute_aurc(mc_out["preds"], y_true, mc_out["confidence"], True),
            "aurc_mc_entropy": compute_aurc(mc_out["preds"], y_true, mc_out["entropy"], False),
            "coverage_operating_points": coverage_table,
        },
        "inference_ablation": inference_ablation,
    }

    tables_dir = outputs_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    with (tables_dir / f"{exp_name}_reliability_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Export per-sample test predictions for cross-model plots & difficult-case error analysis
    pred_df = pd.DataFrame(
        {
            "sample_idx": np.arange(len(y_true)),
            "true_label": y_true,
            "pred_label": mc_out["preds"],
            "correct": (mc_out["preds"] == y_true).astype(int),
            "conf_uncal": np.max(probs_uncal, axis=-1),
            "conf_cal": mc_out["confidence"],
            "entropy": mc_out["entropy"],
            "mutual_information": mc_out["mutual_information"],
        }
    )
    pred_df.to_csv(tables_dir / f"{exp_name}_test_predictions.csv", index=False)

    # Save diagnostic plots
    fig_dir = outputs_dir / "figures"
    plot_confusion_matrix(
        np.array(cls_metrics["confusion_matrix"]),
        class_names,
        fig_dir / f"{exp_name}_confusion_matrix.png",
        title=f"{exp_name} — Test Confusion Matrix",
    )
    plot_reliability_diagram(
        probs_uncal,
        y_true,
        fig_dir / f"{exp_name}_reliability_before.png",
        title=f"{exp_name} — Before Temperature Scaling",
        n_bins=n_bins,
    )
    plot_reliability_diagram(
        mc_out["mean_probs"],
        y_true,
        fig_dir / f"{exp_name}_reliability_after.png",
        title=f"{exp_name} — After Temperature Scaling (T={learned_t:.2f}, MC={mc_samples})",
        n_bins=n_bins,
    )
    plot_uncertainty_histograms(
        mc_out["entropy"],
        mc_out["preds"] == y_true,
        fig_dir / f"{exp_name}_entropy_correct_vs_error.png",
    )
    plot_risk_coverage_comparison(
        {
            "Deterministic (Uncalibrated)": compute_risk_coverage_curve(
                np.argmax(probs_uncal, axis=-1), y_true, np.max(probs_uncal, axis=-1), True
            ),
            f"Calibrated (MC={mc_samples}, Conf)": compute_risk_coverage_curve(
                mc_out["preds"], y_true, mc_out["confidence"], True
            ),
            f"Calibrated (MC={mc_samples}, Entropy)": compute_risk_coverage_curve(
                mc_out["preds"], y_true, mc_out["entropy"], False
            ),
        },
        fig_dir / f"{exp_name}_risk_coverage.png",
    )

    print(f"[{exp_name}] Calibration:", json.dumps(results["calibration"], indent=2))
    print(f"[{exp_name}] Error Detection AUROC:", json.dumps(results["uncertainty"], indent=2))
    print(f"[{exp_name}] Matched Coverage Points:", [(round(r['target_coverage'], 2), round(r['coverage'], 3), round(r['selective_accuracy'], 3)) for r in coverage_table])


if __name__ == "__main__":
    main()
