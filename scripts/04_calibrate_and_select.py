"""Step 4: Fit Temperature Scaling on Val, run MC Dropout uncertainty & Selective Classification on Test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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
from trustskin.uncertainty import collect_logits_and_targets, mc_dropout_predict
from trustskin.utils import get_device, load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate model on Val and evaluate uncertainty + selective risk on Test."
    )
    parser.add_argument("--config", type=str, default="configs/improved_mc_cnn.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device = get_device(cfg["experiment"]["device"])
    exp_name = cfg["experiment"]["name"]
    outputs_dir = Path(cfg["paths"]["outputs_dir"])

    loaders, class_names = create_dataloaders(cfg)
    model = build_model(cfg["model"]).to(device)
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
    print(f"Learned Temperature on Val split: T = {learned_t:.4f}")

    # 2. Deterministic Test evaluation (before & after Temperature Scaling)
    test_logits, test_targets = collect_logits_and_targets(model, loaders["test"], device)
    y_true = test_targets.numpy()
    probs_uncal = F.softmax(test_logits, dim=-1).numpy()
    probs_cal = scaler.predict_probs(test_logits)

    # 3. MC Dropout Test evaluation (with learned temperature)
    mc_samples = int(cfg["uncertainty"]["mc_samples"])
    mc_out = mc_dropout_predict(
        model,
        loaders["test"],
        device=device,
        mc_samples=mc_samples,
        temperature=learned_t,
    )

    n_bins = int(cfg["calibration"]["n_bins"])
    cls_metrics = compute_classification_metrics(y_true, mc_out["preds"], class_names=class_names)

    # Select coverage thresholds strictly on Validation calibrated confidence
    val_probs_cal = scaler.predict_probs(val_logits)
    val_conf_cal = np.max(val_probs_cal, axis=-1)
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
        "selective": {
            "aurc_uncal_conf": compute_aurc(
                np.argmax(probs_uncal, axis=-1), y_true, np.max(probs_uncal, axis=-1), True
            ),
            "aurc_mc_conf": compute_aurc(mc_out["preds"], y_true, mc_out["confidence"], True),
            "aurc_mc_entropy": compute_aurc(mc_out["preds"], y_true, mc_out["entropy"], False),
            "coverage_operating_points": coverage_table,
        },
    }

    out_json = outputs_dir / "tables" / f"{exp_name}_reliability_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

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
        fig_dir / f"{exp_name}_reliability_after_mc.png",
        title=f"{exp_name} — After Temperature Scaling + MC Dropout",
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
            "MC Dropout + Temp Scaling (Conf)": compute_risk_coverage_curve(
                mc_out["preds"], y_true, mc_out["confidence"], True
            ),
            "MC Dropout + Temp Scaling (Entropy)": compute_risk_coverage_curve(
                mc_out["preds"], y_true, mc_out["entropy"], False
            ),
        },
        fig_dir / f"{exp_name}_risk_coverage.png",
    )

    print(json.dumps(results["calibration"], indent=2))
    print(json.dumps({k: v for k, v in results["selective"].items() if k != "coverage_operating_points"}, indent=2))


if __name__ == "__main__":
    main()
