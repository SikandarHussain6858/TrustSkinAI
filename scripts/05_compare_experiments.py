"""Step 5: Aggregate all saved experiment results into comparison CSV, combined Risk-Coverage plot, and training curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from trustskin.evaluation import plot_risk_coverage_comparison
from trustskin.selective import compute_risk_coverage_curve


def plot_training_histories(logs_dir: Path, save_path: Path) -> None:
    """Plot train/val loss and validation Macro-F1 across all logged experiments."""
    history_files = sorted(logs_dir.glob("*_history.json"))
    if not history_files:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for hfile in history_files:
        label = hfile.stem.replace("_history", "")
        with hfile.open("r", encoding="utf-8") as f:
            hist = json.load(f)
        epochs = [r["epoch"] for r in hist]
        ax1.plot(epochs, [r["val_loss"] for r in hist], marker="o", markersize=3, label=f"{label} (val)")
        ax1.plot(epochs, [r["train_loss"] for r in hist], linestyle="--", alpha=0.5, label=f"{label} (train)")
        ax2.plot(epochs, [r["val_macro_f1"] for r in hist], marker="o", markersize=3, label=label)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Macro F1")
    ax2.set_title("Validation Macro F1 Across Epochs")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile comparison table & combined plots across experiments.")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    logs_dir = out_dir / "logs"

    rows = []
    for json_file in sorted(tables_dir.glob("*.json")):
        if json_file.name in {"eda_summary.json", "difficult_cases_summary.json"}:
            continue
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if "classification" in data and "calibration" in data:
            cls = data["classification"]
            cal = data["calibration"]
            sel = data["selective"]
            unc = data.get("uncertainty", {})
            rows.append(
                {
                    "experiment": data.get("experiment", json_file.stem),
                    "accuracy": round(cls["accuracy"], 4),
                    "balanced_accuracy": round(cls["balanced_accuracy"], 4),
                    "macro_f1": round(cls["macro_f1"], 4),
                    "ece_before": round(cal["ece_uncalibrated"], 4),
                    "ece_after": round(cal["ece_mc_calibrated"], 4),
                    "brier_after": round(cal["brier_mc_calibrated"], 4),
                    "aurc": round(sel["aurc_mc_conf"], 4),
                    "error_auroc_ent": round(unc.get("error_auroc_mc_entropy", float("nan")), 4),
                    "temperature": round(data.get("learned_temperature", 1.0), 4),
                }
            )
        elif "accuracy" in data and "macro_f1" in data:
            rows.append(
                {
                    "experiment": json_file.stem.replace("_test_metrics", ""),
                    "accuracy": round(data["accuracy"], 4),
                    "balanced_accuracy": round(data["balanced_accuracy"], 4),
                    "macro_f1": round(data["macro_f1"], 4),
                    "ece_before": round(data.get("ece", float("nan")), 4),
                    "ece_after": round(data.get("ece", float("nan")), 4),
                    "brier_after": round(data.get("brier_score", float("nan")), 4),
                    "aurc": round(data.get("aurc_confidence", float("nan")), 4),
                    "error_auroc_ent": round(data.get("error_auroc_entropy", float("nan")), 4),
                    "temperature": 1.0,
                }
            )

    if rows:
        df = pd.DataFrame(rows)
        out_csv = tables_dir / "experiment_comparison.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved comparison table to {out_csv}:")
        print(df.to_string(index=False))
    else:
        print(f"No experiment result JSON files found in {tables_dir}.")

    # Combined multi-model Risk-Coverage overlay from per-sample CSVs
    curves = {}
    for pred_csv in sorted(tables_dir.glob("*_test_predictions.csv")):
        label = pred_csv.stem.replace("_test_predictions", "")
        pdf = pd.read_csv(pred_csv)
        curves[label] = compute_risk_coverage_curve(
            preds=pdf["pred_label"].to_numpy(),
            targets=pdf["true_label"].to_numpy(),
            scores=pdf["conf_cal"].to_numpy(),
            higher_is_more_confident=True,
        )
    if curves:
        plot_risk_coverage_comparison(
            curves,
            figures_dir / "all_models_risk_coverage.png",
            title="TrustSkin AI — Controlled Risk–Coverage Comparison Across All Models",
        )
        print(f"Saved combined Risk-Coverage plot to {figures_dir / 'all_models_risk_coverage.png'}")

    plot_training_histories(logs_dir, figures_dir / "training_curves.png")


if __name__ == "__main__":
    main()
