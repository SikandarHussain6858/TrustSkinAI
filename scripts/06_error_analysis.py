"""Step 6: Difficult-Case, Confused-Class, and Malignant-Safety Error Analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from trustskin.data import LocalMedIMetaDataset
from trustskin.utils import load_config

SHORT_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
MALIGNANT_CLASSES = {0, 2}  # Melanoma (0), Basal cell carcinoma (2)


def analyze_difficult_cases(pred_df: pd.DataFrame) -> dict:
    """Compute top confused class pairs, per-class error rates, and malignant referral safety."""
    y_true = pred_df["true_label"].to_numpy()
    y_pred = pred_df["pred_label"].to_numpy()
    conf = pred_df["conf_cal"].to_numpy()
    errors = y_true != y_pred

    # Top confused pairs (true -> pred where true != pred)
    pair_counts: dict[str, int] = {}
    for t, p in zip(y_true[errors], y_pred[errors]):
        key = f"{SHORT_NAMES[int(t)]} -> {SHORT_NAMES[int(p)]}"
        pair_counts[key] = pair_counts.get(key, 0) + 1
    top_confusions = sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]

    # Per-class error rate & mean uncertainty
    per_class = {}
    for cls_idx, name in enumerate(SHORT_NAMES):
        mask = y_true == cls_idx
        if np.any(mask):
            per_class[name] = {
                "count": int(mask.sum()),
                "error_rate": round(float(np.mean(errors[mask])), 4),
                "mean_entropy": round(float(np.mean(pred_df["entropy"].to_numpy()[mask])), 4),
                "mean_confidence": round(float(np.mean(conf[mask])), 4),
            }

    # Malignant referral capture at 80% and 70% coverage
    malignant_mask = np.isin(y_true, list(MALIGNANT_CLASSES))
    malignant_errors = malignant_mask & errors
    safety = {}
    for cov in [0.9, 0.8, 0.7]:
        th = float(np.quantile(conf, 1.0 - cov))
        referred = conf < th
        caught_mal_errors = int(np.sum(malignant_errors & referred))
        total_mal_errors = int(np.sum(malignant_errors))
        safety[f"coverage_{int(cov * 100)}"] = {
            "threshold": round(th, 4),
            "malignant_errors_total": total_mal_errors,
            "malignant_errors_referred": caught_mal_errors,
            "malignant_error_capture_rate": round(
                float(caught_mal_errors / max(total_mal_errors, 1)), 4
            ),
        }

    return {
        "total_test_samples": int(len(y_true)),
        "total_errors": int(errors.sum()),
        "top_confused_pairs": [{"pair": k, "count": v} for k, v in top_confusions],
        "per_class_reliability": per_class,
        "malignant_referral_safety": safety,
    }


def save_case_grid(
    dataset: LocalMedIMetaDataset,
    subset_df: pd.DataFrame,
    save_path: Path,
    title: str,
    max_items: int = 12,
) -> None:
    """Render a 3x4 grid of dermoscopic test images with true/pred labels and uncertainty."""
    rows_df = subset_df.head(max_items)
    n = len(rows_df)
    if n == 0:
        return
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.1, nrows * 3.3))
    axes = np.atleast_2d(axes)

    for i, (_, row) in enumerate(rows_df.iterrows()):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        img_path = dataset.dataset_dir / dataset.rel_paths[int(row["sample_idx"])]
        with Image.open(img_path) as img:
            ax.imshow(img.convert("RGB"))
        t_lbl = SHORT_NAMES[int(row["true_label"])]
        p_lbl = SHORT_NAMES[int(row["pred_label"])]
        status = "OK" if int(row["correct"]) == 1 else "ERR"
        ax.set_title(
            f"True: {t_lbl} | Pred: {p_lbl} ({status})\n"
            f"Conf: {row['conf_cal']:.2f} | Ent: {row['entropy']:.2f}",
            fontsize=8.5,
            color="green" if status == "OK" else "crimson",
        )
        ax.axis("off")

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run difficult-case & failure analysis on test predictions.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--predictions", type=str, required=True, help="Path to *_test_predictions.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pred_path = Path(args.predictions)
    exp_name = pred_path.stem.replace("_test_predictions", "")
    outputs_dir = Path(cfg["paths"]["outputs_dir"])

    pred_df = pd.read_csv(pred_path)
    summary = analyze_difficult_cases(pred_df)
    summary["experiment"] = exp_name

    out_json = outputs_dir / "tables" / f"{exp_name}_difficult_cases.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    dataset_dir = Path(cfg["paths"]["data_root"]) / cfg["dataset"]["dataset_id"]
    if dataset_dir.exists():
        test_ds = LocalMedIMetaDataset(
            dataset_dir, task_name=cfg["dataset"]["task_name"], split="test"
        )
        fig_dir = outputs_dir / "figures"
        # 1. Top-12 highest-uncertainty test lesions
        high_unc = pred_df.sort_values("entropy", ascending=False)
        save_case_grid(
            test_ds,
            high_unc,
            fig_dir / f"{exp_name}_highest_uncertainty_grid.png",
            title=f"{exp_name} — Top 12 Highest-Uncertainty Test Lesions",
        )
        # 2. Top-12 most confident misclassifications (failure mode analysis)
        conf_errs = pred_df[pred_df["correct"] == 0].sort_values("conf_cal", ascending=False)
        save_case_grid(
            test_ds,
            conf_errs,
            fig_dir / f"{exp_name}_confident_errors_grid.png",
            title=f"{exp_name} — Top 12 Most Confident Misclassifications (Failure Modes)",
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
