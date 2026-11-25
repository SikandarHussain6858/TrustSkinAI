"""Step 7: Formal Difficult-Case Criteria, Bootstrap 95% CIs, and Paired McNemar Tests (CS320 P08).

Evaluates every saved `*_test_predictions.csv` on three pre-defined difficult-case subsets:
1. `hard_6_non_nv`: All 6 non-majority classes (MEL, BCC, AKIEC, BKL, DF, VASC).
2. `confusable_triad`: Visually overlapping pigmented lesions (MEL, NV, BKL).
3. `baseline_failures`: The test samples misclassified by the CS320 Baseline CNN (analysis only).

Outputs:
- outputs/tables/difficult_case_metrics.csv
- outputs/tables/bootstrap_ci.csv
- outputs/tables/mcnemar_tests.csv
- outputs/figures/difficult_case_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from trustskin.selective import compute_aurc


def exact_mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float | int]:
    """Exact two-sided McNemar test comparing paired binary correctness of model A vs model B."""
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    n01 = int(np.sum((~a) & b))  # A wrong, B right (fixed by B)
    n10 = int(np.sum(a & (~b)))  # A right, B wrong (regressed by B)
    discordant = n01 + n10
    p_value = 1.0 if discordant == 0 else float(binomtest(n01, discordant, 0.5, alternative="two-sided").pvalue)
    return {
        "n_fixed_by_model": n01,
        "n_regressed_by_model": n10,
        "net_gain": n01 - n10,
        "p_value": p_value,
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    conf: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, tuple[float, float, float]]:
    """Compute point estimates and 95% percentile bootstrap CIs for Acc, Bal Acc, Macro F1, AURC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs, bals, f1s, aurcs = [], [], [], []

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt, yp, cf = y_true[idx], y_pred[idx], conf[idx]
        accs.append(float(accuracy_score(yt, yp)))
        bals.append(float(balanced_accuracy_score(yt, yp)))
        f1s.append(float(f1_score(yt, yp, average="macro", zero_division=0)))
        aurcs.append(float(compute_aurc(yp, yt, cf, higher_is_more_confident=True)))

    def _summarize(point: float, samples: list[float]) -> tuple[float, float, float]:
        lo, hi = np.percentile(samples, [2.5, 97.5])
        return round(point, 4), round(float(lo), 4), round(float(hi), 4)

    return {
        "accuracy": _summarize(float(accuracy_score(y_true, y_pred)), accs),
        "balanced_accuracy": _summarize(float(balanced_accuracy_score(y_true, y_pred)), bals),
        "macro_f1": _summarize(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), f1s),
        "aurc": _summarize(float(compute_aurc(y_pred, y_true, conf, True)), aurcs),
    }


def subset_metrics(y_true: np.ndarray, y_pred: np.ndarray, conf: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    """Compute classification and selective-risk metrics on a boolean subset mask."""
    yt, yp, cf = y_true[mask], y_pred[mask], conf[mask]
    n = int(len(yt))
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "balanced_accuracy": 0.0, "macro_f1": 0.0, "mel_recall": 0.0, "aurc": 0.0}

    mel_mask = yt == 0
    mel_recall = float(np.mean(yp[mel_mask] == 0)) if np.any(mel_mask) else float("nan")
    return {
        "n": n,
        "accuracy": round(float(accuracy_score(yt, yp)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(yt, yp)), 4),
        "macro_f1": round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
        "mel_recall": round(mel_recall, 4) if not np.isnan(mel_recall) else float("nan"),
        "aurc": round(float(compute_aurc(yp, yt, cf, higher_is_more_confident=True)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run difficult-case subset evaluation, bootstrap CIs, and McNemar tests.")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"

    pred_files = sorted(tables_dir.glob("*_test_predictions.csv"))
    if not pred_files:
        print(f"No *_test_predictions.csv files found in {tables_dir}.")
        return

    preds_by_model: dict[str, pd.DataFrame] = {}
    for pf in pred_files:
        name = pf.stem.replace("_test_predictions", "")
        preds_by_model[name] = pd.read_csv(pf)

    # Define difficult subsets strictly from ground-truth classes + baseline failure analysis subset
    ref_df = preds_by_model.get("baseline_cnn", next(iter(preds_by_model.values())))
    y_true_ref = ref_df["true_label"].to_numpy()
    subsets: dict[str, np.ndarray] = {
        "all_test": np.ones(len(y_true_ref), dtype=bool),
        "hard_6_non_nv": y_true_ref != 1,
        "confusable_triad_mel_nv_bkl": np.isin(y_true_ref, [0, 1, 4]),
    }
    if "baseline_cnn" in preds_by_model:
        subsets["baseline_failures"] = preds_by_model["baseline_cnn"]["correct"].to_numpy() == 0

    # 1. Difficult-case subset table
    diff_rows = []
    for model_name, df in preds_by_model.items():
        yt = df["true_label"].to_numpy()
        yp = df["pred_label"].to_numpy()
        cf = df["conf_cal"].to_numpy()
        for sub_name, mask in subsets.items():
            row = {"model": model_name, "subset": sub_name}
            row.update(subset_metrics(yt, yp, cf, mask))
            diff_rows.append(row)

    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(tables_dir / "difficult_case_metrics.csv", index=False)

    # 2. Bootstrap 95% CIs on full test set
    boot_rows = []
    for model_name, df in preds_by_model.items():
        cis = bootstrap_metrics(
            df["true_label"].to_numpy(),
            df["pred_label"].to_numpy(),
            df["conf_cal"].to_numpy(),
            n_resamples=args.n_boot,
            seed=args.seed,
        )
        for metric_name, (pt, lo, hi) in cis.items():
            boot_rows.append(
                {
                    "model": model_name,
                    "metric": metric_name,
                    "point": pt,
                    "ci_95_low": lo,
                    "ci_95_high": hi,
                }
            )
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(tables_dir / "bootstrap_ci.csv", index=False)

    # 3. Exact paired McNemar tests vs baseline_cnn
    mcnemar_rows = []
    if "baseline_cnn" in preds_by_model:
        base_correct = preds_by_model["baseline_cnn"]["correct"].to_numpy()
        for model_name, df in preds_by_model.items():
            if model_name == "baseline_cnn":
                continue
            mod_correct = df["correct"].to_numpy()
            for sub_name, mask in subsets.items():
                res = exact_mcnemar_test(base_correct[mask], mod_correct[mask])
                mcnemar_rows.append({"comparison": f"{model_name} vs baseline_cnn", "subset": sub_name, **res})
    mcnemar_df = pd.DataFrame(mcnemar_rows)
    mcnemar_df.to_csv(tables_dir / "mcnemar_tests.csv", index=False)

    # 4. Grouped bar chart of accuracy across difficult subsets
    pivot = diff_df.pivot(index="subset", columns="model", values="accuracy")
    ax = pivot.plot(kind="bar", figsize=(10.5, 5.2), rot=15)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("TrustSkin AI — Performance Across Formal Difficult-Case Subsets")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / "difficult_case_comparison.png", dpi=200)
    plt.close()

    print("=== Difficult-Case Subset Metrics ===")
    print(diff_df.to_string(index=False))
    print("\n=== Paired McNemar Tests vs baseline_cnn ===")
    print(mcnemar_df.to_string(index=False))


if __name__ == "__main__":
    main()
