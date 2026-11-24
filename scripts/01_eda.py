"""Step 1: Exploratory Data Analysis for MedIMeta Dermatoscopy (CS320 P08)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trustskin.data import load_medimeta_splits
from trustskin.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA on MedIMeta Dermatoscopy splits.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    (outputs_dir / "tables").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "figures").mkdir(parents=True, exist_ok=True)

    splits, class_names = load_medimeta_splits(
        data_root=cfg["paths"]["data_root"],
        dataset_id=ds_cfg["dataset_id"],
        task_name=ds_cfg["task_name"],
        image_size=ds_cfg["image_size"],
        augment_train=False,
    )

    summary_rows = []
    split_counts: dict[str, np.ndarray] = {}

    for split_name, ds in splits.items():
        labels = np.array([int(ds[i][1]) for i in range(len(ds))], dtype=np.int64)
        counts = np.bincount(labels, minlength=len(class_names))
        split_counts[split_name] = counts
        for cls_idx, cls_name in enumerate(class_names):
            summary_rows.append(
                {
                    "split": split_name,
                    "class_idx": cls_idx,
                    "class_name": cls_name,
                    "count": int(counts[cls_idx]),
                    "fraction": float(counts[cls_idx] / max(len(labels), 1)),
                }
            )

    df = pd.DataFrame(summary_rows)
    df.to_csv(outputs_dir / "tables" / "eda_class_distribution.csv", index=False)

    sample_img, sample_lbl = splits["train"][0]
    meta = {
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "total_images": sum(len(v) for v in splits.values()),
        "num_classes": len(class_names),
        "class_names": class_names,
        "sample_tensor_shape": list(sample_img.shape),
        "sample_dtype": str(sample_img.dtype),
        "train_imbalance_ratio": float(
            split_counts["train"].max() / max(split_counts["train"].min(), 1)
        ),
    }
    with (outputs_dir / "tables" / "eda_summary.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Bar chart of class counts across splits
    x = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, split_name in enumerate(["train", "val", "test"]):
        ax.bar(x + i * width, split_counts[split_name], width=width, label=split_name)
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=25, ha="right")
    ax.set_ylabel("Number of Images")
    ax.set_title("MedIMeta Dermatoscopy — Class Distribution Across Predefined Splits")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(outputs_dir / "figures" / "eda_class_distribution.png", dpi=200)
    plt.close(fig)

    print("EDA complete. Summary:")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
