"""Step 1: Exploratory Data Analysis for MedIMeta Dermatoscopy (CS320 P08).

Generates:
- outputs/tables/eda_class_distribution.csv
- outputs/tables/eda_summary.json
- outputs/figures/eda_class_distribution.png (linear + log scale)
- outputs/figures/eda_sample_grid.png (7 classes x 3 real dermoscopic samples)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from trustskin.data import LocalMedIMetaDataset, get_split_labels
from trustskin.utils import load_config


SHORT_CLASS_NAMES = [
    "MEL (Melanoma)",
    "NV (Nevus)",
    "BCC (Basal Cell)",
    "AKIEC (Actinic Ker.)",
    "BKL (Benign Ker.)",
    "DF (Dermatofibroma)",
    "VASC (Vascular)",
]


def compute_train_channel_stats(
    dataset: LocalMedIMetaDataset,
    max_samples: int = 2000,
    seed: int = 42,
) -> tuple[list[float], list[float]]:
    """Estimate channel-wise RGB mean/std in [0, 1] strictly from the training split."""
    rng = np.random.default_rng(seed)
    n = len(dataset)
    chosen = rng.choice(n, size=min(n, max_samples), replace=False)

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for idx in chosen:
        img_path = dataset.dataset_dir / dataset.rel_paths[int(idx)]
        with Image.open(img_path) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
        channel_sum += arr.sum(axis=(0, 1))
        channel_sq_sum += (arr ** 2).sum(axis=(0, 1))
        pixel_count += arr.shape[0] * arr.shape[1]

    mean = channel_sum / pixel_count
    std = np.sqrt(np.maximum(channel_sq_sum / pixel_count - mean ** 2, 1e-12))
    return [round(float(m), 4) for m in mean], [round(float(s), 4) for s in std]


def plot_sample_grid(
    dataset: LocalMedIMetaDataset,
    short_names: list[str],
    save_path: Path,
    samples_per_class: int = 3,
) -> None:
    """Save a 7x3 grid of representative dermoscopic images per class."""
    num_classes = len(short_names)
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(samples_per_class * 2.6, num_classes * 2.1))

    for cls_idx in range(num_classes):
        cls_indices = np.where(dataset.labels == cls_idx)[0][:samples_per_class]
        for col in range(samples_per_class):
            ax = axes[cls_idx, col]
            if col < len(cls_indices):
                img_path = dataset.dataset_dir / dataset.rel_paths[int(cls_indices[col])]
                with Image.open(img_path) as img:
                    ax.imshow(img.convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(short_names[cls_idx], fontsize=9, fontweight="bold", rotation=0, labelpad=65, va="center")

    fig.suptitle("MedIMeta Dermatoscopy — Representative Training Images per Class", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0.08, 0.0, 1.0, 0.97])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA on MedIMeta Dermatoscopy splits.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]
    data_root = Path(cfg["paths"]["data_root"])
    dataset_id = ds_cfg["dataset_id"]
    task_name = ds_cfg["task_name"]
    dataset_dir = data_root / dataset_id

    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    (outputs_dir / "tables").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "figures").mkdir(parents=True, exist_ok=True)

    train_ds = LocalMedIMetaDataset(dataset_dir, task_name=task_name, split="train")
    val_ds = LocalMedIMetaDataset(dataset_dir, task_name=task_name, split="val")
    test_ds = LocalMedIMetaDataset(dataset_dir, task_name=task_name, split="test")
    class_names = train_ds.class_names
    num_classes = len(class_names)

    split_datasets = {"train": train_ds, "val": val_ds, "test": test_ds}
    split_counts: dict[str, np.ndarray] = {}
    summary_rows = []

    for split_name, ds in split_datasets.items():
        labels = get_split_labels(data_root, dataset_id=dataset_id, task_name=task_name, split=split_name)
        counts = np.bincount(labels, minlength=num_classes)
        split_counts[split_name] = counts
        for cls_idx, cls_name in enumerate(class_names):
            summary_rows.append(
                {
                    "split": split_name,
                    "class_idx": cls_idx,
                    "short_name": SHORT_CLASS_NAMES[cls_idx] if cls_idx < len(SHORT_CLASS_NAMES) else cls_name,
                    "class_name": cls_name,
                    "count": int(counts[cls_idx]),
                    "percent": round(float(100.0 * counts[cls_idx] / max(len(labels), 1)), 2),
                }
            )

    df = pd.DataFrame(summary_rows)
    df.to_csv(outputs_dir / "tables" / "eda_class_distribution.csv", index=False)

    # Check split leakage & metadata integrity
    split_sets = {name: set(ds.rel_paths) for name, ds in split_datasets.items()}
    overlaps = {
        "train_val": len(split_sets["train"] & split_sets["val"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "val_test": len(split_sets["val"] & split_sets["test"]),
    }

    annotations_path = dataset_dir / "annotations.csv"
    null_count = 0
    unique_sources = None
    if annotations_path.exists():
        ann_df = pd.read_csv(annotations_path)
        null_count = int(ann_df.isnull().sum().sum())
        if "original_filepath" in ann_df.columns:
            unique_sources = int(ann_df["original_filepath"].nunique())

    train_mean, train_std = compute_train_channel_stats(train_ds, max_samples=2000, seed=cfg["experiment"]["seed"])

    sample_img, _ = train_ds[0]
    sample_arr = np.asarray(sample_img)

    meta = {
        "dataset_id": dataset_id,
        "task_name": task_name,
        "split_sizes": {k: len(v) for k, v in split_datasets.items()},
        "total_images": sum(len(v) for v in split_datasets.values()),
        "num_classes": num_classes,
        "class_names": class_names,
        "image_shape_hwc": list(sample_arr.shape),
        "image_dtype": str(sample_arr.dtype),
        "split_overlaps": overlaps,
        "annotations_null_cells": null_count,
        "unique_original_filepaths": unique_sources,
        "train_imbalance_ratio": round(float(split_counts["train"].max() / max(split_counts["train"].min(), 1)), 2),
        "val_min_class_count": int(split_counts["val"].min()),
        "train_rgb_mean_estimated": train_mean,
        "train_rgb_std_estimated": train_std,
    }
    with (outputs_dir / "tables" / "eda_summary.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Dual-panel class distribution chart (Linear + Log scale)
    x = np.arange(num_classes)
    width = 0.26
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, split_name in enumerate(["train", "val", "test"]):
        ax1.bar(x + i * width, split_counts[split_name], width=width, label=split_name)
        ax2.bar(x + i * width, split_counts[split_name], width=width, label=split_name)

    for ax, title, log_scale in [
        (ax1, "Class Distribution (Linear Scale)", False),
        (ax2, "Class Distribution (Log Scale — Highlights Minority Classes)", True),
    ]:
        ax.set_xticks(x + width)
        ax.set_xticklabels(SHORT_CLASS_NAMES[:num_classes], rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Number of Images")
        ax.set_title(title)
        if log_scale:
            ax.set_yscale("log")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(outputs_dir / "figures" / "eda_class_distribution.png", dpi=200)
    plt.close(fig)

    # Representative lesion grid
    plot_sample_grid(
        train_ds,
        SHORT_CLASS_NAMES[:num_classes],
        outputs_dir / "figures" / "eda_sample_grid.png",
        samples_per_class=3,
    )

    print("EDA complete! Summary:")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
