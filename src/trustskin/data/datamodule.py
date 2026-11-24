"""MedIMeta Dermatoscopy data loading, transforms, and class weight computation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Standard 7-class Dermatoscopy (HAM10000 / MedIMeta `derm`) class names (fallback)
DEFAULT_CLASS_NAMES = [
    "Melanoma (MEL)",
    "Melanocytic Nevus (NV)",
    "Basal Cell Carcinoma (BCC)",
    "Actinic Keratosis (AKIEC)",
    "Benign Keratosis (BKL)",
    "Dermatofibroma (DF)",
    "Vascular Lesion (VASC)",
]


def build_transforms(
    image_size: int = 224,
    mean: list[float] | tuple[float, ...] = (0.485, 0.456, 0.406),
    std: list[float] | tuple[float, ...] = (0.229, 0.224, 0.225),
    augment: bool = False,
) -> transforms.Compose:
    """Build deterministic or augmented preprocessing transforms for 224x224 RGB dermoscopy."""
    ops: list[Any] = [transforms.Resize((image_size, image_size))]
    if augment:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )
    return transforms.Compose(ops)


class _LabelCastWrapper(Dataset):
    """Ensure labels returned by MedIMeta are 1D torch.long class indices."""

    def __init__(self, base_dataset: Dataset) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.base_dataset[idx]
        if isinstance(label, torch.Tensor):
            label = label.squeeze().long()
        else:
            label = torch.tensor(int(np.asarray(label).squeeze()), dtype=torch.long)
        return image, label


def load_medimeta_splits(
    data_root: str | Path,
    dataset_id: str = "derm",
    task_name: str = "Disease",
    image_size: int = 224,
    mean: list[float] | tuple[float, ...] = (0.485, 0.456, 0.406),
    std: list[float] | tuple[float, ...] = (0.229, 0.224, 0.225),
    augment_train: bool = False,
) -> tuple[dict[str, Dataset], list[str]]:
    """Load predefined train, val, and test splits from MedIMeta.

    Test and validation splits never receive data augmentation.
    """
    try:
        from medimeta import MedIMeta
    except ImportError as exc:
        raise ImportError(
            "The `medimeta` package is required to load the MedIMeta dataset. "
            "Install it with `pip install medimeta`."
        ) from exc

    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Data directory not found: {root}. Place the downloaded MedIMeta "
            f"`{dataset_id}` dataset inside `{root}`."
        )

    train_tf = build_transforms(image_size=image_size, mean=mean, std=std, augment=augment_train)
    eval_tf = build_transforms(image_size=image_size, mean=mean, std=std, augment=False)

    splits: dict[str, Dataset] = {}
    raw_train = MedIMeta(
        data_path=str(root),
        dataset_id=dataset_id,
        task_name=task_name,
        split="train",
        transform=train_tf,
    )
    raw_val = MedIMeta(
        data_path=str(root),
        dataset_id=dataset_id,
        task_name=task_name,
        split="val",
        transform=eval_tf,
    )
    raw_test = MedIMeta(
        data_path=str(root),
        dataset_id=dataset_id,
        task_name=task_name,
        split="test",
        transform=eval_tf,
    )

    splits["train"] = _LabelCastWrapper(raw_train)
    splits["val"] = _LabelCastWrapper(raw_val)
    splits["test"] = _LabelCastWrapper(raw_test)

    class_names = list(getattr(raw_train, "class_names", DEFAULT_CLASS_NAMES))
    return splits, class_names


def compute_class_weights(labels: np.ndarray | list[int], num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency balanced class weights from training labels only."""
    arr = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(arr, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = len(arr) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(
    cfg: dict[str, Any],
) -> tuple[dict[str, DataLoader], list[str]]:
    """Create train, val, and test DataLoaders from a project config dict."""
    ds_cfg = cfg["dataset"]
    splits, class_names = load_medimeta_splits(
        data_root=cfg["paths"]["data_root"],
        dataset_id=ds_cfg.get("dataset_id", "derm"),
        task_name=ds_cfg.get("task_name", "Disease"),
        image_size=ds_cfg.get("image_size", 224),
        mean=ds_cfg.get("normalize_mean", [0.485, 0.456, 0.406]),
        std=ds_cfg.get("normalize_std", [0.229, 0.224, 0.225]),
        augment_train=bool(ds_cfg.get("augment", False)),
    )

    batch_size = int(ds_cfg.get("batch_size", 64))
    num_workers = int(ds_cfg.get("num_workers", 2))
    pin_memory = bool(ds_cfg.get("pin_memory", True)) and torch.cuda.is_available()

    loaders = {
        "train": DataLoader(
            splits["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            splits["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            splits["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
    return loaders, class_names
