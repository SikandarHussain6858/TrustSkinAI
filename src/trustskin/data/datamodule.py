"""MedIMeta Dermatoscopy data loading, transforms, and class weight computation.

Supports both:
1. Official `medimeta.MedIMeta` loader (preferred when installed, e.g., on Google Colab).
2. Lightweight built-in `LocalMedIMetaDataset` reading `info.yaml`, `splits/{split}.txt`,
   `task_labels/{task_name}.npy`, and `images/*.tiff` directly via PIL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

DEFAULT_CLASS_NAMES = [
    "Melanoma",
    "Melanocytic nevus",
    "Basal cell carcinoma",
    "Actinic keratosis / Bowen's disease",
    "Benign keratosis",
    "Dermatofibroma",
    "Vascular lesion",
]


def _numpy_normalize_transform(
    image_size: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> Callable[[Image.Image], torch.Tensor]:
    """Fallback PIL -> normalized CHW float32 tensor when torchvision is not installed."""
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

    def _transform(img: Image.Image) -> torch.Tensor:
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        arr = (arr - mean_arr) / std_arr
        return torch.from_numpy(arr)

    return _transform


def build_transforms(
    image_size: int = 224,
    mean: list[float] | tuple[float, ...] = (0.485, 0.456, 0.406),
    std: list[float] | tuple[float, ...] = (0.229, 0.224, 0.225),
    augment: bool = False,
) -> Callable[[Image.Image], torch.Tensor]:
    """Build deterministic or augmented preprocessing transforms for 224x224 RGB dermoscopy."""
    try:
        from torchvision import transforms
    except ImportError:
        if augment:
            raise ImportError(
                "Data augmentation requires `torchvision`. Install with `pip install torchvision`."
            )
        return _numpy_normalize_transform(image_size, tuple(mean), tuple(std))

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


class LocalMedIMetaDataset(Dataset):
    """Direct reader for an extracted MedIMeta dataset directory (e.g. `data/derm/`)."""

    def __init__(
        self,
        dataset_dir: str | Path,
        task_name: str = "disease category",
        split: str = "train",
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.task_name = task_name
        self.split = split
        self.transform = transform

        info_path = self.dataset_dir / "info.yaml"
        if not info_path.exists():
            raise FileNotFoundError(f"Missing MedIMeta info.yaml at {info_path}")

        with info_path.open("r", encoding="utf-8") as f:
            self.info = yaml.safe_load(f)

        split_file = self.dataset_dir / "splits" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")

        self.rel_paths = [
            line.strip()
            for line in split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.indices = np.array([int(Path(p).stem) for p in self.rel_paths], dtype=np.int64)

        labels_file = self.dataset_dir / "task_labels" / f"{task_name}.npy"
        if not labels_file.exists():
            raise FileNotFoundError(f"Missing task label file: {labels_file}")

        all_labels = np.load(labels_file)
        self.labels = all_labels[self.indices].astype(np.int64)

        # Extract ordered class names from info.yaml
        self.class_names = list(DEFAULT_CLASS_NAMES)
        for task in self.info.get("tasks", []):
            if task.get("task_name") == task_name and "labels" in task:
                label_map = task["labels"]
                self.class_names = [label_map[i] for i in sorted(label_map.keys())]
                break

    def __len__(self) -> int:
        return len(self.rel_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | Image.Image, torch.Tensor]:
        img_path = self.dataset_dir / self.rel_paths[idx]
        with Image.open(img_path) as img:
            rgb = img.convert("RGB")
        x = self.transform(rgb) if self.transform is not None else rgb
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y


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


def get_split_labels(
    data_root: str | Path,
    dataset_id: str = "derm",
    task_name: str = "disease category",
    split: str = "train",
) -> np.ndarray:
    """Fast O(1)-image-IO label lookup directly from MedIMeta `.npy` and `splits/*.txt`."""
    dataset_dir = Path(data_root) / dataset_id
    split_file = dataset_dir / "splits" / f"{split}.txt"
    labels_file = dataset_dir / "task_labels" / f"{task_name}.npy"
    rel_paths = [
        line.strip()
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indices = np.array([int(Path(p).stem) for p in rel_paths], dtype=np.int64)
    all_labels = np.load(labels_file)
    return all_labels[indices].astype(np.int64)


def load_medimeta_splits(
    data_root: str | Path,
    dataset_id: str = "derm",
    task_name: str = "disease category",
    image_size: int = 224,
    mean: list[float] | tuple[float, ...] = (0.485, 0.456, 0.406),
    std: list[float] | tuple[float, ...] = (0.229, 0.224, 0.225),
    augment_train: bool = False,
    prefer_local: bool = False,
) -> tuple[dict[str, Dataset], list[str]]:
    """Load predefined train, val, and test splits from MedIMeta.

    Test and validation splits never receive data augmentation.
    """
    root = Path(data_root)
    dataset_dir = root / dataset_id
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. "
            f"Expected extracted MedIMeta `{dataset_id}` folder inside `{root}`."
        )

    train_tf = build_transforms(image_size=image_size, mean=mean, std=std, augment=augment_train)
    eval_tf = build_transforms(image_size=image_size, mean=mean, std=std, augment=False)

    use_medimeta_pkg = False
    if not prefer_local:
        try:
            from medimeta import MedIMeta

            use_medimeta_pkg = True
        except ImportError:
            use_medimeta_pkg = False

    if use_medimeta_pkg:
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
        splits: dict[str, Dataset] = {
            "train": _LabelCastWrapper(raw_train),
            "val": _LabelCastWrapper(raw_val),
            "test": _LabelCastWrapper(raw_test),
        }
        class_names = list(getattr(raw_train, "class_names", DEFAULT_CLASS_NAMES))
        return splits, class_names

    local_train = LocalMedIMetaDataset(dataset_dir, task_name=task_name, split="train", transform=train_tf)
    local_val = LocalMedIMetaDataset(dataset_dir, task_name=task_name, split="val", transform=eval_tf)
    local_test = LocalMedIMetaDataset(dataset_dir, task_name=task_name, split="test", transform=eval_tf)
    return {"train": local_train, "val": local_val, "test": local_test}, local_train.class_names


def compute_class_weights(
    labels: np.ndarray | list[int],
    num_classes: int,
    power: float = 1.0,
) -> torch.Tensor:
    """Compute (optionally dampened) inverse-frequency class weights from training labels only.

    Args:
        labels: 1D array/list of integer class indices from the training split.
        num_classes: Total number of classes.
        power: Dampening exponent (1.0 = full linear inverse frequency; 0.5 = square-root).
    """
    arr = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(arr, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = (len(arr) / (num_classes * counts)) ** float(power)
    weights = weights / float(np.mean(weights))
    return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(
    cfg: dict[str, Any],
) -> tuple[dict[str, DataLoader], list[str]]:
    """Create train, val, and test DataLoaders from a project config dict."""
    ds_cfg = cfg["dataset"]
    splits, class_names = load_medimeta_splits(
        data_root=cfg["paths"]["data_root"],
        dataset_id=ds_cfg.get("dataset_id", "derm"),
        task_name=ds_cfg.get("task_name", "disease category"),
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
