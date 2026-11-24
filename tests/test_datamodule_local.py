"""Smoke test for loading the real MedIMeta Dermatoscopy dataset at data/derm/."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from trustskin.data import compute_class_weights, get_split_labels, load_medimeta_splits

DERM_DIR = Path("data/derm")


@pytest.mark.skipif(not DERM_DIR.exists(), reason="MedIMeta derm dataset not present locally.")
def test_local_medimeta_derm_splits_and_shapes() -> None:
    splits, class_names = load_medimeta_splits(
        data_root="data",
        dataset_id="derm",
        task_name="disease category",
        image_size=224,
        augment_train=False,
        prefer_local=True,
    )

    assert len(class_names) == 7
    assert len(splits["train"]) == 10015
    assert len(splits["val"]) == 193
    assert len(splits["test"]) == 1512

    img, label = splits["train"][0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert 0 <= int(label) < 7

    train_labels = get_split_labels("data", "derm", "disease category", "train")
    weights = compute_class_weights(train_labels, num_classes=7)
    assert weights.shape == (7,)
    # Dermatofibroma (class 5) is rarest, Nevus (class 1) is most common
    assert weights[5] > weights[1]
