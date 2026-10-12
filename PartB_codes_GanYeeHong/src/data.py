"""Chest X-ray data loading for Part B.

Intensities are scaled to [0, 1] and then standardised with the mean and
standard deviation of the training partition, as the assignment requires.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

CANDIDATE_PATHS = (
    PROJECT_DIR / "chestxray_binary_64.npz",
    PROJECT_DIR.parent / "chestxray_binary_64.npz",
)


def find_dataset() -> Path:
    for path in CANDIDATE_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(
        "chestxray_binary_64.npz not found. Place it in the project root or "
        "one directory above it."
    )


def load_data(path=None):
    """Return the three partitions plus the training statistics used to scale them."""
    raw = np.load(path or find_dataset())

    train_images = raw["train_images"].astype(np.float32) / 255.0
    mean = float(train_images.mean())
    std = float(train_images.std())

    def partition(prefix):
        images = raw[f"{prefix}_images"].astype(np.float32) / 255.0
        images = (images - mean) / std
        x = torch.from_numpy(images).unsqueeze(1)
        y = torch.from_numpy(raw[f"{prefix}_labels"].astype(np.float32))
        return TensorDataset(x, y)

    return {
        "train": partition("train"),
        "validation": partition("val"),
        "test": partition("test"),
        "mean": mean,
        "std": std,
    }


def class_counts(dataset):
    labels = dataset.tensors[1]
    return {0: int((labels == 0).sum()), 1: int((labels == 1).sum())}


def combined_training_set(data):
    """Train and validation merged, used for the final retrain before testing.

    The standardisation statistics are not refitted: the assignment specifies
    normalising with the mean and standard deviation of the training set, so
    those are kept throughout.
    """
    images = torch.cat([data["train"].tensors[0], data["validation"].tensors[0]])
    labels = torch.cat([data["train"].tensors[1], data["validation"].tensors[1]])
    return TensorDataset(images, labels)
