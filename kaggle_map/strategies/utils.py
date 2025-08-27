"""Utilities for strategy implementations."""

from datetime import UTC, datetime

import numpy as np
import torch
from loguru import logger
from pydantic import BaseModel

from kaggle_map.core.models import TrainingRow

# Data split constants
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.debug(
            "Device selection: using MPS (Apple Metal)",
            device_type="mps",
            available_backends=["mps", "cuda", "cpu"],
        )
        return device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cuda_count = torch.cuda.device_count()
        logger.debug(
            "Device selection: using CUDA",
            device_type="cuda",
            cuda_devices=cuda_count,
            available_backends=["cuda", "cpu"],
        )
        return device
    device = torch.device("cpu")
    logger.debug(
        "Device selection: fallback to CPU",
        device_type="cpu",
        available_backends=["cpu"],
    )
    return device


class ModelParameters(BaseModel):
    """Parameters for model training and evaluation."""

    train_split: float
    random_seed: int
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]
    timestamp: str
    total_samples: int

    @classmethod
    def create(
        cls,
        train_split: float,
        random_seed: int,
        train_indices: list[int],
        val_indices: list[int],
        test_indices: list[int],
        total_samples: int,
    ) -> "ModelParameters":
        """Create ModelParameters with current timestamp."""
        return cls(
            train_split=train_split,
            random_seed=random_seed,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            timestamp=datetime.now(UTC).isoformat(),
            total_samples=total_samples,
        )


def split_training_data(
    training_rows: list[TrainingRow],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    random_seed: int = 42,
) -> tuple[list[TrainingRow], list[TrainingRow], list[TrainingRow]]:
    """Split training data into train/validation/test sets.

    Args:
        training_rows: List of training data rows
        train_ratio: Fraction of data for training (default: 0.70)
        val_ratio: Fraction of data for validation (default: 0.15)
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (train_rows, val_rows, test_rows)
    """
    assert training_rows, "Training data cannot be empty"
    assert train_ratio + val_ratio <= 1.0, "train_ratio + val_ratio must be <= 1.0"

    n_samples = len(training_rows)
    indices = np.arange(n_samples)

    # Set random seed for reproducibility
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    # Calculate split points
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)

    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # Create splits
    train_rows = [training_rows[i] for i in train_indices]
    val_rows = [training_rows[i] for i in val_indices]
    test_rows = [training_rows[i] for i in test_indices]

    logger.debug(
        f"Split {n_samples} samples into train={len(train_rows)}, "
        f"val={len(val_rows)}, test={len(test_rows)}"
    )

    assert train_rows, "Training split cannot be empty"
    assert val_rows, "Validation split cannot be empty"

    return train_rows, val_rows, test_rows


def get_split_indices(
    n_samples: int,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    random_seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Get indices for train/validation/test splits.

    Args:
        n_samples: Total number of samples
        train_ratio: Fraction of data for training (default: 0.70)
        val_ratio: Fraction of data for validation (default: 0.15)
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    indices = np.arange(n_samples)

    # Set random seed for reproducibility
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    # Calculate split points
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)

    # Split indices
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size : train_size + val_size].tolist()
    test_indices = indices[train_size + val_size :].tolist()

    return train_indices, val_indices, test_indices
