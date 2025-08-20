"""Utilities for strategy implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch
from loguru import logger
from torch import nn, optim

if TYPE_CHECKING:
    import numpy as np

    from kaggle_map.models import QuestionId

    from .mlp import MLPNet

# Data structures for MLP training pipeline - replacing complex tuples with self-documenting types


class TrainingData(NamedTuple):
    """Training data prepared for MLP model training.

    Replaces: tuple[np.ndarray, np.ndarray, dict[QuestionId, np.ndarray], np.ndarray]

    Benefits over tuple:
    - Self-documenting: result.embeddings vs result[0]
    - Type safety: IDE autocompletion and static analysis
    - Immutable: Cannot accidentally modify components
    - Easy to extend: Add new fields without breaking existing code
    """

    embeddings: np.ndarray  # Shape: (n_samples, embedding_dim) - text embeddings
    correctness: np.ndarray  # Shape: (n_samples,) - binary correctness labels
    misconception_labels: dict[QuestionId, np.ndarray]  # Question-specific misconception labels
    question_ids: np.ndarray  # Shape: (n_samples,) - question ID for each sample


class ProcessedRows(NamedTuple):
    """Intermediate processing results from training row processing.

    Replaces: tuple[list, list, list, dict]

    This represents the intermediate state before conversion to numpy arrays.
    Using NamedTuple for immutability since this is a pure data container.
    """

    embeddings: list[np.ndarray]  # List of embedding arrays
    correctness: list[float]  # List of correctness values
    question_ids: list[QuestionId]  # List of question IDs
    misconception_labels: dict[QuestionId, list[np.ndarray]]  # Per-question misconception labels


@dataclass(frozen=True)
class DatasetItem:
    """Single dataset item for PyTorch training.

    Replaces: tuple[torch.Tensor, dict[str, torch.Tensor], int, int]

    Using dataclass instead of NamedTuple because:
    - Contains complex nested types (dict)
    - May need validation logic in the future
    - More readable than tuple unpacking
    """

    features: torch.Tensor  # Shape: (embedding_dim,) - input features
    labels: dict[str, torch.Tensor]  # Multi-head labels (correctness, misconceptions, etc.)
    question_id: QuestionId  # Question identifier
    sample_index: int  # Index in dataset


@dataclass(frozen=True)
class TrainingSetup:
    """Training setup components for MLP model.

    Replaces: tuple[MLPNet, dict[str, nn.Module], optim.Adam]

    Using dataclass for:
    - Clear component naming
    - Type safety
    - Easy extension for additional setup components
    """

    model: MLPNet  # The neural network model
    criterions: dict[str, nn.Module]  # Loss functions for each head
    optimizer: optim.Adam  # Optimizer for training


class BatchData(NamedTuple):
    """Batched data for training.

    Replaces: tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]

    Using NamedTuple because:
    - Pure data container
    - Immutable batch data
    - Clear component access
    """

    features: torch.Tensor  # Shape: (batch_size, embedding_dim)
    multi_labels: dict[str, torch.Tensor]  # Batched multi-head labels
    question_ids: torch.Tensor  # Shape: (batch_size,)
    indices: torch.Tensor  # Shape: (batch_size,) - sample indices


def collate_multihead_batch(  # noqa: PLR0912
    batch: list, device: torch.device | None = None
) -> BatchData:
    """Custom collate function for multi-head training with variable tensor shapes.

    Handles padding of misconception and category labels to max batch size,
    while keeping correctness labels consistent. Also moves tensors to target device.
    Essential for DataLoader when samples have different numbers of misconceptions/categories.

    Args:
        batch: List of (features, multi_labels, question_id, index) tuples
        device: Target device for tensors (if None, uses device from first feature)

    Returns:
        BatchData with features, padded multi_labels, question_ids, and indices
    """
    features = torch.stack([item[0] for item in batch])
    question_ids = torch.tensor([item[2] for item in batch])
    indices = torch.tensor([item[3] for item in batch])

    # Handle multi-labels with different shapes per sample
    multi_labels = {}

    # Correctness labels (consistent shape across samples)
    multi_labels["correctness"] = torch.stack([item[1]["correctness"] for item in batch])

    # Misconception labels - pad to max size in batch
    misc_labels = [item[1]["misconceptions"] for item in batch]
    if misc_labels:
        max_misc_size = max(label.size(0) for label in misc_labels)
        padded_misc = []
        for label in misc_labels:
            if label.size(0) < max_misc_size:
                padded = torch.zeros(max_misc_size, device=label.device)
                padded[: label.size(0)] = label
                padded_misc.append(padded)
            else:
                padded_misc.append(label)
        multi_labels["misconceptions"] = torch.stack(padded_misc)

    # Correct category labels - pad to max size in batch
    correct_cat_labels = [item[1].get("correct_categories") for item in batch if "correct_categories" in item[1]]
    if correct_cat_labels:
        max_correct_size = max(label.size(0) for label in correct_cat_labels)
        padded_correct = []
        for item in batch:
            if "correct_categories" in item[1]:
                label = item[1]["correct_categories"]
                if label.size(0) < max_correct_size:
                    padded = torch.zeros(max_correct_size, device=label.device)
                    padded[: label.size(0)] = label
                    padded_correct.append(padded)
                else:
                    padded_correct.append(label)
            else:
                padded_correct.append(torch.zeros(max_correct_size, device=features.device))
        multi_labels["correct_categories"] = torch.stack(padded_correct)

    # Incorrect category labels - pad to max size in batch
    incorrect_cat_labels = [item[1].get("incorrect_categories") for item in batch if "incorrect_categories" in item[1]]
    if incorrect_cat_labels:
        max_incorrect_size = max(label.size(0) for label in incorrect_cat_labels)
        padded_incorrect = []
        for item in batch:
            if "incorrect_categories" in item[1]:
                label = item[1]["incorrect_categories"]
                if label.size(0) < max_incorrect_size:
                    padded = torch.zeros(max_incorrect_size, device=label.device)
                    padded[: label.size(0)] = label
                    padded_incorrect.append(padded)
                else:
                    padded_incorrect.append(label)
            else:
                padded_incorrect.append(torch.zeros(max_incorrect_size, device=features.device))
        multi_labels["incorrect_categories"] = torch.stack(padded_incorrect)

    # Move all tensors to target device if specified
    if device is not None:
        features = features.to(device)
        question_ids = question_ids.to(device)
        indices = indices.to(device)

        # Move all labels to device
        for key, tensor in multi_labels.items():
            multi_labels[key] = tensor.to(device)

    return BatchData(
        features=features,
        multi_labels=multi_labels,
        question_ids=question_ids,
        indices=indices,
    )


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
