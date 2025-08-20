"""PyTorch dataset and data handling utilities for MLP training.

Provides device-agnostic dataset implementation with structured return types
for multi-head MLP training. Dataset returns CPU tensors; device movement
is handled by collate functions during training.
"""

from typing import NamedTuple

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

from kaggle_map.models import QuestionId


class DatasetItem(NamedTuple):
    """Structured return type from MLPDataset.

    Replaces complex tuple returns with self-documenting named fields.
    """

    features: torch.Tensor  # Embedding features, shape: (embedding_dim,)
    labels: dict[str, torch.Tensor]  # Multi-head labels
    question_id: QuestionId  # Question identifier
    sample_index: int  # Index in dataset


class MLPDataset(Dataset):
    """PyTorch dataset for multi-head MLP training.

    Device-agnostic dataset that provides:
    - Embeddings (question/answer/explanation)
    - Correctness labels (binary)
    - Misconception labels (when applicable)

    Returns CPU tensors; device movement handled by collate functions.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        correctness: np.ndarray,
        misconception_labels: dict[QuestionId, np.ndarray],
        question_ids: np.ndarray,
    ) -> None:
        """Initialize dataset with training data.

        Args:
            embeddings: Input embeddings, shape (n_samples, embedding_dim)
            correctness: Binary correctness labels, shape (n_samples,)
            misconception_labels: Per-question misconception labels
            question_ids: Question IDs for each sample, shape (n_samples,)
        """
        assert len(embeddings) == len(correctness) == len(question_ids), (
            f"Data length mismatch: embeddings={len(embeddings)}, "
            f"correctness={len(correctness)}, question_ids={len(question_ids)}"
        )

        # Store as CPU tensors (device-agnostic)
        self.embeddings = torch.FloatTensor(embeddings)
        self.correctness = torch.FloatTensor(correctness).unsqueeze(1)
        self.question_ids = question_ids

        # Build label mappings for training
        self.misconception_labels = []
        self._build_label_mappings(misconception_labels, question_ids)

    def _build_label_mappings(
        self,
        misconception_labels: dict[QuestionId, np.ndarray],
        question_ids: np.ndarray,
    ) -> None:
        """Build mappings for multi-head training labels."""
        # Create local index trackers for each question
        question_local_indices = {}
        for qid in set(question_ids):
            question_local_indices[qid] = 0

        # Build labels in same order as global dataset
        for _global_idx, qid in enumerate(question_ids):
            local_idx = question_local_indices[qid]
            if qid in misconception_labels and local_idx < len(misconception_labels[qid]):
                misc_label = misconception_labels[qid][local_idx]
                self.misconception_labels.append(torch.FloatTensor(misc_label))
            else:
                # Default: no misconception (zero vector)
                self.misconception_labels.append(torch.zeros(1))

            question_local_indices[qid] += 1

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> DatasetItem:
        """Get training sample with multi-head labels.

        Args:
            idx: Sample index

        Returns:
            DatasetItem with structured fields instead of tuple
        """
        assert 0 <= idx < len(self.embeddings), f"Index {idx} out of range [0, {len(self.embeddings)})"
        question_id = self.question_ids[idx]

        # Features: embedding only (no ground truth correctness)
        features = self.embeddings[idx]

        # Multi-head labels
        labels = {
            "correctness": self.correctness[idx],  # Ground truth correctness for training
            "misconceptions": self.misconception_labels[idx],
        }

        logger.debug(
            "Dataset item retrieved",
            idx=idx,
            question_id=question_id,
            features_shape=list(features.shape),
            labels_keys=list(labels.keys()),
        )

        return DatasetItem(
            features=features,
            labels=labels,
            question_id=question_id,
            sample_index=idx,
        )
