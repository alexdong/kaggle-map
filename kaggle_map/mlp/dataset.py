"""Dataset for MLP training."""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from kaggle_map.core.models import Answer, QuestionId

__all__ = ["DatasetArrays", "MLPDataset", "TrainingSample"]


@dataclass
class TrainingSample:
    """A single training sample for MLP model."""

    embedding: torch.Tensor
    question_id: torch.Tensor
    label: torch.Tensor
    is_correct_idx: torch.Tensor


@dataclass
class DatasetArrays:
    """Arrays needed for dataset creation."""

    embeddings: np.ndarray
    question_ids: np.ndarray
    predictions: np.ndarray
    mc_answers: np.ndarray


@dataclass
class DatasetEncoders:
    """Encoders and reference data for the dataset."""

    correct_answers: dict[QuestionId, Answer]
    true_label_encoders: dict[QuestionId, LabelEncoder]
    false_label_encoders: dict[QuestionId, LabelEncoder]


class MLPDataset(Dataset):
    """PyTorch dataset for MLP training."""

    def __init__(self, arrays: DatasetArrays, encoders: DatasetEncoders) -> None:
        """Initialize dataset with arrays and encoders.

        Args:
            arrays: Data arrays (embeddings, question_ids, predictions, mc_answers)
            encoders: Label encoders and reference data
        """
        self.embeddings = torch.FloatTensor(arrays.embeddings)
        self.question_ids = torch.LongTensor(arrays.question_ids)
        self.predictions = arrays.predictions
        self.correct_answers = encoders.correct_answers
        self.mc_answers = arrays.mc_answers
        self.true_label_encoders = encoders.true_label_encoders
        self.false_label_encoders = encoders.false_label_encoders

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> TrainingSample:
        """Get a single training sample."""
        qid = int(self.question_ids[idx].item())
        prediction = self.predictions[idx]
        mc_answer = self.mc_answers[idx]

        is_correct = mc_answer == self.correct_answers.get(qid, "")
        is_correct_idx = torch.tensor(1 if is_correct else 0, dtype=torch.long)

        label_encoder = self.true_label_encoders.get(qid) if is_correct else self.false_label_encoders.get(qid)

        if (
            label_encoder is not None
            and hasattr(label_encoder, "classes_")
            and prediction in getattr(label_encoder, "classes_", [])
        ):
            label = label_encoder.transform([prediction])[0]
        else:
            label = 0  # Default to first class if not found

        return TrainingSample(
            embedding=self.embeddings[idx],
            question_id=self.question_ids[idx],
            label=torch.tensor(label, dtype=torch.long),
            is_correct_idx=is_correct_idx,
        )
