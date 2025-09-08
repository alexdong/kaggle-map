"""Dataset for MLP training."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from kaggle_map.core.models import Answer, QuestionId
from kaggle_map.mlp.label_encoder import LabelEncoders

__all__ = ["DatasetArrays", "MLPDataset", "TrainingSample"]


@dataclass
class TrainingSample:
    """A single training sample for MLP model."""

    embedding: torch.Tensor
    question_id: torch.Tensor
    label: torch.Tensor
    is_correct: torch.Tensor


@dataclass
class DatasetArrays:
    """Arrays needed for dataset creation."""

    embeddings: np.ndarray
    question_ids: np.ndarray
    predictions: np.ndarray
    mc_answers: np.ndarray


class MLPDataset(Dataset):
    """PyTorch dataset for MLP training."""

    def __init__(
        self,
        arrays: DatasetArrays,
        correct_answers: dict[QuestionId, Answer],
        label_encoders: LabelEncoders,
    ) -> None:
        """Initialize dataset with arrays and encoders.

        Args:
            arrays: Data arrays (embeddings, question_ids, predictions, mc_answers)
            correct_answers: Mapping of question_id to correct answer
            label_encoders: Encoder for converting predictions to labels
        """
        assert len(arrays.embeddings) == len(arrays.question_ids) == len(arrays.predictions) == len(arrays.mc_answers)

        # Convert to tensors
        self.embeddings = torch.FloatTensor(arrays.embeddings)
        self.question_ids = torch.LongTensor(arrays.question_ids)

        # Pre-compute correctness flags and labels for efficiency
        self.is_correct = torch.zeros(len(arrays.mc_answers), dtype=torch.long)
        self.labels = torch.zeros(len(arrays.predictions), dtype=torch.long)

        for i in range(len(arrays.mc_answers)):
            qid = int(arrays.question_ids[i])
            is_correct = arrays.mc_answers[i] == correct_answers.get(qid, "")
            self.is_correct[i] = 1 if is_correct else 0
            self.labels[i] = label_encoders.encode(qid, str(arrays.predictions[i]), is_correct=is_correct)

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> TrainingSample:
        """Get a single training sample.

        Returns:
            TrainingSample with all necessary tensors
        """
        return TrainingSample(
            embedding=self.embeddings[idx],
            question_id=self.question_ids[idx],
            label=self.labels[idx],
            is_correct=self.is_correct[idx],
        )
