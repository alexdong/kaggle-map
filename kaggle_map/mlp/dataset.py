"""Dataset for MLP training."""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from kaggle_map.core.models import Answer, QuestionId


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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single training sample.

        Returns:
            Tuple of (embedding, question_id, label, is_correct_idx)
        """
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

        return (
            self.embeddings[idx],
            self.question_ids[idx],
            torch.tensor(label, dtype=torch.long),
            is_correct_idx,
        )
