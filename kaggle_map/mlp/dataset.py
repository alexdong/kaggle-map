"""Dataset for MLP training."""

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from kaggle_map.core.models import Answer, QuestionId


class MLPDataset(Dataset):
    """PyTorch dataset for MLP training."""

    def __init__(
        self,
        embeddings: np.ndarray,
        question_ids: np.ndarray,
        predictions: np.ndarray,  # Full prediction strings like "True_Correct:NA"
        correct_answers: dict[QuestionId, Answer],
        mc_answers: np.ndarray,  # Student answers
        true_label_encoders: dict[QuestionId, LabelEncoder],
        false_label_encoders: dict[QuestionId, LabelEncoder],
    ) -> None:
        self.embeddings = torch.FloatTensor(embeddings)
        self.question_ids = torch.LongTensor(question_ids)
        self.predictions = predictions
        self.correct_answers = correct_answers
        self.mc_answers = mc_answers
        self.true_label_encoders = true_label_encoders
        self.false_label_encoders = false_label_encoders

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

        # Determine if answer is correct
        is_correct = mc_answer == self.correct_answers.get(qid, "")
        is_correct_idx = torch.tensor(1 if is_correct else 0, dtype=torch.long)

        # Get appropriate label encoder
        label_encoder = self.true_label_encoders.get(qid) if is_correct else self.false_label_encoders.get(qid)

        # Encode the prediction label
        if label_encoder and prediction in label_encoder.classes_:
            label = label_encoder.transform([prediction])[0]
        else:
            label = 0  # Default to first class if not found

        return (
            self.embeddings[idx],
            self.question_ids[idx],
            torch.tensor(label, dtype=torch.long),
            is_correct_idx,
        )
