"""Neural network model for misconception prediction."""

from dataclasses import dataclass
from typing import Literal

import torch
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from torch import nn

from kaggle_map.core.models import QuestionId

ArchitectureSize = Literal["medium", "large", "xlarge"]


@dataclass(frozen=True)
class EvaluationResult:
    """Key for model evaluation outputs by question and correctness."""

    question_id: QuestionId
    is_correct: bool

    def __hash__(self) -> int:
        return hash((self.question_id, self.is_correct))

    def __str__(self) -> str:
        return f"Q{self.question_id}_{'correct' if self.is_correct else 'incorrect'}"


@dataclass(frozen=True)
class Architecture:
    """Architecture configuration for MLP model."""

    size: ArchitectureSize
    layers: list[int]  # Layer dimensions including input
    dropout: float = 0.3
    activation: str = "gelu"


# Simplified architectures for 4096+ dim embeddings only
ARCHITECTURES = {
    # For 4096-dim embeddings (semantic strategy)
    "medium_4096": Architecture("medium", [4128, 2048, 1024]),
    "large_4096": Architecture("large", [4128, 2048, 1024, 512]),
    "xlarge_4096": Architecture("xlarge", [4128, 2048, 1024, 512, 256]),
    # For 8192-dim embeddings (double-blind strategy)
    "medium_8192": Architecture("medium", [8224, 4096, 2048]),
    "large_8192": Architecture("large", [8224, 4096, 2048, 1024]),
    "xlarge_8192": Architecture("xlarge", [8224, 4096, 2048, 1024, 512]),
}


EMBEDDING_DIM_THRESHOLD = 6000  # Threshold to choose between 4096 and 8192 architectures


def get_architecture(size: str, embedding_dim: int) -> Architecture:
    """Get architecture config for given size and embedding dimension."""
    dim_key = "4096" if embedding_dim <= EMBEDDING_DIM_THRESHOLD else "8192"

    key = f"{size}_{dim_key}"
    assert key in ARCHITECTURES, f"Architecture {key} not found"
    return ARCHITECTURES[key]


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "leaky_relu": nn.LeakyReLU(0.2),
        "silu": nn.SiLU(),
    }
    return activations.get(name, nn.GELU())


class QuestionSpecificMLP(nn.Module):
    """MLP with shared trunk and question-specific prediction heads."""

    def __init__(
        self,
        question_predictions: dict[QuestionId, list[str]],
        embedding_dim: int,
        architecture_size: str = "xlarge",
        dropout: float = 0.3,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        correctness_embedding_dim = 32
        self.correctness_embedding = nn.Embedding(2, correctness_embedding_dim)

        arch = get_architecture(architecture_size, embedding_dim)
        activation_fn = get_activation(activation)

        layers = []
        for i in range(len(arch.layers) - 1):
            layers.append(nn.Linear(arch.layers[i], arch.layers[i + 1]))
            layers.append(nn.LayerNorm(arch.layers[i + 1]))
            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout))

        self.trunk = nn.Sequential(*layers)
        self.output_dim = arch.layers[-1]

        self.true_heads = nn.ModuleDict()
        self.false_heads = nn.ModuleDict()
        self.true_label_encoders = {}
        self.false_label_encoders = {}

        for question_id, predictions in question_predictions.items():
            true_preds = [p for p in predictions if p.startswith("True_")]
            false_preds = [p for p in predictions if p.startswith("False_")]

            if true_preds:
                self.true_heads[str(question_id)] = nn.Linear(self.output_dim, len(true_preds))
                encoder = LabelEncoder()
                encoder.fit(sorted(true_preds))
                self.true_label_encoders[question_id] = encoder

            if false_preds:
                self.false_heads[str(question_id)] = nn.Linear(self.output_dim, len(false_preds))
                encoder = LabelEncoder()
                encoder.fit(sorted(false_preds))
                self.false_label_encoders[question_id] = encoder

        logger.info(f"Created model with {len(self.true_heads)} true heads and {len(self.false_heads)} false heads")

    def forward(
        self, x: torch.Tensor, question_ids: torch.Tensor, is_correct: torch.Tensor
    ) -> dict[EvaluationResult, torch.Tensor]:
        """Forward pass returning logits per question, split by correctness.

        Args:
            x: [batch_size, embedding_dim] - embeddings
            question_ids: [batch_size] - question IDs
            is_correct: [batch_size] - correctness indices (0 or 1)

        Returns:
            Dictionary mapping EvaluationResult to logits tensor
        """
        correct_emb = self.correctness_embedding(is_correct.long())

        combined = torch.cat([x, correct_emb], dim=-1)

        shared_features = self.trunk(combined)

        outputs = {}
        unique_questions = torch.unique(question_ids)

        for qid in unique_questions:
            qid_int = int(qid.item())
            mask = question_ids == qid

            if not mask.any():
                continue

            question_features = shared_features[mask]
            question_correctness = is_correct[mask]

            correct_mask = question_correctness > 0
            if correct_mask.any() and str(qid_int) in self.true_heads:
                correct_features = question_features[correct_mask]
                eval_key = EvaluationResult(question_id=qid_int, is_correct=True)
                outputs[eval_key] = self.true_heads[str(qid_int)](correct_features)

            incorrect_mask = ~correct_mask
            if incorrect_mask.any() and str(qid_int) in self.false_heads:
                incorrect_features = question_features[incorrect_mask]
                eval_key = EvaluationResult(question_id=qid_int, is_correct=False)
                outputs[eval_key] = self.false_heads[str(qid_int)](incorrect_features)

        return outputs
