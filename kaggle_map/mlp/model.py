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
class Architecture:
    """Architecture configuration for MLP model."""

    size: ArchitectureSize
    layers: list[int]  # Layer dimensions including input
    dropout: float = 0.3
    activation: str = "gelu"


# Simplified architectures for 4096+ dim embeddings only
ARCHITECTURES = {
    # For 4096-dim embeddings (semantic strategy)
    "medium_4096": Architecture("medium", [4128, 2048, 1024, 512, 256]),
    "large_4096": Architecture("large", [4128, 3072, 1536, 768, 384]),
    "xlarge_4096": Architecture("xlarge", [4128, 4096, 2048, 1024, 512]),
    # For 8192-dim embeddings (double-blind strategy)
    "medium_8192": Architecture("medium", [8224, 4096, 2048, 1024, 512]),
    "large_8192": Architecture("large", [8224, 6144, 3072, 1536, 768]),
    "xlarge_8192": Architecture("xlarge", [8224, 8192, 4096, 2048, 1024]),
}


def get_architecture(size: str, embedding_dim: int) -> Architecture:
    """Get architecture config for given size and embedding dimension."""
    # Round to nearest supported dimension
    dim_key = "4096" if embedding_dim <= 6000 else "8192"

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

        # Learnable correctness embedding
        correctness_embedding_dim = 32
        self.correctness_embedding = nn.Embedding(2, correctness_embedding_dim)

        # Get architecture for this embedding dimension
        arch = get_architecture(architecture_size, embedding_dim)
        activation_fn = get_activation(activation)

        # Build trunk layers
        layers = []
        for i in range(len(arch.layers) - 1):
            layers.append(nn.Linear(arch.layers[i], arch.layers[i + 1]))
            layers.append(nn.LayerNorm(arch.layers[i + 1]))
            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout))

        self.trunk = nn.Sequential(*layers)
        self.output_dim = arch.layers[-1]

        # Create question-specific heads
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

        logger.info(
            f"Created model with {len(self.true_heads)} true heads and {len(self.false_heads)} false heads"
        )

    def forward(
        self, x: torch.Tensor, question_ids: torch.Tensor, is_correct: torch.Tensor
    ) -> dict[tuple[int, bool], torch.Tensor]:
        """Forward pass returning logits per question, split by correctness.

        Args:
            x: [batch_size, embedding_dim] - embeddings
            question_ids: [batch_size] - question IDs
            is_correct: [batch_size] - correctness indices (0 or 1)

        Returns:
            Dictionary mapping (question_id, is_correct) to logits tensor
        """
        # Add learnable correctness embeddings
        correct_emb = self.correctness_embedding(is_correct.long())

        # Concatenate with input embeddings
        combined = torch.cat([x, correct_emb], dim=-1)

        # Pass through trunk
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

            # Process correct answers
            correct_mask = question_correctness > 0
            if correct_mask.any() and str(qid_int) in self.true_heads:
                correct_features = question_features[correct_mask]
                outputs[(qid_int, True)] = self.true_heads[str(qid_int)](correct_features)

            # Process incorrect answers
            incorrect_mask = ~correct_mask
            if incorrect_mask.any() and str(qid_int) in self.false_heads:
                incorrect_features = question_features[incorrect_mask]
                outputs[(qid_int, False)] = self.false_heads[str(qid_int)](incorrect_features)

        return outputs
