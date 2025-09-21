"""Neural network model for misconception prediction."""

from dataclasses import dataclass

import torch
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from torch import nn

from kaggle_map.core.models import (
    ActivationType,
    ArchitectureSize,
    EmbeddingModel,
    EmbeddingStrategy,
    QuestionId,
)
from kaggle_map.embeddings import get_input_embeddings_dimension
from kaggle_map.utils.logger_config import configure_logger

configure_logger(__name__)


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
    activation: ActivationType = ActivationType.GELU


CORRECTNESS_EMBEDDING_DIMENSIONS = 32


def get_architecture(
    size: ArchitectureSize,
    embedding_model: EmbeddingModel,
    embedding_strategy: EmbeddingStrategy,
    correctness_embedding_dim: int = CORRECTNESS_EMBEDDING_DIMENSIONS,
) -> Architecture:
    """Compose architecture layers based on embedding sources and size.

    Args:
        size: Architecture size (MEDIUM, LARGE, XLARGE)
        embedding_model: Embedding backend used to generate explanations embeddings
        embedding_strategy: Strategy describing how text inputs are combined
        correctness_embedding_dim: Dimensionality of the correctness embedding head
    """
    embedding_dim = get_input_embeddings_dimension(embedding_strategy, embedding_model)
    total_input_dim = embedding_dim + correctness_embedding_dim

    # Create architecture layers based on input dimension
    # The hidden layers progressively reduce dimensionality
    if size == ArchitectureSize.MEDIUM:
        # Medium: 2 hidden layers
        hidden_1 = min(2048, total_input_dim // 2)
        hidden_2 = hidden_1 // 2
        layers = [total_input_dim, hidden_1, hidden_2]
    elif size == ArchitectureSize.LARGE:
        # Large: 3 hidden layers
        hidden_1 = min(2048, total_input_dim // 2)
        hidden_2 = hidden_1 // 2
        hidden_3 = hidden_2 // 2
        layers = [total_input_dim, hidden_1, hidden_2, hidden_3]
    else:  # XLARGE
        # XLarge: 4 hidden layers
        hidden_1 = min(2048, total_input_dim // 2)
        hidden_2 = hidden_1 // 2
        hidden_3 = hidden_2 // 2
        hidden_4 = hidden_3 // 2
        layers = [total_input_dim, hidden_1, hidden_2, hidden_3, hidden_4]

    return Architecture(size, layers)


def get_activation(activation_type: ActivationType) -> nn.Module:
    """Get activation function by type."""
    activations = {
        ActivationType.RELU: nn.ReLU(),
        ActivationType.GELU: nn.GELU(),
        ActivationType.LEAKY_RELU: nn.LeakyReLU(0.2),
        ActivationType.SILU: nn.SiLU(),
    }
    return activations.get(activation_type, nn.GELU())


class QuestionSpecificMLP(nn.Module):
    """MLP with shared trunk and question-specific prediction heads."""

    def __init__(  # noqa: PLR0913
        self,
        question_predictions: dict[QuestionId, list[str]],
        embedding_model: EmbeddingModel,
        embedding_strategy: EmbeddingStrategy,
        architecture_size: ArchitectureSize = ArchitectureSize.XLARGE,
        dropout: float = 0.3,
        activation: ActivationType = ActivationType.GELU,
        correctness_embedding_dim: int = CORRECTNESS_EMBEDDING_DIMENSIONS,
    ) -> None:
        super().__init__()

        assert correctness_embedding_dim > 0, "Correctness embedding dimension must be positive"

        self.embedding_model = embedding_model
        self.embedding_strategy = embedding_strategy
        self.correctness_embedding_dim = correctness_embedding_dim

        embedding_dim = get_input_embeddings_dimension(embedding_strategy, embedding_model)
        self.embedding_dim = embedding_dim

        self.correctness_embedding = nn.Embedding(2, correctness_embedding_dim)

        arch = get_architecture(
            size=architecture_size,
            embedding_model=embedding_model,
            embedding_strategy=embedding_strategy,
            correctness_embedding_dim=correctness_embedding_dim,
        )
        activation_fn = get_activation(activation)

        layers = []
        for i in range(len(arch.layers) - 1):
            layers.append(nn.Linear(arch.layers[i], arch.layers[i + 1]))
            layers.append(nn.LayerNorm(arch.layers[i + 1]))
            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout))

        self.trunk = nn.Sequential(*layers)
        self.output_dim = arch.layers[-1]
        self.total_input_dim = arch.layers[0]

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
        self, input_embeddings: torch.Tensor, question_ids: torch.Tensor, mc_answer_correctnesses: torch.Tensor
    ) -> dict[EvaluationResult, torch.Tensor]:
        """Forward pass returning logits per question, split by correctness.

        Args:
            input_embeddings: [batch_size, embedding_dim] - explanation embeddings
            question_ids: [batch_size] - question IDs
            mc_answer_correctnesses: [batch_size] - correctness indices (0 or 1)

        Returns:
            Dictionary mapping EvaluationResult to logits tensor
        """
        correct_emb = self.correctness_embedding(mc_answer_correctnesses.long())

        combined = torch.cat([input_embeddings, correct_emb], dim=-1)

        shared_features = self.trunk(combined)

        outputs = {}
        unique_questions = torch.unique(question_ids)

        for qid in unique_questions:
            qid_int = int(qid.item())
            mask = question_ids == qid

            if not mask.any():
                continue

            question_features = shared_features[mask]
            question_correctness = mc_answer_correctnesses[mask]

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
