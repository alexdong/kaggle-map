"""MLPNet neural network model for student misconception prediction.

Multi-head MLP architecture with dynamic embedding dimensions that adapts
to different embedding models. Provides separate heads for correctness
prediction and question-specific misconception detection.
"""

import torch
from loguru import logger
from torch import nn

from kaggle_map.core.embeddings.embedding_models import EmbeddingModel
from kaggle_map.core.models import Misconception, QuestionId
from kaggle_map.strategies.mlp.config import (
    DROPOUT_RATE,
    HIDDEN_DIMS,
    MAX_MISCONCEPTIONS_PER_QUESTION,
)


class MLPNet(nn.Module):
    """Multi-head MLP for direct category prediction aligned with competition format.

    Architecture:
    1. Shared embedding trunk: processes question/answer/explanation (dynamic input size)
    2. Correctness head: predicts if student answer is correct
    3. Misconception heads: predict misconception distributions per question

    This directly mirrors the baseline's two-step process:
    - Determine correctness (neural vs rule-based)
    - Select misconception within correctness state (neural vs frequency-based)
    """

    def __init__(
        self,
        question_misconceptions: dict[QuestionId, list[Misconception]],
        embedding_model: EmbeddingModel,
    ) -> None:
        """Initialize multi-head MLP with dynamic embedding dimensions.

        Args:
            question_misconceptions: Misconceptions per question (using proper type)
            embedding_model: Embedding model providing dynamic dimension
        """
        super().__init__()

        # Store metadata for prediction reconstruction
        self.question_misconceptions = question_misconceptions
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_model.dim  # Dynamic dimension from model

        logger.debug(
            "Initializing MLPNet with dynamic embedding dimension",
            embedding_model=embedding_model.model_id,
            embedding_dim=self.embedding_dim,
            total_questions=len(question_misconceptions),
            max_misconceptions_per_question=MAX_MISCONCEPTIONS_PER_QUESTION,
        )

        # Shared trunk: dynamic embedding input size
        self.shared_trunk = nn.Sequential(
            nn.Linear(self.embedding_dim, HIDDEN_DIMS[0]),  # Dynamic input
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1]),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_DIMS[1], HIDDEN_DIMS[2]),
        )

        # Correctness prediction head (binary classification)
        self.correctness_head = nn.Linear(HIDDEN_DIMS[2], 1)

        # Misconception prediction heads - use fixed size for all questions
        self.misconception_heads = nn.ModuleDict()
        for question_id in question_misconceptions:
            qid_str = str(question_id)
            self.misconception_heads[qid_str] = nn.Linear(HIDDEN_DIMS[2], MAX_MISCONCEPTIONS_PER_QUESTION)

        logger.info(
            "MLPNet initialized successfully",
            embedding_dim=self.embedding_dim,
            hidden_dims=HIDDEN_DIMS,
            misconception_heads=len(self.misconception_heads),
            total_parameters=sum(p.numel() for p in self.parameters()),
        )

    def forward(self, x: torch.Tensor, question_id: int) -> dict[str, torch.Tensor]:
        """Multi-head forward pass returning all prediction components.

        Args:
            x: Input tensor (embedding only, shape: [batch_size, embedding_dim])
            question_id: Question ID for question-specific heads

        Returns:
            Dictionary with keys:
            - 'correctness': Correctness prediction logits [batch_size, 1]
            - 'misconceptions': Misconception logits (if question has misconceptions)
        """
        assert isinstance(x, torch.Tensor), f"Input must be a tensor, got {type(x)}"
        assert x.shape[-1] == self.embedding_dim, (
            f"Input dimension mismatch: expected {self.embedding_dim}, got {x.shape[-1]}"
        )

        qid_str = str(question_id)

        # Ensure input tensor is on the same device as model
        model_device = next(self.parameters()).device
        assert x.device.type == model_device.type, (
            f"Input device type mismatch: model on {model_device.type}, input on {x.device.type}"
        )

        # Shared feature extraction
        shared_features = self.shared_trunk(x)

        # Multi-head outputs
        outputs = {}

        # Correctness prediction (always present)
        outputs["correctness"] = self.correctness_head(shared_features)

        # Misconception predictions (if question has misconceptions)
        if qid_str in self.misconception_heads:
            outputs["misconceptions"] = self.misconception_heads[qid_str](shared_features)

        # Ensure all outputs are on correct device
        for key, tensor in outputs.items():
            assert tensor.device.type == model_device.type, (
                f"{key} device type mismatch: expected {model_device.type}, got {tensor.device.type}"
            )

        logger.debug(
            "Forward pass completed",
            question_id=question_id,
            input_shape=list(x.shape),
            output_heads=list(outputs.keys()),
            shared_features_shape=list(shared_features.shape),
        )

        return outputs

    @property
    def device(self) -> torch.device:
        """Get the device this model is on."""
        return next(self.parameters()).device

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension this model expects."""
        return self.embedding_dim
