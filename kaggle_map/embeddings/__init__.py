"""Utilities for computing embeddings with Qwen3-Embedding-8B.

This module provides functions to compute embeddings using the Qwen3-8B model
with Q8_0 quantization for efficient processing.

  1. Double Blind Embedding Strategy (compute_double_blind_strategy_embeddings)

  - Creates two separate embeddings:
    - Embedding 1: (question, correct_answer) - 4096 dims
    - Embedding 2: (mc_answer, student_explanation) - 4096 dims
  - Returns concatenated 8192-dimensional torch.Tensor

  2. Semantic Injection Embedding Strategy (compute_semantic_strategy_embeddings)

  - Creates one unified embedding containing all components:
    - (question, correct_answer, mc_answer, student_explanation)
  - Returns 4096-dimensional torch.Tensor
"""

import torch

from kaggle_map.core.models import EmbeddingModel, EmbeddingStrategy, EvaluationRow
from kaggle_map.embeddings.gemma import GemmaEmbeddingModel
from kaggle_map.embeddings.qwen import QwenEmbeddingModel


def get_input_embeddings_dimension(strategy: EmbeddingStrategy, model: EmbeddingModel) -> int:
    base_dimension = 8192 if model == EmbeddingModel.QWEN else 768
    return base_dimension * 2 if strategy == EmbeddingStrategy.DOUBLE_BLIND else base_dimension


def get_model(model: EmbeddingModel) -> "QwenEmbeddingModel | GemmaEmbeddingModel":
    if model == EmbeddingModel.QWEN:
        return QwenEmbeddingModel.get_instance()
    return GemmaEmbeddingModel.get_instance()


def encode(row: EvaluationRow, strategy: EmbeddingStrategy, model: EmbeddingModel) -> torch.Tensor:
    assert strategy in EmbeddingStrategy, f"Invalid embedding strategy: {strategy}"
    assert model in EmbeddingModel, f"Invalid embedding model: {model}"
    assert row.correct_answer is not None, "Correct answer is required for double blind embeddings"

    if strategy == EmbeddingStrategy.SEMANTIC:
        unified_text = (
            f"Question: {row.question_text}\n"
            f"Correct Answer: {row.correct_answer}\n"
            f"Student Answer: {row.mc_answer}\n"
            f"Student Explanation: {row.student_explanation}"
        )
        return torch.Tensor(get_model(model).encode(unified_text))
    # DOUBLE_BLIND
    question_correct_text = f"Question: {row.question_text}\nCorrect Answer: {row.correct_answer}"
    answer_explanation_text = f"Student Answer: {row.mc_answer}\nStudent Explanation: {row.student_explanation}"
    question_correct_embedding = torch.Tensor(get_model(model).encode(question_correct_text))
    answer_explanation_embedding = torch.Tensor(get_model(model).encode(answer_explanation_text))
    return torch.cat([question_correct_embedding, answer_explanation_embedding])
