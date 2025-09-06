"""Utilities for computing embeddings with Qwen3-Embedding-8B.

This module provides functions to compute embeddings using the Qwen3-8B model
with Q8_0 quantization for efficient processing.

  1. Double Blind Embedding Strategy (compute_double_blind_strategy_embeddings)

  - Creates two separate embeddings:
    - Embedding 1: (question, correct_answer) - 4096 dims
    - Embedding 2: (mc_answer, student_explanation) - 4096 dims
  - Returns concatenated 8192-dimensional embeddings

  2. Semantic Injection Embedding Strategy (compute_semantic_strategy_embeddings)

  - Creates one unified embedding containing all components:
    - (question, correct_answer, mc_answer, student_explanation)
  - Returns 4096-dimensional embeddings
"""

from typing import TYPE_CHECKING

import numpy as np

from kaggle_map.embeddings.qwen import QwenEmbeddingModel

if TYPE_CHECKING:
    from kaggle_map.core.models import TrainingRow


def compute_double_blind_strategy_embeddings(row: "TrainingRow") -> np.ndarray:
    model = QwenEmbeddingModel.get_instance()
    assert row.correct_answer is not None, "Correct answer is required for double blind embeddings"

    correct_answer = row.correct_answer
    question_correct_text = f"Question: {row.question_text}\nCorrect Answer: {correct_answer}"
    answer_explanation_text = f"Student Answer: {row.mc_answer}\nStudent Explanation: {row.student_explanation}"
    question_correct_embedding = model.encode(question_correct_text)
    answer_explanation_embedding = model.encode(answer_explanation_text)
    return np.concatenate([question_correct_embedding, answer_explanation_embedding])


def compute_semantic_strategy_embedding(row: "TrainingRow") -> np.ndarray:
    model = QwenEmbeddingModel.get_instance()
    assert row.correct_answer is not None, "Correct answer is required for double blind embeddings"
    correct_answer = row.correct_answer if row.correct_answer else row.mc_answer
    unified_text = (
        f"Question: {row.question_text}\n"
        f"Correct Answer: {correct_answer}\n"
        f"Student Answer: {row.mc_answer}\n"
        f"Student Explanation: {row.student_explanation}"
    )
    return model.encode(unified_text)


__all__ = [
    "compute_double_blind_strategy_embeddings",
    "compute_semantic_strategy_embedding",
]
