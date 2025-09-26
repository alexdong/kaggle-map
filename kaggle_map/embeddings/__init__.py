"""Utilities for computing embeddings for MLP training."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from kaggle_map.core.models import EmbeddingModel, EmbeddingStrategy, EvaluationRow

if TYPE_CHECKING:
    from kaggle_map.embeddings.gemma import GemmaEmbeddingModel
    from kaggle_map.embeddings.qwen import QwenEmbeddingModel


def get_input_embeddings_dimension(strategy: EmbeddingStrategy, model: EmbeddingModel) -> int:
    """Return embedding dimensionality for the given model and strategy."""

    # DOUBLE_BLIND concatenates two embeddings, GOAL_DRIVEN uses single
    # QWEN produces 8192-dim embeddings, GEMMA produces 768-dim embeddings
    base_dim = 8192 if model == EmbeddingModel.QWEN else 768
    return base_dim * 2 if strategy == EmbeddingStrategy.DOUBLE_BLIND else base_dim


def get_model(model: EmbeddingModel) -> "QwenEmbeddingModel | GemmaEmbeddingModel":
    if model == EmbeddingModel.QWEN:
        from kaggle_map.embeddings.qwen import QwenEmbeddingModel

        return QwenEmbeddingModel.get_instance()

    from kaggle_map.embeddings.gemma import GemmaEmbeddingModel

    return GemmaEmbeddingModel.get_instance()


def _encode_single(
    row: EvaluationRow, strategy: EmbeddingStrategy, model_instance: "GemmaEmbeddingModel | QwenEmbeddingModel"
) -> torch.Tensor:
    """Encode a single evaluation row."""
    assert row.correct_answer is not None, "Correct answer is required for embeddings"

    if strategy == EmbeddingStrategy.DOUBLE_BLIND:
        # DOUBLE_BLIND: Create two separate embeddings and concatenate
        question_correct_text = f"Question: {row.question_text}\nCorrect Answer: {row.correct_answer}"
        answer_explanation_text = f"Student Answer: {row.mc_answer}\nStudent Explanation: {row.student_explanation}"

        question_correct_embedding = model_instance.encode(question_correct_text)
        answer_explanation_embedding = model_instance.encode(answer_explanation_text)

        return torch.cat([question_correct_embedding, answer_explanation_embedding])

    if strategy == EmbeddingStrategy.GOAL_DRIVEN:
        # GOAL_DRIVEN uses unified approach
        unified_text = (
            f"Question: {row.question_text}\n"
            f"Correct Answer: {row.correct_answer}\n"
            f"Student Answer: {row.mc_answer}\n"
            f"Student Explanation: {row.student_explanation}"
        )
        return model_instance.encode(unified_text)

    msg = f"Unsupported strategy: {strategy}"
    raise ValueError(msg)


def _encode_batch(
    rows: Sequence[EvaluationRow],
    strategy: EmbeddingStrategy,
    model_instance: "GemmaEmbeddingModel | QwenEmbeddingModel",
) -> torch.Tensor:
    """Encode a batch of evaluation rows."""
    row_list = rows if isinstance(rows, list) else list(rows)

    if not row_list:
        # Use QWEN as default for empty batches
        expected_dim = get_input_embeddings_dimension(strategy, EmbeddingModel.QWEN)
        return torch.zeros(0, expected_dim)

    # Validate all rows have correct_answer
    for row in row_list:
        assert row.correct_answer is not None, "Correct answer is required for embeddings"

    if strategy == EmbeddingStrategy.DOUBLE_BLIND:
        # DOUBLE_BLIND: Create two separate embeddings for each row and concatenate
        question_correct_texts = [
            f"Question: {row.question_text}\nCorrect Answer: {row.correct_answer}" for row in row_list
        ]
        answer_explanation_texts = [
            f"Student Answer: {row.mc_answer}\nStudent Explanation: {row.student_explanation}" for row in row_list
        ]

        question_correct_embeddings = model_instance.encode(question_correct_texts)
        answer_explanation_embeddings = model_instance.encode(answer_explanation_texts)

        return torch.cat([question_correct_embeddings, answer_explanation_embeddings], dim=1)

    if strategy == EmbeddingStrategy.GOAL_DRIVEN:
        # GOAL_DRIVEN uses unified approach
        texts = [
            f"Question: {row.question_text}\n"
            f"Correct Answer: {row.correct_answer}\n"
            f"Student Answer: {row.mc_answer}\n"
            f"Student Explanation: {row.student_explanation}"
            for row in row_list
        ]
        return model_instance.encode(texts)

    msg = f"Unsupported strategy: {strategy}"
    raise ValueError(msg)


def encode(
    row: EvaluationRow | Sequence[EvaluationRow],
    strategy: EmbeddingStrategy,
    model: EmbeddingModel,
) -> torch.Tensor:
    """Encode evaluation row(s) into embeddings.

    Args:
        row: Single EvaluationRow or list of EvaluationRows
        strategy: Embedding strategy to use
        model: Embedding model to use

    Returns:
        torch.Tensor: Embeddings (1D for single row, 2D for batch)
    """
    assert strategy in EmbeddingStrategy, f"Invalid embedding strategy: {strategy}"
    assert model in EmbeddingModel, f"Invalid embedding model: {model}"

    if isinstance(row, EvaluationRow):
        assert row.correct_answer is not None, "Correct answer is required for embeddings"
        model_instance = get_model(model)
        return _encode_single(row, strategy, model_instance)
    if isinstance(row, Sequence):
        row_sequence = row if isinstance(row, list) else list(row)
        for evaluation_row in row_sequence:
            assert evaluation_row.correct_answer is not None, "Correct answer is required for embeddings"

        model_instance = get_model(model)
        return _encode_batch(row_sequence, strategy, model_instance)
    msg = f"Expected EvaluationRow or Sequence[EvaluationRow], got {type(row)}"
    raise TypeError(msg)
