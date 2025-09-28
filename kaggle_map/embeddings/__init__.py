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

    # NOTE: We read the real embedding dimensionalities from the model specs
    # rather than the cache artefacts. Qwen3-Embedding-8B (via llama.cpp) yields
    # 4096-d vectors, while EmbeddingGemma-300M produces 768-d vectors. The
    # DOUBLE_BLIND strategy concatenates two embeddings, effectively doubling
    # the model-specific dimensionality before we append the correctness head.
    base_dim = 4096 if model == EmbeddingModel.QWEN else 768
    return base_dim * 2 if strategy == EmbeddingStrategy.DOUBLE_BLIND else base_dim


def get_model(model: EmbeddingModel) -> "QwenEmbeddingModel | GemmaEmbeddingModel":
    if model == EmbeddingModel.QWEN:
        try:
            from kaggle_map.embeddings.qwen import QwenEmbeddingModel  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - defensive guard
            msg = (
                "Failed to import QwenEmbeddingModel. Remove stale artefacts with "
                "`rm -rf .cache/embeddings` and reinstall embedding extras via "
                "`uv pip install -e .[embeddings]`."
            )
            raise AssertionError(msg) from exc

        return QwenEmbeddingModel.get_instance()

    try:
        from kaggle_map.embeddings.gemma import GemmaEmbeddingModel  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - defensive guard
        msg = (
            "Failed to import GemmaEmbeddingModel. Remove stale artefacts with "
            "`rm -rf .cache/embeddings` and reinstall embedding extras via "
            "`uv pip install -e .[embeddings]`."
        )
        raise AssertionError(msg) from exc

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
    model: EmbeddingModel,
    strategy: EmbeddingStrategy,
    row: EvaluationRow | Sequence[EvaluationRow],
) -> torch.Tensor:
    """Encode evaluation row(s) using the requested embedding model and strategy.

    Args:
        model: Logical embedding model to load (e.g. Qwen, Gemma).
        strategy: Embedding composition strategy to apply.
        row: Single evaluation row or an iterable of rows to embed.

    Returns:
        torch.Tensor: Embedding tensor (1D for a single row, 2D for batches).
    """
    assert model in EmbeddingModel, f"Invalid embedding model: {model}"
    assert strategy in EmbeddingStrategy, f"Invalid embedding strategy: {strategy}"

    model_instance = get_model(model)

    if isinstance(row, EvaluationRow):
        return _encode_single(row, strategy, model_instance)

    if isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)):
        rows = row if isinstance(row, list) else list(row)
        return _encode_batch(rows, strategy, model_instance)

    msg = f"Expected EvaluationRow or Sequence[EvaluationRow], got {type(row)}"
    raise TypeError(msg)
