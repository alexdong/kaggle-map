"""Utilities for computing embeddings for MLP training."""

from collections.abc import Sequence

import torch

from kaggle_map.core.models import EmbeddingModel, EmbeddingStrategy, EvaluationRow
from kaggle_map.embeddings.gemma import GemmaEmbeddingModel
from kaggle_map.embeddings.qwen import QwenEmbeddingModel


def get_input_embeddings_dimension(model: EmbeddingModel, strategy: EmbeddingStrategy) -> int:
    """Return embedding dimensionality for the given model and strategy."""

    # NOTE: We read the real embedding dimensionalities from the model specs
    # rather than the cache artefacts. Qwen3-Embedding-8B (via llama.cpp) yields
    # 4096-d vectors, while EmbeddingGemma-300M produces 768-d vectors. The
    # DOUBLE_BLIND strategy concatenates two embeddings, effectively doubling
    # the model-specific dimensionality before we append the correctness head.
    base_dim = 4096 if model == EmbeddingModel.QWEN else 768
    return base_dim * 2 if strategy == EmbeddingStrategy.DOUBLE_BLIND else base_dim


def encode(
    model: EmbeddingModel,
    strategy: EmbeddingStrategy,
    rows: Sequence[EvaluationRow],
) -> torch.Tensor:
    assert model in EmbeddingModel, f"Invalid embedding model: {model}"
    assert strategy in EmbeddingStrategy, f"Invalid embedding strategy: {strategy}"
    model_instance = (
        QwenEmbeddingModel.get_instance() if model == EmbeddingModel.QWEN else GemmaEmbeddingModel.get_instance()
    )

    for row in rows:
        assert row.correct_answer is not None, "Correct answer is required for embeddings"

    if strategy == EmbeddingStrategy.DOUBLE_BLIND:
        # DOUBLE_BLIND: Create two separate embeddings for each row and concatenate
        question_correct_texts = [
            f"Question: {row.question_text}\nCorrect Answer: {row.correct_answer}" for row in rows
        ]
        answer_explanation_texts = [
            f"Student Answer: {row.mc_answer}\nStudent Explanation: {row.student_explanation}" for row in rows
        ]

        question_correct_embeddings = model_instance.encode(question_correct_texts)
        answer_explanation_embeddings = model_instance.encode(answer_explanation_texts)

        return torch.cat([question_correct_embeddings, answer_explanation_embeddings], dim=1)

    # GOAL_DRIVEN uses unified approach
    texts = [
        f"Question: {row.question_text}\n"
        f"Correct Answer: {row.correct_answer}\n"
        f"Student Answer: {row.mc_answer}\n"
        f"Student Explanation: {row.student_explanation}"
        for row in rows
    ]
    return model_instance.encode(texts)


if __name__ == "__main__":
    test_texts = [
        EvaluationRow(
            row_id="1",
            question_id="2",
            question_text="What is 2 + 2?",
            mc_answer="4",
            student_explanation="It's basic arithmetic.",
            correct_answer="4",
        ),
    ]
    embeddings = encode(EmbeddingModel.QWEN, EmbeddingStrategy.DOUBLE_BLIND, test_texts)
    print("Qwen embeddings shape:", embeddings.shape)
    print(
        "Embedding dimension should be: ",
        get_input_embeddings_dimension(EmbeddingModel.QWEN, EmbeddingStrategy.DOUBLE_BLIND),
    )
