"""Utilities for computing embeddings with Qwen3-Embedding-8B.

This module provides functions to compute embeddings using the Qwen3-8B model
with Q8_0 quantization for efficient processing.
"""

from typing import Any

import numpy as np
from loguru import logger

from kaggle_map.embeddings.embedding_models import QwenEmbeddingModel
from kaggle_map.utils.device import get_device


def compute_concatenated_embeddings(
    training_data: list[Any],
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Compute embeddings for training data using Qwen3-8B.

    Args:
        training_data: List of training rows with question_text, mc_answer,
                      student_explanation, question_id, and prediction attributes
        device: Device to use for computation (default: auto-detect)

    Returns:
        Tuple of (embeddings, question_ids, extra_data)
        - embeddings: np.ndarray of shape (n_samples, 4096)
        - question_ids: np.ndarray of question IDs
        - extra_data: dict with 'predictions' and 'mc_answers' arrays
    """
    if device is None:
        device = str(get_device())

    logger.info(f"Computing embeddings with Qwen3-8B Q8_0 on device: {device}")

    tokenizer = QwenEmbeddingModel()

    # Prepare combined texts
    combined_texts = []
    question_ids_list = []
    predictions_list: list[str] = []
    mc_answers_list = []

    for row in training_data:
        # Combine question, answer, and explanation into single text
        combined_text = (
            f"Question: {row.question_text}\n"
            f"Answer: {row.mc_answer}\n"
            f"Explanation: {row.student_explanation}"
        )
        combined_texts.append(combined_text)
        question_ids_list.append(row.question_id)
        predictions_list.append(str(row.prediction))
        mc_answers_list.append(row.mc_answer)

    logger.info(f"Encoding {len(combined_texts)} samples...")

    # Process in batches for memory efficiency
    batch_size = 32  # Adjust based on available memory
    embeddings_list = []

    for i in range(0, len(combined_texts), batch_size):
        batch = combined_texts[i:i + batch_size]
        batch_embeddings = tokenizer.encode(batch)
        embeddings_list.append(batch_embeddings)

        if i % (batch_size * 10) == 0:
            logger.debug(f"Processed {i}/{len(combined_texts)} samples")

    # Combine all embeddings
    embeddings = np.vstack(embeddings_list)
    logger.info(f"Computed embeddings with shape: {embeddings.shape}")

    return (
        embeddings,
        np.array(question_ids_list),
        {"predictions": np.array(predictions_list), "mc_answers": np.array(mc_answers_list)},
    )


def compute_single_embeddings(
    training_data: list[Any],
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Compute single embeddings using combined text.

    This is now identical to compute_concatenated_embeddings since we're
    using a single model rather than concatenating separate embeddings.

    Args:
        training_data: List of training rows
        device: Device to use for computation (default: auto-detect)

    Returns:
        Tuple of (embeddings, question_ids, extra_data)
        - embeddings: np.ndarray of shape (n_samples, 4096)
        - question_ids: np.ndarray of question IDs
        - extra_data: dict with 'predictions' and 'mc_answers' arrays
    """
    return compute_concatenated_embeddings(training_data, device)

