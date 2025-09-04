"""Utilities for computing embeddings with consistent concatenation approach.

This module provides functions to compute embeddings using the standardized approach:
- Separate encoding of questions and answers
- Concatenation to create 768-dimensional vectors (2x base dimension)
- Optimized batch processing for GPU utilization
"""

from typing import Any

import numpy as np
from loguru import logger

from kaggle_map.core.embeddings.embedding_models import EmbeddingModel, get_tokenizer
from kaggle_map.utils.device import get_device

# Constants for batch sizing
SMALL_MODEL_THRESHOLD = 384
GPU_BATCH_SIZE_SMALL = 64
GPU_BATCH_SIZE_LARGE = 32
CPU_BATCH_SIZE_SMALL = 32
CPU_BATCH_SIZE_LARGE = 16


def compute_concatenated_embeddings(
    training_data: list[Any],
    embedding_model_name: str = "MINI_LM",
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Compute concatenated embeddings (question + answer) for training data.

    This is the standardized approach used throughout the system for consistent
    768-dimensional embeddings regardless of the base model dimension.

    Args:
        training_data: List of training rows with question_text, mc_answer,
                      student_explanation, question_id, and prediction attributes
        embedding_model_name: Name of the embedding model (default: "MINI_LM")
        device: Device to use for computation (default: auto-detect)

    Returns:
        Tuple of (concatenated_embeddings, question_ids, extra_data)
        - concatenated_embeddings: np.ndarray of shape (n_samples, 2 * base_dim)
        - question_ids: np.ndarray of question IDs
        - extra_data: dict with 'predictions' and 'mc_answers' arrays
    """
    if device is None:
        device = str(get_device())

    # Get the embedding model enum
    embedding_model = getattr(EmbeddingModel, embedding_model_name)
    logger.info(
        f"Computing concatenated embeddings with {embedding_model_name} "
        f"(base_dim={embedding_model.base_dim}, final_dim={embedding_model.dim}) on device: {device}"
    )

    tokenizer = get_tokenizer(model=embedding_model, device=device)

    # Prepare texts for batch encoding
    question_texts = []
    answer_texts = []
    question_ids_list = []
    predictions_list: list[str] = []
    mc_answers_list = []

    for row in training_data:
        question_texts.append(row.question_text)
        answer_texts.append(f"Answer: {row.mc_answer}; Explanation: {row.student_explanation}")
        question_ids_list.append(row.question_id)
        predictions_list.append(str(row.prediction))
        mc_answers_list.append(row.mc_answer)

    # Batch encode all texts at once for better GPU utilization
    logger.info(f"Batch encoding {len(question_texts)} questions and answers...")

    # Adjust batch size based on embedding dimensions to avoid OOM
    batch_size = (
        (GPU_BATCH_SIZE_LARGE if device != "cpu" else CPU_BATCH_SIZE_LARGE)
        if embedding_model.base_dim > SMALL_MODEL_THRESHOLD
        else (GPU_BATCH_SIZE_SMALL if device != "cpu" else CPU_BATCH_SIZE_SMALL)
    )

    # Encode questions in batches
    question_embeddings = tokenizer.encode(question_texts, batch_size=batch_size, show_progress_bar=True)

    # Encode answers in batches
    answer_embeddings = tokenizer.encode(answer_texts, batch_size=batch_size, show_progress_bar=True)

    # Concatenate question and answer embeddings
    combined_embeddings = np.concatenate([question_embeddings, answer_embeddings], axis=1)
    logger.info(f"Computed concatenated embeddings with shape: {combined_embeddings.shape}")

    return (
        combined_embeddings,
        np.array(question_ids_list),
        {"predictions": np.array(predictions_list), "mc_answers": np.array(mc_answers_list)},
    )


def compute_single_embeddings(
    training_data: list[Any],
    embedding_model_name: str = "MINI_LM",
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Compute single embeddings using combined text (legacy approach).

    This function is provided for backward compatibility but is not recommended
    for new code. Use compute_concatenated_embeddings instead.

    Args:
        training_data: List of training rows
        embedding_model_name: Name of the embedding model (default: "MINI_LM")
        device: Device to use for computation (default: auto-detect)

    Returns:
        Tuple of (embeddings, question_ids, extra_data)
        - embeddings: np.ndarray of shape (n_samples, base_dim)
        - question_ids: np.ndarray of question IDs
        - extra_data: dict with 'predictions' and 'mc_answers' arrays
    """
    if device is None:
        device = str(get_device())

    # Get the embedding model enum
    embedding_model = getattr(EmbeddingModel, embedding_model_name)
    logger.info(
        f"Computing single embeddings with {embedding_model_name} (dim={embedding_model.base_dim}) on device: {device}"
    )

    tokenizer = get_tokenizer(model=embedding_model, device=device)

    # Prepare combined texts
    combined_texts = []
    question_ids_list = []
    predictions_list: list[str] = []
    mc_answers_list = []

    for row in training_data:
        combined_texts.append(row.to_embedding_text())
        question_ids_list.append(row.question_id)
        predictions_list.append(str(row.prediction))
        mc_answers_list.append(row.mc_answer)

    # Batch encode
    batch_size = GPU_BATCH_SIZE_SMALL if device != "cpu" else CPU_BATCH_SIZE_SMALL
    embeddings = tokenizer.encode(combined_texts, batch_size=batch_size, show_progress_bar=True)

    logger.info(f"Computed single embeddings with shape: {embeddings.shape}")

    return (
        embeddings,
        np.array(question_ids_list),
        {"predictions": np.array(predictions_list), "mc_answers": np.array(mc_answers_list)},
    )
