"""Diverse text sampling using embeddings for maximum variety selection."""

from collections.abc import Sequence

import torch
from loguru import logger
from torch.nn.functional import cosine_similarity

from kaggle_map.embeddings.gemma import GemmaEmbeddingModel
from kaggle_map.utils.logger_config import configure_logger

configure_logger(__name__)

# Constants
MIN_EMBEDDINGS_FOR_DIVERSITY = 2
MAX_DIVERSITY_RANGE = 2
MAX_SHORT_TEXT_LENGTH = 200


def calculate_diversity(embeddings: Sequence[torch.Tensor]) -> float:
    """Calculate average pairwise cosine distance as diversity score.

    Args:
        embeddings: List of embedding tensors

    Returns:
        Average pairwise distance (0-2 range, higher = more diverse)
    """
    assert len(embeddings) >= MIN_EMBEDDINGS_FOR_DIVERSITY, (
        f"Need at least {MIN_EMBEDDINGS_FOR_DIVERSITY} embeddings, got {len(embeddings)}"
    )

    embedding_list = embeddings if isinstance(embeddings, list) else list(embeddings)

    n = len(embedding_list)
    logger.debug(f"Calculating diversity for {n} embeddings")

    if n == MIN_EMBEDDINGS_FOR_DIVERSITY:
        # Simple case: just two embeddings
        distance = 1 - cosine_similarity(embedding_list[0], embedding_list[1], dim=0).item()
        logger.debug(f"  Distance between 2 embeddings: {distance:.3f}")
        return distance

    # Calculate all pairwise distances
    distances: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embedding_list[i], embedding_list[j], dim=0).item()
            dist: float = 1 - sim
            distances.append(dist)
            logger.debug(f"  Distance [{i},{j}]: {dist:.3f}")

    avg_distance = sum(distances) / len(distances)
    logger.debug(f"  Average pairwise distance: {avg_distance:.3f} from {len(distances)} pairs")

    return avg_distance


def select_diverse_samples(  # noqa: C901
    texts: Sequence[str],
    n_samples: int = 3,
) -> tuple[list[int], float]:
    """Select n most diverse text samples using greedy selection.

    Uses a greedy algorithm that iteratively selects the text that
    maximizes the minimum distance to already selected texts.

    Args:
        texts: List of text samples to select from
        n_samples: Number of diverse samples to select

    Returns:
        (selected_indices, diversity_score)
        - selected_indices: Indices of selected texts
        - diversity_score: Average pairwise distance (0-2)
    """
    text_list = texts if isinstance(texts, list) else list(texts)

    assert text_list, "Cannot select from empty text sequence"
    assert n_samples > 0, f"n_samples must be positive, got {n_samples}"

    logger.info(f"Selecting {n_samples} diverse samples from {len(text_list)} texts")

    # Handle edge cases
    if len(text_list) <= n_samples:
        logger.debug(f"  Texts ({len(text_list)}) <= n_samples ({n_samples}), returning all")
        return list(range(len(text_list))), 0.0

    if n_samples == 1:
        # Just return the longest text for richness
        longest_idx = max(range(len(text_list)), key=lambda i: len(text_list[i]))
        logger.debug(f"  Single sample requested, returning longest text at index {longest_idx}")
        return [longest_idx], 0.0

    # Get embedding model singleton
    logger.debug("Loading embedding model")
    model = GemmaEmbeddingModel.get_instance()

    # Encode all texts
    logger.debug(f"Encoding {len(text_list)} texts")
    embeddings = []
    for i, text in enumerate(text_list):
        truncated = text[:MAX_SHORT_TEXT_LENGTH] if len(text) > MAX_SHORT_TEXT_LENGTH else text
        logger.debug(f"  Encoding text {i}: '{truncated}...'")
        embeddings.append(model.encode(text))

    logger.debug(f"  Embedding dimensions: {embeddings[0].shape}")

    # Greedy selection
    selected = []

    # Start with longest text (most informative)
    first_idx = max(range(len(text_list)), key=lambda i: len(text_list[i]))
    selected.append(first_idx)
    logger.debug(f"  Selected first (longest): index {first_idx}, length {len(text_list[first_idx])}")

    # Select remaining samples
    for round_num in range(n_samples - 1):
        logger.debug(f"  Selection round {round_num + 2}/{n_samples}")

        max_min_dist = -1
        best_idx = -1

        for candidate_idx in range(len(embeddings)):
            if candidate_idx in selected:
                continue

            # Find minimum distance to already selected embeddings
            min_dist_to_selected = float("inf")
            for selected_idx in selected:
                sim = cosine_similarity(embeddings[candidate_idx], embeddings[selected_idx], dim=0).item()
                dist = 1 - sim
                min_dist_to_selected = min(min_dist_to_selected, dist)

            # Track the candidate with maximum minimum distance
            if min_dist_to_selected > max_min_dist:
                max_min_dist = min_dist_to_selected
                best_idx = candidate_idx

        selected.append(best_idx)
        logger.debug(f"    Selected index {best_idx} with min distance {max_min_dist:.3f}")

    # Calculate final diversity score
    selected_embeddings = [embeddings[i] for i in selected]
    diversity_score = calculate_diversity(selected_embeddings)

    logger.success(f"Selected {len(selected)} diverse samples with diversity score {diversity_score:.3f}")

    return selected, diversity_score


if __name__ == "__main__":
    """Standalone validation of diverse sampling."""

    logger.info("=== Diverse Sampler Module Validation ===")

    # Test texts with varying similarity
    test_texts = [
        "The student multiplied 3 by 2 and got 5",  # Math error
        "Student thinks 3 times 2 equals 5",  # Similar to above
        "The answer is wrong because they added instead of multiplying",  # Different explanation
        "They confused multiplication with addition",  # Similar to above
        "The student doesn't understand fractions",  # Different topic
        "Fractions are hard for this student",  # Similar to above
        "The student made a sign error",  # Different error type
        "Negative numbers confuse the student",  # Related to above
    ]
    logger.info(f"\nTest data: {len(test_texts)} texts")
    for i, text in enumerate(test_texts):
        logger.info(f"  {i}: {text[:50]}...")

    # Test 1: Select 3 diverse samples
    logger.info("\n1. Testing diverse selection (n=3):")
    indices, score = select_diverse_samples(test_texts, n_samples=3)
    logger.info(f"  Selected indices: {indices}")
    logger.info(f"  Diversity score: {score:.3f}")
    logger.info("  Selected texts:")
    for idx in indices:
        logger.info(f"    [{idx}]: {test_texts[idx]}")
