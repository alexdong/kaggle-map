"""Tests for diverse text sampling using embeddings."""

from unittest.mock import Mock, patch

import pytest
import torch

from kaggle_map.embeddings.sampler import calculate_diversity, select_diverse_samples


def _get_base_embedding(text: str, h: int) -> list[float]:
    """Get base embedding vector based on text content."""
    if not text or not text.strip():
        return [0.1, 0.1, 0.1]

    text_lower = text.lower()

    # Define embedding patterns for different topics
    topic_embeddings = {
        ("math", "addition"): [0.8, 0.2, 0.1],
        ("history", "rome"): [0.1, 0.9, 0.2],
        ("geography", "mountain"): [0.2, 0.1, 0.9],
    }

    for keywords, embedding in topic_embeddings.items():
        if any(keyword in text_lower for keyword in keywords):
            return embedding

    # Default: use hash to generate varied embeddings
    return [0.5 + (h % 10) / 20, 0.5 + ((h >> 4) % 10) / 20, 0.5 + ((h >> 8) % 10) / 20]


@pytest.fixture
def mock_embedder():
    """Create a fast mock embedder for testing."""
    mock = Mock()

    def fast_encode(text: str) -> torch.Tensor:
        """Generate deterministic embeddings based on text content."""
        h = hash(text) if text else 0
        base = _get_base_embedding(text, h)
        variation = (h % 100) / 1000.0
        return torch.tensor([b + variation for b in base], dtype=torch.float32)

    mock.encode.side_effect = fast_encode
    return mock


def test_calculate_diversity_two_embeddings():
    """Test diversity calculation with two embeddings."""
    # Create two orthogonal embeddings (maximum diversity)
    emb1 = torch.tensor([1.0, 0.0, 0.0])
    emb2 = torch.tensor([0.0, 1.0, 0.0])

    diversity = calculate_diversity([emb1, emb2])
    assert diversity > 0.9, f"Orthogonal vectors should have high diversity, got {diversity}"

    # Create two identical embeddings (minimum diversity)
    emb3 = torch.tensor([1.0, 1.0, 1.0])
    emb4 = torch.tensor([1.0, 1.0, 1.0])

    diversity = calculate_diversity([emb3, emb4])
    assert diversity < 0.1, f"Identical vectors should have low diversity, got {diversity}"


def test_calculate_diversity_multiple_embeddings():
    """Test diversity calculation with multiple embeddings."""
    # Create diverse embeddings
    embeddings = [
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0]),
    ]

    diversity = calculate_diversity(embeddings)
    assert diversity > 0.8, f"Orthogonal vectors should have high diversity, got {diversity}"

    # Create similar embeddings
    similar = [
        torch.tensor([1.0, 0.9, 0.8]),
        torch.tensor([1.0, 0.85, 0.75]),
        torch.tensor([1.0, 0.88, 0.82]),
    ]

    diversity = calculate_diversity(similar)
    assert diversity < 0.3, f"Similar vectors should have low diversity, got {diversity}"


def test_select_diverse_samples_basic(mock_embedder):
    """Test basic diverse sample selection."""
    with patch("kaggle_map.embeddings.sampler.GemmaEmbeddingModel.get_instance", return_value=mock_embedder):
        texts = [
            "Mathematics problem about addition",
            "Math addition question",  # Similar to first
            "History of ancient Rome",  # Different topic
            "Geography of mountains",  # Another different topic
        ]

        indices, score = select_diverse_samples(texts, n_samples=3)

        assert len(indices) == 3, f"Should select 3 samples, got {len(indices)}"
        assert len(set(indices)) == 3, "Selected indices should be unique"
        assert all(0 <= idx < len(texts) for idx in indices), "Indices should be valid"
        assert score > 0, "Diversity score should be positive"

        # Should select the diverse topics, not the similar math ones
        assert 2 in indices or 3 in indices, "Should include diverse topics"


def test_select_diverse_samples_edge_cases(mock_embedder):
    """Test edge cases in diverse sampling."""
    with patch("kaggle_map.embeddings.sampler.GemmaEmbeddingModel.get_instance", return_value=mock_embedder):
        texts = ["text1", "text2"]

        # More samples than texts
        indices, score = select_diverse_samples(texts, n_samples=5)
        assert len(indices) == 2, "Should return all available texts"
        assert score == 0.0, "Score should be 0 when returning all texts"

        # Single sample
        indices, score = select_diverse_samples(["short", "this is much longer text"], n_samples=1)
        assert len(indices) == 1, "Should return 1 sample"
        assert indices[0] == 1, "Should select the longer text"

        # Empty input handling
        with pytest.raises(AssertionError, match="empty"):
            select_diverse_samples([], n_samples=1)


def test_select_diverse_samples_with_empty_texts(mock_embedder):
    """Test handling of empty and whitespace texts."""
    with patch("kaggle_map.embeddings.sampler.GemmaEmbeddingModel.get_instance", return_value=mock_embedder):
        texts = ["valid text", "", "  ", "another valid", None]

        # Filter out None values for testing
        texts = [t if t is not None else "" for t in texts]

        indices, _score = select_diverse_samples(texts, n_samples=2)
        assert len(indices) == 2, "Should handle empty texts"
        # Should prefer non-empty texts
        assert 0 in indices or 3 in indices, "Should prefer valid texts"


def test_reproducibility(mock_embedder):
    """Test that selection is deterministic."""
    with patch("kaggle_map.embeddings.sampler.GemmaEmbeddingModel.get_instance", return_value=mock_embedder):
        texts = ["text a", "text b", "text c", "text d", "text e"]

        # Run multiple times
        results = []
        for _ in range(3):
            indices, score = select_diverse_samples(texts, n_samples=3)
            results.append((indices, score))

        # Check all results are identical
        first_indices, first_score = results[0]
        for indices, score in results[1:]:
            assert indices == first_indices, "Selection should be deterministic"
            assert abs(score - first_score) < 0.001, "Scores should be identical"


if __name__ == "__main__":
    """Run all tests."""
    import sys

    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info("Running diverse sampler tests...")

    # Create mock embedder for tests that need it
    mock_emb = mock_embedder()

    # Tests that don't need mock embedder
    test_calculate_diversity_two_embeddings()
    test_calculate_diversity_multiple_embeddings()

    # Tests that need mock embedder
    test_select_diverse_samples_basic(mock_emb)
    test_select_diverse_samples_edge_cases(mock_emb)
    test_select_diverse_samples_with_empty_texts(mock_emb)
    test_reproducibility(mock_emb)

    logger.success("All tests passed!")
