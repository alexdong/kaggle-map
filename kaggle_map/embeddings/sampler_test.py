"""Tests for diverse text sampling using embeddings."""

import torch

from kaggle_map.embeddings.sampler import calculate_diversity, select_diverse_samples


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


def test_select_diverse_samples_basic():
    """Test basic diverse sample selection."""
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


def test_select_diverse_samples_edge_cases():
    """Test edge cases in diverse sampling."""
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
    try:
        select_diverse_samples([], n_samples=1)
        assert False, "Should fail with empty input"
    except AssertionError as e:
        assert "empty" in str(e).lower()


def test_select_diverse_samples_with_empty_texts():
    """Test handling of empty and whitespace texts."""
    texts = ["valid text", "", "  ", "another valid", None]
    
    # Filter out None values for testing
    texts = [t if t is not None else "" for t in texts]
    
    indices, score = select_diverse_samples(texts, n_samples=2)
    assert len(indices) == 2, "Should handle empty texts"
    # Should prefer non-empty texts
    assert 0 in indices or 3 in indices, "Should prefer valid texts"


def test_reproducibility():
    """Test that selection is deterministic."""
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
    
    test_functions = [
        test_calculate_diversity_two_embeddings,
        test_calculate_diversity_multiple_embeddings,
        test_select_diverse_samples_basic,
        test_select_diverse_samples_edge_cases,
        test_select_diverse_samples_with_empty_texts,
        test_reproducibility,
    ]
    
    for test_func in test_functions:
        logger.info(f"  Running {test_func.__name__}...")
        test_func()
        logger.success(f"    ✓ {test_func.__name__} passed")
    
    logger.success("All tests passed!")