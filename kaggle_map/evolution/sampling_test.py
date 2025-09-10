"""Tests for balanced sampling functionality."""

import pandas as pd

from kaggle_map.evolution.sampling import stratified_sample


def test_stratified_sample_basic() -> None:
    """Test basic stratified sampling functionality."""
    # Create test data with multiple strata
    data = pd.DataFrame(
        {
            "QuestionId": [1, 1, 1, 2, 2, 2, 3, 3, 3] * 3,
            "Category": ["A", "B", "C"] * 9,
            "MC_Answer": ["X", "Y", "Z"] * 9,
            "value": range(27),
        }
    )

    # Sample 10% of data
    sampled = stratified_sample(data, sample_ratio=0.1, random_seed=42)

    # Should have approximately 10% of original size
    assert 2 <= len(sampled) <= 4  # Allow some variance due to rounding

    # All original columns should be present
    assert list(sampled.columns) == list(data.columns)


def test_stratified_sample_preserves_distribution() -> None:
    """Test that sampling preserves category distribution."""
    # Create imbalanced data
    data = pd.DataFrame(
        {
            "QuestionId": [1] * 100 + [2] * 20 + [3] * 10,
            "Category": ["Correct"] * 100 + ["Misconception"] * 20 + ["Neither"] * 10,
            "MC_Answer": ["A"] * 130,
            "value": range(130),
        }
    )

    # Sample 20% of data
    sampled = stratified_sample(data, sample_ratio=0.2, random_seed=42)

    # Check that rare categories are included
    assert "Neither" in sampled["Category"].to_numpy()
    assert "Misconception" in sampled["Category"].to_numpy()
    assert "Correct" in sampled["Category"].to_numpy()


def test_stratified_sample_min_samples() -> None:
    """Test minimum samples per stratum."""
    # Create data with rare combinations
    data = pd.DataFrame(
        {
            "QuestionId": [1] * 50 + [2] * 2 + [3] * 1,
            "Category": ["Common"] * 50 + ["Rare"] * 2 + ["VeryRare"] * 1,
            "MC_Answer": ["A"] * 53,
            "value": range(53),
        }
    )

    # Sample with min_samples_per_stratum
    sampled = stratified_sample(data, sample_ratio=0.1, min_samples_per_stratum=2, random_seed=42)

    # Since we allow oversampling to preserve rare strata, check that small strata are preserved
    # Rare category should have at least 1 sample
    rare_count = len(sampled[sampled["Category"] == "Rare"])
    assert rare_count >= 1  # Should have at least 1 of the 2 rare samples

    # Very rare should include its only sample
    very_rare_count = len(sampled[sampled["Category"] == "VeryRare"])
    assert very_rare_count == 1  # Should include the single very rare sample

    # Total sample size should be reasonable (allow some oversampling)
    assert 5 <= len(sampled) <= 10  # Target was 5, allow up to 10 for rare preservation


def test_stratified_sample_reproducible() -> None:
    """Test that sampling with same seed produces same results."""
    data = pd.DataFrame(
        {
            "QuestionId": list(range(100)),
            "Category": ["A", "B", "C", "D"] * 25,
            "MC_Answer": ["X", "Y"] * 50,
            "value": range(100),
        }
    )

    # Sample twice with same seed
    sample1 = stratified_sample(data, sample_ratio=0.2, random_seed=123)
    sample2 = stratified_sample(data, sample_ratio=0.2, random_seed=123)

    # Should be identical
    pd.testing.assert_frame_equal(sample1, sample2)

    # Sample with different seed should be different
    sample3 = stratified_sample(data, sample_ratio=0.2, random_seed=456)
    assert not sample1.equals(sample3)


def test_stratified_sample_handles_empty_strata() -> None:
    """Test handling of empty strata after filtering."""
    data = pd.DataFrame(
        {
            "QuestionId": [1, 2, 3],
            "Category": ["A", "B", "C"],
            "MC_Answer": ["X", "Y", "Z"],
            "value": [1, 2, 3],
        }
    )

    # Sample 50% - each stratum has only 1 item
    sampled = stratified_sample(data, sample_ratio=0.5, random_seed=42)

    # Should handle single-item strata gracefully
    assert len(sampled) >= 1
    assert len(sampled) <= 3
