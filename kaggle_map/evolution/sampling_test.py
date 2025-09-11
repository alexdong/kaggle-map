"""Tests for balanced sampling functionality."""

import logging

import pandas as pd
import pytest

from kaggle_map.evolution.sampling import stratified_sample

# Set debug logging for tests
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def basic_test_data() -> pd.DataFrame:
    """Create basic test data with multiple strata for testing."""
    return pd.DataFrame(
        {
            "QuestionId": [1, 1, 1, 2, 2, 2, 3, 3, 3] * 3,
            "Category": ["A", "B", "C"] * 9,
            "MC_Answer": ["X", "Y", "Z"] * 9,
            "value": range(27),
        }
    )


@pytest.fixture
def imbalanced_test_data() -> pd.DataFrame:
    """Create imbalanced test data for distribution testing."""
    return pd.DataFrame(
        {
            "QuestionId": [1] * 100 + [2] * 20 + [3] * 10,
            "Category": ["Correct"] * 100 + ["Misconception"] * 20 + ["Neither"] * 10,
            "MC_Answer": ["A"] * 130,
            "value": range(130),
        }
    )


@pytest.fixture
def rare_strata_data() -> pd.DataFrame:
    """Create test data with rare combinations."""
    return pd.DataFrame(
        {
            "QuestionId": [1] * 50 + [2] * 2 + [3] * 1,
            "Category": ["Common"] * 50 + ["Rare"] * 2 + ["VeryRare"] * 1,
            "MC_Answer": ["A"] * 53,
            "value": range(53),
        }
    )


def test_stratified_sample_basic(basic_test_data: pd.DataFrame) -> None:
    """Test basic stratified sampling functionality."""
    # Sample 10% of data
    sampled = stratified_sample(basic_test_data, sample_ratio=0.1, random_seed=42)

    # Should have approximately 10% of original size
    expected_min = 2
    expected_max = 4
    assert expected_min <= len(sampled) <= expected_max, (
        f"Sample size {len(sampled)} should be between {expected_min} and {expected_max} for 10% of {len(basic_test_data)} rows"
    )

    # All original columns should be present
    assert list(sampled.columns) == list(basic_test_data.columns), "Sampled data should preserve all original columns"


def test_stratified_sample_preserves_distribution(imbalanced_test_data: pd.DataFrame) -> None:
    """Test that sampling preserves category distribution."""
    # Sample 20% of data
    sampled = stratified_sample(imbalanced_test_data, sample_ratio=0.2, random_seed=42)

    # Check that rare categories are included
    sampled_categories = set(sampled["Category"].to_numpy())
    assert "Neither" in sampled_categories, "Rare 'Neither' category should be preserved in sample"
    assert "Misconception" in sampled_categories, "Minority 'Misconception' category should be preserved in sample"
    assert "Correct" in sampled_categories, "Majority 'Correct' category should be preserved in sample"


def test_stratified_sample_min_samples(rare_strata_data: pd.DataFrame) -> None:
    """Test minimum samples per stratum."""
    # Sample with min_samples_per_stratum
    sampled = stratified_sample(rare_strata_data, sample_ratio=0.1, min_samples_per_stratum=2, random_seed=42)

    # Since we allow oversampling to preserve rare strata, check that small strata are preserved
    rare_count = len(sampled[sampled["Category"] == "Rare"])
    assert rare_count >= 1, f"Rare category should have at least 1 sample, got {rare_count}"

    # Very rare should include its only sample
    very_rare_count = len(sampled[sampled["Category"] == "VeryRare"])
    assert very_rare_count == 1, f"VeryRare category should include its single sample, got {very_rare_count}"

    # Total sample size should be reasonable (allow some oversampling)
    target_size = int(len(rare_strata_data) * 0.1)  # ~5
    max_allowed = target_size * 2  # Allow up to 2x for rare preservation
    assert target_size <= len(sampled) <= max_allowed, (
        f"Sample size {len(sampled)} should be between {target_size} and {max_allowed}"
    )


@pytest.mark.parametrize("random_seed", [42, 123, 456])
def test_stratified_sample_reproducible(basic_test_data: pd.DataFrame, random_seed: int) -> None:
    """Test that sampling with same seed produces same results."""
    # Sample twice with same seed
    sample1 = stratified_sample(basic_test_data, sample_ratio=0.2, random_seed=random_seed)
    sample2 = stratified_sample(basic_test_data, sample_ratio=0.2, random_seed=random_seed)

    # Should be identical
    pd.testing.assert_frame_equal(sample1, sample2, check_names=True)


def test_stratified_sample_different_seeds_produce_different_results(basic_test_data: pd.DataFrame) -> None:
    """Test that different seeds produce different sampling results."""
    # Sample with different seeds
    sample1 = stratified_sample(basic_test_data, sample_ratio=0.2, random_seed=123)
    sample2 = stratified_sample(basic_test_data, sample_ratio=0.2, random_seed=456)

    # Should be different (with high probability)
    assert not sample1.equals(sample2), "Different random seeds should produce different samples"


@pytest.mark.parametrize("sample_ratio", [0.3, 0.5, 0.7])
def test_stratified_sample_handles_small_strata(sample_ratio: float) -> None:
    """Test handling of single-item strata."""
    data = pd.DataFrame(
        {
            "QuestionId": [1, 2, 3],
            "Category": ["A", "B", "C"],
            "MC_Answer": ["X", "Y", "Z"],
            "value": [1, 2, 3],
        }
    )

    # Sample with various ratios - each stratum has only 1 item
    sampled = stratified_sample(data, sample_ratio=sample_ratio, random_seed=42)

    # Should handle single-item strata gracefully
    min_expected = 1
    max_expected = len(data)
    assert min_expected <= len(sampled) <= max_expected, (
        f"Sample size {len(sampled)} should be between {min_expected} and {max_expected} for ratio {sample_ratio}"
    )

    # All sampled rows should be from original data
    for idx in sampled.index:
        assert idx in data.index, f"Sampled index {idx} should exist in original data"
