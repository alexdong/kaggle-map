"""Tests for failure analysis module."""

import logging

import pandas as pd
import pytest

from kaggle_map.evolution.analysis import (
    ErrorPattern,
    analyze_error_patterns,
    group_failures_by_type,
    summarize_for_gpt5,
)

# Set debug logging for tests
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def pattern_analysis_data() -> pd.DataFrame:
    """Create test data with various patterns for pattern analysis."""
    return pd.DataFrame(
        {
            "QuestionId": [100, 100, 100, 101, 101, 102] * 2,
            "Category": ["True_Misconception"] * 6 + ["True_Correct"] * 6,
            "MC_Answer": ["A", "A", "B", "A", "B", "C"] * 2,
            "actual_misconception": ["Wrong_Method"] * 12,
            "predicted_category": ["True_Correct"] * 12,
            "predicted_misconception": ["NA"] * 12,
        }
    )


@pytest.fixture
def failure_grouping_data() -> pd.DataFrame:
    """Create test data for grouping failures by error type."""
    return pd.DataFrame(
        {
            "Category": ["True_Misconception", "True_Correct", "True_Neither", "True_Misconception"],
            "predicted_category": ["True_Correct", "True_Misconception", "True_Neither", "True_Misconception"],
            "actual_misconception": ["Wrong_Method", "NA", "NA", "Calculation_Error"],
            "predicted_misconception": ["NA", "Wrong_Method", "NA", "Wrong_Method"],
        }
    )


@pytest.fixture
def gpt5_summary_data() -> pd.DataFrame:
    """Create test data for GPT-5 summary generation."""
    return pd.DataFrame(
        {
            "QuestionId": [100] * 10,
            "Category": ["True_Misconception"] * 10,
            "MC_Answer": ["A"] * 5 + ["B"] * 5,
            "actual_misconception": ["Wrong_Method"] * 10,
            "predicted_category": ["True_Correct"] * 10,
            "predicted_misconception": ["NA"] * 10,
            "QuestionText": ["What is 2+2?"] * 10,
            "StudentExplanation": [f"Explanation {i}" for i in range(10)],
        }
    )


@pytest.mark.parametrize("max_patterns", [1, 3, 5, 10])
def test_analyze_error_patterns_basic(pattern_analysis_data: pd.DataFrame, max_patterns: int) -> None:
    """Test extracting error patterns from error_prediction.csv."""
    patterns = analyze_error_patterns(pattern_analysis_data, max_patterns=max_patterns)

    # Should return list of ErrorPattern objects within limit
    assert len(patterns) <= max_patterns, f"Should return at most {max_patterns} patterns, got {len(patterns)}"
    assert all(isinstance(p, ErrorPattern) for p in patterns), "All returned items should be ErrorPattern objects"

    # Most common pattern should be first (patterns should be sorted by frequency)
    if len(patterns) > 1:
        first_count = patterns[0].count
        last_count = patterns[-1].count
        assert first_count >= last_count, f"Patterns should be sorted by count: first={first_count}, last={last_count}"


def test_group_failures_by_type(failure_grouping_data: pd.DataFrame) -> None:
    """Test grouping failures by error type."""
    grouped = group_failures_by_type(failure_grouping_data)

    # Should have expected group keys
    expected_groups = {"wrong_category", "wrong_misconception"}
    actual_groups = set(grouped.keys())
    assert expected_groups.issubset(actual_groups), f"Should include groups {expected_groups}, got {actual_groups}"

    # Verify specific row classifications
    # Row 0: True_Misconception predicted as True_Correct (wrong category)
    # Row 1: True_Correct predicted as True_Misconception (wrong category)
    # Row 2: True_Neither predicted as True_Neither (correct)
    # Row 3: True_Misconception predicted as True_Misconception but wrong misconception

    # Row 0 should be in wrong_category or both_wrong
    row_0_groups = [group for group, indices in grouped.items() if 0 in indices]
    assert len(row_0_groups) > 0, "Row 0 (category mismatch) should be classified in at least one error group"

    # Row 1 should be in wrong_category or both_wrong
    row_1_groups = [group for group, indices in grouped.items() if 1 in indices]
    assert len(row_1_groups) > 0, "Row 1 (category mismatch) should be classified in at least one error group"

    # Row 3 should be in wrong_misconception (same category, wrong misconception)
    assert 3 in grouped.get("wrong_misconception", []), "Row 3 should be classified as wrong misconception"


@pytest.mark.parametrize(("max_patterns", "max_examples"), [
    (1, 1),
    (3, 2),
    (5, 3),
])
def test_summarize_for_gpt5(gpt5_summary_data: pd.DataFrame, max_patterns: int, max_examples: int) -> None:
    """Test generating summary for GPT-5 context."""
    summary = summarize_for_gpt5(gpt5_summary_data, max_patterns=max_patterns, max_examples=max_examples)

    # Should be a readable string
    assert isinstance(summary, str), "Summary should be a string"
    assert len(summary) > 0, "Summary should not be empty"

    # Should mention the pattern or error type
    contains_pattern = "True_Misconception" in summary or "wrong category" in summary.lower()
    assert contains_pattern, "Summary should mention the error pattern or category"

    # Should be concise (not too long for GPT context)
    max_length = 5000
    assert len(summary) < max_length, f"Summary should be under {max_length} characters for GPT context, got {len(summary)}"


def test_summarize_for_gpt5_empty_data() -> None:
    """Test summarize_for_gpt5 handles empty data gracefully."""
    empty_df = pd.DataFrame({
        "QuestionId": [],
        "Category": [],
        "MC_Answer": [],
        "actual_misconception": [],
        "predicted_category": [],
        "predicted_misconception": [],
        "QuestionText": [],
        "StudentExplanation": [],
    })

    summary = summarize_for_gpt5(empty_df, max_patterns=3, max_examples=2)

    # Should handle empty data gracefully
    assert isinstance(summary, str), "Summary should be a string even for empty data"
    # Could be empty or contain a message about no patterns found
    assert len(summary) >= 0, "Summary should not be None"
