"""Tests for failure analysis module."""

import pandas as pd

from kaggle_map.evolution.analysis import (
    ErrorPattern,
    analyze_error_patterns,
    group_failures_by_type,
    summarize_for_gpt5,
)


def test_analyze_error_patterns_basic() -> None:
    """Test extracting error patterns from error_prediction.csv."""
    # Create test data with various patterns
    error_df = pd.DataFrame(
        {
            "QuestionId": [100, 100, 100, 101, 101, 102] * 2,
            "Category": ["True_Misconception"] * 6 + ["True_Correct"] * 6,
            "MC_Answer": ["A", "A", "B", "A", "B", "C"] * 2,
            "actual_misconception": ["Wrong_Method"] * 12,
            "predicted_category": ["True_Correct"] * 12,
            "predicted_misconception": ["NA"] * 12,
        }
    )

    patterns = analyze_error_patterns(error_df, max_patterns=5)

    # Should return list of ErrorPattern objects
    assert len(patterns) <= 5
    assert all(isinstance(p, ErrorPattern) for p in patterns)

    # Most common pattern should be first
    assert patterns[0].count >= patterns[-1].count


def test_group_failures_by_type() -> None:
    """Test grouping failures by error type."""
    failures_df = pd.DataFrame(
        {
            "Category": ["True_Misconception", "True_Correct", "True_Neither", "True_Misconception"],
            "predicted_category": ["True_Correct", "True_Misconception", "True_Neither", "True_Misconception"],
            "actual_misconception": ["Wrong_Method", "NA", "NA", "Calculation_Error"],
            "predicted_misconception": ["NA", "Wrong_Method", "NA", "Wrong_Method"],
        }
    )

    grouped = group_failures_by_type(failures_df)

    # Should have wrong_category and wrong_misconception groups
    assert "wrong_category" in grouped
    assert "wrong_misconception" in grouped

    # First row is wrong category (True_Misconception predicted as True_Correct)
    # Second row is also wrong category (True_Correct predicted as True_Misconception)
    assert 0 in grouped["wrong_category"] or 0 in grouped["both_wrong"]
    assert 1 in grouped["wrong_category"] or 1 in grouped["both_wrong"]

    # Last row is wrong misconception (both have same category but different misconception)
    assert 3 in grouped["wrong_misconception"]


def test_summarize_for_gpt5() -> None:
    """Test generating summary for GPT-5 context."""
    error_df = pd.DataFrame(
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

    summary = summarize_for_gpt5(error_df, max_patterns=3, max_examples=2)

    # Should be a readable string
    assert isinstance(summary, str)
    assert len(summary) > 0

    # Should mention the pattern
    assert "True_Misconception" in summary or "wrong category" in summary.lower()

    # Should be concise (not too long for GPT context)
    assert len(summary) < 5000  # Reasonable limit for context
