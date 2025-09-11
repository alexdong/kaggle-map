"""Tests for evaluation pipeline."""

import logging
from pathlib import Path

import pandas as pd
import pytest

from kaggle_map.evolution import PromptCandidate
from kaggle_map.evolution.evaluator import extract_failure_cases

# Set debug logging for tests
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def mixed_failure_data() -> pd.DataFrame:
    """Create test data with various failure types for testing."""
    return pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4, 5],
            "QuestionId": [100, 101, 102, 103, 104],
            "Category": [
                "True_Misconception",
                "True_Correct",
                "True_Neither",
                "True_Misconception",
                "False_Misconception",
            ],
            "actual_misconception": ["Wrong_Method", "NA", "NA", "Calculation_Error", "Conceptual_Error"],
            "predicted_category": [
                "True_Correct",
                "True_Correct",
                "True_Neither",
                "True_Misconception",
                "True_Misconception",
            ],
            "predicted_misconception": ["NA", "NA", "NA", "Wrong_Method", "Wrong_Method"],
            "MC_Answer": ["A", "B", "C", "D", "A"],
            "QuestionText": ["Q1", "Q2", "Q3", "Q4", "Q5"],
            "StudentExplanation": ["Exp1", "Exp2", "Exp3", "Exp4", "Exp5"],
            "map_score": [0.0, 1.0, 1.0, 0.33, 0.0],  # Failures: 1, 4, 5
        }
    )


@pytest.fixture
def priority_failure_data() -> pd.DataFrame:
    """Create data with complete misses and partial failures for priority testing."""
    return pd.DataFrame(
        {
            "row_id": range(20),
            "QuestionId": range(100, 120),
            "Category": ["True_Misconception"] * 20,
            "actual_misconception": ["Error"] * 20,
            "predicted_category": ["True_Correct"] * 10 + ["True_Misconception"] * 10,
            "predicted_misconception": ["NA"] * 10 + ["Wrong"] * 10,
            "MC_Answer": ["A"] * 20,
            "QuestionText": [f"Q{i}" for i in range(20)],
            "StudentExplanation": [f"Exp{i}" for i in range(20)],
            "map_score": [0.0] * 5 + [0.5] * 10 + [1.0] * 5,  # 5 complete misses, 10 partial, 5 correct
        }
    )


@pytest.fixture
def diverse_failure_data() -> pd.DataFrame:
    """Create data for diverse sampling testing."""
    return pd.DataFrame(
        {
            "row_id": range(30),
            "QuestionId": [100, 100, 100] * 10,  # Same question repeated
            "Category": ["True_Misconception"] * 15 + ["True_Correct"] * 15,
            "actual_misconception": ["Error"] * 30,
            "predicted_category": ["True_Correct"] * 30,
            "predicted_misconception": ["NA"] * 30,
            "MC_Answer": ["A", "B", "C"] * 10,  # Different answers
            "QuestionText": ["Q"] * 30,
            "StudentExplanation": [f"Exp{i}" for i in range(30)],
            "map_score": [0.0] * 30,  # All failures
        }
    )


def test_extract_failure_cases_basic(mixed_failure_data: pd.DataFrame) -> None:
    """Test extracting failure cases from predictions."""
    failures = extract_failure_cases(mixed_failure_data, max_failures=10)

    # Should include failures (where map_score < 1.0)
    expected_failure_count = 3
    assert len(failures) == expected_failure_count, f"Should extract exactly {expected_failure_count} failure cases"

    # Check that failures are actual failures (rows 1, 4, 5 have map_score < 1.0)
    expected_failure_ids = {1, 4, 5}
    actual_failure_ids = {failure.row_id for failure in failures}
    assert actual_failure_ids == expected_failure_ids, (
        f"Failure row IDs should be {expected_failure_ids}, got {actual_failure_ids}"
    )


def test_extract_failure_cases_prioritizes_complete_misses(priority_failure_data: pd.DataFrame) -> None:
    """Test that complete misses (map_score=0) are prioritized."""
    failures = extract_failure_cases(priority_failure_data, max_failures=10)

    # Should get all 5 complete misses first (row_ids 0-4 have map_score=0.0)
    complete_miss_ids = {f.row_id for f in failures if f.row_id < 5}
    expected_complete_misses = 5
    assert len(complete_miss_ids) == expected_complete_misses, (
        f"Should prioritize all {expected_complete_misses} complete misses"
    )

    # Total should be exactly max_failures
    expected_total_failures = 10
    assert len(failures) == expected_total_failures, f"Should return exactly {expected_total_failures} failures"


def test_extract_failure_cases_diverse_sampling(diverse_failure_data: pd.DataFrame) -> None:
    """Test diverse sampling across question/category combinations."""
    failures = extract_failure_cases(diverse_failure_data, max_failures=10)

    # Should sample diverse MC_Answers
    mc_answers = {f.mc_answer for f in failures}
    min_expected_answers = 2
    assert len(mc_answers) >= min_expected_answers, (
        f"Should sample at least {min_expected_answers} different MC answers, got {len(mc_answers)}: {mc_answers}"
    )


@pytest.mark.parametrize("max_failures", [1, 5, 10, 15])
def test_extract_failure_cases_max_failures_limit(mixed_failure_data: pd.DataFrame, max_failures: int) -> None:
    """Test that max_failures parameter limits returned results correctly."""
    failures = extract_failure_cases(mixed_failure_data, max_failures=max_failures)

    # Should never exceed max_failures
    assert len(failures) <= max_failures, f"Should not exceed max_failures={max_failures}, got {len(failures)}"

    # Should return all available failures if max_failures is larger
    total_failures_available = len(mixed_failure_data[mixed_failure_data["map_score"] < 1.0])
    expected_count = min(max_failures, total_failures_available)
    assert len(failures) == expected_count, f"Should return {expected_count} failures for max_failures={max_failures}"


def test_evaluate_candidate_basic(tmp_path: Path) -> None:
    """Test basic candidate evaluation interface."""
    # Create a test prompt template
    template_path = tmp_path / "test.j2"
    template_content = """Student answered: {{ mc_answer }}
Student explained: {{ student_explanation }}
Predictions:
{% for pred in predictions %}
{{ loop.index }}. {{ pred }}
{% endfor %}
Your output:"""
    template_path.write_text(template_content)

    # Create candidate
    candidate = PromptCandidate(
        generation=0,
        candidate_id="gen_00_candidate_0",
        prompt=template_path.read_text(),
        hypothesis="Test hypothesis",
        parent_ids=[],
    )

    # Test candidate creation and properties
    assert candidate.candidate_id == "gen_00_candidate_0", "Candidate ID should be preserved"
    assert candidate.prompt == template_content, "Prompt content should match template"
    assert candidate.hypothesis == "Test hypothesis", "Hypothesis should be preserved"
    assert candidate.parent_ids == [], "Parent IDs should be empty for new candidate"
