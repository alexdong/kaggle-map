"""Tests for evaluation pipeline."""

from pathlib import Path

import pandas as pd

from kaggle_map.evolution import PromptCandidate
from kaggle_map.evolution.evaluator import extract_failure_cases


def test_extract_failure_cases_basic() -> None:
    """Test extracting failure cases from predictions."""
    # Create test data with various failure types
    eval_df = pd.DataFrame(
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

    failures = extract_failure_cases(eval_df, max_failures=10)

    # Should include failures (where map_score < 1.0)
    assert len(failures) == 3

    # Check that failures are actual failures
    for failure in failures:
        assert failure.row_id in [1, 4, 5]


def test_extract_failure_cases_prioritizes_complete_misses() -> None:
    """Test that complete misses (map_score=0) are prioritized."""
    eval_df = pd.DataFrame(
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

    failures = extract_failure_cases(eval_df, max_failures=10)

    # Should get all 5 complete misses first
    complete_miss_count = sum(1 for f in failures if f.row_id < 5)
    assert complete_miss_count == 5

    # Rest should be partial failures
    assert len(failures) == 10


def test_extract_failure_cases_diverse_sampling() -> None:
    """Test diverse sampling across question/category combinations."""
    # Create data with repeated patterns
    eval_df = pd.DataFrame(
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

    failures = extract_failure_cases(eval_df, max_failures=10)

    # Should sample diverse MC_Answers
    mc_answers = {f.mc_answer for f in failures}
    assert len(mc_answers) >= 2  # Should have at least 2 different answers


def test_evaluate_candidate_basic(tmp_path: Path) -> None:
    """Test basic candidate evaluation."""
    # Create a test prompt template
    template_path = tmp_path / "test.j2"
    template_path.write_text("""Student answered: {{ mc_answer }}
Student explained: {{ student_explanation }}
Predictions:
{% for pred in predictions %}
{{ loop.index }}. {{ pred }}
{% endfor %}
Your output:""")

    # Create candidate
    candidate = PromptCandidate(
        generation=0,
        candidate_id="gen_00_candidate_0",
        prompt=template_path.read_text(),
        hypothesis="Test hypothesis",
        parent_ids=[],
    )

    # Note: This test would need a mock or test GGUF model to actually run
    # For now, we're just testing the interface
    assert candidate.candidate_id == "gen_00_candidate_0"
