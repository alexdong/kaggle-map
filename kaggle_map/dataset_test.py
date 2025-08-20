"""Tests for dataset analysis and processing utilities."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from kaggle_map.dataset import (
    build_category_frequencies,
    extract_correct_answers,
    extract_most_common_misconceptions,
    is_answer_correct,
    parse_training_data,
)
from kaggle_map.models import Category, Prediction, TrainingRow


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_training_data():
    """Create sample training data for testing."""
    return [
        TrainingRow(
            row_id=1,
            question_id=100,
            question_text="What is 2+2?",
            mc_answer="4",
            student_explanation="I added them correctly",
            prediction=Prediction(category=Category.TRUE_CORRECT, misconception="NA")
        ),
        TrainingRow(
            row_id=2,
            question_id=100,
            question_text="What is 2+2?",
            mc_answer="5",
            student_explanation="I counted wrong",
            prediction=Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Adding_across")
        ),
        TrainingRow(
            row_id=3,
            question_id=100,
            question_text="What is 2+2?",
            mc_answer="3",
            student_explanation="I subtracted instead",
            prediction=Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Subtraction_error")
        ),
        TrainingRow(
            row_id=4,
            question_id=101,
            question_text="What is 3*3?",
            mc_answer="9",
            student_explanation="Correct multiplication",
            prediction=Prediction(category=Category.TRUE_CORRECT, misconception="NA")
        ),
        TrainingRow(
            row_id=5,
            question_id=101,
            question_text="What is 3*3?",
            mc_answer="6",
            student_explanation="I added instead",
            prediction=Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Addition_instead_multiplication")
        ),
        TrainingRow(
            row_id=6,
            question_id=101,
            question_text="What is 3*3?",
            mc_answer="6",
            student_explanation="Same mistake again",
            prediction=Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Addition_instead_multiplication")
        ),
        TrainingRow(
            row_id=7,
            question_id=102,
            question_text="What is 5-2?",
            mc_answer="3",
            student_explanation="Correct subtraction",
            prediction=Prediction(category=Category.TRUE_CORRECT, misconception="NA")
        ),
        TrainingRow(
            row_id=8,
            question_id=102,
            question_text="What is 5-2?",
            mc_answer="7",
            student_explanation="I don't know",
            prediction=Prediction(category=Category.FALSE_NEITHER, misconception="NA")
        ),
    ]


@pytest.fixture
def temp_training_csv():
    """Create temporary training CSV file with comprehensive test data."""
    training_data = {
        "row_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "QuestionId": [100, 100, 100, 101, 101, 101, 102, 102],
        "QuestionText": [
            "What is 2+2?",
            "What is 2+2?",
            "What is 2+2?",
            "What is 3*3?",
            "What is 3*3?",
            "What is 3*3?",
            "What is 5-2?",
            "What is 5-2?"
        ],
        "MC_Answer": ["4", "5", "3", "9", "6", "6", "3", "7"],
        "StudentExplanation": [
            "I added them correctly",
            "I counted wrong",
            "I subtracted instead",
            "Correct multiplication",
            "I added instead",
            "Same mistake again",
            "Correct subtraction",
            "I don't know"
        ],
        "Category": [
            "True_Correct",
            "False_Misconception",
            "False_Misconception",
            "True_Correct",
            "False_Misconception",
            "False_Misconception",
            "True_Correct",
            "False_Neither"
        ],
        "Misconception": [
            None,
            "Adding_across",
            "Subtraction_error",
            None,
            "Addition_instead_multiplication",
            "Addition_instead_multiplication",
            None,
            None
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        pd.DataFrame(training_data).to_csv(f.name, index=False)
        yield Path(f.name)
        Path(f.name).unlink()  # Cleanup


# =============================================================================
# parse_training_data Function Tests
# =============================================================================


def test_parse_training_data_creates_strongly_typed_training_rows(temp_training_csv):
    """parse_training_data creates properly typed TrainingRow objects."""
    training_rows = parse_training_data(temp_training_csv)
    
    assert len(training_rows) == 8
    assert all(isinstance(row, TrainingRow) for row in training_rows)
    
    # Check first row details
    first_row = training_rows[0]
    assert first_row.row_id == 1
    assert first_row.question_id == 100
    assert first_row.question_text == "What is 2+2?"
    assert first_row.mc_answer == "4"
    assert first_row.category == Category.TRUE_CORRECT
    assert first_row.misconception == "NA"


def test_parse_training_data_handles_nan_misconceptions(temp_training_csv):
    """parse_training_data properly converts pandas NaN to None for misconceptions."""
    training_rows = parse_training_data(temp_training_csv)
    
    # Rows with None misconceptions should have misconception="NA"
    rows_without_misconceptions = [row for row in training_rows if row.misconception == "NA"]
    assert len(rows_without_misconceptions) == 4  # Rows 1, 4, 7, 8
    
    # Rows with actual misconceptions should preserve them
    rows_with_misconceptions = [row for row in training_rows if row.misconception != "NA"]
    assert len(rows_with_misconceptions) == 4  # Rows 2, 3, 5, 6


def test_parse_training_data_raises_error_for_missing_file():
    """parse_training_data raises clear error for non-existent files."""
    missing_path = Path("nonexistent_file.csv")
    
    with pytest.raises(AssertionError, match="Training file not found"):
        parse_training_data(missing_path)


def test_parse_training_data_raises_error_for_empty_csv():
    """parse_training_data raises error for empty CSV files."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Create empty CSV
        pd.DataFrame().to_csv(f.name, index=False)
        temp_path = Path(f.name)
        
        try:
            with pytest.raises(pd.errors.EmptyDataError):
                parse_training_data(temp_path)
        finally:
            temp_path.unlink()


# =============================================================================
# extract_correct_answers Function Tests
# =============================================================================


def test_extract_correct_answers_finds_true_correct_answers(sample_training_data):
    """extract_correct_answers identifies correct answers from True_Correct categories."""
    correct_answers = extract_correct_answers(sample_training_data)
    
    expected_answers = {100: "4", 101: "9", 102: "3"}
    assert correct_answers == expected_answers


def test_extract_correct_answers_handles_single_correct_answer_per_question(sample_training_data):
    """extract_correct_answers works when each question has only one correct answer."""
    correct_answers = extract_correct_answers(sample_training_data)
    
    # Each question should have exactly one correct answer
    assert len(correct_answers) == 3
    assert all(isinstance(qid, int) for qid in correct_answers.keys())
    assert all(isinstance(answer, str) for answer in correct_answers.values())


def test_extract_correct_answers_uses_first_correct_answer_when_multiple_exist():
    """extract_correct_answers uses the first correct answer when multiple exist for same question."""
    conflicting_data = [
        TrainingRow(
            row_id=1, question_id=100, question_text="Test", mc_answer="A",
            student_explanation="Test", prediction=Prediction(category=Category.TRUE_CORRECT, misconception="NA")
        ),
        TrainingRow(
            row_id=2, question_id=100, question_text="Test", mc_answer="B",
            student_explanation="Test", prediction=Prediction(category=Category.TRUE_CORRECT, misconception="NA")
        ),
    ]
    
    correct_answers = extract_correct_answers(conflicting_data)
    # Should use the first correct answer found
    assert correct_answers[100] == "A"


def test_extract_correct_answers_raises_error_for_empty_data():
    """extract_correct_answers raises error for empty training data."""
    with pytest.raises(AssertionError, match="Training data cannot be empty"):
        extract_correct_answers([])


def test_extract_correct_answers_raises_error_when_no_correct_answers_found():
    """extract_correct_answers raises error when no True_Correct categories exist."""
    no_correct_data = [
        TrainingRow(
            row_id=1, question_id=100, question_text="Test", mc_answer="A",
            student_explanation="Test", prediction=Prediction(category=Category.FALSE_NEITHER, misconception="NA")
        ),
    ]
    
    with pytest.raises(AssertionError, match="Must find at least one correct answer"):
        extract_correct_answers(no_correct_data)


# =============================================================================
# is_answer_correct Function Tests
# =============================================================================


def test_is_answer_correct_returns_true_for_matching_answers():
    """is_answer_correct returns True when student answer matches correct answer."""
    correct_answers = {100: "4", 101: "9"}
    
    assert is_answer_correct(100, "4", correct_answers) is True
    assert is_answer_correct(101, "9", correct_answers) is True


def test_is_answer_correct_returns_false_for_non_matching_answers():
    """is_answer_correct returns False when student answer doesn't match correct answer."""
    correct_answers = {100: "4", 101: "9"}
    
    assert is_answer_correct(100, "5", correct_answers) is False
    assert is_answer_correct(101, "6", correct_answers) is False


def test_is_answer_correct_returns_false_for_unknown_questions():
    """is_answer_correct returns False for questions not in correct_answers dict."""
    correct_answers = {100: "4"}
    
    assert is_answer_correct(999, "4", correct_answers) is False


# =============================================================================
# build_category_frequencies Function Tests
# =============================================================================


def test_build_category_frequencies_creates_correctness_patterns(sample_training_data):
    """build_category_frequencies creates frequency patterns based on answer correctness."""
    correct_answers = extract_correct_answers(sample_training_data)
    frequencies = build_category_frequencies(sample_training_data, correct_answers)
    
    # Question 100: correct answer="4", student answers: "4"(correct), "5"(wrong), "3"(wrong)
    assert 100 in frequencies
    assert True in frequencies[100]  # Correct answers
    assert False in frequencies[100]  # Incorrect answers
    
    # For correct answers (answer="4"), should have True_Correct category
    assert Category.TRUE_CORRECT in frequencies[100][True]
    
    # For incorrect answers (answers="5", "3"), should have False_Misconception categories
    assert Category.FALSE_MISCONCEPTION in frequencies[100][False]


def test_build_category_frequencies_orders_by_frequency(sample_training_data):
    """build_category_frequencies orders categories by frequency (most common first)."""
    correct_answers = extract_correct_answers(sample_training_data)
    frequencies = build_category_frequencies(sample_training_data, correct_answers)
    
    # Question 101 has 2 False_Misconception entries with same answer "6"
    # So False_Misconception should be the most frequent for incorrect answers
    incorrect_categories = frequencies[101][False]
    assert incorrect_categories[0] == Category.FALSE_MISCONCEPTION


def test_build_category_frequencies_raises_error_for_empty_data():
    """build_category_frequencies raises error for empty training data."""
    with pytest.raises(AssertionError, match="Training data cannot be empty"):
        build_category_frequencies([], {100: "A"})


def test_build_category_frequencies_raises_error_for_empty_correct_answers():
    """build_category_frequencies raises error for empty correct answers."""
    sample_data = [
        TrainingRow(
            row_id=1, question_id=100, question_text="Test", mc_answer="A",
            student_explanation="Test", prediction=Prediction(category=Category.TRUE_CORRECT, misconception="NA")
        )
    ]
    
    with pytest.raises(AssertionError, match="Correct answers cannot be empty"):
        build_category_frequencies(sample_data, {})


# =============================================================================
# extract_most_common_misconceptions Function Tests
# =============================================================================


def test_extract_most_common_misconceptions_finds_most_frequent(sample_training_data):
    """extract_most_common_misconceptions identifies most common misconception per question."""
    misconceptions = extract_most_common_misconceptions(sample_training_data)
    
    # Question 100: "Adding_across", "Subtraction_error" -> "Adding_across" appears once each
    # So either could be first, but both should be present as possibilities
    assert 100 in misconceptions
    assert misconceptions[100] in ["Adding_across", "Subtraction_error"]
    
    # Question 101: "Addition_instead_multiplication" appears twice -> most common
    assert misconceptions[101] == "Addition_instead_multiplication"
    
    # Question 102: no misconceptions -> "NA"
    assert 102 in misconceptions
    assert misconceptions[102] == "NA"


def test_extract_most_common_misconceptions_handles_no_misconceptions():
    """extract_most_common_misconceptions handles questions with no misconceptions."""
    data_without_misconceptions = [
        TrainingRow(
            row_id=1, question_id=100, question_text="Test", mc_answer="A",
            student_explanation="Test", prediction=Prediction(category=Category.TRUE_CORRECT, misconception="NA")
        ),
        TrainingRow(
            row_id=2, question_id=100, question_text="Test", mc_answer="B",
            student_explanation="Test", prediction=Prediction(category=Category.FALSE_NEITHER, misconception="NA")
        ),
    ]
    
    misconceptions = extract_most_common_misconceptions(data_without_misconceptions)
    
    # Should have entry for question 100 but with "NA" value
    assert 100 in misconceptions
    assert misconceptions[100] == "NA"


def test_extract_most_common_misconceptions_raises_error_for_empty_data():
    """extract_most_common_misconceptions raises error for empty training data."""
    with pytest.raises(AssertionError, match="Training data cannot be empty"):
        extract_most_common_misconceptions([])
