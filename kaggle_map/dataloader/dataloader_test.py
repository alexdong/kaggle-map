"""Tests for dataloader functions."""

from kaggle_map.dataloader import load_error, load_training


def test_load_training_returns_correct_count():
    """Test that load_training returns 1766 rows for question 33474."""
    # Question 33474 has exactly 1766 rows in the training dataset
    question_id = 33474

    training_rows = load_training(question_id)

    assert len(training_rows) == 1766, f"Expected 1766 rows but got {len(training_rows)}"

    # Verify all rows have the correct question_id
    for row in training_rows:
        assert row.question_id == question_id, f"Row has wrong question_id: {row.question_id}"


def test_load_error_returns_correct_count():
    """Test that load_error returns 341 rows for question 33474."""
    # Question 33474 has 341 rows in the error prediction dataset
    question_id = 33474

    error_rows = load_error(question_id)

    assert len(error_rows) == 341, f"Expected 341 rows but got {len(error_rows)}"

    # Verify all rows have the correct question_id
    for row in error_rows:
        assert row.question_id == question_id, f"Row has wrong question_id: {row.question_id}"


def test_training_row_structure():
    """Test that TrainingRow objects have the expected structure."""
    question_id = 31772

    training_rows = load_training(question_id)

    # Check first row has expected attributes
    first_row = training_rows[0]

    assert hasattr(first_row, "row_id"), "TrainingRow missing row_id"
    assert hasattr(first_row, "question_id"), "TrainingRow missing question_id"
    assert hasattr(first_row, "question_text"), "TrainingRow missing question_text"
    assert hasattr(first_row, "mc_answer"), "TrainingRow missing mc_answer"
    assert hasattr(first_row, "student_explanation"), "TrainingRow missing student_explanation"
    assert hasattr(first_row, "category"), "TrainingRow missing category"
    assert hasattr(first_row, "misconception"), "TrainingRow missing misconception"

    # Check types
    assert isinstance(first_row.row_id, int), "row_id should be int"
    assert isinstance(first_row.question_id, int), "question_id should be int"
    assert isinstance(first_row.question_text, str), "question_text should be str"
    assert isinstance(first_row.mc_answer, str), "mc_answer should be str"
    assert isinstance(first_row.student_explanation, str), "student_explanation should be str"


def test_error_row_structure():
    """Test that error prediction rows have the expected structure."""
    question_id = 31772

    error_rows = load_error(question_id)

    # Check first row has expected attributes
    first_row = error_rows[0]

    assert hasattr(first_row, "row_id"), "TrainingRow missing row_id"
    assert hasattr(first_row, "question_id"), "TrainingRow missing question_id"
    assert hasattr(first_row, "question_text"), "TrainingRow missing question_text"
    assert hasattr(first_row, "mc_answer"), "TrainingRow missing mc_answer"
    assert hasattr(first_row, "student_explanation"), "TrainingRow missing student_explanation"
    assert hasattr(first_row, "category"), "TrainingRow missing category"
    assert hasattr(first_row, "misconception"), "TrainingRow missing misconception"

    # Check types
    assert isinstance(first_row.row_id, int), "row_id should be int"
    assert isinstance(first_row.question_id, int), "question_id should be int"
    assert isinstance(first_row.question_text, str), "question_text should be str"
    assert isinstance(first_row.mc_answer, str), "mc_answer should be str"
    assert isinstance(first_row.student_explanation, str), "student_explanation should be str"
