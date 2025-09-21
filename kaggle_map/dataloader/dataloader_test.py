"""Tests for dataloader functions."""

from pathlib import Path

import pandas as pd

from kaggle_map.core.models import Category
from kaggle_map.dataloader import load_rows_by_ids, load_training_data, load_validation_data

FULL_TRAIN_CSV = Path("datasets/33474_full_train.csv")
TINY_VALIDATION_CSV = Path("datasets/33474_tiny_train.csv")


def test_load_training_data_returns_correct_count():
    """Test that load_training_data returns 1766 rows for question 33474."""
    # Question 33474 has exactly 1766 rows in the training dataset
    question_id = 33474

    training_rows = load_training_data(FULL_TRAIN_CSV, question_id)

    assert len(training_rows) == 1766, f"Expected 1766 rows but got {len(training_rows)}"

    # Verify all rows have the correct question_id
    for row in training_rows:
        assert row.question_id == question_id, f"Row has wrong question_id: {row.question_id}"


def test_load_validation_data_returns_correct_count():
    """Test that load_validation_data returns 17 rows for question 33474."""
    # 33474 has 17 labelled rows in the tiny validation subset
    question_id = 33474

    error_pairs = load_validation_data(TINY_VALIDATION_CSV, question_id)

    # 33474_tiny_train.csv contains 17 labelled rows for question 33474
    assert len(error_pairs) == 17, f"Expected 17 rows but got {len(error_pairs)}"

    # Verify all rows have the correct question_id
    for eval_row, _prediction in error_pairs:
        assert eval_row.question_id == question_id, f"Row has wrong question_id: {eval_row.question_id}"


def test_training_row_structure():
    """Test that TrainingRow objects have the expected structure."""
    question_id = 33474

    training_rows = load_training_data(FULL_TRAIN_CSV, question_id)

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


def test_validation_data_structure():
    """Test that validation data rows have the expected structure."""
    question_id = 33474

    error_pairs = load_validation_data(TINY_VALIDATION_CSV, question_id)

    # Check first pair has expected structure
    assert len(error_pairs) > 0, "Should have at least one error pair"
    first_eval_row, first_prediction = error_pairs[0]

    # Check EvaluationRow attributes
    assert hasattr(first_eval_row, "row_id"), "EvaluationRow missing row_id"
    assert hasattr(first_eval_row, "question_id"), "EvaluationRow missing question_id"
    assert hasattr(first_eval_row, "question_text"), "EvaluationRow missing question_text"
    assert hasattr(first_eval_row, "mc_answer"), "EvaluationRow missing mc_answer"
    assert hasattr(first_eval_row, "student_explanation"), "EvaluationRow missing student_explanation"

    # Check Prediction attributes
    assert hasattr(first_prediction, "category"), "Prediction missing category"
    assert hasattr(first_prediction, "misconception"), "Prediction missing misconception"

    # Check types
    assert isinstance(first_eval_row.row_id, int), "row_id should be int"
    assert isinstance(first_eval_row.question_id, int), "question_id should be int"
    assert isinstance(first_eval_row.question_text, str), "question_text should be str"
    assert isinstance(first_eval_row.mc_answer, str), "mc_answer should be str"
    assert isinstance(first_eval_row.student_explanation, str), "student_explanation should be str"
    assert isinstance(first_prediction.category, Category), "category should be Category enum"
    assert isinstance(first_prediction.misconception, str), "misconception should be str"


def test_load_all_training_data(tmp_path):
    """Test that load_training_data can load all data when question_id is None."""
    sample_df = pd.DataFrame(
        [
            {
                "row_id": 10,
                "QuestionId": 1,
                "QuestionText": "Q1",
                "MC_Answer": "A",
                "StudentExplanation": "Exp1",
                "Category": "True_Correct",
                "Misconception": "NA",
            },
            {
                "row_id": 11,
                "QuestionId": 2,
                "QuestionText": "Q2",
                "MC_Answer": "B",
                "StudentExplanation": "Exp2",
                "Category": "False_Misconception",
                "Misconception": "Calculation",
            },
        ]
    )
    csv_path = tmp_path / "sample_train.csv"
    sample_df.to_csv(csv_path, index=False)

    all_training_rows = load_training_data(csv_path)
    assert len(all_training_rows) == 2, f"Expected 2 rows but got {len(all_training_rows)}"

    single_question_rows = load_training_data(csv_path, 1)
    assert len(single_question_rows) == 1, f"Expected 1 row for question 1 but got {len(single_question_rows)}"

    first_row = all_training_rows[0]
    assert hasattr(first_row, "question_id"), "TrainingRow should have question_id"
    assert isinstance(first_row.question_id, int), "question_id should be int"


def test_load_all_validation_data(tmp_path):
    """Test that load_validation_data can load all data when question_id is None."""
    sample_df = pd.DataFrame(
        [
            {
                "row_id": 20,
                "QuestionId": 1,
                "QuestionText": "Q1",
                "MC_Answer": "A",
                "StudentExplanation": "Exp1",
                "Category": "True_Correct",
                "Misconception": "NA",
            },
            {
                "row_id": 21,
                "QuestionId": 2,
                "QuestionText": "Q2",
                "MC_Answer": "B",
                "StudentExplanation": "Exp2",
                "Category": "False_Misconception",
                "Misconception": "Reason",
            },
        ]
    )
    csv_path = tmp_path / "sample_validation.csv"
    sample_df.to_csv(csv_path, index=False)

    all_validation_pairs = load_validation_data(csv_path)
    assert len(all_validation_pairs) == 2, f"Expected 2 pairs but got {len(all_validation_pairs)}"

    single_question_pairs = load_validation_data(csv_path, 1)
    assert len(single_question_pairs) == 1, f"Expected 1 pair for question 1 but got {len(single_question_pairs)}"

    first_eval_row, first_prediction = all_validation_pairs[0]
    assert hasattr(first_eval_row, "question_id"), "EvaluationRow should have question_id"
    assert isinstance(first_eval_row.question_id, int), "question_id should be int"
    assert isinstance(first_prediction.category, Category), "category should be Category enum"


def test_load_rows_by_ids(tmp_path):
    """Test loading specific rows by ID."""
    # Create test CSV
    test_data = pd.DataFrame(
        [
            {
                "row_id": 1,
                "QuestionId": 100,
                "QuestionText": "Q1",
                "MC_Answer": "A",
                "StudentExplanation": "Exp1",
                "Category": "True_Correct",
                "Misconception": "NA",
            },
            {
                "row_id": 2,
                "QuestionId": 101,
                "QuestionText": "Q2",
                "MC_Answer": "B",
                "StudentExplanation": "Exp2",
                "Category": "True_Correct",
                "Misconception": "NA",
            },
            {
                "row_id": 3,
                "QuestionId": 102,
                "QuestionText": "Q3",
                "MC_Answer": "C",
                "StudentExplanation": "Exp3",
                "Category": "True_Misconception",
                "Misconception": "Test",
            },
            {
                "row_id": 4,
                "QuestionId": 103,
                "QuestionText": "Q4",
                "MC_Answer": "D",
                "StudentExplanation": "Exp4",
                "Category": "True_Neither",
                "Misconception": "NA",
            },
        ]
    )

    test_file = tmp_path / "test_data.csv"
    test_data.to_csv(test_file, index=False)

    # Load specific rows
    result = load_rows_by_ids(test_file, [1, 3])

    assert len(result) == 2
    assert result.iloc[0]["row_id"] == 1
    assert result.iloc[1]["row_id"] == 3
    assert result.iloc[0]["QuestionText"] == "Q1"
    assert result.iloc[1]["QuestionText"] == "Q3"
