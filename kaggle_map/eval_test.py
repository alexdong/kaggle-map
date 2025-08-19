"""Tests for MAP@3 evaluation metric implementation."""

from pathlib import Path
import pytest
import pandas as pd
import tempfile
from unittest.mock import patch, MagicMock
from kaggle_map.eval import (
    evaluate, 
    _load_ground_truth, 
    _load_submissions, 
    _parse_prediction_string,
    _calculate_average_precision,
    _predictions_match,
    _prepare_cross_validation_data,
    _save_submission_csv
)
from kaggle_map.models import (
    EvaluationResult, 
    Prediction, 
    Category, 
    EvaluationRow,
    SubmissionRow
)


@pytest.fixture
def temp_csv_files(tmp_path):
    """Helper to create temporary CSV files for testing."""
    def _create_files(ground_truth_data, submission_data):
        gt_path = tmp_path / "ground_truth.csv"
        sub_path = tmp_path / "submission.csv"
        
        pd.DataFrame(ground_truth_data).to_csv(gt_path, index=False)
        pd.DataFrame(submission_data).to_csv(sub_path, index=False)
        
        return gt_path, sub_path
    return _create_files


@pytest.mark.parametrize("test_case,expected_map,expected_perfect", [
    # Perfect predictions: all correct in first position
    (
        {
            "ground_truth": {
                "row_id": [1, 2, 3],
                "Category": ["True_Correct", "False_Misconception", "True_Misconception"],
                "Misconception": ["Adding_across", "Denominator-only_change", "Incorrect_equivalent_fraction_addition"]
            },
            "submission": {
                "row_id": [1, 2, 3],
                "predictions": [
                    "True_Correct:Adding_across False_Neither:Other_tag True_Misconception:Another_tag",
                    "False_Misconception:Denominator-only_change True_Correct:Some_tag False_Neither:Bad_tag", 
                    "True_Misconception:Incorrect_equivalent_fraction_addition False_Neither:Wrong_tag True_Correct:Other_tag"
                ]
            }
        },
        1.0,
        3
    ),
    
    # Second position predictions: all correct in second position
    (
        {
            "ground_truth": {
                "row_id": [1, 2, 3],
                "Category": ["True_Correct", "False_Misconception", "True_Misconception"],
                "Misconception": ["Adding_across", "Denominator-only_change", "Incorrect_equivalent_fraction_addition"]
            },
            "submission": {
                "row_id": [1, 2, 3],
                "predictions": [
                    "False_Neither:Other_tag True_Correct:Adding_across True_Neither:Another_tag",
                    "True_Neither:Some_tag False_Misconception:Denominator-only_change False_Neither:Bad_tag",
                    "False_Neither:Wrong_tag True_Misconception:Incorrect_equivalent_fraction_addition True_Neither:Other_tag"
                ]
            }
        },
        0.5,
        0
    ),
    
    # Third position predictions: all correct in third position
    (
        {
            "ground_truth": {
                "row_id": [1, 2],
                "Category": ["True_Correct", "False_Misconception"],
                "Misconception": ["Adding_across", "Denominator-only_change"]
            },
            "submission": {
                "row_id": [1, 2],
                "predictions": [
                    "False_Neither:Other_tag True_Neither:Another_tag True_Correct:Adding_across",
                    "True_Neither:Some_tag False_Neither:Bad_tag False_Misconception:Denominator-only_change"
                ]
            }
        },
        1/3,
        0
    ),
    
    # Mixed positions: predictions at different positions
    (
        {
            "ground_truth": {
                "row_id": [1, 2, 3, 4],
                "Category": ["True_Correct", "False_Misconception", "True_Misconception", "False_Neither"],
                "Misconception": ["Adding_across", "Denominator-only_change", "Incorrect_equivalent_fraction_addition", "NA"]
            },
            "submission": {
                "row_id": [1, 2, 3, 4],
                "predictions": [
                    "True_Correct:Adding_across False_Neither:Other_tag True_Neither:Another_tag",         # 1st position: AP = 1.0
                    "False_Neither:Other_tag False_Misconception:Denominator-only_change True_Neither:Another_tag",  # 2nd position: AP = 0.5
                    "False_Neither:Other_tag True_Neither:Another_tag True_Misconception:Incorrect_equivalent_fraction_addition",  # 3rd position: AP = 1/3
                    "False_Neither:NA True_Neither:Another_tag False_Neither:Bad_tag"                   # 1st position: AP = 1.0
                ]
            }
        },
        (1.0 + 0.5 + 1/3 + 1.0) / 4,
        2
    ),
    
    # No correct predictions
    (
        {
            "ground_truth": {
                "row_id": [1, 2],
                "Category": ["True_Correct", "False_Misconception"],
                "Misconception": ["Adding_across", "Denominator-only_change"]
            },
            "submission": {
                "row_id": [1, 2],
                "predictions": [
                    "False_Neither:Other_tag True_Neither:Another_tag False_Neither:Bad_tag",
                    "True_Neither:Some_tag False_Neither:Wrong_tag True_Neither:Bad_tag"
                ]
            }
        },
        0.0,
        0
    ),
])
def test_map_calculation(test_case, expected_map, expected_perfect, temp_csv_files):
    """Test MAP@3 calculation for various prediction scenarios."""
    gt_path, sub_path = temp_csv_files(test_case["ground_truth"], test_case["submission"])
    
    result = evaluate(gt_path, sub_path)
    
    assert result.map_score == pytest.approx(expected_map)
    assert result.perfect_predictions == expected_perfect
    assert result.total_observations == len(test_case["ground_truth"]["row_id"])


def test_load_ground_truth(temp_csv_files):
    """Test loading ground truth CSV into Prediction objects."""
    ground_truth_data = {
        "row_id": [1, 2, 3],
        "Category": ["True_Correct", "False_Misconception", "True_Neither"],
        "Misconception": ["Adding_across", "Denominator-only_change", "NA"]
    }
    
    gt_path, _ = temp_csv_files(ground_truth_data, {})
    
    result = _load_ground_truth(gt_path)
    
    assert len(result) == 3
    assert result[1] == Prediction(category=Category.TRUE_CORRECT, misconception="Adding_across")
    assert result[2] == Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Denominator-only_change")
    assert result[3] == Prediction(category=Category.TRUE_NEITHER, misconception=None)


def test_load_submissions(temp_csv_files):
    """Test loading submission CSV into Prediction objects."""
    submission_data = {
        "row_id": [1, 2, 3],
        "predictions": [
            "True_Correct:Adding_across False_Neither:Other_tag",
            "False_Misconception:Denominator-only_change",
            "True_Neither False_Correct"
        ]
    }
    
    _, sub_path = temp_csv_files({}, submission_data)
    
    result = _load_submissions(sub_path)
    
    assert len(result) == 3
    assert len(result[1]) == 2
    assert result[1][0] == Prediction(category=Category.TRUE_CORRECT, misconception="Adding_across")
    assert result[1][1] == Prediction(category=Category.FALSE_NEITHER, misconception="Other_tag")
    assert len(result[2]) == 1
    assert result[2][0] == Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Denominator-only_change")
    assert len(result[3]) == 2
    assert result[3][0] == Prediction(category=Category.TRUE_NEITHER, misconception=None)
    assert result[3][1] == Prediction(category=Category.FALSE_CORRECT, misconception=None)


def test_parse_prediction_string():
    """Test parsing prediction strings into Prediction objects."""
    # Test with misconception
    pred1 = _parse_prediction_string("True_Correct:Adding_across")
    assert pred1 == Prediction(category=Category.TRUE_CORRECT, misconception="Adding_across")
    
    # Test without misconception
    pred2 = _parse_prediction_string("False_Neither")
    assert pred2 == Prediction(category=Category.FALSE_NEITHER, misconception=None)
    
    # Test with empty misconception
    pred3 = _parse_prediction_string("True_Misconception:")
    assert pred3 == Prediction(category=Category.TRUE_MISCONCEPTION, misconception=None)
    
    # Test with whitespace
    pred4 = _parse_prediction_string(" False_Correct : Some_tag ")
    assert pred4 == Prediction(category=Category.FALSE_CORRECT, misconception="Some_tag")


def test_calculate_average_precision():
    """Test average precision calculation for single observations."""
    ground_truth = Prediction(category=Category.TRUE_CORRECT, misconception="Adding_across")
    
    # First position match
    predictions1 = [
        Prediction(category=Category.TRUE_CORRECT, misconception="Adding_across"),
        Prediction(category=Category.FALSE_NEITHER, misconception="Other_tag")
    ]
    assert _calculate_average_precision(ground_truth, predictions1) == 1.0
    
    # Second position match
    predictions2 = [
        Prediction(category=Category.FALSE_NEITHER, misconception="Other_tag"),
        Prediction(category=Category.TRUE_CORRECT, misconception="Adding_across")
    ]
    assert _calculate_average_precision(ground_truth, predictions2) == 0.5
    
    # Third position match
    predictions3 = [
        Prediction(category=Category.FALSE_NEITHER, misconception="Other_tag"),
        Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Wrong_tag"),
        Prediction(category=Category.TRUE_CORRECT, misconception="Adding_across")
    ]
    assert _calculate_average_precision(ground_truth, predictions3) == pytest.approx(1/3)
    
    # No match
    predictions4 = [
        Prediction(category=Category.FALSE_NEITHER, misconception="Other_tag"),
        Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Wrong_tag")
    ]
    assert _calculate_average_precision(ground_truth, predictions4) == 0.0
    
    # Empty predictions
    assert _calculate_average_precision(ground_truth, []) == 0.0


def test_predictions_match():
    """Test prediction matching logic."""
    pred1 = Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Adding_across")
    pred2 = Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Adding_across")
    pred3 = Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Different_tag")
    pred4 = Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Adding_across")
    pred5 = Prediction(category=Category.TRUE_NEITHER, misconception=None)
    pred6 = Prediction(category=Category.TRUE_NEITHER, misconception=None)
    
    # Exact matches
    assert _predictions_match(pred1, pred2) is True
    assert _predictions_match(pred5, pred6) is True
    
    # Different misconceptions
    assert _predictions_match(pred1, pred3) is False
    
    # Different categories
    assert _predictions_match(pred1, pred4) is False


def test_prepare_cross_validation_data(tmp_path):
    """Test preparation of cross-validation data from train.csv."""
    # Create sample train.csv
    train_data = {
        "row_id": [1, 2, 3],
        "QuestionId": [101, 102, 103],
        "QuestionText": ["What is 2+2?", "What is 3+3?", "What is 4+4?"],
        "MC_Answer": ["A", "B", "C"],
        "StudentExplanation": ["I think it's 4", "I think it's 6", "I think it's 8"],
        "Category": ["True_Correct", "False_Misconception", "True_Neither"],
        "Misconception": ["Adding_across", "Denominator-only_change", "NA"]
    }
    
    train_csv_path = tmp_path / "train.csv"
    pd.DataFrame(train_data).to_csv(train_csv_path, index=False)
    
    test_rows, ground_truth_data = _prepare_cross_validation_data(train_csv_path)
    
    # Check test rows
    assert len(test_rows) == 3
    assert test_rows[0].row_id == 1
    assert test_rows[0].question_id == 101
    assert test_rows[0].question_text == "What is 2+2?"
    assert test_rows[0].mc_answer == "A"
    assert test_rows[0].student_explanation == "I think it's 4"
    
    # Check ground truth data
    assert len(ground_truth_data) == 3
    assert list(ground_truth_data.columns) == ["row_id", "Category", "Misconception"]
    assert ground_truth_data.iloc[0]["row_id"] == 1
    assert ground_truth_data.iloc[0]["Category"] == "True_Correct"
    assert ground_truth_data.iloc[0]["Misconception"] == "Adding_across"


def test_save_submission_csv(tmp_path):
    """Test saving predictions in submission CSV format."""
    # Create sample submission rows
    predictions = [
        SubmissionRow(
            row_id=1,
            predicted_categories=[
                Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Adding_across"),
                Prediction(category=Category.FALSE_NEITHER, misconception=None)
            ]
        ),
        SubmissionRow(
            row_id=2,
            predicted_categories=[
                Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Denominator-only_change")
            ]
        )
    ]
    
    submission_path = tmp_path / "submission.csv"
    _save_submission_csv(predictions, submission_path)
    
    # Read back and verify
    result_df = pd.read_csv(submission_path)
    
    assert len(result_df) == 2
    assert result_df.iloc[0]["row_id"] == 1
    assert result_df.iloc[0]["predictions"] == "True_Misconception:Adding_across False_Neither:NA"
    assert result_df.iloc[1]["row_id"] == 2
    assert result_df.iloc[1]["predictions"] == "False_Misconception:Denominator-only_change"


def test_load_ground_truth_file_assertions(tmp_path):
    """Test that _load_ground_truth properly validates input files."""
    # Test missing file
    missing_path = tmp_path / "missing.csv"
    with pytest.raises(AssertionError, match="Ground truth file not found"):
        _load_ground_truth(missing_path)
    
    # Test empty file - create CSV with no data rows but with header
    empty_path = tmp_path / "empty.csv"
    with open(empty_path, 'w') as f:
        f.write("row_id,Category,Misconception\n")
    with pytest.raises(AssertionError, match="Ground truth file cannot be empty"):
        _load_ground_truth(empty_path)


def test_load_submissions_file_assertions(tmp_path):
    """Test that _load_submissions properly validates input files."""
    # Test missing file
    missing_path = tmp_path / "missing.csv"
    with pytest.raises(AssertionError, match="Submission file not found"):
        _load_submissions(missing_path)


def test_load_submissions_with_invalid_predictions(temp_csv_files):
    """Test that _load_submissions handles invalid prediction strings gracefully."""
    submission_data = {
        "row_id": [1, 2, 3],
        "predictions": [
            "True_Correct:Adding_across Invalid_Category:Some_tag",  # Invalid category
            "False_Misconception:Denominator-only_change",  # Valid
            "nan"  # NaN value
        ]
    }
    
    _, sub_path = temp_csv_files({}, submission_data)
    
    result = _load_submissions(sub_path)
    
    # First row should only have the valid prediction
    assert len(result[1]) == 1
    assert result[1][0] == Prediction(category=Category.TRUE_CORRECT, misconception="Adding_across")
    
    # Second row should have the valid prediction
    assert len(result[2]) == 1
    assert result[2][0] == Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Denominator-only_change")
    
    # Third row should be empty due to nan
    assert len(result[3]) == 0


def test_prepare_cross_validation_data_assertions(tmp_path):
    """Test that _prepare_cross_validation_data validates input properly."""
    # Test empty train.csv - create CSV with headers but no data
    empty_path = tmp_path / "empty_train.csv"
    with open(empty_path, 'w') as f:
        f.write("row_id,QuestionId,QuestionText,MC_Answer,StudentExplanation,Category,Misconception\n")
    
    with pytest.raises(AssertionError, match="Training CSV cannot be empty"):
        _prepare_cross_validation_data(empty_path)


