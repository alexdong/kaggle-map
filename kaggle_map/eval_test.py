"""Tests for MAP@3 evaluation metric implementation."""

from pathlib import Path
import pytest
import pandas as pd
from kaggle_map.eval import (
    evaluate, 
    _load_ground_truth, 
    _load_submissions
)
from kaggle_map.metrics import calculate_map_at_3
from kaggle_map.models import (
    Prediction, 
    Category
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
    
    assert result == pytest.approx(expected_map)


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
    assert result[3] == Prediction(category=Category.TRUE_NEITHER, misconception="NA")


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
    assert result[3][0] == Prediction(category=Category.TRUE_NEITHER, misconception="NA")
    assert result[3][1] == Prediction(category=Category.FALSE_CORRECT, misconception="NA")


def test_parse_prediction_string():
    """Test parsing prediction strings into Prediction objects using Prediction.from_string."""
    # Test with misconception
    pred1 = Prediction.from_string("True_Correct:Adding_across")
    assert pred1 == Prediction(category=Category.TRUE_CORRECT, misconception="Adding_across")
    
    # Test without misconception
    pred2 = Prediction.from_string("False_Neither")
    assert pred2 == Prediction(category=Category.FALSE_NEITHER, misconception="NA")
    
    # Test with empty misconception
    pred3 = Prediction.from_string("True_Misconception:")
    assert pred3 == Prediction(category=Category.TRUE_MISCONCEPTION, misconception="NA")
    
    # Test with whitespace
    pred4 = Prediction.from_string(" False_Correct : Some_tag ")
    assert pred4 == Prediction(category=Category.FALSE_CORRECT, misconception="Some_tag")


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


def test_evaluate_with_custom_metric_function(temp_csv_files):
    """Test that evaluate function works with custom metric functions."""
    ground_truth_data = {
        "row_id": [1, 2],
        "Category": ["True_Correct", "False_Misconception"],
        "Misconception": ["Adding_across", "Denominator-only_change"]
    }
    
    submission_data = {
        "row_id": [1, 2],
        "predictions": [
            "True_Correct:Adding_across False_Neither:Other_tag",
            "False_Neither:Other_tag False_Misconception:Denominator-only_change"
        ]
    }
    
    gt_path, sub_path = temp_csv_files(ground_truth_data, submission_data)
    
    # Custom metric that always returns 0.8
    def custom_metric(ground_truth, predictions):
        return 0.8
    
    result = evaluate(gt_path, sub_path, metric_fn=custom_metric)
    
    # Should use custom metric, not MAP@3
    assert result == 0.8


def test_evaluate_with_default_map_at_3_metric(temp_csv_files):
    """Test that evaluate function uses MAP@3 by default."""
    ground_truth_data = {
        "row_id": [1, 2],
        "Category": ["True_Correct", "False_Misconception"],
        "Misconception": ["Adding_across", "Denominator-only_change"]
    }
    
    submission_data = {
        "row_id": [1, 2],
        "predictions": [
            "True_Correct:Adding_across False_Neither:Other_tag",  # 1st position: 1.0
            "False_Neither:Other_tag False_Misconception:Denominator-only_change"  # 2nd position: 0.5
        ]
    }
    
    gt_path, sub_path = temp_csv_files(ground_truth_data, submission_data)
    
    # Test with default metric (should be MAP@3)
    result_default = evaluate(gt_path, sub_path)
    
    # Test with explicit MAP@3 metric
    result_explicit = evaluate(gt_path, sub_path, metric_fn=calculate_map_at_3)
    
    # Both should give the same result
    assert result_default == result_explicit
    assert result_default == pytest.approx((1.0 + 0.5) / 2)