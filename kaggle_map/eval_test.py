"""Tests for MAP@3 evaluation metric implementation."""

from pathlib import Path
import pytest
import pandas as pd
from pydantic import BaseModel
from kaggle_map.eval import evaluate_map3, EvaluationResult


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
                    "True_Correct:Adding_across False_Neither:Other_tag True_Other:Another_tag",
                    "False_Misconception:Denominator-only_change True_Right:Some_tag False_Bad:Bad_tag", 
                    "True_Misconception:Incorrect_equivalent_fraction_addition False_Neither:Wrong_tag True_Other:Other_tag"
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
                    "False_Neither:Other_tag True_Correct:Adding_across True_Other:Another_tag",
                    "True_Right:Some_tag False_Misconception:Denominator-only_change False_Bad:Bad_tag",
                    "False_Neither:Wrong_tag True_Misconception:Incorrect_equivalent_fraction_addition True_Other:Other_tag"
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
                    "False_Neither:Other_tag True_Other:Another_tag True_Correct:Adding_across",
                    "True_Right:Some_tag False_Bad:Bad_tag False_Misconception:Denominator-only_change"
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
                "Misconception": ["Adding_across", "Denominator-only_change", "Incorrect_equivalent_fraction_addition", "Some_other_tag"]
            },
            "submission": {
                "row_id": [1, 2, 3, 4],
                "predictions": [
                    "True_Correct:Adding_across False_Neither:Other_tag True_Other:Another_tag",         # 1st position: AP = 1.0
                    "False_Neither:Other_tag False_Misconception:Denominator-only_change True_Other:Another_tag",  # 2nd position: AP = 0.5
                    "False_Neither:Other_tag True_Other:Another_tag True_Misconception:Incorrect_equivalent_fraction_addition",  # 3rd position: AP = 1/3
                    "False_Neither:Other_tag True_Other:Another_tag False_Bad:Bad_tag"                   # Not found: AP = 0.0
                ]
            }
        },
        (1.0 + 0.5 + 1/3 + 0.0) / 4,
        1
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
                    "False_Neither:Other_tag True_Other:Another_tag False_Bad:Bad_tag",
                    "True_Wrong:Some_tag False_Other:Wrong_tag True_Bad:Bad_tag"
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
    
    result = evaluate_map3(gt_path, sub_path)
    
    assert result.map_score == pytest.approx(expected_map)
    assert result.perfect_predictions == expected_perfect
    assert result.total_observations == len(test_case["ground_truth"]["row_id"])


