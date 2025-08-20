"""Tests for model fitting functionality."""

from pathlib import Path
import pytest
import pandas as pd
from kaggle_map.models import Category, Prediction
from kaggle_map.strategies.baseline import BaselineStrategy


@pytest.fixture
def temp_train_csv(tmp_path):
    """Helper to create temporary training CSV files for testing."""
    def _create_file(training_data):
        train_path = tmp_path / "train.csv"
        pd.DataFrame(training_data).to_csv(train_path, index=False)
        return train_path
    return _create_file


@pytest.mark.parametrize("test_case", [
    # Basic fitting test with simple data
    {
        "training_data": {
            "row_id": [1, 2, 3, 4, 5, 6],
            "QuestionId": [101, 101, 101, 102, 102, 102],
            "QuestionText": ["Q1"] * 3 + ["Q2"] * 3,
            "MC_Answer": ["A", "A", "B", "X", "X", "Y"],
            "StudentExplanation": ["Correct reasoning", "Good answer", "Wrong choice", "Right", "Correct", "Incorrect"],
            "Category": ["True_Correct", "True_Neither", "False_Misconception", "True_Correct", "True_Misconception", "False_Neither"],
            "Misconception": ["NA", "NA", "AdditionError", "NA", "MultiplicationMistake", "NA"]
        },
        "expected_questions": 2,
        "expected_correct_answers": {"101": "A", "102": "X"},
        "expected_misconceptions": {"101": "AdditionError", "102": "MultiplicationMistake"}
    }
])
def test_model_fitting(test_case, temp_train_csv):
    """Test basic model fitting functionality."""
    train_path = temp_train_csv(test_case["training_data"])
    
    # Fit the model
    model = BaselineStrategy.fit(train_path)
    
    # Check basic structure
    assert len(model.correct_answers) == test_case["expected_questions"]
    assert len(model.category_frequencies) == test_case["expected_questions"]
    assert len(model.common_misconceptions) == test_case["expected_questions"]
    
    # Check correct answers
    for qid, answer in test_case["expected_correct_answers"].items():
        assert model.correct_answers[int(qid)] == answer
    
    # Check misconceptions
    for qid, misconception in test_case["expected_misconceptions"].items():
        assert model.common_misconceptions[int(qid)] == misconception


def test_model_serialization(temp_train_csv):
    """Test model saving and loading."""
    training_data = {
        "row_id": [1, 2],
        "QuestionId": [101, 101],
        "QuestionText": ["Q1", "Q1"],
        "MC_Answer": ["A", "A"],
        "StudentExplanation": ["Good", "Bad"],
        "Category": ["True_Correct", "False_Neither"],
        "Misconception": ["NA", "NA"]
    }
    
    train_path = temp_train_csv(training_data)
    original_model = BaselineStrategy.fit(train_path)
    
    # Save and load
    save_path = train_path.parent / "test_model.json"
    original_model.save(save_path)
    loaded_model = BaselineStrategy.load(save_path)
    
    # Verify loaded model matches original
    assert loaded_model.correct_answers == original_model.correct_answers
    assert len(loaded_model.category_frequencies) == len(original_model.category_frequencies)
    assert loaded_model.common_misconceptions == original_model.common_misconceptions


def test_model_prediction_format(temp_train_csv):
    """Test that model produces correct prediction format."""
    training_data = {
        "row_id": [1, 2, 3],
        "QuestionId": [101, 101, 101],
        "QuestionText": ["Q1", "Q1", "Q1"],
        "MC_Answer": ["A", "A", "B"],
        "StudentExplanation": ["Good", "OK", "Wrong"],
        "Category": ["True_Correct", "True_Neither", "False_Misconception"],
        "Misconception": ["NA", "NA", "TestError"]
    }
    
    train_path = temp_train_csv(training_data)
    model = BaselineStrategy.fit(train_path)
    
    # Create test data
    from kaggle_map.models import EvaluationRow
    test_row = EvaluationRow(
        row_id=999,
        question_id=101,
        question_text="Test question",
        mc_answer="A",
        student_explanation="Test explanation"
    )
    
    prediction = model.predict(test_row)
    
    # Check prediction structure
    assert prediction.row_id == 999
    assert len(prediction.predicted_categories) <= 3
    
    # Check all predictions are valid Prediction objects
    for pred in prediction.predicted_categories:
        assert isinstance(pred, Prediction)
        assert ":" in str(pred)  # Should be in Category:Misconception format