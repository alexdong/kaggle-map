"""Tests for model prediction functionality."""

from pathlib import Path
import pytest
import pandas as pd
from kaggle_map.models import Category, Prediction, EvaluationRow
from kaggle_map.strategies.baseline import BaselineStrategy


@pytest.fixture
def temp_test_csv(tmp_path):
    """Helper to create temporary test CSV files for testing."""
    def _create_file(test_data):
        test_path = tmp_path / "test.csv"
        pd.DataFrame(test_data).to_csv(test_path, index=False)
        return test_path
    return _create_file


@pytest.fixture
def simple_model():
    """Create a simple fitted model for testing predictions."""
    from kaggle_map.models import Category
    
    # Create a basic model with known patterns
    correct_answers = {101: "A", 102: "X"}
    
    category_frequencies = {
        101: {
            True: [Category.TRUE_CORRECT, Category.TRUE_NEITHER],
            False: [Category.FALSE_MISCONCEPTION, Category.FALSE_NEITHER]
        },
        102: {
            True: [Category.TRUE_CORRECT, Category.TRUE_MISCONCEPTION],
            False: [Category.FALSE_NEITHER, Category.FALSE_MISCONCEPTION]
        }
    }
    
    common_misconceptions = {101: "AdditionError", 102: "MultiplicationMistake"}
    
    return BaselineStrategy(
        correct_answers=correct_answers,
        category_frequencies=category_frequencies,
        common_misconceptions=common_misconceptions
    )


@pytest.mark.parametrize("test_case", [
    # Basic prediction test
    {
        "test_data": {
            "row_id": [1, 2, 3],
            "QuestionId": [101, 101, 102],
            "QuestionText": ["Q1", "Q1", "Q2"],
            "MC_Answer": ["A", "B", "X"],
            "StudentExplanation": ["Correct", "Wrong", "Right"]
        },
        "expected_rows": 3
    }
])
def test_model_prediction(test_case, simple_model, temp_test_csv):
    """Test basic model prediction functionality."""
    test_path = temp_test_csv(test_case["test_data"])
    
    # Load test data
    test_df = pd.read_csv(test_path)
    test_rows = []
    for _, row in test_df.iterrows():
        test_rows.append(EvaluationRow(
            row_id=int(row["row_id"]),
            question_id=int(row["QuestionId"]),
            question_text=str(row["QuestionText"]),
            mc_answer=str(row["MC_Answer"]),
            student_explanation=str(row["StudentExplanation"])
        ))
    
    # Make predictions
    predictions = simple_model.predict(test_rows)
    
    # Check basic structure
    assert len(predictions) == test_case["expected_rows"]
    
    for pred in predictions:
        assert hasattr(pred, 'row_id')
        assert hasattr(pred, 'predicted_categories')
        assert len(pred.predicted_categories) <= 3  # MAP@3 limit


def test_prediction_format(simple_model):
    """Test that predictions have correct format."""
    test_row = EvaluationRow(
        row_id=999,
        question_id=101,
        question_text="Test question",
        mc_answer="A",
        student_explanation="Test explanation"
    )
    
    predictions = simple_model.predict([test_row])
    
    # Check prediction format
    assert len(predictions) == 1
    prediction = predictions[0]
    
    assert prediction.row_id == 999
    assert isinstance(prediction.predicted_categories, list)
    
    # All predictions should be Prediction objects with correct format
    for pred in prediction.predicted_categories:
        assert isinstance(pred, Prediction)
        pred_str = str(pred)
        assert ":" in pred_str  # Should be Category:Misconception format
        category_part, misconception_part = pred_str.split(":", 1)
        
        # Category should be valid
        # Using assert to validate category instead of try/except
        try:
            category = Category(category_part)
            assert category is not None, f"Category validation failed for: {category_part}"
        except ValueError as e:
            pytest.fail(f"Invalid category in prediction: {category_part} - {e}")


def test_correct_vs_incorrect_predictions(simple_model):
    """Test that model behaves differently for correct vs incorrect answers."""
    # Correct answer
    correct_row = EvaluationRow(
        row_id=1,
        question_id=101,
        question_text="Test",
        mc_answer="A",  # This is the correct answer for question 101
        student_explanation="Correct"
    )
    
    # Incorrect answer
    incorrect_row = EvaluationRow(
        row_id=2,
        question_id=101,
        question_text="Test",
        mc_answer="B",  # This is incorrect for question 101
        student_explanation="Wrong"
    )
    
    correct_predictions = simple_model.predict([correct_row])
    incorrect_predictions = simple_model.predict([incorrect_row])
    
    # Should get different category patterns
    correct_cats = [str(p) for p in correct_predictions[0].predicted_categories]
    incorrect_cats = [str(p) for p in incorrect_predictions[0].predicted_categories]
    
    # Categories should be different (they follow different frequency patterns)
    assert correct_cats != incorrect_cats


def test_misconception_handling(simple_model):
    """Test that misconceptions are properly applied to misconception categories."""
    test_row = EvaluationRow(
        row_id=1,
        question_id=101,
        question_text="Test",
        mc_answer="B",  # Incorrect answer
        student_explanation="Wrong"
    )
    
    predictions = simple_model.predict([test_row])
    prediction_strings = [str(p) for p in predictions[0].predicted_categories]
    
    # Should have some predictions with misconceptions
    misconception_predictions = [p for p in prediction_strings if p.endswith("_Misconception:AdditionError")]
    
    # At least one misconception category should use the common misconception
    assert len(misconception_predictions) > 0