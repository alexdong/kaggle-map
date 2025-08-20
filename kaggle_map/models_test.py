"""Tests for core data structures and models."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from kaggle_map.models import (
    Category,
    EvaluationResult,
    Prediction,
    SubmissionRow,
    EvaluationRow,
    TrainingRow,
)
from kaggle_map.strategies.baseline import BaselineStrategy


# =============================================================================
# Category Enum Tests
# =============================================================================


def test_category_enum_has_all_expected_competition_values():
    """Category enum contains all competition categories."""
    expected_values = {
        "True_Correct",
        "True_Neither", 
        "True_Misconception",
        "False_Correct",
        "False_Neither",
        "False_Misconception"
    }
    
    actual_values = {category.value for category in Category}
    assert actual_values == expected_values


def test_misconception_categories_end_with_misconception_suffix():
    """Categories ending with '_Misconception' are identified as misconception categories."""
    misconception_categories = [
        Category.TRUE_MISCONCEPTION,
        Category.FALSE_MISCONCEPTION
    ]
    
    for category in misconception_categories:
        assert category.is_misconception, f"{category.value} should be a misconception category"


def test_correct_answer_categories_start_with_true_prefix():
    """Categories starting with 'True_' are identified as correct answer categories."""
    correct_answer_categories = [
        Category.TRUE_CORRECT,
        Category.TRUE_NEITHER,
        Category.TRUE_MISCONCEPTION
    ]
    
    for category in correct_answer_categories:
        assert category.is_correct_answer, f"{category.value} should be a correct answer category"


def test_by_truth_value_returns_correct_categories_for_true():
    """by_truth_value with is_true=True returns all TRUE_* categories."""
    true_categories = Category.by_truth_value(is_true=True)
    
    expected_categories = {
        Category.TRUE_CORRECT,
        Category.TRUE_NEITHER,
        Category.TRUE_MISCONCEPTION
    }
    
    assert set(true_categories) == expected_categories
    assert len(true_categories) == 3


def test_by_truth_value_returns_correct_categories_for_false():
    """by_truth_value with is_true=False returns all FALSE_* categories."""
    false_categories = Category.by_truth_value(is_true=False)
    
    expected_categories = {
        Category.FALSE_CORRECT,
        Category.FALSE_NEITHER,
        Category.FALSE_MISCONCEPTION
    }
    
    assert set(false_categories) == expected_categories
    assert len(false_categories) == 3


# =============================================================================
# Prediction Class Tests
# =============================================================================


def test_prediction_formats_misconception_categories_with_tag():
    """Misconception categories include the misconception tag in value."""
    prediction = Prediction(
        category=Category.TRUE_MISCONCEPTION, 
        misconception="Adding_across"
    )
    
    assert str(prediction) == "True_Misconception:Adding_across"
    assert str(prediction) == "True_Misconception:Adding_across"


def test_prediction_formats_non_misconception_categories_with_na():
    """Non-misconception categories get ':NA' suffix regardless of misconception field."""
    prediction = Prediction(
        category=Category.TRUE_CORRECT,
        misconception="SomeValue"  # Should be ignored
    )
    
    assert str(prediction) == "True_Correct:NA"


def test_prediction_ignores_misconception_for_non_misconception_categories():
    """Misconception field is ignored for categories that aren't misconceptions."""
    pred_with_misconception = Prediction(
        category=Category.FALSE_NEITHER,
        misconception="TestError"
    )
    pred_without_misconception = Prediction(
        category=Category.FALSE_NEITHER
    )
    
    assert str(pred_with_misconception) == str(pred_without_misconception)
    assert str(pred_with_misconception) == "False_Neither:NA"


def test_prediction_string_representation_matches_value_property():
    """String representation equals value property for easy usage."""
    prediction = Prediction(
        category=Category.FALSE_MISCONCEPTION,
        misconception="Denominator-only_change"
    )
    
    assert str(prediction) == str(prediction)  # This test is now redundant but shows __str__ works
    assert str(prediction) == "False_Misconception:Denominator-only_change"


# =============================================================================
# EvaluationResult Validation Tests
# =============================================================================


def test_evaluation_result_accepts_map_scores_between_zero_and_one():
    """Valid MAP scores between 0.0 and 1.0 are accepted."""
    valid_scores = [0.0, 0.5, 0.8333, 1.0]
    
    for score in valid_scores:
        result = EvaluationResult(
            map_score=score,
            total_observations=10,
            perfect_predictions=5,
        )
        assert result.map_score == score


def test_evaluation_result_rejects_negative_map_scores():
    """MAP scores below 0.0 are rejected with clear error."""
    with pytest.raises(ValidationError, match="MAP score must be between 0 and 1"):
        EvaluationResult(
            map_score=-0.1,
            total_observations=10,
            perfect_predictions=0,
        )


def test_evaluation_result_rejects_map_scores_above_one():
    """MAP scores above 1.0 are rejected with clear error."""
    with pytest.raises(ValidationError, match="MAP score must be between 0 and 1"):
        EvaluationResult(
            map_score=1.1,
            total_observations=10,
            perfect_predictions=0,
        )


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_training_data():
    """Sample training data for testing."""
    return [
        TrainingRow(
            row_id=1,
            question_id=100,
            question_text="What is 2+2?",
            mc_answer="4",
            student_explanation="I added them",
            category=Category.TRUE_CORRECT,
            misconception=None
        ),
        TrainingRow(
            row_id=2,
            question_id=100,
            question_text="What is 2+2?",
            mc_answer="5",
            student_explanation="I counted wrong",
            category=Category.FALSE_MISCONCEPTION,
            misconception="Adding_across"
        ),
        TrainingRow(
            row_id=3,
            question_id=101,
            question_text="What is 3+3?",
            mc_answer="6",
            student_explanation="Correct answer",
            category=Category.TRUE_CORRECT,
            misconception=None
        )
    ]


@pytest.fixture
def sample_test_data():
    """Sample test data for predictions."""
    return [
        EvaluationRow(
            row_id=1001,
            question_id=100,
            question_text="What is 2+2?",
            mc_answer="4",
            student_explanation="Simple addition"
        ),
        EvaluationRow(
            row_id=1002,
            question_id=101,
            question_text="What is 3+3?",
            mc_answer="7",
            student_explanation="Wrong answer"
        )
    ]


@pytest.fixture
def temp_training_csv():
    """Create temporary training CSV file."""
    training_data = {
        "row_id": [1, 2, 3, 4],
        "QuestionId": [100, 100, 101, 101],
        "QuestionText": ["What is 2+2?", "What is 2+2?", "What is 3+3?", "What is 3+3?"],
        "MC_Answer": ["4", "5", "6", "7"],
        "StudentExplanation": ["Correct", "Wrong", "Right", "Mistake"],
        "Category": ["True_Correct", "False_Misconception", "True_Correct", "False_Neither"],
        "Misconception": [None, "Adding_across", None, None]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        pd.DataFrame(training_data).to_csv(f.name, index=False)
        yield Path(f.name)
        Path(f.name).unlink()  # Cleanup


# =============================================================================
# BaselineStrategy Core Behavior Tests
# =============================================================================


def test_model_fit_extracts_correct_answers_from_training_data(temp_training_csv):
    """Model extracts correct answer for each question from True_Correct entries."""
    model = BaselineStrategy.fit(temp_training_csv)
    
    expected_answers = {100: "4", 101: "6"}
    assert model.correct_answers == expected_answers


def test_model_fit_builds_category_patterns_by_answer_correctness(temp_training_csv):
    """Model builds category frequency patterns based on answer correctness."""
    model = BaselineStrategy.fit(temp_training_csv)
    
    # Question 100: student answered "4" (correct) -> True_Correct, student answered "5" (wrong) -> False_Misconception
    assert Category.TRUE_CORRECT in model.category_frequencies[100][True]
    assert Category.FALSE_MISCONCEPTION in model.category_frequencies[100][False]


def test_model_fit_finds_most_common_misconceptions_per_question(temp_training_csv):
    """Model identifies the most common misconception for each question."""
    model = BaselineStrategy.fit(temp_training_csv)
    
    assert model.common_misconceptions[100] == "Adding_across"
    # Question 101 has no misconceptions, so it won't be in the dictionary
    assert 101 not in model.common_misconceptions or model.common_misconceptions[101] is None


def test_model_predict_returns_up_to_three_predictions_per_row(temp_training_csv, sample_test_data):
    """Model predictions contain at most 3 categories per test row."""
    model = BaselineStrategy.fit(temp_training_csv)
    predictions = [model.predict(row) for row in sample_test_data]
    
    for prediction in predictions:
        assert len(prediction.predicted_categories) <= 3
        assert isinstance(prediction, SubmissionRow)


def test_model_predict_applies_misconceptions_to_misconception_categories(temp_training_csv):
    """Model applies misconception tags to misconception categories in predictions."""
    model = BaselineStrategy.fit(temp_training_csv)
    
    # Test with incorrect answer to trigger misconception categories
    test_row = EvaluationRow(
        row_id=9999,
        question_id=100,
        question_text="What is 2+2?",
        mc_answer="5",  # Wrong answer
        student_explanation="Wrong explanation"
    )
    
    prediction = model.predict(test_row)
    prediction_values = [str(pred) for pred in prediction.predicted_categories]
    
    # Should contain misconception category with the actual misconception
    assert any("False_Misconception:Adding_across" in val for val in prediction_values)


def test_model_predict_uses_category_frequencies_for_ordering(temp_training_csv):
    """Model orders predictions by frequency from training data."""
    model = BaselineStrategy.fit(temp_training_csv)
    
    test_row = EvaluationRow(
        row_id=9999,
        question_id=100,
        question_text="What is 2+2?", 
        mc_answer="4",  # Correct answer
        student_explanation="Right"
    )
    
    prediction = model.predict(test_row)
    
    # First prediction should be the most frequent category for correct answers
    assert prediction.predicted_categories[0].category == Category.TRUE_CORRECT


# =============================================================================
# BaselineStrategy Edge Cases & Error Handling Tests
# =============================================================================


def test_model_fit_handles_training_data_without_misconceptions():
    """Model handles training data where no misconceptions are present."""
    training_data = {
        "row_id": [1, 2],
        "QuestionId": [100, 100],
        "QuestionText": ["Test", "Test"],
        "MC_Answer": ["A", "B"],
        "StudentExplanation": ["Exp1", "Exp2"],
        "Category": ["True_Correct", "False_Neither"],
        "Misconception": [None, None]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        pd.DataFrame(training_data).to_csv(f.name, index=False)
        temp_path = Path(f.name)
        
        try:
            model = BaselineStrategy.fit(temp_path)
            # No misconceptions means the question won't be in the dictionary
            assert 100 not in model.common_misconceptions or model.common_misconceptions[100] is None
        finally:
            temp_path.unlink()


def test_model_fit_raises_error_for_conflicting_correct_answers():
    """Model raises clear error when multiple correct answers exist for same question."""
    training_data = {
        "row_id": [1, 2],
        "QuestionId": [100, 100],
        "QuestionText": ["Test", "Test"],
        "MC_Answer": ["A", "B"],  # Conflicting correct answers
        "StudentExplanation": ["Exp1", "Exp2"],
        "Category": ["True_Correct", "True_Correct"],  # Both marked as correct
        "Misconception": [None, None]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        pd.DataFrame(training_data).to_csv(f.name, index=False)
        temp_path = Path(f.name)
        
        try:
            with pytest.raises(AssertionError, match="Conflicting correct answers"):
                BaselineStrategy.fit(temp_path)
        finally:
            temp_path.unlink()


def test_model_fit_requires_at_least_one_correct_answer():
    """Model raises error when no correct answers are found in training data."""
    training_data = {
        "row_id": [1, 2],
        "QuestionId": [100, 100],
        "QuestionText": ["Test", "Test"],
        "MC_Answer": ["A", "B"],
        "StudentExplanation": ["Exp1", "Exp2"],
        "Category": ["False_Neither", "False_Misconception"],  # No True_Correct
        "Misconception": [None, "SomeError"]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        pd.DataFrame(training_data).to_csv(f.name, index=False)
        temp_path = Path(f.name)
        
        try:
            with pytest.raises(AssertionError, match="Must find at least one correct answer"):
                BaselineStrategy.fit(temp_path)
        finally:
            temp_path.unlink()


def test_model_predict_handles_questions_not_in_training_data(temp_training_csv):
    """Model handles test questions that weren't in training data."""
    model = BaselineStrategy.fit(temp_training_csv)
    
    # Test with question ID not in training data
    test_row = EvaluationRow(
        row_id=9999,
        question_id=999,  # Not in training data
        question_text="New question",
        mc_answer="X",
        student_explanation="New explanation"
    )
    
    with pytest.raises(AssertionError):
        model.predict(test_row)


def test_model_predict_handles_empty_test_data_gracefully(temp_training_csv):
    """Model can be used with empty test data by not calling predict."""
    model = BaselineStrategy.fit(temp_training_csv)
    test_data = []
    predictions = [model.predict(row) for row in test_data]
    
    assert predictions == []


# =============================================================================
# BaselineStrategy Serialization Tests  
# =============================================================================


def test_model_serialization_round_trip_preserves_all_data(temp_training_csv):
    """Model serialization and deserialization preserves all model data."""
    original_model = BaselineStrategy.fit(temp_training_csv)
    
    # Serialize to dict and back
    model_dict = original_model.to_dict()
    reconstructed_model = BaselineStrategy.from_dict(model_dict)
    
    assert reconstructed_model.correct_answers == original_model.correct_answers
    assert reconstructed_model.category_frequencies == original_model.category_frequencies
    assert reconstructed_model.common_misconceptions == original_model.common_misconceptions


def test_model_to_dict_creates_json_serializable_format(temp_training_csv):
    """Model to_dict creates format that can be JSON serialized."""
    model = BaselineStrategy.fit(temp_training_csv)
    model_dict = model.to_dict()
    
    # Should be JSON serializable without errors
    json_str = json.dumps(model_dict)
    assert isinstance(json_str, str)
    assert len(json_str) > 0


def test_model_from_dict_recreates_equivalent_model(temp_training_csv):
    """Model from_dict recreates functionally equivalent model."""
    original_model = BaselineStrategy.fit(temp_training_csv)
    
    # Create test data
    test_row = EvaluationRow(
        row_id=1,
        question_id=100,
        question_text="Test",
        mc_answer="4",
        student_explanation="Test"
    )
    
    # Get predictions from original
    original_prediction = original_model.predict(test_row)
    
    # Recreate model and get predictions
    model_dict = original_model.to_dict()
    reconstructed_model = BaselineStrategy.from_dict(model_dict)
    reconstructed_prediction = reconstructed_model.predict(test_row)
    
    # Predictions should be identical
    assert original_prediction.row_id == reconstructed_prediction.row_id
    assert len(original_prediction.predicted_categories) == len(reconstructed_prediction.predicted_categories)
    for o_pred, r_pred in zip(original_prediction.predicted_categories, reconstructed_prediction.predicted_categories):
        assert str(o_pred) == str(r_pred)


# =============================================================================
# File I/O Function Tests
# =============================================================================


def test_save_load_model_round_trip_with_temp_file(temp_training_csv):
    """Save and load model preserves all functionality."""
    original_model = BaselineStrategy.fit(temp_training_csv)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_model_path = Path(f.name)
        
        try:
            # Save and load
            original_model.save(temp_model_path)
            loaded_model = BaselineStrategy.load(temp_model_path)
            
            # Verify equivalence
            assert loaded_model.correct_answers == original_model.correct_answers
            assert loaded_model.category_frequencies == original_model.category_frequencies
            assert loaded_model.common_misconceptions == original_model.common_misconceptions
            
        finally:
            temp_model_path.unlink()


def test_load_model_raises_clear_error_for_missing_file():
    """Load model gives clear error message for missing files."""
    missing_path = Path("nonexistent_model.json")
    
    with pytest.raises(AssertionError, match="Model file not found"):
        BaselineStrategy.load(missing_path)


# =============================================================================
# Data Structure Tests
# =============================================================================


def test_training_row_creation_with_valid_category_enum():
    """TrainingRow accepts valid Category enum values."""
    row = TrainingRow(
        row_id=1,
        question_id=100,
        question_text="Test question",
        mc_answer="A",
        student_explanation="Test explanation",
        category=Category.TRUE_CORRECT,
        misconception=None
    )
    
    assert row.category == Category.TRUE_CORRECT
    assert row.misconception is None


def test_training_row_handles_none_misconceptions():
    """TrainingRow properly handles None misconceptions."""
    row_with_none = TrainingRow(
        row_id=1,
        question_id=100,
        question_text="Test",
        mc_answer="A",
        student_explanation="Test",
        category=Category.FALSE_NEITHER,
        misconception=None
    )
    
    row_with_misconception = TrainingRow(
        row_id=2,
        question_id=100,
        question_text="Test",
        mc_answer="B",
        student_explanation="Test",
        category=Category.FALSE_MISCONCEPTION,
        misconception="SomeError"
    )
    
    assert row_with_none.misconception is None
    assert row_with_misconception.misconception == "SomeError"


def test_submission_row_limits_predictions_to_three_items():
    """SubmissionRow can hold up to 3 predictions as per competition rules."""
    predictions = [
        Prediction(category=Category.TRUE_CORRECT),
        Prediction(category=Category.FALSE_NEITHER),  
        Prediction(category=Category.TRUE_MISCONCEPTION, misconception="TestError")
    ]
    
    submission = SubmissionRow(
        row_id=1,
        predicted_categories=predictions
    )
    
    assert len(submission.predicted_categories) == 3
    assert all(isinstance(pred, Prediction) for pred in submission.predicted_categories)


def test_evaluation_row_automatically_normalizes_text_fields():
    """EvaluationRow automatically normalizes text fields using field validators."""
    row = EvaluationRow(
        row_id=1,
        question_id=100,
        question_text="  What is 2+2?  ",  # Extra whitespace
        mc_answer="\\frac{3}{4}",  # LaTeX fraction
        student_explanation="  I think it is four  "  # Extra whitespace
    )
    
    # Text should be normalized automatically
    assert row.question_text == "What is 2+2?"
    assert row.mc_answer == "3/4"  # LaTeX fraction normalized
    assert row.student_explanation == "I think it is four"


def test_training_row_inherits_normalization_from_evaluation_row():
    """TrainingRow inherits automatic normalization from EvaluationRow."""
    row = TrainingRow(
        row_id=1,
        question_id=100,
        question_text="  What is 2+2?  ",  # Extra whitespace
        mc_answer="\\( \\frac{1}{2} \\)",  # LaTeX with parentheses
        student_explanation="  I think it is four  ",  # Extra whitespace
        category=Category.TRUE_CORRECT,
        misconception=None
    )
    
    # Text should be normalized automatically via inheritance
    assert row.question_text == "What is 2+2?"
    assert row.mc_answer == "1/2"  # LaTeX normalized to fraction
    assert row.student_explanation == "I think it is four"
    assert row.category == Category.TRUE_CORRECT