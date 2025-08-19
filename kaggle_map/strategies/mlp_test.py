"""Tests for MLP neural network strategy."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from kaggle_map.models import Category, EvaluationRow, Prediction, SubmissionRow, TrainingRow
from kaggle_map.strategies.mlp import MLPNet, MLPStrategy
from kaggle_map.utils.embedding_models import EmbeddingModel


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_training_data():
    """Sample training data for MLP testing."""
    return [
        TrainingRow(
            row_id=1,
            question_id=100,
            question_text="What is 2+2?",
            mc_answer="4",
            student_explanation="I added them correctly",
            category=Category.TRUE_CORRECT,
            misconception=None,
        ),
        TrainingRow(
            row_id=2,
            question_id=100,
            question_text="What is 2+2?",
            mc_answer="5",
            student_explanation="I made an error",
            category=Category.FALSE_MISCONCEPTION,
            misconception="Adding_across",
        ),
        TrainingRow(
            row_id=3,
            question_id=100,
            question_text="What is 2+2?",
            mc_answer="3",
            student_explanation="Wrong calculation",
            category=Category.FALSE_MISCONCEPTION,
            misconception="Subtraction_instead",
        ),
        TrainingRow(
            row_id=4,
            question_id=101,
            question_text="What is 3*3?",
            mc_answer="9",
            student_explanation="Multiplication",
            category=Category.TRUE_CORRECT,
            misconception=None,
        ),
        TrainingRow(
            row_id=5,
            question_id=101,
            question_text="What is 3*3?",
            mc_answer="6",
            student_explanation="I added instead",
            category=Category.FALSE_MISCONCEPTION,
            misconception="Addition_instead",
        ),
    ]


@pytest.fixture
def temp_training_csv():
    """Create temporary training CSV file for MLP testing."""
    training_data = {
        "row_id": [1, 2, 3, 4, 5],
        "QuestionId": [100, 100, 100, 101, 101],
        "QuestionText": [
            "What is 2+2?",
            "What is 2+2?", 
            "What is 2+2?",
            "What is 3*3?",
            "What is 3*3?",
        ],
        "MC_Answer": ["4", "5", "3", "9", "6"],
        "StudentExplanation": [
            "Correct addition",
            "Made error",
            "Wrong calc", 
            "Multiplication",
            "Added instead",
        ],
        "Category": [
            "True_Correct",
            "False_Misconception",
            "False_Misconception",
            "True_Correct", 
            "False_Misconception",
        ],
        "Misconception": [None, "Adding_across", "Subtraction_instead", None, "Addition_instead"],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        pd.DataFrame(training_data).to_csv(f.name, index=False)
        yield Path(f.name)
        Path(f.name).unlink()  # Cleanup


@pytest.fixture
def sample_test_data():
    """Sample test data for predictions."""
    return [
        EvaluationRow(
            row_id=1001,
            question_id=100,
            question_text="What is 2+2?",
            mc_answer="4",
            student_explanation="Simple addition",
        ),
        EvaluationRow(
            row_id=1002,
            question_id=101,
            question_text="What is 3*3?",
            mc_answer="6",
            student_explanation="Wrong calculation",
        ),
    ]


# =============================================================================
# MLPNet Architecture Tests
# =============================================================================


def test_mlp_net_initialization():
    """Test MLPNet model initialization with question-specific heads."""
    question_misconceptions = {
        100: ["Adding_across", "Subtraction_instead", "NA"],
        101: ["Addition_instead", "NA"],
    }
    
    model = MLPNet(question_misconceptions)
    
    # Check shared trunk
    assert isinstance(model.shared_trunk, torch.nn.Sequential)
    
    # Check question-specific heads
    assert "100" in model.question_heads
    assert "101" in model.question_heads
    
    # Check head output sizes - all should be max size (3 in this case)
    assert model.question_heads["100"].out_features == 3  # Max size
    assert model.question_heads["101"].out_features == 3  # Max size (padded from 2)


def test_mlp_net_forward_pass():
    """Test MLPNet forward pass with sample input."""
    question_misconceptions = {100: ["Misconception1", "NA"]}
    model = MLPNet(question_misconceptions)
    
    # Create sample input: 385 dimensions (384 embedding + 1 correctness)
    sample_input = torch.randn(1, 385)
    
    output = model(sample_input, 100)
    
    # Should output logits for misconceptions
    assert output.shape == (1, 2)  # 1 misconception + NA
    assert isinstance(output, torch.Tensor)


# =============================================================================
# MLPStrategy Core Functionality Tests
# =============================================================================


def test_mlp_strategy_properties():
    """Test MLPStrategy name and description properties."""
    # Create minimal strategy for testing properties
    question_misconceptions = {100: ["Test", "NA"]}
    model = MLPNet(question_misconceptions)
    
    strategy = MLPStrategy(
        model=model,
        correct_answers={100: "4"},
        question_misconceptions=question_misconceptions,
        embedding_model=EmbeddingModel.MINI_LM,
    )
    
    assert strategy.name == "mlp"
    assert "MLP" in strategy.description
    assert "misconception" in strategy.description.lower()


def test_extract_correct_answers(sample_training_data):
    """Test extraction of correct answers from training data."""
    correct_answers = MLPStrategy._extract_correct_answers(sample_training_data)
    
    expected_answers = {100: "4", 101: "9"}
    assert correct_answers == expected_answers


def test_extract_question_misconceptions(sample_training_data):
    """Test extraction of question-specific misconceptions."""
    misconceptions = MLPStrategy._extract_question_misconceptions(sample_training_data)
    
    # Check question 100 has both misconceptions + NA
    assert 100 in misconceptions
    assert "Adding_across" in misconceptions[100]
    assert "Subtraction_instead" in misconceptions[100]
    assert "NA" in misconceptions[100]
    assert len(misconceptions[100]) == 3
    
    # Check question 101 has one misconception + NA
    assert 101 in misconceptions
    assert "Addition_instead" in misconceptions[101]
    assert "NA" in misconceptions[101] 
    assert len(misconceptions[101]) == 2


def test_parse_training_data(temp_training_csv):
    """Test parsing CSV into TrainingRow objects."""
    training_data = MLPStrategy._parse_training_data(temp_training_csv)
    
    assert len(training_data) == 5
    assert all(isinstance(row, TrainingRow) for row in training_data)
    
    # Check first row
    first_row = training_data[0]
    assert first_row.row_id == 1
    assert first_row.question_id == 100
    assert first_row.category == Category.TRUE_CORRECT
    assert first_row.misconception is None


def test_is_answer_correct():
    """Test answer correctness checking."""
    question_misconceptions = {100: ["Test", "NA"]}
    model = MLPNet(question_misconceptions)
    
    strategy = MLPStrategy(
        model=model,
        correct_answers={100: "4", 101: "9"},
        question_misconceptions=question_misconceptions,
        embedding_model=EmbeddingModel.MINI_LM,
    )
    
    assert strategy._is_answer_correct(100, "4") is True
    assert strategy._is_answer_correct(100, "5") is False
    assert strategy._is_answer_correct(101, "9") is True
    assert strategy._is_answer_correct(999, "X") is False  # Unknown question


def test_reconstruct_predictions():
    """Test reconstruction of predictions from misconception probabilities."""
    question_misconceptions = {100: ["Adding_across", "Subtraction_instead", "NA"]}
    model = MLPNet(question_misconceptions)
    
    strategy = MLPStrategy(
        model=model,
        correct_answers={100: "4"},
        question_misconceptions=question_misconceptions,
        embedding_model=EmbeddingModel.MINI_LM,
    )
    
    # Test case: misconception detected
    probs = np.array([0.8, 0.1, 0.2])  # High prob for Adding_across
    predictions = strategy._reconstruct_predictions(probs, is_correct=False, question_id=100)
    
    assert len(predictions) >= 1
    assert predictions[0].category == Category.FALSE_MISCONCEPTION
    assert predictions[0].misconception == "Adding_across"
    
    # Test case: no misconception detected (all low probs except NA)
    probs = np.array([0.1, 0.2, 0.9])  # High prob for NA
    predictions = strategy._reconstruct_predictions(probs, is_correct=True, question_id=100)
    
    assert len(predictions) >= 1
    assert predictions[0].category == Category.TRUE_NEITHER


def test_create_default_prediction():
    """Test creation of default predictions for unknown questions."""
    question_misconceptions = {100: ["Test", "NA"]}
    model = MLPNet(question_misconceptions)
    
    strategy = MLPStrategy(
        model=model,
        correct_answers={100: "4"},
        question_misconceptions=question_misconceptions,
        embedding_model=EmbeddingModel.MINI_LM,
    )
    
    # Test with correct answer
    row = EvaluationRow(
        row_id=1,
        question_id=999,  # Unknown question
        question_text="Unknown question",
        mc_answer="4",  # Would be correct for question 100
        student_explanation="Test",
    )
    
    predictions = strategy._create_default_prediction(row)
    
    assert len(predictions) == 1
    assert predictions[0].category == Category.FALSE_NEITHER  # Unknown question -> False


# =============================================================================
# Integration Tests (require small training)
# =============================================================================


@pytest.mark.slow  
def test_mlp_strategy_fit_and_predict_integration(temp_training_csv):
    """Test full fit and predict pipeline with minimal training."""
    # This test requires actual training, so mark as slow
    strategy = MLPStrategy.fit(temp_training_csv)
    
    # Verify strategy was created correctly
    assert len(strategy.correct_answers) == 2  # Questions 100 and 101
    assert len(strategy.question_misconceptions) == 2
    assert strategy.model is not None
    
    # Test prediction
    test_data = [
        EvaluationRow(
            row_id=1001,
            question_id=100,
            question_text="What is 2+2?",
            mc_answer="4",
            student_explanation="Correct answer",
        )
    ]
    
    predictions = strategy.predict(test_data)
    
    assert len(predictions) == 1
    assert isinstance(predictions[0], SubmissionRow)
    assert predictions[0].row_id == 1001
    assert len(predictions[0].predicted_categories) <= 3
    assert all(isinstance(pred, Prediction) for pred in predictions[0].predicted_categories)


def test_mlp_strategy_empty_test_data(temp_training_csv):
    """Test strategy handles empty test data gracefully."""
    strategy = MLPStrategy.fit(temp_training_csv)
    predictions = strategy.predict([])
    
    assert predictions == []


def test_mlp_strategy_unknown_question_handling(temp_training_csv):
    """Test strategy handles unknown questions in test data."""
    strategy = MLPStrategy.fit(temp_training_csv)
    
    test_data = [
        EvaluationRow(
            row_id=1001,
            question_id=999,  # Not in training data
            question_text="Unknown question",
            mc_answer="X",
            student_explanation="Unknown",
        )
    ]
    
    predictions = strategy.predict(test_data)
    
    assert len(predictions) == 1
    assert len(predictions[0].predicted_categories) >= 1
    # Should default to Neither category


# =============================================================================
# Serialization Tests
# =============================================================================


@pytest.mark.slow
def test_mlp_strategy_save_load_round_trip(temp_training_csv):
    """Test saving and loading preserves model functionality."""
    original_strategy = MLPStrategy.fit(temp_training_csv)
    
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        temp_model_path = Path(f.name)
        
        try:
            # Save and load
            original_strategy.save(temp_model_path)
            loaded_strategy = MLPStrategy.load(temp_model_path)
            
            # Verify metadata preservation
            assert loaded_strategy.correct_answers == original_strategy.correct_answers
            assert loaded_strategy.question_misconceptions == original_strategy.question_misconceptions
            
            # Test that loaded model can make predictions
            test_data = [
                EvaluationRow(
                    row_id=1001,
                    question_id=100,
                    question_text="Test",
                    mc_answer="4",
                    student_explanation="Test",
                )
            ]
            
            original_predictions = original_strategy.predict(test_data)
            loaded_predictions = loaded_strategy.predict(test_data)
            
            # Predictions should have same structure (may differ due to model randomness)
            assert len(original_predictions) == len(loaded_predictions)
            assert original_predictions[0].row_id == loaded_predictions[0].row_id
            
        finally:
            temp_model_path.unlink()


def test_mlp_strategy_load_nonexistent_file():
    """Test loading from nonexistent file raises clear error."""
    nonexistent_path = Path("nonexistent_model.pkl")
    
    with pytest.raises(AssertionError, match="Model file not found"):
        MLPStrategy.load(nonexistent_path)


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_extract_correct_answers_empty_data():
    """Test extraction fails gracefully with empty training data."""
    with pytest.raises(AssertionError, match="Training data cannot be empty"):
        MLPStrategy._extract_correct_answers([])


def test_extract_correct_answers_no_correct_category():
    """Test extraction fails when no TRUE_CORRECT categories exist."""
    training_data = [
        TrainingRow(
            row_id=1,
            question_id=100,
            question_text="Test",
            mc_answer="A",
            student_explanation="Test",
            category=Category.FALSE_NEITHER,
            misconception=None,
        )
    ]
    
    with pytest.raises(AssertionError, match="Must find at least one correct answer"):
        MLPStrategy._extract_correct_answers(training_data)


def test_extract_correct_answers_conflicting_answers():
    """Test extraction fails with conflicting correct answers for same question."""
    training_data = [
        TrainingRow(
            row_id=1,
            question_id=100,
            question_text="Test",
            mc_answer="A",
            student_explanation="Test",
            category=Category.TRUE_CORRECT,
            misconception=None,
        ),
        TrainingRow(
            row_id=2,
            question_id=100,
            question_text="Test",
            mc_answer="B",  # Conflicting answer
            student_explanation="Test",
            category=Category.TRUE_CORRECT,
            misconception=None,
        ),
    ]
    
    with pytest.raises(AssertionError, match="Conflicting correct answers"):
        MLPStrategy._extract_correct_answers(training_data)


def test_parse_training_data_missing_file():
    """Test parsing fails with clear error for missing CSV file."""
    missing_path = Path("nonexistent.csv")
    
    with pytest.raises(AssertionError, match="Training file not found"):
        MLPStrategy._parse_training_data(missing_path)


def test_extract_question_misconceptions_empty_data():
    """Test misconception extraction fails gracefully with empty data."""
    with pytest.raises(AssertionError, match="Training data cannot be empty"):
        MLPStrategy._extract_question_misconceptions([])