"""Tests for LLM evaluator parsing logic."""

from kaggle_map.core.models import Category, Prediction
from kaggle_map.llm.evaluator import parse_predictions

# Constants
EXPECTED_PREDICTIONS = 3  # LLM should return exactly 3 predictions


def test_prediction_string_representation() -> None:
    """Test that Prediction.__str__ returns the correct format."""
    # Test misconception case
    pred1 = Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Division")
    assert str(pred1) == "True_Misconception:Division"

    # Test NA case
    pred2 = Prediction(category=Category.FALSE_CORRECT, misconception="NA")
    assert str(pred2) == "False_Correct:NA"

    # Test that category.value gives the right string
    pred3 = Prediction(category=Category.TRUE_NEITHER, misconception="NA")
    assert pred3.category.value == "True_Neither"


def test_parse_predictions_single_line_format() -> None:
    """Test parsing of LLM output in the expected format: three predictions on one line."""
    # This is the format the prompt asks for
    response = "True_Correct:NA True_Neither:NA True_Misconception:Division"

    predictions = parse_predictions(response)

    assert len(predictions) == EXPECTED_PREDICTIONS
    assert predictions[0].category == Category.TRUE_CORRECT
    assert predictions[0].misconception == "NA"
    assert predictions[1].category == Category.TRUE_NEITHER
    assert predictions[1].misconception == "NA"
    assert predictions[2].category == Category.TRUE_MISCONCEPTION
    assert predictions[2].misconception == "Division"


def test_parse_predictions_with_misconception_ids() -> None:
    """Test parsing when misconceptions are numeric IDs."""
    response = "False_Misconception:123 False_Neither:NA False_Correct:NA"

    predictions = parse_predictions(response)

    assert len(predictions) == EXPECTED_PREDICTIONS
    assert predictions[0].category == Category.FALSE_MISCONCEPTION
    assert predictions[0].misconception == "123"
    assert predictions[1].category == Category.FALSE_NEITHER
    assert predictions[1].misconception == "NA"
    assert predictions[2].category == Category.FALSE_CORRECT
    assert predictions[2].misconception == "NA"


def test_parse_predictions_with_extra_whitespace() -> None:
    """Test parsing handles extra whitespace correctly."""
    response = "  True_Correct:NA   True_Neither:NA   True_Misconception:Subtraction  "

    predictions = parse_predictions(response)

    assert len(predictions) == EXPECTED_PREDICTIONS
    assert predictions[0].category == Category.TRUE_CORRECT
    assert predictions[1].category == Category.TRUE_NEITHER
    assert predictions[2].category == Category.TRUE_MISCONCEPTION
    assert predictions[2].misconception == "Subtraction"


def test_parse_predictions_with_newlines() -> None:
    """Test that parser handles response with newlines (takes first valid line)."""
    response = """Some preamble text
    True_Correct:NA True_Neither:NA True_Misconception:Division
    Some trailing text"""

    predictions = parse_predictions(response)

    assert len(predictions) == EXPECTED_PREDICTIONS
    assert predictions[0].category == Category.TRUE_CORRECT


def test_parse_predictions_incomplete() -> None:
    """Test parsing when LLM returns fewer than 3 predictions."""
    response = "True_Correct:NA"

    predictions = parse_predictions(response)

    # Should pad with defaults
    assert len(predictions) == EXPECTED_PREDICTIONS
    assert predictions[0].category == Category.TRUE_CORRECT
    assert predictions[1].category == Category.FALSE_NEITHER  # Default
    assert predictions[2].category == Category.FALSE_NEITHER  # Default


def test_parse_predictions_invalid_format() -> None:
    """Test parsing handles invalid formats gracefully."""
    response = "Invalid response without proper format"

    predictions = parse_predictions(response)

    # Should return all defaults
    assert len(predictions) == EXPECTED_PREDICTIONS
    assert all(p.category == Category.FALSE_NEITHER for p in predictions)
    assert all(p.misconception == "NA" for p in predictions)


def test_parse_predictions_mixed_valid_invalid() -> None:
    """Test parsing with mix of valid and invalid predictions."""
    response = "True_Correct:NA InvalidPrediction False_Neither:NA"

    predictions = parse_predictions(response)

    assert len(predictions) == EXPECTED_PREDICTIONS
    assert predictions[0].category == Category.TRUE_CORRECT
    assert predictions[1].category == Category.FALSE_NEITHER  # Second valid prediction
    assert predictions[2].category == Category.FALSE_NEITHER  # Default padding
