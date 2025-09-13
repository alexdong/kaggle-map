"""Tests for LLM evaluation utilities."""

from kaggle_map.core.models import Category
from kaggle_map.llm.utils import parse_predictions


def test_parse_predictions_valid_format():
    """Test parsing valid prediction format."""
    response = "True_Correct:NA True_Neither:NA True_Misconception:Division"
    predictions = parse_predictions(response)

    assert len(predictions) == 3
    assert predictions[0].category == Category.TRUE_CORRECT
    assert predictions[0].misconception == "NA"
    assert predictions[2].category == Category.TRUE_MISCONCEPTION
    assert predictions[2].misconception == "Division"


def test_parse_predictions_partial():
    """Test parsing with less than 3 predictions."""
    response = "True_Misconception:Fractions"
    predictions = parse_predictions(response)

    assert len(predictions) == 3
    assert predictions[0].category == Category.TRUE_MISCONCEPTION
    assert predictions[0].misconception == "Fractions"
    # Should pad with defaults
    assert predictions[1].category == Category.TRUE_CORRECT
    assert predictions[1].misconception == "NA"
