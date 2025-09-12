"""Tests for LLM evaluation utilities."""

from unittest.mock import MagicMock

import pandas as pd
from jinja2 import Template

from kaggle_map.core.models import Category, EvaluationResult
from kaggle_map.llm.utils import evaluate_dataframe, parse_predictions


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


def test_evaluate_dataframe():
    """Test evaluation of a DataFrame with mock LLM."""
    # Create test DataFrame
    df = pd.DataFrame(
        [
            {
                "row_id": 1,
                "QuestionId": 100,
                "QuestionText": "What is 2+2?",
                "MC_Answer": "C",
                "StudentExplanation": "I added them together",
                "Category": "True_Correct",
                "Misconception": "NA",
            },
            {
                "row_id": 2,
                "QuestionId": 101,
                "QuestionText": "What is 3x4?",
                "MC_Answer": "B",
                "StudentExplanation": "I multiplied",
                "Category": "True_Misconception",
                "Misconception": "Multiplication",
            },
        ]
    )

    # Mock LLM that returns predictable results
    mock_llm = MagicMock()
    mock_llm.return_value = {"choices": [{"text": "True_Correct:NA True_Neither:NA True_Misconception:Division"}]}

    # Simple template
    template = Template("Question: {{ question_text }}")

    # Run evaluation
    results, avg_score = evaluate_dataframe(df, template, mock_llm)

    assert len(results) == 2
    assert isinstance(results[0], EvaluationResult)
    assert results[0].row_id == 1
    assert results[0].mc_answer == "C"
    assert 0 <= avg_score <= 1
