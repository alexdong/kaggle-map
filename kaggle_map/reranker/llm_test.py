"""Tests for LLM reranker functionality."""

import pytest

from kaggle_map.core.models import Category, EvaluationRow, Prediction
from kaggle_map.reranker.rerank import RerankingRequest, build_reranking_prompt, parse_reranking_response


def test_build_reranking_prompt():
    # Arrange - Real fraction division problem
    evaluation_row = EvaluationRow(
        row_id=1,
        question_id=100,
        question_text="Calculate \\( \\frac{1}{2} \\div 6 \\)",
        correct_answer="\\( \\frac{1}{12} \\)",
        mc_answer="\\( 3 \\)",
        student_explanation="1 / 2 of 6 is 3, so the answer is B.",
    )

    predictions = [
        Prediction(category=Category.TRUE_MISCONCEPTION, misconception="SwapDividend"),
        Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Mult"),
        Prediction(category=Category.TRUE_NEITHER, misconception="NA"),
    ]

    request = RerankingRequest(
        evaluation_row=evaluation_row,
        candidate_predictions=predictions,
    )

    # Act
    prompt = build_reranking_prompt(request)

    # Assert - Updated for new prompt format
    assert "Student answered: 3" in prompt
    assert "Student explained: 1 / 2 of 6 is 3" in prompt
    assert "1. True_Misconception:SwapDividend" in prompt
    assert "2. False_Misconception:Mult" in prompt
    assert "3. True_Neither:NA" in prompt
    assert "Output format: numbers only, comma-separated" in prompt
    assert "Example outputs:" in prompt


def test_build_reranking_prompt_with_none_correct_answer():
    # Arrange - Real algebra problem
    evaluation_row = EvaluationRow(
        row_id=1,
        question_id=100,
        question_text="\\( 2y = 24 \\) What is the value of \\( y \\)?",
        correct_answer=None,  # Sometimes correct answer is not provided
        mc_answer="\\( 12 \\)",
        student_explanation="i think because 2*12is 24 so that's why im sort of guessing",
    )

    predictions = [
        Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Not_variable"),
    ]

    request = RerankingRequest(
        evaluation_row=evaluation_row,
        candidate_predictions=predictions,
    )

    # Act
    prompt = build_reranking_prompt(request)

    # Assert - Updated for new prompt format (doesn't include correct answer anymore)
    assert "Student answered: 12" in prompt
    assert "Student explained: i think because 2*12is 24" in prompt


def test_parse_reranking_response_simple():
    # Arrange - Real misconception types
    original_predictions = [
        Prediction(category=Category.TRUE_MISCONCEPTION, misconception="SwapDividend"),
        Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Mult"),
        Prediction(category=Category.TRUE_CORRECT, misconception="NA"),
    ]
    response = "3,1,2"

    # Act
    reordered = parse_reranking_response(response, original_predictions)

    # Assert
    assert len(reordered) == 3
    assert reordered[0].misconception == "NA"
    assert reordered[1].misconception == "SwapDividend"
    assert reordered[2].misconception == "Mult"


def test_parse_reranking_response_with_extra_text():
    # Arrange - Real fraction misconceptions
    original_predictions = [
        Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Wrong_Fraction"),
        Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Additive"),
    ]
    response = "The best order is: 2, 1."

    # Act
    reordered = parse_reranking_response(response, original_predictions)

    # Assert
    assert len(reordered) == 2
    assert reordered[0].misconception == "Additive"
    assert reordered[1].misconception == "Wrong_Fraction"


def test_parse_reranking_response_no_numbers():
    # Arrange - Single misconception case
    original_predictions = [
        Prediction(category=Category.TRUE_MISCONCEPTION, misconception="FlipChange"),
    ]
    response = "No numbers here"

    # Act & Assert
    with pytest.raises(AssertionError, match="No numbers found in reranking response"):
        parse_reranking_response(response, original_predictions)


def test_parse_reranking_response_invalid_index():
    # Arrange - Common misconceptions for fractions
    original_predictions = [
        Prediction(category=Category.TRUE_MISCONCEPTION, misconception="WNB"),
        Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Irrelevant"),
    ]
    response = "1,2,3"  # Index 3 is out of bounds (3-1=2, but only indices 0,1 exist)

    # Act & Assert
    with pytest.raises(AssertionError, match="Invalid indices in reranking response"):
        parse_reranking_response(response, original_predictions)


def test_parse_reranking_response_missing_indices():
    # Arrange - Mix of real misconception categories
    original_predictions = [
        Prediction(category=Category.TRUE_MISCONCEPTION, misconception="SwapDividend"),
        Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Not_variable"),
        Prediction(category=Category.TRUE_CORRECT, misconception="NA"),
    ]
    response = "1,2"  # Missing index 3

    # Act & Assert
    with pytest.raises(AssertionError, match="Missing indices in reranking: expected 3, got 2"):
        parse_reranking_response(response, original_predictions)


def test_parse_reranking_response_duplicate_indices():
    # Arrange - Common algebra misconceptions
    original_predictions = [
        Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Additive"),
        Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Mult"),
    ]
    response = "1,1"  # Duplicate index

    # Act & Assert
    with pytest.raises(AssertionError, match="Missing indices in reranking: expected 2, got 1"):
        parse_reranking_response(response, original_predictions)
