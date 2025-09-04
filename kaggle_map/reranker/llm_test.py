"""Tests for LLM reranker functionality."""

import pytest

from kaggle_map.core.models import Category, EvaluationRow, Prediction
from kaggle_map.reranker.llm import build_reranking_prompt, parse_reranking_response
from kaggle_map.reranker.models import RerankingRequest


def test_build_reranking_prompt():
    # Arrange
    evaluation_row = EvaluationRow(
        row_id=1,
        question_id=100,
        question_text="What is 2 + 2?",
        correct_answer="4",
        mc_answer="5",
        student_explanation="I added 2 and 3 instead",
    )
    
    predictions = [
        Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Addition Error"),
        Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Counting Error"),
        Prediction(category=Category.TRUE_NEITHER, misconception="NA"),
    ]
    
    request = RerankingRequest(
        evaluation_row=evaluation_row,
        candidate_predictions=predictions,
    )
    
    # Act
    prompt = build_reranking_prompt(request)
    
    # Assert
    assert "What is 2 + 2?" in prompt
    assert "Correct Answer: 4" in prompt
    assert "Student Answer: 5" in prompt
    assert "I added 2 and 3 instead" in prompt
    assert "1. True_Misconception:Addition Error" in prompt
    assert "2. False_Misconception:Counting Error" in prompt
    assert "3. True_Neither:NA" in prompt
    assert "Reply with ONLY the reordered numbers" in prompt


def test_build_reranking_prompt_with_none_correct_answer():
    # Arrange
    evaluation_row = EvaluationRow(
        row_id=1,
        question_id=100,
        question_text="Solve for x",
        correct_answer=None,
        mc_answer="x = 3",
        student_explanation="I guessed",
    )
    
    predictions = [
        Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Algebra Error"),
    ]
    
    request = RerankingRequest(
        evaluation_row=evaluation_row,
        candidate_predictions=predictions,
    )
    
    # Act
    prompt = build_reranking_prompt(request)
    
    # Assert
    assert "Correct Answer: Not provided" in prompt


def test_parse_reranking_response_simple():
    # Arrange
    original_predictions = [
        Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Type A"),
        Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Type B"),
        Prediction(category=Category.TRUE_CORRECT, misconception="NA"),
    ]
    response = "3,1,2"
    
    # Act
    reordered = parse_reranking_response(response, original_predictions)
    
    # Assert
    assert len(reordered) == 3
    assert reordered[0].misconception == "NA"
    assert reordered[1].misconception == "Type A"
    assert reordered[2].misconception == "Type B"


def test_parse_reranking_response_with_extra_text():
    # Arrange
    original_predictions = [
        Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Type A"),
        Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Type B"),
    ]
    response = "The best order is: 2, 1."
    
    # Act
    reordered = parse_reranking_response(response, original_predictions)
    
    # Assert
    assert len(reordered) == 2
    assert reordered[0].misconception == "Type B"
    assert reordered[1].misconception == "Type A"


def test_parse_reranking_response_no_numbers():
    # Arrange
    original_predictions = [
        Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Type A"),
    ]
    response = "No numbers here"
    
    # Act & Assert
    with pytest.raises(AssertionError, match="No numbers found in reranking response"):
        parse_reranking_response(response, original_predictions)


def test_parse_reranking_response_invalid_index():
    # Arrange
    original_predictions = [
        Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Type A"),
        Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Type B"),
    ]
    response = "1,2,3"  # Index 3 is out of bounds (3-1=2, but only indices 0,1 exist)
    
    # Act & Assert
    with pytest.raises(AssertionError, match="Invalid indices in reranking response"):
        parse_reranking_response(response, original_predictions)


def test_parse_reranking_response_missing_indices():
    # Arrange
    original_predictions = [
        Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Type A"),
        Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Type B"),
        Prediction(category=Category.TRUE_CORRECT, misconception="NA"),
    ]
    response = "1,2"  # Missing index 3
    
    # Act & Assert
    with pytest.raises(AssertionError, match="Missing indices in reranking: expected 3, got 2"):
        parse_reranking_response(response, original_predictions)


def test_parse_reranking_response_duplicate_indices():
    # Arrange
    original_predictions = [
        Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Type A"),
        Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Type B"),
    ]
    response = "1,1"  # Duplicate index
    
    # Act & Assert
    with pytest.raises(AssertionError, match="Missing indices in reranking: expected 2, got 1"):
        parse_reranking_response(response, original_predictions)