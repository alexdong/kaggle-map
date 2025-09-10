"""Tests for the reranking prompt template."""

from pathlib import Path

import pytest
from jinja2 import Template

from kaggle_map.core.models import Category, EvaluationRow, Prediction
from kaggle_map.reranker.rerank import RerankingRequest, build_reranking_prompt


def test_prompt_template_exists():
    """Test that the prompt template file exists."""
    template_path = Path(__file__).parent / "prompts" / "prompt.j2"
    assert template_path.exists(), "prompt.j2 template file should exist"


def test_template_renders_all_parameters():
    """Test that the template correctly renders all required parameters."""
    template_path = Path(__file__).parent / "prompts" / "prompt.j2"
    template = Template(template_path.read_text())
    
    # Test data
    mc_answer = "B"
    student_explanation = "The answer is B because..."
    predictions = [
        Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Misconception A"),
        Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Misconception B"),
        Prediction(category=Category.TRUE_CORRECT, misconception="NA"),
    ]
    
    rendered = template.render(
        mc_answer=mc_answer,
        student_explanation=student_explanation,
        predictions=predictions
    )
    
    # Verify all parameters are in the rendered output
    assert mc_answer in rendered, "mc_answer should be in rendered prompt"
    assert student_explanation in rendered, "student_explanation should be in rendered prompt"
    # Check that predictions are numbered correctly
    assert "1." in rendered, "First prediction should be numbered"
    assert "2." in rendered, "Second prediction should be numbered"
    assert "3." in rendered, "Third prediction should be numbered"
    
    # Verify structure
    assert "Student answered:" in rendered
    assert "Student explained:" in rendered
    assert "Predictions:" in rendered
    assert "Output format: numbers only, comma-separated" in rendered
    assert "Your output:" in rendered


def test_build_reranking_prompt_integration():
    """Test that build_reranking_prompt correctly uses the template."""
    # Create test data
    evaluation_row = EvaluationRow(
        row_id=1,
        question_id=100,
        question_text="What is 2 + 2?",
        mc_answer="C",
        student_explanation="I chose C because it makes the most sense"
    )
    
    predictions = [
        Prediction(category=Category.TRUE_MISCONCEPTION, misconception="First misconception"),
        Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Second misconception"),
        Prediction(category=Category.TRUE_NEITHER, misconception="NA"),
    ]
    
    request = RerankingRequest(
        evaluation_row=evaluation_row,
        candidate_predictions=predictions
    )
    
    # Build prompt
    prompt = build_reranking_prompt(request)
    
    # Verify content
    assert "Student answered: C" in prompt
    assert "I chose C because it makes the most sense" in prompt
    assert "1." in prompt  # Check numbering exists
    assert "2." in prompt
    assert "3." in prompt
    assert "Output format: numbers only, comma-separated" in prompt


def test_template_handles_special_characters():
    """Test that the template correctly handles special characters."""
    template_path = Path(__file__).parent / "prompts" / "prompt.j2"
    template = Template(template_path.read_text())
    
    # Test with special characters that might break templating
    mc_answer = "A & B"
    student_explanation = "The answer has {{ braces }} and < tags >"
    predictions = [
        Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Uses 'quotes' and \"double quotes\""),
        Prediction(category=Category.FALSE_MISCONCEPTION, misconception="Has $pecial chars"),
        Prediction(category=Category.TRUE_CORRECT, misconception="NA"),
    ]
    
    rendered = template.render(
        mc_answer=mc_answer,
        student_explanation=student_explanation,
        predictions=predictions
    )
    
    # Verify special characters are preserved
    assert "A & B" in rendered
    assert "{{ braces }}" in rendered
    assert "< tags >" in rendered
