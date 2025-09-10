"""Tests for GPT-5 prompt generator."""

import json
from unittest.mock import MagicMock, patch

from kaggle_map.evolution import EvolutionContext
from kaggle_map.evolution.generator import (
    generate_candidates,
    parse_gpt5_response,
    parse_structured_response,
    validate_template_variables,
)


def test_validate_template_variables() -> None:
    """Test template variable validation."""
    # Valid template with all required variables
    valid_template = """
    Question: {{ question_text }}
    Category: {{ category }}
    MC Answer: {{ mc_answer }}
    Student Explanation: {{ student_explanation }}
    """
    assert validate_template_variables(valid_template) is True

    # Missing required variable
    invalid_template = """
    Question: {{ question_text }}
    Category: {{ category }}
    """
    assert validate_template_variables(invalid_template) is False


def test_parse_structured_response() -> None:
    """Test parsing structured JSON response."""
    response_json = json.dumps(
        {
            "candidates": [
                {
                    "hypothesis": "Using chain-of-thought reasoning",
                    "template": "Question: {{ question_text }}\nCategory: {{ category }}\nMC Answer: {{ mc_answer }}\nStudent: {{ student_explanation }}",
                },
                {
                    "hypothesis": "Adding misconception examples",
                    "template": "Q: {{ question_text }}\nCat: {{ category }}\nAnswer: {{ mc_answer }}\nExplanation: {{ student_explanation }}",
                },
            ]
        }
    )

    candidates = parse_structured_response(response_json, generation_id=1)

    assert candidates is not None
    assert len(candidates) == 2
    assert candidates[0].hypothesis == "Using chain-of-thought reasoning"
    assert candidates[1].candidate_id == "gen_01_candidate_1"
    assert "{{ question_text }}" in candidates[0].prompt


def test_parse_gpt5_response() -> None:
    """Test parsing GPT-5 response into candidates."""
    response_text = """
    CANDIDATE 1
    HYPOTHESIS: Adding more context about math misconceptions will improve accuracy
    TEMPLATE:
    Question: {{ question_text }}
    Category: {{ category }}
    MC Answer: {{ mc_answer }}
    Student Explanation: {{ student_explanation }}
    Common misconceptions in this area include...

    CANDIDATE 2
    HYPOTHESIS: Chain-of-thought reasoning helps identify misconceptions
    TEMPLATE:
    Let's think step by step about this problem.
    Question: {{ question_text }}
    Category: {{ category }}
    MC Answer: {{ mc_answer }}
    Student Explanation: {{ student_explanation }}
    """

    candidates = parse_gpt5_response(response_text, generation_id=1)

    assert len(candidates) == 2
    assert candidates[0].candidate_id == "gen_01_candidate_0"
    assert "more context" in candidates[0].hypothesis
    assert "{{ question_text }}" in candidates[0].prompt

    assert candidates[1].candidate_id == "gen_01_candidate_1"
    assert "Chain-of-thought" in candidates[1].hypothesis
    assert "step by step" in candidates[1].prompt


@patch("kaggle_map.evolution.generator.OpenAI")
def test_generate_candidates(mock_openai: MagicMock) -> None:
    """Test generating candidates with mocked GPT-5."""
    # Mock the OpenAI client
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    # Mock the response
    mock_response = MagicMock()
    mock_response.content = """
    CANDIDATE 1
    HYPOTHESIS: Test hypothesis
    TEMPLATE:
    Question: {{ question_text }}
    Category: {{ category }}
    MC Answer: {{ mc_answer }}
    Student Explanation: {{ student_explanation }}
    """
    mock_client.responses.create.return_value = mock_response

    # Create test context
    context = EvolutionContext(
        current_best_prompt="gen_00_candidate_0",
        current_best_score=0.65,
        parent_prompts=[],
        failure_patterns={},
        competition_context="Test competition context",
        next_generation_id=1,
    )

    # Generate candidates
    candidates = generate_candidates(context, num_candidates=1)

    assert len(candidates) == 1
    assert candidates[0].generation == 1
    assert mock_client.responses.create.called
