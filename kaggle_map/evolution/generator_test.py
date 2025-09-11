"""Tests for GPT-5 prompt generator."""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from kaggle_map.evolution import EvolutionContext
from kaggle_map.evolution.generator import (
    generate_candidates,
    parse_gpt5_response,
    parse_structured_response,
    validate_template_variables,
)

# Set debug logging for tests
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def valid_template() -> str:
    """Create a valid Jinja2 template with all required variables."""
    return """
    Question: {{ question_text }}
    Category: {{ category }}
    MC Answer: {{ mc_answer }}
    Student Explanation: {{ student_explanation }}
    """


@pytest.fixture
def invalid_template() -> str:
    """Create an invalid template missing required variables."""
    return """
    Question: {{ question_text }}
    Category: {{ category }}
    """


@pytest.fixture
def sample_structured_response() -> str:
    """Create a sample structured JSON response for testing."""
    return json.dumps(
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


@pytest.fixture
def sample_gpt5_response() -> str:
    """Create a sample GPT-5 text response for testing."""
    return """
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


@pytest.fixture
def sample_evolution_context() -> EvolutionContext:
    """Create a sample evolution context for testing."""
    return EvolutionContext(
        current_best_prompt="gen_00_candidate_0",
        current_best_score=0.65,
        parent_prompts=[],
        failure_patterns={},
        competition_context="Test competition context",
        next_generation_id=1,
    )


def test_validate_template_variables_valid(valid_template: str) -> None:
    """Test template variable validation with valid template."""
    is_valid = validate_template_variables(valid_template)
    assert is_valid is True, "Valid template with all required variables should pass validation"


def test_validate_template_variables_invalid(invalid_template: str) -> None:
    """Test template variable validation with invalid template."""
    is_valid = validate_template_variables(invalid_template)
    assert is_valid is False, "Invalid template missing required variables should fail validation"


@pytest.mark.parametrize(
    "missing_var",
    [
        "question_text",
        "category",
        "mc_answer",
        "student_explanation",
    ],
)
def test_validate_template_variables_missing_specific(missing_var: str) -> None:
    """Test template validation when specific required variables are missing."""
    # Create template missing one specific variable
    all_vars = {
        "question_text": "{{ question_text }}",
        "category": "{{ category }}",
        "mc_answer": "{{ mc_answer }}",
        "student_explanation": "{{ student_explanation }}",
    }

    # Remove the specified variable
    del all_vars[missing_var]

    template = "\n".join(f"{key.replace('_', ' ').title()}: {value}" for key, value in all_vars.items())

    is_valid = validate_template_variables(template)
    assert is_valid is False, f"Template missing {missing_var} should fail validation"


@pytest.mark.parametrize("generation_id", [0, 1, 5, 10])
def test_parse_structured_response(sample_structured_response: str, generation_id: int) -> None:
    """Test parsing structured JSON response."""
    candidates = parse_structured_response(sample_structured_response, generation_id=generation_id)

    assert candidates is not None, "Should successfully parse valid JSON response"
    assert len(candidates) == 2, "Should parse exactly 2 candidates from the response"

    # Check first candidate
    assert candidates[0].hypothesis == "Using chain-of-thought reasoning", "First candidate hypothesis should match"
    assert candidates[0].generation == generation_id, f"First candidate should have generation ID {generation_id}"
    assert candidates[0].candidate_id == f"gen_{generation_id:02d}_candidate_0", (
        "First candidate should have correct ID format"
    )
    assert "{{ question_text }}" in candidates[0].prompt, "First candidate template should contain required variables"

    # Check second candidate
    assert candidates[1].hypothesis == "Adding misconception examples", "Second candidate hypothesis should match"
    assert candidates[1].generation == generation_id, f"Second candidate should have generation ID {generation_id}"
    assert candidates[1].candidate_id == f"gen_{generation_id:02d}_candidate_1", (
        "Second candidate should have correct ID format"
    )
    assert "{{ question_text }}" in candidates[1].prompt, "Second candidate template should contain required variables"


@pytest.mark.parametrize("generation_id", [1, 3, 5])
def test_parse_gpt5_response(sample_gpt5_response: str, generation_id: int) -> None:
    """Test parsing GPT-5 response into candidates."""
    candidates = parse_gpt5_response(sample_gpt5_response, generation_id=generation_id)

    assert len(candidates) == 2, "Should parse exactly 2 candidates from GPT-5 response"

    # Check first candidate
    assert candidates[0].candidate_id == f"gen_{generation_id:02d}_candidate_0", (
        "First candidate should have correct ID format"
    )
    assert "more context" in candidates[0].hypothesis, "First candidate hypothesis should contain key phrase"
    assert "{{ question_text }}" in candidates[0].prompt, (
        "First candidate prompt should contain required template variables"
    )
    assert candidates[0].generation == generation_id, f"First candidate should have generation ID {generation_id}"

    # Check second candidate
    assert candidates[1].candidate_id == f"gen_{generation_id:02d}_candidate_1", (
        "Second candidate should have correct ID format"
    )
    assert "Chain-of-thought" in candidates[1].hypothesis, "Second candidate hypothesis should contain key phrase"
    assert "step by step" in candidates[1].prompt, "Second candidate prompt should contain key phrase"
    assert candidates[1].generation == generation_id, f"Second candidate should have generation ID {generation_id}"


@patch("kaggle_map.evolution.generator.OpenAI")
def test_generate_candidates(mock_openai: MagicMock, sample_evolution_context: EvolutionContext) -> None:
    """Test generating candidates with mocked GPT-5."""
    # Mock the OpenAI client
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    # Mock the response
    mock_response = MagicMock()
    mock_response.content = """
    CANDIDATE 1
    HYPOTHESIS: Test hypothesis for improved performance
    TEMPLATE:
    Question: {{ question_text }}
    Category: {{ category }}
    MC Answer: {{ mc_answer }}
    Student Explanation: {{ student_explanation }}
    Additional context for better understanding...
    """
    mock_client.responses.create.return_value = mock_response

    # Generate candidates
    candidates = generate_candidates(sample_evolution_context, num_candidates=1)

    # Verify results
    assert len(candidates) == 1, "Should generate exactly 1 candidate as requested"
    assert candidates[0].generation == 1, "Generated candidate should have correct generation ID"
    assert candidates[0].candidate_id == "gen_01_candidate_0", "Generated candidate should have correct ID format"
    assert "Test hypothesis" in candidates[0].hypothesis, "Generated candidate should preserve hypothesis from response"
    assert "{{ question_text }}" in candidates[0].prompt, (
        "Generated candidate should contain required template variables"
    )

    # Verify OpenAI client was called
    assert mock_client.responses.create.called, "OpenAI client should have been called to generate response"


@patch("kaggle_map.evolution.generator.OpenAI")
@pytest.mark.parametrize("num_candidates", [1, 2, 3])
def test_generate_candidates_multiple(
    mock_openai: MagicMock, sample_evolution_context: EvolutionContext, num_candidates: int
) -> None:
    """Test generating multiple candidates."""
    # Mock the OpenAI client
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    # Create properly formatted response with multiple candidates that will pass validation
    candidates_text = [
        f"""
CANDIDATE {i + 1}
HYPOTHESIS: Test hypothesis {i + 1} for improved performance
TEMPLATE:
Question: {{{{ question_text }}}}
Category: {{{{ category }}}}
MC Answer: {{{{ mc_answer }}}}
Student Explanation: {{{{ student_explanation }}}}
Additional content for candidate {i + 1} that provides specific improvements...
        """.strip()
        for i in range(num_candidates)
    ]

    mock_response = MagicMock()
    mock_response.content = "\n\n".join(candidates_text)
    mock_client.responses.create.return_value = mock_response

    # Generate candidates
    candidates = generate_candidates(sample_evolution_context, num_candidates=num_candidates)

    # Verify results
    assert len(candidates) == num_candidates, f"Should generate exactly {num_candidates} candidates as requested"

    for i, candidate in enumerate(candidates):
        expected_id = f"gen_01_candidate_{i}"
        assert candidate.candidate_id == expected_id, f"Candidate {i} should have correct ID format"
        assert f"Test hypothesis {i + 1}" in candidate.hypothesis, f"Candidate {i} should have correct hypothesis"
        assert "{{ question_text }}" in candidate.prompt, f"Candidate {i} should contain required template variables"
