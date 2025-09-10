"""Minimal tests for evolution data models using real data from error_prediction.csv."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from kaggle_map.core.models import Category, Prediction
from kaggle_map.evolution import (
    CandidateID,
    EvaluationResult,
    EvolutionContext,
    FailureCase,
    Generation,
    GenerationID,
    PromptCandidate,
)


def test_prompt_candidate_creation() -> None:
    """Test creating a prompt candidate with all required fields."""
    candidate = PromptCandidate(
        generation=0,
        candidate_id="gen_00_candidate_0",
        prompt=(
            "Student answered: {{ mc_answer }}\n"
            "Student explained: {{ student_explanation }}\n"
            "Predictions:\n"
            "{% for pred in predictions %}\n"
            "{{ loop.index }}. {{ pred }}\n"
            "{% endfor %}\n"
            "Output format: numbers only, comma-separated\n"
            "Your output:"
        ),
        hypothesis="Baseline prompt for reranking predictions",
        parent_ids=[],
    )

    assert candidate.generation == 0
    assert candidate.candidate_id == "gen_00_candidate_0"
    assert "mc_answer" in candidate.prompt
    assert candidate.hypothesis == "Baseline prompt for reranking predictions"
    assert candidate.parent_ids == []


def test_evaluation_result_map_score_validation() -> None:
    """Test MAP score must be between 0 and 1."""
    # Valid score
    result = EvaluationResult(
        candidate_id="gen_00_candidate_0",
        map_score=0.75,
        failure_samples=[],
    )
    assert result.map_score == 0.75

    # Invalid scores should raise validation error
    with pytest.raises(ValidationError, match="MAP score must be between 0 and 1, got 1.5"):
        EvaluationResult(
            candidate_id="gen_00_candidate_0",
            map_score=1.5,  # Too high
            failure_samples=[],
        )

    with pytest.raises(ValidationError, match="MAP score must be between 0 and 1, got -0.1"):
        EvaluationResult(
            candidate_id="gen_00_candidate_0",
            map_score=-0.1,  # Negative
            failure_samples=[],
        )


def test_failure_case_with_real_data() -> None:
    """Test FailureCase with real data from error_prediction.csv."""
    # Real data from row_id=107 in error_prediction.csv
    failure = FailureCase(
        row_id=107,
        question_id=31772,
        question_text=(
            "What fraction of the shape is not shaded? Give your answer in its simplest form. "
            "[Image: A triangle split into 9 equal smaller triangles. 6 of them are shaded.]"
        ),
        mc_answer=r"\( \frac{1}{3} \)",
        student_explanation="3 out of 9 parts aren't shaded.",
        prediction=Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Incomplete"),
        predicted=[
            Prediction(category=Category.TRUE_CORRECT, misconception="NA"),
            Prediction(category=Category.TRUE_NEITHER, misconception="NA"),
            Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Incomplete"),
        ],
    )

    # Should have all TrainingRow fields
    assert failure.row_id == 107
    assert failure.question_id == 31772
    assert failure.category == Category.TRUE_MISCONCEPTION

    # Plus the predicted field
    assert len(failure.predicted) == 3
    assert failure.predicted[0].category == Category.TRUE_CORRECT
    assert failure.predicted[2].misconception == "Incomplete"


def test_failure_case_wnb_misconception() -> None:
    """Test FailureCase with WNB misconception from real data."""
    # Real data from row_id=518 with WNB misconception
    failure = FailureCase(
        row_id=518,
        question_id=31772,
        question_text=(
            "What fraction of the shape is not shaded? Give your answer in its simplest form. "
            "[Image: A triangle split into 9 equal smaller triangles. 6 of them are shaded.]"
        ),
        mc_answer=r"\( \frac{1}{3} \)",
        student_explanation="Because there are 6 triangles and 3 are white. That is 3/6 simplified to 1/3.",
        prediction=Prediction(category=Category.TRUE_MISCONCEPTION, misconception="WNB"),
        predicted=[
            Prediction(category=Category.TRUE_CORRECT, misconception="NA"),
            Prediction(category=Category.TRUE_NEITHER, misconception="NA"),
            Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Incomplete"),
        ],
    )

    assert failure.misconception == "WNB"
    assert failure.predicted[2].misconception == "Incomplete"  # Model predicted wrong misconception


def test_generation_ordering() -> None:
    """Test Generation properly orders evaluations by MAP score."""
    gen = Generation(
        generation_id=0,
        candidates=[
            PromptCandidate(
                generation=0,
                candidate_id="gen_00_candidate_0",
                prompt="Test {{ variable }} prompt",
                hypothesis="Test hypothesis",
                parent_ids=[],
            ),
        ],
        evaluations=[
            EvaluationResult(candidate_id="c1", map_score=0.5, failure_samples=[]),
            EvaluationResult(candidate_id="c2", map_score=0.8, failure_samples=[]),
            EvaluationResult(candidate_id="c3", map_score=0.3, failure_samples=[]),
        ],
        timestamp=datetime.now(),
    )

    # Should be ordered by map_score descending
    assert gen.evaluations[0].map_score == 0.8
    assert gen.evaluations[1].map_score == 0.5
    assert gen.evaluations[2].map_score == 0.3


def test_evolution_context_with_real_prompt() -> None:
    """Test EvolutionContext with realistic prompt template."""
    # Use real baseline prompt structure
    baseline_prompt = """Student answered: {{ mc_answer }}
Student explained: {{ student_explanation }}

Predictions:
{% for pred in predictions %}
{{ loop.index }}. {{ pred.category }}:{{ pred.misconception }}
{% endfor %}

Task: Rank these predictions from most to least likely.
Output format: numbers only, comma-separated
Your output:"""

    candidate = PromptCandidate(
        generation=0,
        candidate_id="gen_00_candidate_0",
        prompt=baseline_prompt,
        hypothesis="Baseline prompt with clear task description",
        parent_ids=[],
    )

    # Create failure case with real data
    failure_case = FailureCase(
        row_id=107,
        question_id=31772,
        question_text="What fraction of the shape is not shaded?",
        mc_answer=r"\( \frac{1}{3} \)",
        student_explanation="3 out of 9 parts aren't shaded.",
        prediction=Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Incomplete"),
        predicted=[
            Prediction(category=Category.TRUE_CORRECT, misconception="NA"),
            Prediction(category=Category.TRUE_NEITHER, misconception="NA"),
            Prediction(category=Category.TRUE_MISCONCEPTION, misconception="Incomplete"),
        ],
    )

    context = EvolutionContext(
        current_best_prompt="gen_00_candidate_0",
        current_best_score=0.72,  # Realistic MAP@3 score
        parent_prompts=[candidate],
        failure_patterns={"gen_00_candidate_0": [failure_case]},
        competition_context="MAP - Charting Student Math Misunderstandings competition context",
        next_generation_id=1,
    )

    assert context.current_best_prompt == "gen_00_candidate_0"
    assert context.current_best_score == 0.72
    assert len(context.parent_prompts) == 1
    assert len(context.failure_patterns["gen_00_candidate_0"]) == 1
    assert context.next_generation_id == 1


def test_generation_id_type() -> None:
    """Test GenerationID is an integer."""
    gen_id: GenerationID = 5
    assert isinstance(gen_id, int)
    assert gen_id == 5


def test_candidate_id_format() -> None:
    """Test CandidateID follows expected string format."""
    cand_id: CandidateID = "gen_03_candidate_2"
    assert isinstance(cand_id, str)
    assert "gen_" in cand_id
    assert "candidate_" in cand_id


def test_prompt_candidate_str_method() -> None:
    """Test PromptCandidate has readable string representation."""
    candidate = PromptCandidate(
        generation=0,
        candidate_id="gen_00_candidate_0",
        prompt="Test {{ variable }} prompt",
        hypothesis="Baseline prompt for testing misconception detection",
        parent_ids=["gen_00_candidate_1", "gen_00_candidate_2"],
    )

    str_repr = str(candidate)
    assert "gen_00_candidate_0" in str_repr
    assert "Baseline prompt for testing misconception detecti" in str_repr  # First 50 chars
    assert "gen_00_candidate_1" in str_repr  # Parent IDs included
