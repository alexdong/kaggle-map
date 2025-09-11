"""Minimal tests for evolution data models using real data from error_prediction.csv."""

import logging
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

# Set debug logging for tests
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def sample_prompt_candidate() -> PromptCandidate:
    """Create a sample prompt candidate for testing."""
    return PromptCandidate(
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


@pytest.fixture
def sample_failure_case() -> FailureCase:
    """Create a sample failure case with real data from error_prediction.csv."""
    return FailureCase(
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


def test_prompt_candidate_creation(sample_prompt_candidate: PromptCandidate) -> None:
    """Test creating a prompt candidate with all required fields."""
    candidate = sample_prompt_candidate

    assert candidate.generation == 0, "Generation should be set correctly"
    assert candidate.candidate_id == "gen_00_candidate_0", "Candidate ID should match expected format"
    assert "mc_answer" in candidate.prompt, "Prompt should contain mc_answer template variable"
    assert candidate.hypothesis == "Baseline prompt for reranking predictions", "Hypothesis should be preserved"
    assert candidate.parent_ids == [], "New baseline candidate should have no parent IDs"


def test_evaluation_result_valid_map_score() -> None:
    """Test MAP score within valid range is accepted."""
    result = EvaluationResult(
        candidate_id="gen_00_candidate_0",
        map_score=0.75,
        failure_samples=[],
    )
    assert result.map_score == pytest.approx(0.75), "Valid MAP score should be preserved exactly"


@pytest.mark.parametrize("invalid_score", [1.5, -0.1, 2.0])
def test_evaluation_result_invalid_map_scores(invalid_score: float) -> None:
    """Test MAP scores outside valid range raise ValidationError."""
    with pytest.raises(ValidationError, match=r"MAP score must be between 0\.0 and 1\.0"):
        EvaluationResult(
            candidate_id="gen_00_candidate_0",
            map_score=invalid_score,
            failure_samples=[],
        )


def test_failure_case_with_real_data(sample_failure_case: FailureCase) -> None:
    """Test FailureCase with real data from error_prediction.csv."""
    failure = sample_failure_case

    # Should have all TrainingRow fields
    assert failure.row_id == 107, "Row ID should match source data"
    assert failure.question_id == 31772, "Question ID should match source data"
    assert failure.category == Category.TRUE_MISCONCEPTION, "Category should be preserved from source"

    # Plus the predicted field
    assert len(failure.predicted) == 3, "Should have exactly 3 predictions"
    assert failure.predicted[0].category == Category.TRUE_CORRECT, "First prediction should be TRUE_CORRECT"
    assert failure.predicted[2].misconception == "Incomplete", "Third prediction misconception should match"


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

    assert failure.misconception == "WNB", "Actual misconception should be WNB"
    assert failure.predicted[2].misconception == "Incomplete", "Model incorrectly predicted Incomplete instead of WNB"


def test_generation_ordering(sample_prompt_candidate: PromptCandidate) -> None:
    """Test Generation properly orders evaluations by MAP score."""
    gen = Generation(
        generation_id=0,
        candidates=[sample_prompt_candidate],
        evaluations=[
            EvaluationResult(candidate_id="c1", map_score=0.5, failure_samples=[]),
            EvaluationResult(candidate_id="c2", map_score=0.8, failure_samples=[]),
            EvaluationResult(candidate_id="c3", map_score=0.3, failure_samples=[]),
        ],
        timestamp=datetime.now(),
    )

    # Should be ordered by map_score descending
    assert gen.evaluations[0].map_score == pytest.approx(0.8), "Highest score should be first"
    assert gen.evaluations[1].map_score == pytest.approx(0.5), "Middle score should be second"
    assert gen.evaluations[2].map_score == pytest.approx(0.3), "Lowest score should be last"


def test_evolution_context_with_real_prompt(sample_prompt_candidate: PromptCandidate, sample_failure_case: FailureCase) -> None:
    """Test EvolutionContext with realistic prompt template."""
    context = EvolutionContext(
        current_best_prompt="gen_00_candidate_0",
        current_best_score=0.72,  # Realistic MAP@3 score
        parent_prompts=[sample_prompt_candidate],
        failure_patterns={"gen_00_candidate_0": [sample_failure_case]},
        competition_context="MAP - Charting Student Math Misunderstandings competition context",
        next_generation_id=1,
    )

    assert context.current_best_prompt == "gen_00_candidate_0", "Best prompt ID should be preserved"
    assert context.current_best_score == pytest.approx(0.72), "Best score should be preserved with float precision"
    assert len(context.parent_prompts) == 1, "Should have exactly one parent prompt"
    assert len(context.failure_patterns["gen_00_candidate_0"]) == 1, "Should have exactly one failure case for best prompt"
    assert context.next_generation_id == 1, "Next generation ID should be incremented"


@pytest.mark.parametrize("generation_id", [0, 1, 5, 99])
def test_generation_id_type(generation_id: int) -> None:
    """Test GenerationID is an integer."""
    gen_id: GenerationID = generation_id
    assert isinstance(gen_id, int), "GenerationID should be an integer type"
    assert gen_id == generation_id, "GenerationID value should be preserved"


@pytest.mark.parametrize("candidate_id", [
    "gen_00_candidate_0",
    "gen_03_candidate_2",
    "gen_99_baseline",
])
def test_candidate_id_format(candidate_id: str) -> None:
    """Test CandidateID follows expected string format."""
    cand_id: CandidateID = candidate_id
    assert isinstance(cand_id, str), "CandidateID should be a string type"
    assert "gen_" in cand_id, "CandidateID should contain 'gen_' prefix"
    assert candidate_id == cand_id, "CandidateID value should be preserved"


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
    assert "gen_00_candidate_0" in str_repr, "String representation should include candidate ID"
    assert "Baseline prompt for testing misconception detecti" in str_repr, "Should include truncated hypothesis"
    assert "gen_00_candidate_1" in str_repr, "Should include parent IDs in representation"
