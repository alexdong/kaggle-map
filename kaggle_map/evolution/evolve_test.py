"""Tests for evolution orchestrator."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from kaggle_map.evolution import (
    EvaluationResult,
    EvolutionContext,
    Generation,
    PromptCandidate,
)
from kaggle_map.evolution.evolve import (
    check_convergence,
    evolve_prompts,
    run_generation,
    select_top_performers,
)


def test_check_convergence() -> None:
    """Test convergence detection."""
    # No improvement over 3 generations
    scores = [0.65, 0.65, 0.65, 0.65]
    assert check_convergence(scores, threshold=0.01, window=3) is True

    # Improvement in recent generation
    scores = [0.60, 0.61, 0.62, 0.65]
    assert check_convergence(scores, threshold=0.01, window=3) is False

    # Not enough generations yet
    scores = [0.60, 0.61]
    assert check_convergence(scores, threshold=0.01, window=3) is False


def test_select_top_performers() -> None:
    """Test selecting top performing candidates."""
    generation = Generation(
        generation_id=0,
        candidates=[
            PromptCandidate(
                generation=0,
                candidate_id="gen_00_candidate_0",
                prompt="Question: {{ question_text }}\nCategory: {{ category }}\nMC Answer: {{ mc_answer }}\nStudent: {{ student_explanation }}",
                hypothesis="Hypothesis 0",
                parent_ids=[],
            ),
            PromptCandidate(
                generation=0,
                candidate_id="gen_00_candidate_1",
                prompt="Q: {{ question_text }}\nCat: {{ category }}\nAnswer: {{ mc_answer }}\nExplanation: {{ student_explanation }}",
                hypothesis="Hypothesis 1",
                parent_ids=[],
            ),
            PromptCandidate(
                generation=0,
                candidate_id="gen_00_candidate_2",
                prompt="{{ question_text }}\n{{ category }}\n{{ mc_answer }}\n{{ student_explanation }}",
                hypothesis="Hypothesis 2",
                parent_ids=[],
            ),
        ],
        evaluations=[
            EvaluationResult(candidate_id="gen_00_candidate_0", map_score=0.9, failure_samples=[]),
            EvaluationResult(candidate_id="gen_00_candidate_1", map_score=0.8, failure_samples=[]),
            EvaluationResult(candidate_id="gen_00_candidate_2", map_score=0.7, failure_samples=[]),
        ],
        timestamp=datetime.now(),
    )

    # Select top 40% (should be 1 out of 3)
    selected = select_top_performers(generation, top_percentage=0.4)
    assert len(selected) == 1
    assert selected[0].candidate_id == "gen_00_candidate_0"

    # Select top 70% (should be 2 out of 3)
    selected = select_top_performers(generation, top_percentage=0.7)
    assert len(selected) == 2
    assert selected[0].candidate_id == "gen_00_candidate_0"
    assert selected[1].candidate_id == "gen_00_candidate_1"


@patch("kaggle_map.evolution.evolve.evaluate_all_candidates")
@patch("kaggle_map.evolution.evolve.generate_candidates")
def test_run_generation(
    mock_generate: MagicMock,
    mock_evaluate: MagicMock,
) -> None:
    """Test running a single generation."""
    # Mock candidates
    mock_candidates = [
        PromptCandidate(
            generation=0,
            candidate_id="gen_00_candidate_0",
            prompt="Question: {{ question_text }}\nCategory: {{ category }}\nMC Answer: {{ mc_answer }}\nStudent: {{ student_explanation }}",
            hypothesis="Test",
            parent_ids=[],
        )
    ]
    mock_generate.return_value = mock_candidates

    # Mock evaluations
    mock_evaluations = [
        EvaluationResult(
            candidate_id="gen_00_candidate_0",
            map_score=0.7,
            failure_samples=[],
        )
    ]
    mock_evaluate.return_value = mock_evaluations

    # Create context
    context = EvolutionContext(
        current_best_prompt="baseline",
        current_best_score=0.6,
        parent_prompts=[],
        failure_patterns={},
        competition_context="Test",
        next_generation_id=0,
    )

    # Run generation
    generation = run_generation(context)

    assert generation.generation_id == 0
    assert len(generation.candidates) == 1
    assert len(generation.evaluations) == 1
    assert mock_generate.called
    assert mock_evaluate.called


@patch("kaggle_map.evolution.evolve.run_generation")
@patch("kaggle_map.evolution.evolve.Storage")
def test_evolve_prompts(
    mock_storage_class: MagicMock,
    mock_run_generation: MagicMock,
    tmp_path: Path,  # noqa: ARG001
) -> None:
    """Test full evolution process."""
    # Mock storage
    mock_storage = MagicMock()
    mock_storage_class.return_value = mock_storage

    # Mock generation returns
    gen = Generation(
        generation_id=0,
        candidates=[
            PromptCandidate(
                generation=0,
                candidate_id="gen_00_candidate_0",
                prompt="Question: {{ question_text }}\nCategory: {{ category }}\nMC Answer: {{ mc_answer }}\nStudent: {{ student_explanation }}",
                hypothesis="Test",
                parent_ids=[],
            )
        ],
        evaluations=[
            EvaluationResult(
                candidate_id="gen_00_candidate_0",
                map_score=0.7,
                failure_samples=[],
            )
        ],
        timestamp=datetime.now(),
    )
    mock_run_generation.return_value = gen

    # Run evolution for 2 generations
    best_prompt = evolve_prompts(max_generations=2)

    assert best_prompt is not None
    assert mock_run_generation.call_count == 2
    assert mock_storage.save_generation.called
    assert mock_storage.save_context.called
