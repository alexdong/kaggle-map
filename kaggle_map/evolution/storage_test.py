"""Tests for storage layer."""

from datetime import datetime
from pathlib import Path

from kaggle_map.evolution import (
    EvaluationResult,
    EvolutionContext,
    Generation,
    PromptCandidate,
)
from kaggle_map.evolution.storage import Storage


def test_storage_paths() -> None:
    """Test that storage paths are correctly computed."""
    storage = Storage(base_dir=Path("/tmp/test"))

    # Test prompt template path
    template_path = storage.get_prompt_template_path("gen_00_candidate_0")
    assert template_path == Path("/tmp/test/reranker/prompts/gen_00_candidate_0.j2")

    # Test generation directory
    gen_dir = storage.get_generation_dir(0)
    assert gen_dir == Path("/tmp/test/reranker/prompts/generations/gen_00")

    # Test context path
    context_path = storage.get_context_path()
    assert context_path == Path("/tmp/test/reranker/prompts/generations/context.json")


def test_save_and_load_prompt_template(tmp_path: Path) -> None:
    """Test saving and loading Jinja2 templates."""
    storage = Storage(base_dir=tmp_path)

    # Create a test prompt
    candidate = PromptCandidate(
        generation=0,
        candidate_id="gen_00_candidate_0",
        prompt="Student answered: {{ mc_answer }}\nPredictions: {{ predictions }}",
        hypothesis="Test hypothesis",
        parent_ids=[],
    )

    # Save the template
    storage.save_prompt_template(candidate)

    # Load it back
    loaded_prompt = storage.load_prompt_template("gen_00_candidate_0")
    assert loaded_prompt == candidate.prompt


def test_save_and_load_generation(tmp_path: Path) -> None:
    """Test saving and loading generation data."""
    storage = Storage(base_dir=tmp_path)

    # Create test generation with minimal data
    generation = Generation(
        generation_id=0,
        candidates=[
            PromptCandidate(
                generation=0,
                candidate_id="gen_00_candidate_0",
                prompt="Test {{ var }} prompt",
                hypothesis="Test hypothesis",
                parent_ids=[],
            ),
        ],
        evaluations=[
            EvaluationResult(
                candidate_id="gen_00_candidate_0",
                map_score=0.75,
                failure_samples=[],
            ),
        ],
        timestamp=datetime.now(),
    )

    # Save generation
    storage.save_generation(generation)

    # Load it back
    loaded = storage.load_generation(0)
    assert loaded.generation_id == generation.generation_id
    assert len(loaded.candidates) == 1
    assert loaded.evaluations[0].map_score == 0.75


def test_save_and_load_context(tmp_path: Path) -> None:
    """Test saving and loading evolution context."""
    storage = Storage(base_dir=tmp_path)

    # Create test context
    context = EvolutionContext(
        current_best_prompt="gen_00_candidate_0",
        current_best_score=0.72,
        parent_prompts=[
            PromptCandidate(
                generation=0,
                candidate_id="gen_00_candidate_0",
                prompt="Test {{ var }} prompt",
                hypothesis="Baseline",
                parent_ids=[],
            ),
        ],
        failure_patterns={},
        competition_context="Test competition context",
        next_generation_id=1,
    )

    # Save context
    storage.save_context(context)

    # Load it back
    loaded = storage.load_context()
    assert loaded.current_best_prompt == "gen_00_candidate_0"
    assert loaded.current_best_score == 0.72
    assert loaded.next_generation_id == 1


def test_list_generations(tmp_path: Path) -> None:
    """Test listing available generations."""
    storage = Storage(base_dir=tmp_path)

    # Initially empty
    assert storage.list_generations() == []

    # Create some generation directories
    gen0 = storage.get_generation_dir(0)
    gen0.mkdir(parents=True)
    gen2 = storage.get_generation_dir(1)
    gen2.mkdir(parents=True)

    # Should list them in order
    generations = storage.list_generations()
    assert generations == [0, 1]
