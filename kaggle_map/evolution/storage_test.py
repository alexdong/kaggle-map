"""Tests for storage layer."""

import logging
from datetime import datetime
from pathlib import Path

import pytest

from kaggle_map.evolution import (
    EvaluationResult,
    EvolutionContext,
    Generation,
    PromptCandidate,
)
from kaggle_map.evolution.storage import Storage

# Set debug logging for tests
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def sample_prompt_candidate() -> PromptCandidate:
    """Create a sample prompt candidate for testing."""
    return PromptCandidate(
        generation=0,
        candidate_id="gen_00_candidate_0",
        prompt="Student answered: {{ mc_answer }}\nPredictions: {{ predictions }}",
        hypothesis="Test hypothesis",
        parent_ids=[],
    )


@pytest.fixture
def sample_generation() -> Generation:
    """Create a sample generation for testing."""
    return Generation(
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


@pytest.fixture
def sample_context() -> EvolutionContext:
    """Create a sample evolution context for testing."""
    return EvolutionContext(
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


@pytest.mark.parametrize("candidate_id,expected_filename", [
    ("gen_00_candidate_0", "gen_00_candidate_0.j2"),
    ("gen_01_candidate_5", "gen_01_candidate_5.j2"),
    ("gen_99_baseline", "gen_99_baseline.j2"),
])
def test_prompt_template_paths(tmp_path: Path, candidate_id: str, expected_filename: str) -> None:
    """Test that prompt template paths are correctly computed."""
    storage = Storage(base_dir=tmp_path)
    
    template_path = storage.get_prompt_template_path(candidate_id)
    expected_path = tmp_path / "reranker" / "prompts" / expected_filename
    
    assert template_path == expected_path, f"Template path should be {expected_path} for candidate {candidate_id}"


@pytest.mark.parametrize("generation_id,expected_dir", [
    (0, "gen_00"),
    (1, "gen_01"), 
    (10, "gen_10"),
    (99, "gen_99"),
])
def test_generation_directory_paths(tmp_path: Path, generation_id: int, expected_dir: str) -> None:
    """Test that generation directory paths are correctly computed."""
    storage = Storage(base_dir=tmp_path)
    
    gen_dir = storage.get_generation_dir(generation_id)
    expected_path = tmp_path / "reranker" / "prompts" / "generations" / expected_dir
    
    assert gen_dir == expected_path, f"Generation directory should be {expected_path} for generation {generation_id}"


def test_context_path(tmp_path: Path) -> None:
    """Test that context path is correctly computed."""
    storage = Storage(base_dir=tmp_path)
    
    context_path = storage.get_context_path()
    expected_path = tmp_path / "reranker" / "prompts" / "generations" / "context.json"
    
    assert context_path == expected_path, "Context path should point to context.json in generations directory"


def test_save_and_load_prompt_template(tmp_path: Path, sample_prompt_candidate: PromptCandidate) -> None:
    """Test saving and loading Jinja2 templates."""
    storage = Storage(base_dir=tmp_path)

    # Save the template
    storage.save_prompt_template(sample_prompt_candidate)

    # Load it back
    loaded_prompt = storage.load_prompt_template(sample_prompt_candidate.candidate_id)
    
    assert loaded_prompt == sample_prompt_candidate.prompt, "Loaded prompt should match original prompt exactly"


def test_save_and_load_generation(tmp_path: Path, sample_generation: Generation) -> None:
    """Test saving and loading generation data."""
    storage = Storage(base_dir=tmp_path)

    # Save generation
    storage.save_generation(sample_generation)

    # Load it back
    loaded = storage.load_generation(sample_generation.generation_id)
    
    assert loaded.generation_id == sample_generation.generation_id, "Generation ID should be preserved"
    assert len(loaded.candidates) == len(sample_generation.candidates), "All candidates should be preserved"
    assert loaded.evaluations[0].map_score == pytest.approx(0.75), "MAP score should be preserved with float precision"
    assert loaded.candidates[0].candidate_id == sample_generation.candidates[0].candidate_id, "Candidate details should be preserved"


def test_save_and_load_context(tmp_path: Path, sample_context: EvolutionContext) -> None:
    """Test saving and loading evolution context."""
    storage = Storage(base_dir=tmp_path)

    # Save context
    storage.save_context(sample_context)

    # Load it back
    loaded = storage.load_context()
    
    assert loaded.current_best_prompt == sample_context.current_best_prompt, "Best prompt ID should be preserved"
    assert loaded.current_best_score == pytest.approx(sample_context.current_best_score), "Best score should be preserved with float precision"
    assert loaded.next_generation_id == sample_context.next_generation_id, "Next generation ID should be preserved"
    assert len(loaded.parent_prompts) == len(sample_context.parent_prompts), "Parent prompts should be preserved"
    assert loaded.competition_context == sample_context.competition_context, "Competition context should be preserved"


def test_list_generations(tmp_path: Path, sample_generation: Generation) -> None:
    """Test listing available generations."""
    storage = Storage(base_dir=tmp_path)

    # Initially empty
    empty_generations = storage.list_generations()
    assert empty_generations == [], "Should return empty list when no generations exist"

    # Create and save some actual generations
    gen0 = Generation(
        generation_id=0,
        candidates=sample_generation.candidates,
        evaluations=sample_generation.evaluations,
        timestamp=sample_generation.timestamp,
    )
    storage.save_generation(gen0)
    
    gen1 = Generation(
        generation_id=1,
        candidates=sample_generation.candidates,
        evaluations=sample_generation.evaluations,
        timestamp=sample_generation.timestamp,
    )
    storage.save_generation(gen1)

    # Should list them in order
    generations = storage.list_generations()
    assert generations == [0, 1], "Should return generations in sorted order"
    assert len(generations) == 2, "Should find exactly 2 generation directories"
