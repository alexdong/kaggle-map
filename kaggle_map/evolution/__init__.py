"""Evolution system data models and types."""

from datetime import datetime

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from kaggle_map.core.models import Prediction, TrainingRow

# Type aliases for clarity
type GenerationID = int  # e.g., 0, 1, 2, ...
type CandidateID = str  # e.g., "gen_03_candidate_2"
type MAPScore = float  # 0.0 to 1.0

# Constants
MAX_PREDICTIONS = 3
MAX_FAILURE_SAMPLES = 10
MAX_PARENT_PROMPTS = 3


class PromptCandidate(BaseModel):
    """A single prompt variation with its metadata."""

    generation: GenerationID
    candidate_id: CandidateID
    prompt: str  # Full Jinja2 template
    hypothesis: str  # Why this might work better
    parent_ids: list[CandidateID]  # For tracking lineage

    @field_validator("generation")
    @classmethod
    def validate_generation(cls, v: int) -> int:
        assert v >= 0, f"Generation must be non-negative, got {v}"
        return v

    @field_validator("candidate_id")
    @classmethod
    def validate_candidate_id(cls, v: str) -> str:
        assert v, "Candidate ID cannot be empty"
        assert "gen_" in v, f"Candidate ID must contain 'gen_': {v}"
        assert "candidate_" in v, f"Candidate ID must contain 'candidate_': {v}"
        return v

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        assert v, "Prompt cannot be empty"
        assert "{{" in v, "Prompt must contain Jinja2 opening brackets {{"
        assert "}}" in v, "Prompt must contain Jinja2 closing brackets }}"
        return v

    def __str__(self) -> str:
        """Readable representation for logging."""
        parents = f" (parents: {', '.join(self.parent_ids)})" if self.parent_ids else " (no parents)"
        return f"[{self.candidate_id}] {self.hypothesis[:50]}...{parents}"


class FailureCase(TrainingRow):
    """A test case where the model failed, inheriting all training data fields."""

    predicted: list[Prediction]  # What the model predicted

    @field_validator("predicted")
    @classmethod
    def validate_predicted(cls, v: list[Prediction]) -> list[Prediction]:
        assert v, "Predicted list cannot be empty"
        assert len(v) <= MAX_PREDICTIONS, f"Maximum {MAX_PREDICTIONS} predictions allowed, got {len(v)}"
        return v

    def __str__(self) -> str:
        """Readable representation for logging."""
        actual = f"{self.category}:{self.misconception}"
        predicted_first = (
            f"{self.predicted[0].category}:{self.predicted[0].misconception}" if self.predicted else "None"
        )
        return f"[Q{self.question_id}] Actual: {actual}, Predicted: {predicted_first}"


class EvaluationResult(BaseModel):
    """Performance metrics for a prompt candidate."""

    candidate_id: CandidateID
    map_score: MAPScore = Field(ge=0.0, le=1.0)  # Mean Average Precision @ 3
    failure_samples: list[FailureCase]  # 10 diverse failures

    @field_validator("failure_samples")
    @classmethod
    def validate_failure_samples(cls, v: list[FailureCase]) -> list[FailureCase]:
        assert len(v) <= MAX_FAILURE_SAMPLES, f"Maximum {MAX_FAILURE_SAMPLES} failure samples, got {len(v)}"
        return v

    def __str__(self) -> str:
        """Readable representation for logging."""
        return f"[{self.candidate_id}] MAP@3: {self.map_score:.3f}, Failures: {len(self.failure_samples)}"


class Generation(BaseModel):
    """A complete evolution generation."""

    generation_id: GenerationID
    candidates: list[PromptCandidate]
    evaluations: list[EvaluationResult]  # Should be ordered by map_score desc
    timestamp: datetime

    @field_validator("generation_id")
    @classmethod
    def validate_generation_id(cls, v: int) -> int:
        assert v >= 0, f"Generation ID must be non-negative, got {v}"
        return v

    @field_validator("evaluations")
    @classmethod
    def validate_and_sort_evaluations(cls, v: list[EvaluationResult]) -> list[EvaluationResult]:
        """Ensure evaluations are sorted by MAP score descending."""
        if not v:
            return v

        # Sort by MAP score descending
        sorted_evals = sorted(v, key=lambda e: e.map_score, reverse=True)

        logger.debug(f"Sorted {len(sorted_evals)} evaluations by MAP score")
        for i, eval_result in enumerate(sorted_evals[:3]):
            logger.debug(f"  #{i + 1}: {eval_result}")

        return sorted_evals

    def __str__(self) -> str:
        """Readable representation for logging."""
        best_score = self.evaluations[0].map_score if self.evaluations else 0.0
        return f"Generation {self.generation_id}: {len(self.candidates)} candidates, best MAP@3: {best_score:.3f}"


class EvolutionContext(BaseModel):
    """Context for generating next batch of prompts."""

    current_best_prompt: CandidateID
    current_best_score: MAPScore = Field(ge=0.0, le=1.0)
    parent_prompts: list[PromptCandidate]  # Top 3 across ALL generations
    failure_patterns: dict[CandidateID, list[FailureCase]]  # 10 failures per top candidate
    competition_context: str  # Content from @docs/competition.md
    next_generation_id: GenerationID

    @field_validator("parent_prompts")
    @classmethod
    def validate_parent_prompts(cls, v: list[PromptCandidate]) -> list[PromptCandidate]:
        assert len(v) <= MAX_PARENT_PROMPTS, f"Maximum {MAX_PARENT_PROMPTS} parent prompts, got {len(v)}"
        return v

    @field_validator("next_generation_id")
    @classmethod
    def validate_next_generation_id(cls, v: int) -> int:
        assert v >= 0, f"Next generation ID must be non-negative, got {v}"
        return v

    @field_validator("competition_context")
    @classmethod
    def validate_competition_context(cls, v: str) -> str:
        assert v, "Competition context cannot be empty"
        return v

    def __str__(self) -> str:
        """Readable representation for logging."""
        return (
            f"EvolutionContext: Best {self.current_best_prompt} (MAP@3: {self.current_best_score:.3f}), "
            f"Next gen: {self.next_generation_id}, Parents: {len(self.parent_prompts)}"
        )
