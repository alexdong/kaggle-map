"""Evolution system data models and types."""

import re
from datetime import datetime

from loguru import logger
from pydantic import BaseModel, field_validator

from kaggle_map.core.models import Prediction, TrainingRow
from kaggle_map.utils.logger_config import configure_logger

configure_logger(__name__)

# Type aliases for clarity
type GenerationID = int
type CandidateID = str
type MAPScore = float

# Constants
MAX_PREDICTIONS = 3
MAX_FAILURE_SAMPLES = 10
MAX_PARENT_PROMPTS = 3
HYPOTHESIS_PREVIEW_LENGTH = 50
LOW_MAP_THRESHOLD = 0.3
HIGH_MAP_THRESHOLD = 0.7
MAX_GENERATION_WARNING = 100
MIN_GENERATION_DIR_PARTS = 2
MAX_DISPLAY_GENERATIONS = 5

# Evaluation thresholds
EXCELLENT_MAP_THRESHOLD = 0.7
GOOD_MAP_THRESHOLD = 0.5
POOR_MAP_THRESHOLD = 0.3
STRONG_RESULT_THRESHOLD = 0.6
MODERATE_RESULT_THRESHOLD = 0.4
PARTIAL_HIT_THRESHOLD = 0.5

# Context validation
MIN_CONTEXT_LENGTH = 50
MAX_CONTEXT_LENGTH = 10000

# Test constants
TEST_MAP_SCORE = 0.42

# Display constants
HYPOTHESIS_DISPLAY_LENGTH = 100
HYPOTHESIS_DEBUG_LENGTH = 80


class PromptCandidate(BaseModel):
    """A single prompt variation with its metadata."""

    generation: GenerationID
    candidate_id: CandidateID
    prompt: str
    hypothesis: str
    parent_ids: list[CandidateID]

    @field_validator("generation")
    @classmethod
    def validate_generation(cls, v: int) -> int:
        assert isinstance(v, int), f"Generation must be an integer, got type {type(v).__name__}"
        assert v >= 0, f"Generation must be non-negative, got {v} which is < 0"
        return v

    @field_validator("candidate_id")
    @classmethod
    def validate_candidate_id(cls, v: str) -> str:
        assert v, f"Candidate ID cannot be empty, got: '{v}'"
        assert v.strip(), f"Candidate ID cannot be whitespace, got: '{v}'"
        assert "gen_" in v, f"Candidate ID must contain 'gen_' prefix for generation tracking, got: '{v}'"
        assert "candidate_" in v, f"Candidate ID must contain 'candidate_' for identification, got: '{v}'"

        try:
            gen_part = v.split("gen_")[1].split("_")[0]
            int(gen_part)
        except (IndexError, ValueError) as e:
            logger.error(f"Invalid candidate ID format: {v} - {e}")
            msg = f"Candidate ID has invalid generation number format: '{v}'"
            raise AssertionError(msg) from e

        return v

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        assert v, f"Prompt cannot be empty, got {len(v)} chars"
        assert v.strip(), f"Prompt cannot be only whitespace, got {len(v)} chars"
        assert "{{" in v, f"Prompt must contain Jinja2 opening brackets '{{{{', template starts with: '{v[:50]}...'"
        assert "}}" in v, f"Prompt must contain Jinja2 closing brackets '}}}}', template ends with: '...{v[-50:]}'"

        required_vars = {"question_text", "category", "mc_answer", "student_explanation"}
        found_vars = set()
        for match in re.finditer(r"{{\s*(\w+)\s*}}", v):
            found_vars.add(match.group(1))

        missing = required_vars - found_vars
        if missing:
            logger.warning(f"Template missing required variables: {missing}")

        return v

    def __str__(self) -> str:
        parents = f" (parents: {', '.join(self.parent_ids)})" if self.parent_ids else " (no parents)"
        if len(self.hypothesis) > HYPOTHESIS_PREVIEW_LENGTH:
            hypothesis_preview = self.hypothesis[:HYPOTHESIS_PREVIEW_LENGTH] + "..."
        else:
            hypothesis_preview = self.hypothesis
        return f"[{self.candidate_id}] {hypothesis_preview}{parents}"


class FailureCase(TrainingRow):
    """A test case where the model failed, inheriting all training data fields."""

    predicted: list[Prediction]

    @field_validator("predicted")
    @classmethod
    def validate_predicted(cls, v: list[Prediction]) -> list[Prediction]:
        assert v, "Predicted list cannot be empty, need at least 1 prediction"
        assert len(v) <= MAX_PREDICTIONS, f"Maximum {MAX_PREDICTIONS} predictions allowed, got {len(v)} predictions"

        for i, pred in enumerate(v):
            assert isinstance(pred, Prediction), f"Prediction {i} is not a Prediction object: {type(pred)}"

        return v

    def __str__(self) -> str:
        actual = f"{self.category}:{self.misconception}"
        predicted_first = (
            f"{self.predicted[0].category}:{self.predicted[0].misconception}" if self.predicted else "None"
        )
        return f"[Q{self.question_id}] Actual: {actual}, Predicted: {predicted_first}"


class EvaluationResult(BaseModel):
    """Performance metrics for a prompt candidate."""

    candidate_id: CandidateID
    map_score: MAPScore
    failure_samples: list[FailureCase]

    @field_validator("map_score")
    @classmethod
    def validate_map_score(cls, v: float) -> float:
        assert isinstance(v, int | float), f"MAP score must be numeric, got {type(v).__name__}: {v}"
        assert 0.0 <= v <= 1.0, f"MAP score must be between 0.0 and 1.0, got {v:.4f} which is out of range"

        if v < LOW_MAP_THRESHOLD:
            logger.warning(f"Low MAP@3 score: {v:.4f} - candidate may need improvement")
        elif v > HIGH_MAP_THRESHOLD:
            logger.info(f"High MAP@3 score: {v:.4f} - strong candidate")

        return v

    @field_validator("failure_samples")
    @classmethod
    def validate_failure_samples(cls, v: list[FailureCase]) -> list[FailureCase]:
        assert len(v) <= MAX_FAILURE_SAMPLES, (
            f"Maximum {MAX_FAILURE_SAMPLES} failure samples allowed, got {len(v)} samples"
        )

        if not v:
            logger.warning("No failure samples provided - may indicate perfect performance or evaluation issue")

        return v

    def __str__(self) -> str:
        return f"[{self.candidate_id}] MAP@3: {self.map_score:.3f}, Failures: {len(self.failure_samples)}"


class Generation(BaseModel):
    """A complete evolution generation."""

    generation_id: GenerationID
    candidates: list[PromptCandidate]
    evaluations: list[EvaluationResult]
    timestamp: datetime

    @field_validator("generation_id")
    @classmethod
    def validate_generation_id(cls, v: int) -> int:
        assert isinstance(v, int), f"Generation ID must be an integer, got {type(v).__name__}: {v}"
        assert v >= 0, f"Generation ID must be non-negative (starts at 0), got {v}"

        if v > MAX_GENERATION_WARNING:
            logger.warning(f"High generation ID ({v}) - consider checking convergence criteria")

        return v

    @field_validator("evaluations")
    @classmethod
    def validate_and_sort_evaluations(cls, v: list[EvaluationResult]) -> list[EvaluationResult]:
        if not v:
            logger.warning("No evaluations provided - generation may have failed")
            return v

        sorted_evals = sorted(v, key=lambda e: (-e.map_score, e.candidate_id))

        scores = [e.map_score for e in sorted_evals]
        best = scores[0] if scores else 0.0
        worst = scores[-1] if scores else 0.0
        avg = sum(scores) / len(scores) if scores else 0.0

        logger.info(f"Evaluation stats: best={best:.4f}, worst={worst:.4f}, avg={avg:.4f}, spread={best - worst:.4f}")

        for i, eval_result in enumerate(sorted_evals[:3]):
            logger.info(f"  Top {i + 1}: {eval_result}")

        return sorted_evals

    def __str__(self) -> str:
        best_score = self.evaluations[0].map_score if self.evaluations else 0.0
        num_evaluated = len(self.evaluations)
        return (
            f"Generation {self.generation_id}: {len(self.candidates)} candidates, "
            f"{num_evaluated} evaluated, best MAP@3: {best_score:.3f}"
        )


class EvolutionContext(BaseModel):
    """Context for generating next batch of prompts."""

    current_best_prompt: CandidateID
    current_best_score: MAPScore
    parent_prompts: list[PromptCandidate]
    failure_patterns: dict[CandidateID, list[FailureCase]]
    competition_context: str
    next_generation_id: GenerationID

    @field_validator("current_best_score")
    @classmethod
    def validate_current_best_score(cls, v: float) -> float:
        assert isinstance(v, int | float), f"Score must be numeric, got {type(v).__name__}: {v}"
        assert 0.0 <= v <= 1.0, f"Current best score must be between 0.0 and 1.0, got {v:.4f}"

        logger.info(f"Current best MAP@3 score across all generations: {v:.4f}")
        return v

    @field_validator("parent_prompts")
    @classmethod
    def validate_parent_prompts(cls, v: list[PromptCandidate]) -> list[PromptCandidate]:
        assert len(v) <= MAX_PARENT_PROMPTS, (
            f"Maximum {MAX_PARENT_PROMPTS} parent prompts allowed, got {len(v)} prompts"
        )

        if not v:
            logger.info("No parent prompts - this may be the initial generation")

        return v

    @field_validator("next_generation_id")
    @classmethod
    def validate_next_generation_id(cls, v: int) -> int:
        assert isinstance(v, int), f"Next generation ID must be an integer, got {type(v).__name__}: {v}"
        assert v >= 0, f"Next generation ID must be non-negative, got {v}"

        logger.info(f"Preparing context for generation {v}")
        return v

    @field_validator("competition_context")
    @classmethod
    def validate_competition_context(cls, v: str) -> str:
        assert v, f"Competition context cannot be empty, got {len(v)} chars"
        assert v.strip(), f"Competition context cannot be only whitespace, got {len(v)} chars"

        if len(v) < MIN_CONTEXT_LENGTH:
            logger.warning(f"Competition context seems short ({len(v)} chars) - may lack detail")
        elif len(v) > MAX_CONTEXT_LENGTH:
            logger.warning(f"Competition context is very long ({len(v)} chars) - consider summarizing")

        return v

    def __str__(self) -> str:
        num_failures = sum(len(failures) for failures in self.failure_patterns.values())
        return (
            f"EvolutionContext: Best {self.current_best_prompt} (MAP@3: {self.current_best_score:.3f}), "
            f"Next gen: {self.next_generation_id}, Parents: {len(self.parent_prompts)}, "
            f"Failure patterns: {num_failures} from {len(self.failure_patterns)} candidates"
        )
