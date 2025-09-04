"""Type definitions for LLM operations using Python 3.13+ syntax.

This module defines specific types for LLM-related operations to improve
code clarity and type safety. Uses Python 3.13's simplified type syntax.
"""

from dataclasses import dataclass
from typing import Literal

from kaggle_map.core.models import (
    Answer,
    EvaluationRow,
    Explanation,
    Misconception,
    Prediction,
    Question,
    QuestionId,
    TrainingRow,
)

# String subtypes for semantic clarity
type Label = str  # "True_Misconception:AddInsteadOfMultiply"
type PromptTemplate = str
type LLMResponse = str

# Model configuration types
type ModelName = Literal["gemma-3-12b-it", "Qwen3-14B"]
type QuantizationLevel = Literal["Q4_K_XL", "Q5_K_XL", "Q6_K_XL"]

# Composite domain objects that group related data
@dataclass(frozen=True)
class StudentWork:
    """Student's attempt at solving a problem."""

    answer: Answer
    explanation: Explanation

    @classmethod
    def from_evaluation_row(cls, row: EvaluationRow) -> "StudentWork":
        """Create from an evaluation row."""
        return cls(answer=row.mc_answer, explanation=row.student_explanation)


@dataclass(frozen=True)
class ProblemContext:
    """Complete context about a math problem."""

    question_id: QuestionId
    question_text: Question
    correct_answer: Answer | None = None
    known_misconceptions: list[Misconception] | None = None

    @classmethod
    def from_evaluation_row(cls, row: EvaluationRow) -> "ProblemContext":
        """Create minimal context from evaluation row."""
        return cls(question_id=row.question_id, question_text=row.question_text)


@dataclass(frozen=True)
class LLMInferenceContext:
    """Everything needed for LLM to make a prediction."""

    problem: ProblemContext
    student_work: StudentWork
    training_examples: list[TrainingRow] | None = None

    @classmethod
    def from_evaluation_row(
        cls, row: EvaluationRow, problem_context: ProblemContext | None = None
    ) -> "LLMInferenceContext":
        """Create inference context from evaluation row."""
        problem = problem_context or ProblemContext.from_evaluation_row(row)
        student_work = StudentWork.from_evaluation_row(row)
        return cls(problem=problem, student_work=student_work)


@dataclass(frozen=True)
class RerankingRequest:
    """Complete request for reranking predictions."""

    problem: ProblemContext
    student_work: StudentWork
    candidate_predictions: list[Prediction]

    @property
    def top_prediction(self) -> Prediction | None:
        """Get the current top prediction."""
        return self.candidate_predictions[0] if self.candidate_predictions else None


@dataclass
class LLMConfig:
    """Configuration for loading and running GGUF models."""

    model_name: ModelName = "gemma-3-12b-it"
    quantization: QuantizationLevel = "Q4_K_XL"
    n_ctx: int = 4096
    n_batch: int = 512
    n_gpu_layers: int = -1  # Use all available layers
    n_threads: int = 8
    verbose: bool = False

    @property
    def model_filename(self) -> str:
        """Get the GGUF filename for this configuration."""
        return f"{self.model_name}-{self.quantization}.gguf"

