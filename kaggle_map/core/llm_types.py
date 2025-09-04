"""Type definitions for LLM operations.

This module defines specific types for LLM-related operations to improve
code clarity and type safety.
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

# Type aliases for semantic clarity
Label = str  # "True_Misconception:AddInsteadOfMultiply"
PromptTemplate = str
LLMResponse = str

# Model configuration types
ModelName = Literal["gemma-3-12b-it", "Qwen3-14B"]
QuantizationLevel = Literal["Q4_K_XL", "Q5_K_XL", "Q6_K_XL"]

# Composite domain objects that group related data
@dataclass(frozen=True)
class StudentWork:
    """Student's attempt at solving a problem."""
    answer: Answer
    explanation: Explanation


@dataclass(frozen=True)
class ProblemContext:
    """Complete context about a math problem."""
    question_id: QuestionId
    question_text: Question
    correct_answer: Answer | None = None
    known_misconceptions: list[Misconception] | None = None


@dataclass(frozen=True)
class LLMInferenceContext:
    """Everything needed for LLM to make a prediction."""
    problem: ProblemContext
    student_work: StudentWork
    training_examples: list[TrainingRow] | None = None

    @classmethod
    def from_evaluation_row(cls, row: EvaluationRow) -> "LLMInferenceContext":
        """Create inference context from evaluation row."""
        problem = ProblemContext(
            question_id=row.question_id,
            question_text=row.question_text
        )
        student_work = StudentWork(
            answer=row.mc_answer,
            explanation=row.student_explanation
        )
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

