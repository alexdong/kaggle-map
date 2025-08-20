"""Core data structures for the Kaggle student misconception prediction competition."""

from enum import Enum
from typing import NamedTuple

from pydantic import BaseModel, field_validator

from kaggle_map.embeddings.formula import normalize_latex_answer, normalize_text

# Domain-specific type aliases
type QuestionId = int
type Answer = str
type Misconception = (
    str  # Specific misconception identifier (e.g., "AddInsteadOfMultiply")
)



class Category(Enum):
    TRUE_CORRECT = "True_Correct"
    TRUE_NEITHER = "True_Neither"
    TRUE_MISCONCEPTION = "True_Misconception"
    FALSE_CORRECT = "False_Correct"
    FALSE_NEITHER = "False_Neither"
    FALSE_MISCONCEPTION = "False_Misconception"

    @property
    def is_misconception(self) -> bool:
        return self.value.endswith("_Misconception")

    @property
    def is_correct_answer(self) -> bool:
        return self.value.startswith("True_")

    @classmethod
    def by_truth_value(cls, *, is_true: bool) -> list["Category"]:
        """Return all Category values corresponding to the given truth value.

        Args:
            is_true: True returns TRUE_* categories, False returns FALSE_* categories

        Returns:
            List of Category enum values matching the boolean prefix
        """
        prefix = "True_" if is_true else "False_"
        return [category for category in cls if category.value.startswith(prefix)]


class Prediction(BaseModel):
    category: Category
    misconception: Misconception | None = None

    def __str__(self) -> str:
        if self.category.is_misconception and self.misconception is not None:
            return f"{self.category.value}:{self.misconception}"
        return f"{self.category.value}:NA"


class EvaluationRow(BaseModel):
    row_id: int
    question_id: QuestionId
    question_text: str
    mc_answer: Answer
    student_explanation: str

    @field_validator("question_text", "student_explanation")
    @classmethod
    def normalize_text_fields(cls, v: str) -> str:
        return normalize_text(v)

    @field_validator("mc_answer")
    @classmethod
    def normalize_answer(cls, v: str) -> str:
        return normalize_latex_answer(v)

    def __repr__(self) -> str:
        """Compose the canonical Q/A/E string used for embeddings.

        Example output:
            "Question: {question}, Answer: {normalized_answer}, Explanation: {explanation}"
        """
        return f"Question: {self.question_text}, Answer: {self.mc_answer}, Explanation: {self.student_explanation}"


class TrainingRow(EvaluationRow):
    category: Category
    misconception: Misconception | None


class SubmissionRow(NamedTuple):
    row_id: int
    predicted_categories: list[Prediction]  # Max 3, ordered by confidence


class EvaluationResult(BaseModel):
    map_score: float
    total_observations: int
    perfect_predictions: int

    @field_validator("map_score")
    @classmethod
    def validate_map_score(cls, v: float) -> float:
        assert 0.0 <= v <= 1.0, f"MAP score must be between 0 and 1, got {v}"
        return v
