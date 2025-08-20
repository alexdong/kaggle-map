"""Core data structures for the Kaggle student misconception prediction competition."""

from enum import Enum
from typing import NamedTuple

from pydantic import BaseModel, field_validator

from kaggle_map.embeddings.formula import normalize_latex_answer, normalize_text

# Domain-specific type aliases
type QuestionId = int
type Answer = str
type Misconception = str


class ResponseContext(NamedTuple):
    """The fundamental unit for probability modeling.

    Represents the context in which a student response occurs:
    - What question they were answering
    - What they selected as their answer
    - What the correct answer actually is

    This triple uniquely defines the "response state" for probability calculations.
    """

    question_id: QuestionId
    selected_answer: Answer
    correct_answer: Answer

    @property
    def is_correct_selection(self) -> bool:
        """True if the student selected the correct answer."""
        return self.selected_answer == self.correct_answer


class Category(Enum):
    """All possible categories in the competition."""

    TRUE_CORRECT = "True_Correct"
    TRUE_NEITHER = "True_Neither"
    TRUE_MISCONCEPTION = "True_Misconception"
    FALSE_CORRECT = "False_Correct"
    FALSE_NEITHER = "False_Neither"
    FALSE_MISCONCEPTION = "False_Misconception"

    @property
    def is_misconception(self) -> bool:
        """Check if this category involves a misconception."""
        return self.value.endswith("_Misconception")

    @property
    def is_correct_answer(self) -> bool:
        """Check if this represents a correct answer."""
        return self.value.startswith("True_")


class Prediction(BaseModel):
    """A prediction in 'Category:Misconception' format for submission."""

    category: Category
    misconception: Misconception | None = None

    @property
    def value(self) -> str:
        """Return the prediction string in 'Category:Misconception' format."""
        if self.category.is_misconception and self.misconception is not None:
            return f"{self.category.value}:{self.misconception}"
        return f"{self.category.value}:NA"

    def __str__(self) -> str:
        """Return the prediction string for easy use."""
        return self.value

    def __repr__(self) -> str:
        """Return a clear representation."""
        return f"Prediction(category={self.category}, misconception={self.misconception!r})"

    def __hash__(self) -> int:
        """Make Prediction hashable for use as dictionary keys."""
        return hash((self.category, self.misconception))

    def __eq__(self, other: object) -> bool:
        """Define equality for Prediction objects."""
        if not isinstance(other, Prediction):
            return False
        return (
            self.category == other.category
            and self.misconception == other.misconception
        )


class EvaluationRow(BaseModel):
    """Single row from test.csv."""

    row_id: int
    question_id: QuestionId
    question_text: str
    mc_answer: Answer
    student_explanation: str

    @field_validator("question_text", "student_explanation")
    @classmethod
    def normalize_text_fields(cls, v: str) -> str:
        """Normalize text fields automatically."""
        return normalize_text(v)

    @field_validator("mc_answer")
    @classmethod
    def normalize_answer(cls, v: str) -> str:
        """Normalize LaTeX answer automatically."""
        return normalize_latex_answer(v)

    def __repr__(self) -> str:
        """Compose the canonical Q/A/E string used for embeddings.

        Example output:
            "Question: {question}, Answer: {normalized_answer}, Explanation: {explanation}"
        """
        return f"Question: {self.question_text}, Answer: {self.mc_answer}, Explanation: {self.student_explanation}"


class TrainingRow(EvaluationRow):
    """Single row from train.csv, extends EvaluationRow with ground truth labels."""

    category: Category
    misconception: Misconception | None


class SubmissionRow(NamedTuple):
    """Prediction result for MAP@3 evaluation."""

    row_id: int
    predicted_categories: list[Prediction]  # Max 3, ordered by confidence


class EvaluationResult(BaseModel):
    """MAP@3 evaluation result with detailed breakdown."""

    map_score: float
    total_observations: int
    perfect_predictions: int

    @field_validator("map_score")
    @classmethod
    def validate_map_score(cls, v: float) -> float:
        assert 0.0 <= v <= 1.0, f"MAP score must be between 0 and 1, got {v}"
        return v
