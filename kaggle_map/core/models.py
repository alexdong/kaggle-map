"""Core data structures for the Kaggle student misconception prediction competition."""

from enum import Enum
from typing import NamedTuple

import pandas as pd
from pydantic import BaseModel, field_validator

from kaggle_map.core.embeddings.formula import normalize_latex_answer, normalize_text

# Domain-specific type aliases
type RowId = int
type QuestionId = int
type Answer = str
type Question = str  # Question text from math problems
type Explanation = str  # Student's explanation of their reasoning
type Misconception = str  # Specific misconception identifier (e.g., "AddInsteadOfMultiply")


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
        prefix = "True_" if is_true else "False_"
        return [category for category in cls if category.value.startswith(prefix)]


class Prediction(BaseModel):
    category: Category
    misconception: Misconception = "NA"

    @classmethod
    def from_ground_truth_row(cls, row: pd.Series) -> "Prediction":
        """Create a Prediction from a ground truth CSV row."""
        category = Category(row["Category"])
        # Handle NaN misconceptions (pandas converts "NA" to NaN)
        misconception = row["Misconception"] if pd.notna(row["Misconception"]) else "NA"
        return cls(category=category, misconception=misconception)

    @classmethod
    def from_string(cls, prediction_str: str) -> "Prediction":
        pred_str = prediction_str.strip()

        assert ":" in pred_str, "Invalid prediction string format"
        category_part, misconception_part = pred_str.split(":", 1)
        category = Category(category_part.strip())
        misconception = misconception_part.strip() if misconception_part.strip() else "NA"
        return cls(category=category, misconception=misconception)

    def __str__(self) -> str:
        if self.category.is_misconception and self.misconception != "NA":
            return f"{self.category.value}:{self.misconception}"
        return f"{self.category.value}:NA"


class EvaluationRow(BaseModel):
    row_id: RowId
    question_id: QuestionId
    question_text: Question
    mc_answer: Answer
    student_explanation: Explanation

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

        Uses __repr__ (not __str__) because this is the exact format required by the
        embedding system via repr(row) calls. This isn't just for debugging - it's
        the data serialization format for ML processing.

        Example output:
            "Question: {Q}, Answer: {A}, Explanation: {E}"

        """
        return f"Question: {self.question_text}, Answer: {self.mc_answer}, Explanation: {self.student_explanation}"


class TrainingRow(EvaluationRow):
    """Training data row: EvaluationRow + Prediction.

    This represents the composition of question/answer/explanation data
    with the ground truth category and misconception prediction.
    """

    prediction: Prediction

    # Expose prediction fields at the top level for backward compatibility
    @property
    def category(self) -> Category:
        """Access the category from the embedded prediction."""
        return self.prediction.category

    @property
    def misconception(self) -> Misconception:
        """Access the misconception from the embedded prediction."""
        return self.prediction.misconception

    @classmethod
    def from_dataframe_row(cls, row: pd.Series) -> "TrainingRow":
        # Create the prediction first
        prediction = Prediction.from_ground_truth_row(row)

        return cls(
            row_id=int(row["row_id"]),
            question_id=int(row["QuestionId"]),
            question_text=str(row["QuestionText"]),
            mc_answer=str(row["MC_Answer"]),
            student_explanation=str(row["StudentExplanation"]),
            prediction=prediction,
        )


class SubmissionRow(NamedTuple):
    row_id: RowId
    predicted_categories: list[Prediction]  # Max 3, ordered by confidence
