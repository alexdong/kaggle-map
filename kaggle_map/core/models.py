"""Core data structures for the Kaggle student misconception prediction competition."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator

from kaggle_map.core.embeddings.formula import normalize_latex_answer, normalize_text

# Domain-specific type aliases
RowId = int
QuestionId = int
Answer = str
Question = str  # Question text from math problems
Explanation = str  # Student's explanation of their reasoning
Misconception = str  # Specific misconception identifier


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

    # Optional context fields for LLM inference
    correct_answer: Answer | None = None
    known_misconceptions: list[Misconception] | None = None

    @field_validator("question_text", "student_explanation")
    @classmethod
    def normalize_text_fields(cls, v: str) -> str:
        return normalize_text(v)

    @field_validator("mc_answer")
    @classmethod
    def normalize_answer(cls, v: str) -> str:
        return normalize_latex_answer(v)

    def to_embedding_text(self) -> str:
        """Generate the canonical Q/A/E string used for embeddings.

        Example output:
            "Question: {}; Student's Answer: {}; Student's Explanation: {}"
        """
        return (
            f"Question: {self.question_text}; "
            f"Student's Answer: {self.mc_answer}; Student's Explanation: {self.student_explanation}"
        )

    def __repr__(self) -> str:
        """Standard repr for debugging."""
        return f"EvaluationRow(row_id={self.row_id}, question_id={self.question_id})"


class TrainingInput(NamedTuple):
    question_id: QuestionId
    embeddings: np.ndarray
    misconception: Misconception


class TrainingRow(EvaluationRow):
    """Training data row: EvaluationRow + Prediction.

    This represents the composition of question/answer/explanation data
    with the ground truth category and misconception prediction.
    """

    prediction: Prediction
    training_examples: list["TrainingRow"] | None = None  # For few-shot prompting

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

    def as_training_input(self) -> TrainingInput:
        # Import here to avoid circular dependency
        from kaggle_map.core.embeddings.tokenizer import get_tokenizer

        tokenizer = get_tokenizer()
        text = self.to_embedding_text()

        # Generate embeddings and convert to numpy array
        embeddings_tensor = tokenizer.encode(text)
        embeddings = np.array(embeddings_tensor)

        return TrainingInput(
            question_id=self.question_id,
            embeddings=embeddings,
            misconception=self.misconception,
        )


class SubmissionRow(NamedTuple):
    row_id: RowId
    predicted_categories: list[Prediction]  # Max 3, ordered by confidence


# Type aliases for LLM operations
Label = str  # "True_Misconception:AddInsteadOfMultiply"
PromptTemplate = str
LLMResponse = str
ModelName = Literal["gemma-3-12b-it", "Qwen3-14B"]
QuantizationLevel = Literal["Q4_K_XL", "Q5_K_XL", "Q6_K_XL"]


@dataclass(frozen=True)
class RerankingRequest:
    """Complete request for reranking predictions."""

    evaluation_row: EvaluationRow
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
