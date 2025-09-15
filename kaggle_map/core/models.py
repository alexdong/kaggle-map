"""Core data structures for the Kaggle student misconception prediction competition."""

from enum import Enum
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, field_validator

from kaggle_map.core.normalise import normalize_latex_answer, normalize_text

# ============================================================================
# Constants
# ============================================================================

RANDOM_SEED = 42  # Fixed seed for reproducibility across all experiments

# ============================================================================
# Type Aliases
# ============================================================================

# Domain-specific type aliases
RowId = int
QuestionId = int
Answer = str
Question = str  # Question text from math problems
Explanation = str  # Student's explanation of their reasoning
Misconception = str  # Specific misconception identifier
Label = str


# ============================================================================
# Core Models
# ============================================================================


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
    def from_csv_string(cls, csv_value: str) -> "Category":
        """Convert CSV format category string to Category enum.

        CSV files use ALL_CAPS format (e.g., 'TRUE_MISCONCEPTION') while
        Category enum uses Title_Case format (e.g., 'True_Misconception').

        Args:
            csv_value: Category string from CSV file (e.g., 'TRUE_MISCONCEPTION')

        Returns:
            Category enum instance

        Example:
            >>> Category.from_csv_string('TRUE_MISCONCEPTION')
            Category.TRUE_MISCONCEPTION
        """
        assert csv_value, "CSV category value cannot be empty"
        assert "_" in csv_value, f"Invalid CSV category format: '{csv_value}'"

        parts = csv_value.split("_")
        expected_parts = 2
        assert len(parts) == expected_parts, f"Invalid CSV category format: '{csv_value}'"

        formatted_value = f"{parts[0].capitalize()}_{parts[1].capitalize()}"
        return cls(formatted_value)

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
        category = Category.from_csv_string(str(row["Category"]))
        # Handle NaN misconceptions (pandas converts "NA" to NaN)
        misconception = row["Misconception"] if pd.notna(row["Misconception"]) else "NA"
        return cls(category=category, misconception=misconception)

    @classmethod
    def from_string(cls, prediction_str: str) -> "Prediction":
        pred_str = prediction_str.strip()

        assert ":" in pred_str, "Invalid prediction string format"
        category_part, misconception_part = pred_str.split(":", 1)
        category_str = category_part.strip()

        # Fix common typos in category names
        typo_corrections = {
            "False_NEither": "False_Neither",
            "False_neither": "False_Neither",
            "True_NEither": "True_Neither",
            "True_neither": "True_Neither",
            "False_Misconcpetion": "False_Misconception",
            "True_Misconcpetion": "True_Misconception",
            "False_Miscon": "False_Misconception",
            "True_Miscon": "True_Misconception",
        }
        category_str = typo_corrections.get(category_str, category_str)

        category = Category(category_str)
        misconception = misconception_part.strip() if misconception_part.strip() else "NA"
        return cls(category=category, misconception=misconception)

    def __str__(self) -> Label:
        if self.category.is_misconception and self.misconception != "NA":
            return f"{self.category.value}:{self.misconception}"
        return f"{self.category.value}:NA"

    @classmethod
    def parse(cls, response: str) -> list["Prediction"]:
        predictions = []
        response_clean = response.strip()
        response = " ".join([s.strip() for s in response_clean.split("\n") if s.strip()])
        prediction_parts = response.split()

        for part in prediction_parts:
            if ":" in part:
                # Standard format: Category:Misconception
                prediction = cls.from_string(part)
            else:
                # Check if it's a valid category without a colon
                category = next((cat for cat in Category if part == cat.value), None)
                if category is None:
                    logger.debug(f"Skipping invalid prediction part: '{part}'")
                    continue
                # Create prediction with NA misconception for categories without colons
                prediction = cls(category=category, misconception="NA")

            predictions.append(prediction)
        return predictions[:3]


def normalize_label(label: Label) -> Label:
    return label.replace("Category.", "").title()


def compare_labels(actual: Label, predicted: Label) -> bool:
    """Compare two labels with normalization."""
    assert actual, f"Actual label cannot be empty: '{actual}'"
    assert predicted, f"Predicted label cannot be empty: '{predicted}'"

    # Fast path: exact match
    if actual == predicted:
        return True

    # Normalize and compare
    actual_norm = normalize_label(actual)
    predicted_norm = normalize_label(predicted)

    # Handle category:misconception format
    if ":" in actual_norm and ":" in predicted_norm:
        actual_parts = actual_norm.split(":", 1)
        pred_parts = predicted_norm.split(":", 1)

        # Category must match exactly
        if actual_parts[0] != pred_parts[0]:
            return False

        # Misconception comparison (case-insensitive)
        return actual_parts[1].lower() == pred_parts[1].lower()

    return actual_norm == predicted_norm


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


class EvaluationResult(BaseModel):
    """Result of evaluating a single row with LLM predictions."""

    row_id: RowId
    question_id: QuestionId
    mc_answer: Answer
    explanation: Explanation
    ground_truth: Prediction
    predictions: list[Prediction]
    score: float

    def __repr__(self) -> str:
        return f"EvaluationResult(row_id={self.row_id}, score={self.score:.2f})"


class SubmissionRow(NamedTuple):
    row_id: RowId
    predicted_categories: list[Prediction]  # Max 3, ordered by confidence


# ============================================================================
# Embedding Strategies
# ============================================================================


class EmbeddingModel(Enum):
    QWEN = "qwen"
    GEMMA = "gemma"


class EmbeddingStrategy(Enum):
    """Strategies for computing embeddings from evaluation rows."""

    GOAL_DRIVEN = "goal_driven"

    @property
    def dimension(self) -> int:
        """Return the output dimension for this embedding strategy."""
        return 8192  # GOAL_DRIVEN uses single embedding (QWEN dimension)

    @classmethod
    def from_string(cls, value: str | None) -> "EmbeddingStrategy":
        """Convert string to enum, with default to GOAL_DRIVEN."""
        if value is None:
            return cls.GOAL_DRIVEN
        return cls(value)


# ============================================================================
# Training Details
# ============================================================================


class ArchitectureSize(Enum):
    """MLP architecture size options."""

    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"


class ActivationType(Enum):
    """Neural network activation function types."""

    RELU = "relu"
    GELU = "gelu"
    LEAKY_RELU = "leaky_relu"
    SILU = "silu"


class OptimizerType(Enum):
    """Optimizer types for training."""

    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"


class SchedulerType(Enum):
    """Learning rate scheduler types."""

    NONE = "none"
    COSINE = "cosine"
    ONECYCLE = "onecycle"


class TrainingConfig(BaseModel):
    """Configuration for MLP training."""

    # Training parameters
    epochs: int = 50
    batch_size: int = 256
    dropout: float = 0.3
    activation: ActivationType = ActivationType.GELU
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # Optimizer and scheduler
    optimizer: OptimizerType = OptimizerType.ADAMW
    scheduler: SchedulerType = SchedulerType.COSINE
    early_stopping_patience: int = 15

    # Data split
    train_split: float = 0.7

    # Architecture and embedding
    embedding_model: EmbeddingModel = EmbeddingModel.QWEN
    embedding_strategy: EmbeddingStrategy = EmbeddingStrategy.GOAL_DRIVEN
    architecture_size: ArchitectureSize = ArchitectureSize.XLARGE

    # File paths
    train_csv_path: Path = Path("datasets/train.csv")

    model_config = {"arbitrary_types_allowed": True}
