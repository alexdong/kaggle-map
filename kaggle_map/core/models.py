"""Core data structures for the Kaggle student misconception prediction competition."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, NamedTuple, get_args

import numpy as np
import pandas as pd
import pydash
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
        category = Category(category_part.strip())
        misconception = misconception_part.strip() if misconception_part.strip() else "NA"
        return cls(category=category, misconception=misconception)

    def __str__(self) -> Label:
        if self.category.is_misconception and self.misconception != "NA":
            return f"{self.category.value}:{self.misconception}"
        return f"{self.category.value}:NA"


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

    DOUBLE_BLIND = "double_blind"
    SEMANTIC = "semantic"

    @property
    def dimension(self) -> int:
        """Return the output dimension for this embedding strategy."""
        return 8192 if self == EmbeddingStrategy.DOUBLE_BLIND else 4096

    @classmethod
    def from_string(cls, value: str | None) -> "EmbeddingStrategy":
        """Convert string to enum, with default to DOUBLE_BLIND."""
        if value is None:
            return cls.DOUBLE_BLIND
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
    embedding_strategy: EmbeddingStrategy = EmbeddingStrategy.DOUBLE_BLIND
    architecture_size: ArchitectureSize = ArchitectureSize.XLARGE

    # File paths
    train_csv_path: Path = Path("datasets/train.csv")

    model_config = {"arbitrary_types_allowed": True}


# ============================================================================
# Reranker Classes
# ============================================================================

# LLM operation type aliases
PromptTemplate = str
LLMResponse = str
RerankerModelName = Literal["Qwen3-14B", "gemma-3-12b-it", "gpt-oss-20b"]
# NOTE: Q4_K_XL and Q5_K_XL have sequential loading conflicts in llama-cpp-python
# Use only one quantization per benchmark session to avoid GPU context corruption
RerankerModelQuantizationLevel = Literal["Q2_K_XL", "Q3_K_XL", "Q4_K_XL", "Q5_K_XL", "Q6_K_XL"]

# Available options derived from type definitions
MODEL_OPTIONS: list[RerankerModelName] = list(get_args(RerankerModelName))
QUANTIZATION_OPTIONS: list[RerankerModelQuantizationLevel] = list(get_args(RerankerModelQuantizationLevel))


class GGUFRepoSpec(NamedTuple):
    """Specification for a GGUF model repository and filename pattern."""

    repo: str  # HuggingFace repository ID
    filename_pattern: str  # Pattern with {quant} placeholder for quantization level
    available_quantizations: list[RerankerModelQuantizationLevel] = QUANTIZATION_OPTIONS


# Model configurations with their HuggingFace patterns
GGUF_MODELS: dict[RerankerModelName, GGUFRepoSpec] = {
    "gpt-oss-20b": GGUFRepoSpec(
        repo="unsloth/gpt-oss-20b-GGUF",
        filename_pattern="gpt-oss-20b-{quant}.gguf",
        available_quantizations=pydash.without(QUANTIZATION_OPTIONS, "Q5_K_XL"),
    ),
    "Qwen3-14B": GGUFRepoSpec(
        repo="unsloth/Qwen3-14B-GGUF",
        filename_pattern="Qwen3-14B-{quant}.gguf",
        # Temporarily test only Q5_K_XL due to sequential loading conflicts
    ),
    "gemma-3-12b-it": GGUFRepoSpec(
        repo="unsloth/gemma-3-12b-it-GGUF",
        filename_pattern="gemma-3-12b-it-{quant}.gguf",
    ),
}


@dataclass
class RerankerLLMLoadConfig:
    """Configuration for loading GGUF models into memory.

    gpt-oss-20b: doesn't follow instruction tuning well.
    Qwen3-14B: Q4: 0.6005; Q6: 0.6021
    gemma-3-12b-it: Q4: 0.6185; Q6: 0.6193

    The Q4 is slightly worse but much much faster, so it's a good trade-off.
    Further, gemma-3 is smaller but slightly better than Qwen3, so it's a good choice
    """

    model_name: RerankerModelName = "gemma-3-12b-it"
    quantization: RerankerModelQuantizationLevel = "Q4_K_XL"
    n_ctx: int = 4096  # Context window size
    n_batch: int = 512  # Batch size for prompt processing
    n_gpu_layers: int = -1  # Use all available GPU layers
    n_threads: int = 8  # CPU threads for inference
    random_seed: int = 42
    verbose: bool = False  # Verbose llama.cpp output

    @property
    def model_filename(self) -> str:
        """Get the GGUF filename for this configuration."""
        return f"{self.model_name}-{self.quantization}.gguf"
