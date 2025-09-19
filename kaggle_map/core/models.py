"""Core data structures for the Kaggle student misconception prediction competition."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pandas as pd
import numpy as np
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
    def from_csv_string(cls, csv_value: str) -> Category:
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
    def by_truth_value(cls, *, is_true: bool) -> list[Category]:
        prefix = "True_" if is_true else "False_"
        return [category for category in cls if category.value.startswith(prefix)]


class Prediction(BaseModel):
    category: Category
    misconception: Misconception = "NA"

    @classmethod
    def from_ground_truth_row(cls, row: pd.Series) -> Prediction:
        """Create a Prediction from a ground truth CSV row."""
        category = Category.from_csv_string(str(row["Category"]))
        # Handle NaN misconceptions (pandas converts "NA" to NaN)
        misconception = row["Misconception"] if pd.notna(row["Misconception"]) else "NA"
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
    def from_dataframe_row(cls, row: pd.Series) -> TrainingRow:
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

    DOUBLE_BLIND = "double_blind"
    GOAL_DRIVEN = "goal_driven"  # Previously called SEMANTIC

    @property
    def dimension(self) -> int:
        """Return the output dimension for this embedding strategy."""
        # DOUBLE_BLIND concatenates two embeddings (2 * 4096 for QWEN = 8192)
        # GOAL_DRIVEN uses single embedding (8192 for QWEN)
        # Since QWEN now produces 8192-dim embeddings natively, both are 8192
        # But DOUBLE_BLIND would be 16384 if we concatenate two 8192 embeddings
        # Let's keep it at historical sizes for compatibility
        return 16384 if self == EmbeddingStrategy.DOUBLE_BLIND else 8192

    @classmethod
    def from_string(cls, value: str | None) -> EmbeddingStrategy:
        """Convert string to enum, with default to DOUBLE_BLIND."""
        if value is None:
            return cls.DOUBLE_BLIND
        # Handle backward compatibility: semantic -> goal_driven
        if value == "semantic":
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
    embedding_strategy: EmbeddingStrategy = EmbeddingStrategy.DOUBLE_BLIND
    architecture_size: ArchitectureSize = ArchitectureSize.XLARGE
    embedding_dim: int | None = None

    # File paths
    train_csv_path: Path = Path("datasets/train.csv")

    model_config = {"arbitrary_types_allowed": True}


# ============================================================================
# Confidence Routing Models
# ============================================================================

# Constants for validation
MAX_PREDICTIONS = 3
PROBABILITY_TOLERANCE = 1e-6
PostInitContext = Mapping[str, object] | None


class MLPPredictionResult(BaseModel):
    """Single MLP prediction with confidence metrics for routing decisions.

    Contains the raw prediction output from MLP along with entropy score
    used to determine whether this prediction should be routed to LLM.
    """

    row_id: RowId
    question_id: QuestionId
    top_predictions: list[Prediction]  # Top 3 predictions ordered by confidence
    top_probabilities: list[float]  # Corresponding softmax probabilities
    entropy: float  # Entropy score: -sum(p_i * log(p_i)) for routing
    prediction_time_ms: float  # Time taken for this prediction

    def model_post_init(self, __context: PostInitContext) -> None:
        """Validate prediction result consistency."""
        assert len(self.top_predictions) <= MAX_PREDICTIONS, "Maximum 3 predictions allowed"
        assert len(self.top_predictions) == len(self.top_probabilities), "Predictions and probabilities must match"
        assert all(0.0 <= p <= 1.0 for p in self.top_probabilities), "Probabilities must be in [0,1]"
        assert abs(sum(self.top_probabilities) - 1.0) < PROBABILITY_TOLERANCE, "Probabilities must sum to 1.0"
        assert self.entropy >= 0.0, "Entropy must be non-negative"


class LLMPredictionResult(BaseModel):
    """LLM prediction result with enhanced reasoning and timing.

    Represents the output from LLM processing, including chain-of-thought
    reasoning and performance metrics for routing evaluation.
    """

    row_id: RowId
    question_id: QuestionId
    predictions: list[Prediction]  # LLM's top 3 predictions
    reasoning: str  # Chain-of-thought explanation
    prediction_time_ms: float  # Time taken for LLM prediction
    success: bool  # Whether LLM prediction completed successfully

    def model_post_init(self, __context: PostInitContext) -> None:
        """Validate LLM prediction result."""
        assert len(self.predictions) <= MAX_PREDICTIONS, "Maximum 3 predictions allowed"
        if self.success:
            assert len(self.predictions) > 0, "Successful predictions must have at least one result"
            assert self.reasoning.strip(), "Successful predictions must have reasoning"


class RoutingDecision(BaseModel):
    """Decision record for routing a prediction to LLM.

    Tracks the decision-making process for whether to route an MLP
    prediction to LLM based on entropy and time constraints.
    """

    row_id: RowId
    entropy: float
    should_route: bool  # Whether this should be routed to LLM
    routing_rank: int | None = None  # Rank in entropy-sorted queue (1-based)
    reason: str = ""  # Explanation for routing decision

    def model_post_init(self, __context: PostInitContext) -> None:
        """Validate routing decision."""
        assert self.entropy >= 0.0, "Entropy must be non-negative"
        if self.should_route:
            assert self.routing_rank is not None, "Routed predictions must have rank"
            assert self.routing_rank > 0, "Routing rank must be positive"


class PredictionState(Enum):
    """State of a prediction in the routing pipeline."""

    MLP_ONLY = "mlp_only"  # Only MLP prediction available
    LLM_PENDING = "llm_pending"  # Queued for LLM processing
    LLM_PROCESSING = "llm_processing"  # Currently being processed by LLM
    LLM_COMPLETE = "llm_complete"  # LLM processing completed successfully
    LLM_FAILED = "llm_failed"  # LLM processing failed, fallback to MLP


class RoutedPrediction(BaseModel):
    """Combined prediction result with routing state and final predictions.

    Represents the complete prediction pipeline result, including both
    MLP and optional LLM predictions with routing decision tracking.
    """

    row_id: RowId
    question_id: QuestionId

    # Core prediction data
    mlp_result: MLPPredictionResult
    llm_result: LLMPredictionResult | None = None

    # Routing information
    routing_decision: RoutingDecision
    state: PredictionState = PredictionState.MLP_ONLY

    # Final output
    final_predictions: list[Prediction] | None = None  # Final top-3 for submission

    @property
    def entropy(self) -> float:
        """Access entropy from MLP result."""
        return self.mlp_result.entropy

    @property
    def was_routed_to_llm(self) -> bool:
        """Whether this prediction was processed by LLM."""
        return self.state in {PredictionState.LLM_COMPLETE, PredictionState.LLM_FAILED}

    @property
    def used_llm_prediction(self) -> bool:
        """Whether final prediction uses LLM result."""
        return self.state == PredictionState.LLM_COMPLETE and self.llm_result is not None

    def get_best_predictions(self) -> list[Prediction]:
        """Get the best available predictions for this row.

        Returns LLM predictions if available and successful,
        otherwise falls back to MLP predictions.
        """
        if self.final_predictions is not None:
            return self.final_predictions

        if self.used_llm_prediction:
            assert self.llm_result is not None
            return self.llm_result.predictions

        return self.mlp_result.top_predictions

    def model_post_init(self, __context: PostInitContext) -> None:
        """Validate routed prediction consistency."""
        assert self.row_id == self.mlp_result.row_id, "Row IDs must match"
        assert self.question_id == self.mlp_result.question_id, "Question IDs must match"

        if self.llm_result is not None:
            assert self.row_id == self.llm_result.row_id, "LLM result row ID must match"
            assert self.question_id == self.llm_result.question_id, "LLM result question ID must match"


class RoutingSession(BaseModel):
    """Session state for the complete routing pipeline execution.

    Tracks the overall progress and timing of processing all predictions
    through the MLP -> entropy sorting -> LLM routing pipeline.
    """

    # Configuration
    total_time_budget_seconds: float  # Maximum time allowed for LLM processing

    # Session state
    predictions: dict[RowId, RoutedPrediction]  # All predictions by row_id
    entropy_sorted_row_ids: list[RowId]  # Row IDs sorted by entropy (high to low)

    # Execution tracking
    session_start_time: float  # Unix timestamp when session started
    mlp_phase_complete: bool = False
    llm_processing_complete: bool = False

    # Statistics
    total_predictions: int = 0
    predictions_routed_to_llm: int = 0
    predictions_completed_by_llm: int = 0
    predictions_failed_by_llm: int = 0
    total_llm_time_used_seconds: float = 0.0

    @property
    def llm_time_remaining_seconds(self) -> float:
        """Calculate remaining time budget for LLM processing."""
        return max(0.0, self.total_time_budget_seconds - self.total_llm_time_used_seconds)

    @property
    def is_time_budget_exhausted(self) -> bool:
        """Check if time budget has been exhausted."""
        return self.total_llm_time_used_seconds >= self.total_time_budget_seconds

    def get_next_prediction_for_llm(self) -> RoutedPrediction | None:
        """Get the next highest-entropy prediction that needs LLM processing.

        Returns None if no predictions need processing or time budget exhausted.
        """
        if self.is_time_budget_exhausted:
            return None

        for row_id in self.entropy_sorted_row_ids:
            prediction = self.predictions[row_id]
            if prediction.routing_decision.should_route and prediction.state == PredictionState.LLM_PENDING:
                return prediction

        return None

    def add_mlp_prediction(self, mlp_result: MLPPredictionResult, routing_decision: RoutingDecision) -> None:
        """Add a new MLP prediction result to the session."""
        routed_prediction = RoutedPrediction(
            row_id=mlp_result.row_id,
            question_id=mlp_result.question_id,
            mlp_result=mlp_result,
            routing_decision=routing_decision,
            state=PredictionState.LLM_PENDING if routing_decision.should_route else PredictionState.MLP_ONLY,
        )

        self.predictions[mlp_result.row_id] = routed_prediction
        self.total_predictions += 1

        if routing_decision.should_route:
            self.predictions_routed_to_llm += 1

    def update_llm_result(self, row_id: RowId, llm_result: LLMPredictionResult) -> None:
        """Update a prediction with LLM processing result."""
        assert row_id in self.predictions, f"Row {row_id} not found in session"

        prediction = self.predictions[row_id]
        prediction.llm_result = llm_result
        prediction.state = PredictionState.LLM_COMPLETE if llm_result.success else PredictionState.LLM_FAILED

        self.total_llm_time_used_seconds += llm_result.prediction_time_ms / 1000.0

        if llm_result.success:
            self.predictions_completed_by_llm += 1
        else:
            self.predictions_failed_by_llm += 1

    def finalize_session(self) -> None:
        """Finalize the routing session and compute final predictions."""
        self.llm_processing_complete = True

        # Set final predictions for all rows
        for prediction in self.predictions.values():
            prediction.final_predictions = prediction.get_best_predictions()

    def get_submission_data(self) -> list[SubmissionRow]:
        """Generate submission data from final predictions."""
        assert self.llm_processing_complete, "Session must be finalized before generating submission"

        submission_rows = []
        for row_id in sorted(self.predictions.keys()):
            prediction = self.predictions[row_id]
            final_preds = prediction.get_best_predictions()

            submission_rows.append(SubmissionRow(row_id=row_id, predicted_categories=final_preds))

        return submission_rows

    def get_performance_summary(self) -> dict[str, float | int]:
        """Get summary statistics for the routing session."""
        llm_success_rate = self.predictions_completed_by_llm / max(1, self.predictions_routed_to_llm)

        return {
            "total_predictions": self.total_predictions,
            "predictions_routed_to_llm": self.predictions_routed_to_llm,
            "predictions_completed_by_llm": self.predictions_completed_by_llm,
            "predictions_failed_by_llm": self.predictions_failed_by_llm,
            "llm_success_rate": llm_success_rate,
            "routing_percentage": self.predictions_routed_to_llm / max(1, self.total_predictions),
            "total_llm_time_used_seconds": self.total_llm_time_used_seconds,
            "time_budget_utilization": self.total_llm_time_used_seconds / max(1e-9, self.total_time_budget_seconds),
        }


