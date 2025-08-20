"""Prediction logic and inference utilities for MLP strategy.

Handles prediction generation, misconception selection, and post-processing
using optimized library functions and proper type handling.
"""

import numpy as np
import torch
from loguru import logger

from kaggle_map.models import (
    Answer,
    Category,
    EvaluationRow,
    Misconception,
    Prediction,
    QuestionId,
)
from kaggle_map.strategies.mlp.config import (
    CORRECTNESS_THRESHOLD,
    MAX_PREDICTIONS,
    MISCONCEPTION_CONFIDENCE_THRESHOLD,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute sigmoid probabilities using PyTorch's optimized implementation.

    Uses PyTorch's battle-tested sigmoid for better performance and numerical
    stability compared to manual implementation. Handles edge cases automatically.

    Args:
        x: Input array for sigmoid computation

    Returns:
        Sigmoid probabilities with same shape as input
    """
    logger.debug(
        "Computing sigmoid using PyTorch",
        input_shape=x.shape,
        input_range=[float(x.min()), float(x.max())],
        input_mean=float(x.mean()),
    )

    # Convert to tensor, apply PyTorch sigmoid, convert back
    # PyTorch handles numerical stability automatically
    x_tensor = torch.from_numpy(x)
    sigmoid_probs = torch.sigmoid(x_tensor).numpy()

    logger.debug(
        "Sigmoid computed successfully",
        output_range=[float(sigmoid_probs.min()), float(sigmoid_probs.max())],
        output_mean=float(sigmoid_probs.mean()),
        values_above_threshold=(sigmoid_probs > CORRECTNESS_THRESHOLD).sum(),
    )

    return sigmoid_probs


def get_best_misconception(
    misconception_probs: np.ndarray,
    question_id: QuestionId,
    question_misconceptions: dict[QuestionId, list[Misconception]],
) -> Misconception | None:
    """Get the most likely misconception for a question using proper types.

    Args:
        misconception_probs: Probability array for misconceptions
        question_id: Question identifier
        question_misconceptions: Available misconceptions per question (proper type)

    Returns:
        Best misconception above threshold, or None
    """
    logger.debug(
        "Finding best misconception",
        question_id=question_id,
        misconception_probs_shape=misconception_probs.shape,
        confidence_threshold=MISCONCEPTION_CONFIDENCE_THRESHOLD,
    )

    if question_id not in question_misconceptions:
        logger.debug(
            "No misconceptions available for question",
            question_id=question_id,
            available_questions=len(question_misconceptions),
        )
        return None

    misconceptions = question_misconceptions[question_id]
    logger.debug(
        "Misconceptions loaded",
        question_id=question_id,
        total_misconceptions=len(misconceptions),
        misconceptions=[str(m) for m in misconceptions[:3]],  # Show first 3 for brevity
    )

    # Exclude NA (last element) from consideration
    if len(misconceptions) > 1:
        # Get probabilities excluding the NA class
        valid_probs = misconception_probs[:-1]
        best_idx = np.argmax(valid_probs)
        best_prob = misconception_probs[best_idx]
        best_misconception = misconceptions[best_idx]

        logger.debug(
            "Best misconception analysis",
            best_index=int(best_idx),
            best_misconception=str(best_misconception),
            best_probability=float(best_prob),
            threshold=MISCONCEPTION_CONFIDENCE_THRESHOLD,
            above_threshold=best_prob > MISCONCEPTION_CONFIDENCE_THRESHOLD,
            na_probability=float(misconception_probs[-1]),
        )

        if best_prob > MISCONCEPTION_CONFIDENCE_THRESHOLD:
            logger.debug(
                "Misconception selected",
                selected_misconception=str(best_misconception),
                confidence=float(best_prob),
            )
            return best_misconception
        logger.debug(
            "No misconception above threshold",
            best_probability=float(best_prob),
            threshold=MISCONCEPTION_CONFIDENCE_THRESHOLD,
        )
    else:
        logger.debug(
            "Insufficient misconceptions for selection",
            misconceptions_count=len(misconceptions),
        )

    return None


class MLPPredictor:
    """Handles MLP model predictions and post-processing.

    Encapsulates prediction logic for cleaner separation of concerns
    and easier testing.
    """

    def __init__(
        self,
        correct_answers: dict[QuestionId, Answer],
        question_misconceptions: dict[QuestionId, list[Misconception]],
    ) -> None:
        """Initialize predictor with required metadata.

        Args:
            correct_answers: Correct answers per question
            question_misconceptions: Available misconceptions per question (proper type)
        """
        self.correct_answers = correct_answers
        self.question_misconceptions = question_misconceptions

    def get_predictions_from_outputs(
        self,
        outputs: dict[str, torch.Tensor],
        question_id: QuestionId,
        row: EvaluationRow,
    ) -> list[Prediction]:
        """Get predictions from model outputs, focusing on misconceptions.

        Args:
            outputs: Model output tensors from forward pass
            question_id: Question identifier
            row: Evaluation row with student data

        Returns:
            List of predictions with proper types
        """
        predictions = []

        # Determine if answer is correct using rule-based approach
        is_correct = self.is_answer_correct(question_id, row.mc_answer)
        prefix = "True_" if is_correct else "False_"

        # Check if we have misconception predictions for this question
        if "misconceptions" in outputs and question_id in self.question_misconceptions:
            misconception_logits = outputs["misconceptions"].squeeze().cpu().numpy()
            misconception_probs = sigmoid(misconception_logits)

            # Get the best misconception using proper types
            best_misconception = get_best_misconception(misconception_probs, question_id, self.question_misconceptions)

            if best_misconception:
                # Create prediction with misconception
                fallback_category = Category(f"{prefix}Neither")
                misconception_prediction = Prediction(category=fallback_category, misconception=best_misconception)
                predictions.append(misconception_prediction)

        # Always add a fallback prediction based on correctness
        fallback_category = Category(f"{prefix}Neither")
        fallback_prediction = Prediction(category=fallback_category)
        predictions.append(fallback_prediction)

        return predictions[:MAX_PREDICTIONS]

    def create_default_prediction(self, row: EvaluationRow) -> list[Prediction]:
        """Create default prediction for unknown questions.

        Args:
            row: Evaluation row for unknown question

        Returns:
            List with default prediction
        """
        logger.debug(
            "Creating default prediction for unknown question",
            question_id=row.question_id,
            row_id=row.row_id,
        )

        # Simple default prediction
        default_category = Category("True_Neither")  # Default fallback
        default_prediction = Prediction(category=default_category)

        logger.debug(
            "Default prediction created",
            question_id=row.question_id,
            available_correct_answers=len(self.correct_answers),
        )

        return [default_prediction]

    def is_answer_correct(self, question_id: QuestionId, student_answer: Answer) -> bool:
        """Check if student answer matches the correct answer.

        Args:
            question_id: Question identifier
            student_answer: Student's answer

        Returns:
            True if answer is correct
        """
        correct_answer = self.correct_answers.get(question_id, "")
        is_correct = student_answer == correct_answer

        logger.debug(
            "Answer correctness check",
            question_id=question_id,
            student_answer=student_answer,
            correct_answer=correct_answer,
            is_correct=is_correct,
        )

        return is_correct
