"""MAP@3 evaluation metric implementation for kaggle competition."""

from pathlib import Path

import pandas as pd
from loguru import logger
from pydantic import BaseModel, field_validator

from .models import Category


class EvaluationResult(BaseModel):
    """MAP@3 evaluation result with detailed breakdown."""

    map_score: float
    total_observations: int
    perfect_predictions: int
    valid_predictions: int
    invalid_predictions: int

    @field_validator("map_score")
    @classmethod
    def validate_map_score(cls, v: float) -> float:
        assert 0.0 <= v <= 1.0, f"MAP score must be between 0 and 1, got {v}"
        return v


def evaluate_map3(ground_truth_path: Path, submission_path: Path) -> EvaluationResult:
    """Calculate MAP@3 score for kaggle submission.

    Args:
        ground_truth_path: Path to ground truth CSV with Category and Misconception columns
        submission_path: Path to submission CSV with predictions column

    Returns:
        EvaluationResult with MAP@3 score and detailed breakdown
    """
    logger.debug(f"Evaluating {submission_path} against {ground_truth_path}")

    # Load and parse files
    ground_truth = _load_ground_truth(ground_truth_path)
    submissions = _load_submissions(submission_path)

    # Calculate MAP@3 over common row_ids
    total_score = 0.0
    perfect_predictions = 0
    valid_predictions = 0
    invalid_predictions = 0

    common_row_ids = set(ground_truth.keys()) & set(submissions.keys())

    for row_id in common_row_ids:
        gt_label = ground_truth[row_id]
        predictions = submissions[row_id]

        # Calculate average precision for this observation
        ap = _calculate_average_precision(gt_label, predictions)
        total_score += ap

        if ap == 1.0:
            perfect_predictions += 1

        # Count prediction validity
        valid_count, invalid_count = _count_prediction_validity(predictions)
        valid_predictions += valid_count
        invalid_predictions += invalid_count

    total_observations = len(common_row_ids)
    map_score = total_score / total_observations if total_observations > 0 else 0.0

    logger.debug(f"MAP@3: {map_score:.4f} over {total_observations} observations")

    return EvaluationResult(
        map_score=map_score,
        total_observations=total_observations,
        perfect_predictions=perfect_predictions,
        valid_predictions=valid_predictions,
        invalid_predictions=invalid_predictions,
    )


def _load_ground_truth(path: Path) -> dict[int, str]:
    """Load ground truth CSV into dict mapping row_id -> category:misconception label."""
    assert path.exists(), f"Ground truth file not found: {path}"

    ground_truth_data = pd.read_csv(path)
    assert not ground_truth_data.empty, "Ground truth file cannot be empty"

    result = {}
    for _, row in ground_truth_data.iterrows():
        row_id = int(row["row_id"])
        category = Category(row["Category"])

        # Handle misconception - NA/None becomes empty, which means just category
        misconception = row["Misconception"]
        if pd.isna(misconception) or misconception == "NA":
            label = category.value
        else:
            label = f"{category.value}:{misconception}"

        result[row_id] = label

    logger.debug(f"Loaded {len(result)} ground truth rows")
    return result


def _load_submissions(path: Path) -> dict[int, list[str]]:
    """Load submission CSV into dict mapping row_id -> list of up to 3 predictions."""
    assert path.exists(), f"Submission file not found: {path}"

    submission_data = pd.read_csv(path)

    result = {}
    for _, row in submission_data.iterrows():
        row_id = int(row["row_id"])
        predictions_str = str(row["predictions"]).strip()

        # Split by spaces and take only first 3 predictions
        if predictions_str and predictions_str != "nan":
            predictions = predictions_str.split()[:3]
        else:
            predictions = []

        result[row_id] = predictions

    logger.debug(f"Loaded {len(result)} submission rows")
    return result


def _calculate_average_precision(
    ground_truth_label: str, predictions: list[str]
) -> float:
    """Calculate average precision for a single observation using MAP@3 formula."""
    if not predictions:
        return 0.0

    # Check each prediction position (1-indexed for precision calculation)
    for k, prediction in enumerate(predictions, 1):
        if _normalize_prediction(prediction) == _normalize_prediction(
            ground_truth_label
        ):
            # Found correct prediction at position k, precision = 1/k
            return 1.0 / k

    # No correct prediction found
    return 0.0


def _normalize_prediction(prediction: str) -> str:
    """Normalize prediction string for consistent comparison."""
    return prediction.strip()


def _count_prediction_validity(predictions: list[str]) -> tuple[int, int]:
    """Count valid and invalid predictions based on Category:Misconception format."""
    valid_count = 0
    invalid_count = 0

    for prediction in predictions:
        if _is_valid_prediction(prediction):
            valid_count += 1
        else:
            invalid_count += 1

    return valid_count, invalid_count


def _is_valid_prediction(prediction: str) -> bool:
    """Check if prediction follows valid Category:Misconception format."""
    prediction = prediction.strip()
    if not prediction:
        return False

    try:
        if ":" in prediction:
            category_part, _ = prediction.split(":", 1)
            Category(category_part.strip())
        else:
            Category(prediction)
        return True
    except ValueError:
        return False


# Backward compatibility alias (avoids shadowing builtin eval in imports)
eval_map3 = evaluate_map3
