"""MAP@3 evaluation metric implementation for kaggle competition."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
from loguru import logger

from kaggle_map.metrics import calculate_map_at_3
from kaggle_map.models import Prediction, RowId


def evaluate(
    ground_truth_path: Path,
    submission_path: Path,
    metric_fn: Callable[[Prediction, list[Prediction]], float] = calculate_map_at_3,
) -> float:
    """Evaluate predictions against ground truth using the specified metric.

    Args:
        ground_truth_path: Path to ground truth CSV file
        submission_path: Path to submission CSV file
        metric_fn: Metric function to use (defaults to MAP@3)

    Returns:
        Average score across all evaluated predictions
    """
    logger.debug(f"Evaluating {submission_path} against {ground_truth_path}")

    # Load and parse files
    ground_truth = _load_ground_truth(ground_truth_path)
    submissions = _load_submissions(submission_path)

    # Calculate metric using provided function
    total_score = 0.0
    common_row_ids = set(ground_truth.keys()) & set(submissions.keys())
    assert len(common_row_ids) > 0, "No common row IDs found"

    for row_id in common_row_ids:
        gt_prediction = ground_truth[row_id]
        submission_predictions = submissions[row_id]
        score = metric_fn(gt_prediction, submission_predictions)
        total_score += score

    total_observations = len(common_row_ids)
    map_score = total_score / total_observations
    logger.debug(f"Metric score: {map_score:.4f} over {total_observations} observations")
    return map_score


def _load_ground_truth(path: Path) -> dict[RowId, Prediction]:
    """Load ground truth CSV into Prediction objects."""
    assert path.exists(), f"Ground truth file not found: {path}"

    ground_truth_data = pd.read_csv(path)
    assert not ground_truth_data.empty, "Ground truth file cannot be empty"

    result = {
        int(row["row_id"]): Prediction.from_ground_truth_row(row)
        for _, row in ground_truth_data.iterrows()
    }

    logger.debug(f"Loaded {len(result)} ground truth rows")
    return result


def _load_submissions(path: Path) -> dict[RowId, list[Prediction]]:
    """Load submission CSV into Prediction objects."""
    assert path.exists(), f"Submission file not found: {path}"

    submission_data = pd.read_csv(path)

    result = {
        int(row["row_id"]): [
            Prediction.from_string(s)
            for s in str(row["Category:Misconception"]).split()
        ]
        for _, row in submission_data.iterrows()
    }

    logger.debug(f"Loaded {len(result)} submission rows")
    return result
