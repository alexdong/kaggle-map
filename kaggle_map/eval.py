"""MAP@3 evaluation metric implementation for kaggle competition."""

from pathlib import Path

import pandas as pd
from loguru import logger

from .models import Category, EvaluationResult, Prediction


def evaluate(ground_truth_path: Path, submission_path: Path) -> EvaluationResult:
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

    common_row_ids = set(ground_truth.keys()) & set(submissions.keys())

    for row_id in common_row_ids:
        gt_prediction = ground_truth[row_id]
        submission_predictions = submissions[row_id]

        # Calculate average precision for this observation
        ap = _calculate_average_precision(gt_prediction, submission_predictions)
        total_score += ap

        if ap == 1.0:
            perfect_predictions += 1

        # Count prediction validity (all parsed predictions are valid by definition)
        valid_predictions += len(submission_predictions)

    total_observations = len(common_row_ids)
    map_score = total_score / total_observations if total_observations > 0 else 0.0

    logger.debug(f"MAP@3: {map_score:.4f} over {total_observations} observations")

    return EvaluationResult(
        map_score=map_score,
        total_observations=total_observations,
        perfect_predictions=perfect_predictions,
        valid_predictions=valid_predictions,
        invalid_predictions=0,  # Invalid predictions are filtered during parsing
    )


def _load_ground_truth(path: Path) -> dict[int, Prediction]:
    """Load ground truth CSV into dict mapping row_id -> Prediction object."""
    assert path.exists(), f"Ground truth file not found: {path}"

    ground_truth_data = pd.read_csv(path)
    assert not ground_truth_data.empty, "Ground truth file cannot be empty"

    result = {}
    for _, row in ground_truth_data.iterrows():
        row_id = int(row["row_id"])
        category = Category(row["Category"])

        # Handle misconception - NA/None becomes None for Prediction object
        misconception = row["Misconception"]
        if pd.isna(misconception) or misconception == "NA":
            misconception = None

        prediction = Prediction(category=category, misconception=misconception)
        result[row_id] = prediction

    logger.debug(f"Loaded {len(result)} ground truth rows")
    return result


def _load_submissions(path: Path) -> dict[int, list[Prediction]]:
    """Load submission CSV into dict mapping row_id -> list of up to 3 Prediction objects."""
    assert path.exists(), f"Submission file not found: {path}"

    submission_data = pd.read_csv(path)

    result = {}
    for _, row in submission_data.iterrows():
        row_id = int(row["row_id"])
        predictions_str = str(row["predictions"]).strip()

        # Split by spaces and parse into Prediction objects
        predictions = []
        if predictions_str and predictions_str != "nan":
            prediction_strings = predictions_str.split()[:3]
            for pred_str in prediction_strings:
                try:
                    prediction = _parse_prediction_string(pred_str)
                    predictions.append(prediction)
                except ValueError:
                    # Invalid prediction format - skip it
                    logger.debug(f"Skipping invalid prediction: {pred_str}")

        result[row_id] = predictions

    logger.debug(f"Loaded {len(result)} submission rows")
    return result


def _parse_prediction_string(pred_str: str) -> Prediction:
    """Parse a prediction string into a Prediction object."""
    pred_str = pred_str.strip()

    if ":" in pred_str:
        category_part, misconception_part = pred_str.split(":", 1)
        category = Category(category_part.strip())
        misconception = (
            misconception_part.strip() if misconception_part.strip() else None
        )
    else:
        category = Category(pred_str)
        misconception = None

    return Prediction(category=category, misconception=misconception)


def _calculate_average_precision(
    ground_truth: Prediction, predictions: list[Prediction]
) -> float:
    """Calculate average precision for a single observation using MAP@3 formula."""
    if not predictions:
        return 0.0

    # Check each prediction position (1-indexed for precision calculation)
    for k, prediction in enumerate(predictions, 1):
        if _predictions_match(ground_truth, prediction):
            # Found correct prediction at position k, precision = 1/k
            return 1.0 / k

    # No correct prediction found
    return 0.0


def _predictions_match(ground_truth: Prediction, prediction: Prediction) -> bool:
    """Check if two predictions match using their value representation."""
    return ground_truth.value == prediction.value


if __name__ == "__main__":
    """Demonstrate MAP@3 evaluation with sample data using Prediction objects."""
    logger.info("Running eval.py demonstration")

    import tempfile
    from pathlib import Path

    import pandas as pd

    # Create sample data for demonstration using the rich type system
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        logger.info("Creating sample evaluation data")

        # Sample ground truth data
        ground_truth_data = {
            "row_id": [1, 2, 3, 4, 5],
            "Category": [
                "True_Correct",
                "False_Misconception",
                "True_Misconception",
                "False_Neither",
                "True_Neither",
            ],
            "Misconception": [
                "Adding_across",
                "Denominator-only_change",
                "Incorrect_equivalent_fraction_addition",
                "NA",
                "NA",
            ],
        }

        # Sample submission data with various prediction accuracies
        submission_data = {
            "row_id": [1, 2, 3, 4, 5],
            "predictions": [
                # Perfect prediction (1st position)
                "True_Correct:Adding_across False_Neither:Other_tag True_Misconception:Wrong_tag",
                # Correct in 2nd position
                "False_Neither:Wrong_tag False_Misconception:Denominator-only_change True_Correct:Other_tag",
                # Correct in 3rd position
                "False_Neither:Wrong_tag True_Correct:Other_tag True_Misconception:Incorrect_equivalent_fraction_addition",
                # Incorrect prediction
                "True_Correct:Wrong_tag False_Misconception:Other_tag True_Misconception:Bad_tag",
                # Correct prediction (category only, no misconception)
                "True_Neither False_Correct True_Misconception:Some_tag",
            ],
        }

        # Save to temporary CSV files
        gt_path = tmp_path / "ground_truth.csv"
        sub_path = tmp_path / "submission.csv"

        pd.DataFrame(ground_truth_data).to_csv(gt_path, index=False)
        pd.DataFrame(submission_data).to_csv(sub_path, index=False)

        logger.info(f"Ground truth saved to: {gt_path}")
        logger.info(f"Submission saved to: {sub_path}")

        # Evaluate MAP@3
        logger.info("Calculating MAP@3 score")
        result = evaluate(gt_path, sub_path)

        # Display results
        print("\n=== MAP@3 EVALUATION RESULTS ===")
        print(f"MAP@3 Score: {result.map_score:.4f}")
        print(f"Total Observations: {result.total_observations}")
        print(f"Perfect Predictions (1st position): {result.perfect_predictions}")
        print(f"Valid Predictions: {result.valid_predictions}")
        print(f"Invalid Predictions: {result.invalid_predictions}")

        # Calculate breakdown by position
        print("\nExpected breakdown:")
        print("  Row 1: Perfect (1st position) -> AP = 1.0")
        print("  Row 2: 2nd position -> AP = 0.5")
        print("  Row 3: 3rd position -> AP = 0.333...")
        print("  Row 4: No match -> AP = 0.0")
        print("  Row 5: Perfect (1st position) -> AP = 1.0")

        expected_map = (1.0 + 0.5 + 1 / 3 + 0.0 + 1.0) / 5
        print(f"  Expected MAP@3: {expected_map:.4f}")
        print(f"  Actual MAP@3: {result.map_score:.4f}")

        # Verify calculation
        tolerance = 0.001
        if abs(result.map_score - expected_map) < tolerance:
            print("✅ MAP@3 calculation verified!")
        else:
            print("❌ MAP@3 calculation mismatch!")

        print("=================================\n")

        # Demonstrate the rich type system
        print("=== PREDICTION OBJECT DEMONSTRATION ===")

        # Show how ground truth is represented as Prediction objects
        gt_data = _load_ground_truth(gt_path)
        print("Ground truth as Prediction objects:")
        for row_id, prediction in gt_data.items():
            print(f"  Row {row_id}: {prediction} -> '{prediction.value}'")

        # Show how submissions are parsed into Prediction objects
        sub_data = _load_submissions(sub_path)
        print("\nSubmissions as Prediction objects:")
        for row_id, predictions in sub_data.items():
            pred_strs = [f"'{pred.value}'" for pred in predictions]
            print(f"  Row {row_id}: {pred_strs}")

        print("========================================\n")

        logger.info("MAP@3 evaluation demonstration completed successfully")
