"""
Confusion matrix analysis for error predictions.

Analyzes predictions from error_prediction.csv and generates misconception confusion matrices
for each QuestionId to understand model performance patterns.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def load_predictions(csv_path: Path) -> pd.DataFrame:
    """Load error predictions from CSV file."""
    logger.debug(f"Loading predictions from {csv_path}")
    predictions_df = pd.read_csv(csv_path)
    logger.info(
        f"Loaded {len(predictions_df)} predictions covering {predictions_df['QuestionId'].nunique()} questions"
    )
    return predictions_df


def extract_prediction_components(prediction_str: str) -> tuple[str, str]:
    """Extract category and misconception from prediction string."""
    if pd.isna(prediction_str) or prediction_str == "NA":
        return "NA", "NA"

    if ":" in prediction_str:
        parts = prediction_str.split(":", 1)
        category = parts[0]
        misconception = parts[1] if len(parts) > 1 else "NA"
        return category, misconception

    return prediction_str, "NA"


def build_misconception_matrix(
    actual_misconceptions: list[str],
    predicted_misconceptions: list[str]
) -> tuple[np.ndarray | None, list[str]]:
    """Build confusion matrix for misconceptions."""
    # Filter out NA pairs
    pairs = [
        (a, p) for a, p in zip(actual_misconceptions, predicted_misconceptions, strict=False)
        if a != "NA" or p != "NA"  # Include if at least one is not NA
    ]

    if not pairs:
        return None, []

    # Get unique labels
    all_misconceptions = set()
    for actual, pred in pairs:
        all_misconceptions.add(actual)
        all_misconceptions.add(pred)

    labels = sorted(all_misconceptions)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    # Build matrix
    n_labels = len(labels)
    matrix = np.zeros((n_labels, n_labels), dtype=int)

    for actual, predicted in pairs:
        actual_idx = label_to_idx[actual]
        pred_idx = label_to_idx[predicted]
        matrix[actual_idx, pred_idx] += 1

    return matrix, labels


def format_confusion_matrix(matrix: np.ndarray | None, labels: list[str]) -> str:
    """Format confusion matrix as readable string."""
    if matrix is None or len(labels) == 0:
        return "\nNo misconception predictions available"

    output = []

    # Prepare label display (truncate if too long)
    max_label_len = 20
    display_labels = [
        label[:max_label_len] if len(label) <= max_label_len else label[:17] + "..."
        for label in labels
    ]

    # Find max width needed for numbers
    max_val = np.max(matrix)
    num_width = max(5, len(str(max_val)) + 2)

    # Header row
    header = "Actual \\ Predicted".ljust(25) + " | " + " | ".join(f"{label:^{num_width}}" for label in display_labels)
    output.append(header)
    output.append("-" * len(header))

    # Data rows
    for i, label in enumerate(display_labels):
        row_values = " | ".join(f"{matrix[i, j]:^{num_width}d}" for j in range(len(labels)))
        output.append(f"{label:25} | {row_values}")

    # Add accuracy info
    total = np.sum(matrix)
    correct = np.trace(matrix)
    accuracy = correct / total if total > 0 else 0
    output.append("-" * len(header))
    output.append(f"Accuracy: {accuracy:.2%} ({correct}/{total} correct)")

    return "\n".join(output)


def analyze_question(df_question: pd.DataFrame) -> dict:
    """Analyze predictions for a single question."""
    # Get question info
    question_id = df_question.iloc[0]["QuestionId"]
    question_text = df_question.iloc[0]["QuestionText"] if not df_question.empty else "Question text not available"

    logger.debug(f"Analyzing QuestionId {question_id} with {len(df_question)} samples")

    # Extract misconceptions
    actual_misconceptions = []
    predicted_misconceptions = []

    for _, row in df_question.iterrows():
        # Get actual misconception
        actual_misc = str(row["actual_misconception"])
        actual_misconceptions.append(actual_misc)

        # Extract predicted misconception
        pred_str = str(row["full_prediction"])
        _, pred_misc = extract_prediction_components(pred_str)
        predicted_misconceptions.append(pred_misc)

    # Build confusion matrix
    matrix, labels = build_misconception_matrix(actual_misconceptions, predicted_misconceptions)

    return {
        "question_id": question_id,
        "question_text": question_text,
        "n_samples": len(df_question),
        "matrix": matrix,
        "labels": labels
    }


def main() -> None:
    """Main analysis function."""
    # Load predictions
    csv_path = Path("datasets/error_prediction.csv")
    if not csv_path.exists():
        logger.error(f"Error predictions file not found: {csv_path}")
        return

    predictions_df = load_predictions(csv_path)

    # Analyze each question
    logger.info("\n" + "=" * 80)
    logger.info("MISCONCEPTION CONFUSION MATRICES BY QUESTION")
    logger.info("=" * 80)

    for question_id in sorted(predictions_df["QuestionId"].unique()):
        df_question = predictions_df[predictions_df["QuestionId"] == question_id]
        result = analyze_question(df_question)

        # Display results
        logger.info(f"\n{'='*80}")
        logger.info(f"QuestionId: {result['question_id']}")
        logger.info(f"Samples: {result['n_samples']}")

        # Show question text (truncated if too long)
        question = result["question_text"]
        max_question_len = 300
        if len(question) > max_question_len:
            question = question[: max_question_len - 3] + "..."
        logger.info(f"Question: {question}")

        # Display confusion matrix
        logger.info("\nMisconception Confusion Matrix:")
        logger.info(format_confusion_matrix(result["matrix"], result["labels"]))


if __name__ == "__main__":
    main()
