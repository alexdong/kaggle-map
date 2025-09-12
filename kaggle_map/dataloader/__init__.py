"""Data loading functions for training and error prediction datasets."""

from pathlib import Path

import pandas as pd
from loguru import logger

from kaggle_map.core.models import EvaluationRow, Prediction, QuestionId, TrainingRow
from kaggle_map.dataloader.sampling import stratification_report as stratification_report
from kaggle_map.dataloader.sampling import stratified_sample as stratified_sample
from kaggle_map.utils.logger_config import configure_logger

configure_logger(__name__)


def load_training_data(csv_path: Path, question_id: QuestionId | None = None) -> list[TrainingRow]:
    """Load training rows from a CSV file, optionally filtered by question ID.

    Args:
        csv_path: Path to the CSV file containing training data
        question_id: Optional question ID to filter by. If None, loads all data.

    Returns:
        List of TrainingRow objects
    """
    assert csv_path.exists(), f"Training data not found at {csv_path}"

    if question_id is not None:
        logger.debug(f"Loading training data for question {question_id} from {csv_path}")
    else:
        logger.debug(f"Loading all training data from {csv_path}")

    df = pd.read_csv(csv_path)

    # Filter for the specific question if provided
    if question_id is not None:
        question_df = df[df["QuestionId"] == question_id]
        logger.debug(f"Found {len(question_df)} rows for question {question_id}")
    else:
        question_df = df
        logger.debug(f"Found {len(question_df)} total rows")

    # Convert each row to TrainingRow
    training_rows = []
    for _, row in question_df.iterrows():
        training_row = TrainingRow.from_dataframe_row(row)
        training_rows.append(training_row)

    if question_id is not None:
        logger.debug(f"Successfully loaded {len(training_rows)} training rows for question {question_id}")
    else:
        logger.debug(f"Successfully loaded {len(training_rows)} total training rows")

    return training_rows


def load_validation_data(
    csv_path: Path, question_id: QuestionId | None = None
) -> list[tuple[EvaluationRow, Prediction]]:
    """Load validation data rows from a CSV file, optionally filtered by question ID.

    Args:
        csv_path: Path to the CSV file containing validation/error prediction data
        question_id: Optional question ID to filter by. If None, loads all data.

    Returns:
        List of tuples containing (EvaluationRow, Prediction) pairs
    """
    assert csv_path.exists(), f"Validation data not found at {csv_path}"

    if question_id is not None:
        logger.debug(f"Loading validation data for question {question_id} from {csv_path}")
    else:
        logger.debug(f"Loading all validation data from {csv_path}")

    df = pd.read_csv(csv_path)

    # Filter for the specific question if provided
    if question_id is not None:
        question_df = df[df["QuestionId"] == question_id]
        logger.debug(f"Found {len(question_df)} rows for question {question_id}")
    else:
        question_df = df
        logger.debug(f"Found {len(question_df)} total rows")

    # Convert each row to EvaluationRow and Prediction tuple
    result_pairs = []
    for _, row in question_df.iterrows():
        # Create EvaluationRow
        eval_row = EvaluationRow(
            row_id=int(row["row_id"]),
            question_id=int(row["QuestionId"]),
            question_text=str(row["QuestionText"]),
            mc_answer=str(row["MC_Answer"]),
            student_explanation=str(row["StudentExplanation"]),
        )

        # Create Prediction from the actual ground truth
        # Handle both column names: "Misconception" (train.csv) and "actual_misconception" (validation.csv)
        misconception_value = row.get("Misconception", row.get("actual_misconception", "NA"))

        # Map Category column (uppercase) to the format expected by Prediction
        mapped_row = pd.Series(
            {
                "Category": row["Category"],  # This is already in uppercase format
                "Misconception": misconception_value,
            }
        )
        prediction = Prediction.from_ground_truth_row(mapped_row)

        result_pairs.append((eval_row, prediction))

    if question_id is not None:
        logger.debug(f"Successfully loaded {len(result_pairs)} validation rows for question {question_id}")
    else:
        logger.debug(f"Successfully loaded {len(result_pairs)} total validation rows")

    return result_pairs


def load_rows_by_ids(data_path: Path, row_ids: list[int]) -> pd.DataFrame:
    """Load specific rows from dataset by row_id values.

    Args:
        data_path: Path to the CSV file containing the data
        row_ids: List of row_id values to load

    Returns:
        DataFrame containing only the requested rows
    """
    assert data_path.exists(), f"Data file not found at {data_path}"
    assert row_ids, "Row IDs list cannot be empty"

    logger.debug(f"Loading {len(row_ids)} specific rows from {data_path}")

    # Load full dataset
    df = pd.read_csv(data_path)

    # Filter by row_id
    filtered_df = df[df["row_id"].isin(row_ids)]

    logger.debug(f"Found {len(filtered_df)} rows out of {len(row_ids)} requested")

    assert not filtered_df.empty, f"No rows found for IDs: {row_ids}"

    return pd.DataFrame(filtered_df)
