"""Data loading functions for training and error prediction datasets."""

from pathlib import Path

import pandas as pd
from loguru import logger

from kaggle_map.core.models import QuestionId, TrainingRow


def load_training(question_id: QuestionId) -> list[TrainingRow]:
    train_csv_path = Path("datasets/train.csv")

    assert train_csv_path.exists(), f"Training data not found at {train_csv_path}"

    logger.debug(f"Loading training data for question {question_id} from {train_csv_path}")

    df = pd.read_csv(train_csv_path)

    # Filter for the specific question
    question_df = df[df["QuestionId"] == question_id]

    logger.debug(f"Found {len(question_df)} rows for question {question_id}")

    # Convert each row to TrainingRow
    training_rows = []
    for _, row in question_df.iterrows():
        training_row = TrainingRow.from_dataframe_row(row)
        training_rows.append(training_row)

    logger.debug(f"Successfully loaded {len(training_rows)} training rows for question {question_id}")

    return training_rows


def load_error(question_id: QuestionId) -> list[TrainingRow]:
    error_csv_path = Path("datasets/error_prediction.csv")

    assert error_csv_path.exists(), f"Error prediction data not found at {error_csv_path}"

    logger.debug(f"Loading error prediction data for question {question_id} from {error_csv_path}")

    df = pd.read_csv(error_csv_path)

    # Filter for the specific question
    question_df = df[df["QuestionId"] == question_id]

    logger.debug(f"Found {len(question_df)} rows for question {question_id}")

    # The error_prediction.csv has different columns, need to map them
    # Map Category column (uppercase) to the format expected by TrainingRow
    training_rows = []
    for _, row in question_df.iterrows():
        # Create a new row with the expected column names
        mapped_row = pd.Series(
            {
                "row_id": row["row_id"],
                "QuestionId": row["QuestionId"],
                "QuestionText": row["QuestionText"],
                "MC_Answer": row["MC_Answer"],
                "StudentExplanation": row["StudentExplanation"],
                "Category": row["Category"],  # This is already in uppercase format
                "Misconception": row.get("actual_misconception", "NA"),
            }
        )

        training_row = TrainingRow.from_dataframe_row(mapped_row)
        training_rows.append(training_row)

    logger.debug(f"Successfully loaded {len(training_rows)} error prediction rows for question {question_id}")

    return training_rows
