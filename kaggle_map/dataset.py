"""Dataset processing utilities for kaggle-map strategies.

This module contains common functions for parsing and processing training data
that are shared across different prediction strategies.
"""

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from loguru import logger

from kaggle_map.models import (
    Answer,
    Category,
    Misconception,
    QuestionId,
    TrainingRow,
)


def parse_training_data(csv_path: Path) -> list[TrainingRow]:
    """Parse CSV into strongly-typed training rows.
    
    Args:
        csv_path: Path to the training CSV file
        
    Returns:
        List of parsed training rows
        
    Raises:
        AssertionError: If CSV file doesn't exist, is empty, or contains no rows
    """
    assert csv_path.exists(), f"Training file not found: {csv_path}"

    training_df = pd.read_csv(csv_path)
    logger.debug(f"Loaded CSV with columns: {list(training_df.columns)}")
    assert not training_df.empty, "Training CSV cannot be empty"

    training_rows = []
    for _, row in training_df.iterrows():
        # Handle NaN misconceptions (pandas converts "NA" to NaN)
        misconception = (
            row["Misconception"] if pd.notna(row["Misconception"]) else None
        )

        training_rows.append(
            TrainingRow(
                row_id=int(row["row_id"]),
                question_id=int(row["QuestionId"]),
                question_text=str(row["QuestionText"]),
                mc_answer=str(row["MC_Answer"]),
                student_explanation=str(row["StudentExplanation"]),
                category=Category(row["Category"]),
                misconception=misconception,
            )
        )

    logger.debug(f"Parsed {len(training_rows)} training rows")
    assert training_rows, "Must parse at least one training row"
    return training_rows


def extract_correct_answers(
    training_data: list[TrainingRow],
) -> dict[QuestionId, Answer]:
    """Extract the correct answer for each question.
    
    Args:
        training_data: List of training rows
        
    Returns:
        Dictionary mapping question IDs to their correct answers
        
    Raises:
        AssertionError: If training data is empty, no correct answers found,
                       or conflicting correct answers for the same question
    """
    assert training_data, "Training data cannot be empty"

    correct_answers = {}

    for row in training_data:
        if row.category == Category.TRUE_CORRECT:
            if row.question_id in correct_answers:
                assert correct_answers[row.question_id] == row.mc_answer, (
                    f"Conflicting correct answers for question {row.question_id}"
                )
            else:
                correct_answers[row.question_id] = row.mc_answer

    logger.debug(f"Extracted correct answers for {len(correct_answers)} questions")
    assert correct_answers, "Must find at least one correct answer"
    assert all(isinstance(qid, int) for qid in correct_answers), (
        "Question IDs must be integers"
    )
    return correct_answers


def build_category_frequencies(
    training_data: list[TrainingRow], correct_answers: dict[QuestionId, Answer]
) -> dict[QuestionId, dict[bool, list[Category]]]:
    """Build frequency-ordered category lists for correct/incorrect answers.
    
    Args:
        training_data: List of training rows
        correct_answers: Dictionary of correct answers by question ID
        
    Returns:
        Dictionary mapping question IDs to correctness-based category frequencies.
        Structure: {question_id: {is_correct: [categories_ordered_by_frequency]}}
        
    Raises:
        AssertionError: If training data or correct answers are empty
    """
    assert training_data, "Training data cannot be empty"
    assert correct_answers, "Correct answers cannot be empty"

    # Group by question and correctness
    question_correctness_categories = defaultdict(lambda: defaultdict(list))

    for row in training_data:
        is_correct = (
            row.question_id in correct_answers
            and row.mc_answer == correct_answers[row.question_id]
        )
        question_correctness_categories[row.question_id][is_correct].append(
            row.category
        )

    # Build frequency-ordered lists
    result = {}
    for question_id, correctness_map in question_correctness_categories.items():
        result[question_id] = {}
        for is_correct, categories in correctness_map.items():
            # Count frequencies and sort by most common
            category_counts = Counter(categories)
            ordered_categories = [
                category for category, _ in category_counts.most_common()
            ]
            result[question_id][is_correct] = ordered_categories

    logger.debug(f"Built category frequencies for {len(result)} questions")
    assert isinstance(result, dict), "Result must be a dictionary"
    return result


def extract_most_common_misconceptions(
    training_data: list[TrainingRow],
) -> dict[QuestionId, Misconception | None]:
    """Find most common misconception per question.
    
    Args:
        training_data: List of training rows
        
    Returns:
        Dictionary mapping question IDs to their most common misconception.
        Value is None if no misconceptions found for that question.
        
    Raises:
        AssertionError: If training data is empty
    """
    assert training_data, "Training data cannot be empty"

    question_misconceptions = defaultdict(list)

    for row in training_data:
        if row.misconception is not None:
            question_misconceptions[row.question_id].append(row.misconception)

    result = {}
    for question_id, misconceptions in question_misconceptions.items():
        if misconceptions:
            most_common = Counter(misconceptions).most_common(1)[0][0]
            result[question_id] = most_common
        else:
            result[question_id] = None

    logger.debug(
        f"Extracted most common misconceptions for {len(result)} questions"
    )
    assert isinstance(result, dict), "Result must be a dictionary"
    return result
