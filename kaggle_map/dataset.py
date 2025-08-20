"""Dataset analysis and processing utilities for student misconception prediction.

This module contains shared functions for understanding the nature of the dataset,
including parsing training data, extracting patterns, and analyzing misconceptions.
These utilities are used by multiple strategy implementations.
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
    assert csv_path.exists(), f"Training file not found: {csv_path}"

    training_df = pd.read_csv(csv_path)
    logger.debug(f"Loaded CSV with columns: {list(training_df.columns)}")
    assert not training_df.empty, "Training CSV cannot be empty"

    training_rows = []
    for _, row in training_df.iterrows():
        training_rows.append(TrainingRow.from_dataframe_row(row))

    logger.debug(f"Parsed {len(training_rows)} training rows")
    assert training_rows, "Must parse at least one training row"
    return training_rows


def extract_correct_answers(
    training_data: list[TrainingRow],
) -> dict[QuestionId, Answer]:
    assert training_data, "Training data cannot be empty"
    correct_answers: dict[QuestionId, Answer] = {}

    for row in training_data:
        if row.category == Category.TRUE_CORRECT and row.question_id not in correct_answers:
            correct_answers[row.question_id] = row.mc_answer

    logger.debug(f"Extracted correct answers for {len(correct_answers)} questions")
    assert correct_answers, "Must find at least one correct answer"
    assert all(isinstance(qid, int) for qid in correct_answers), "Question IDs must be integers"
    return correct_answers


def is_answer_correct(
    question_id: QuestionId,
    student_answer: Answer,
    correct_answers: dict[QuestionId, Answer],
) -> bool:
    correct_answer = correct_answers.get(question_id, "")
    return student_answer == correct_answer


def build_category_frequencies(
    training_data: list[TrainingRow], correct_answers: dict[QuestionId, Answer]
) -> dict[QuestionId, dict[bool, list[Category]]]:
    assert training_data, "Training data cannot be empty"
    assert correct_answers, "Correct answers cannot be empty"

    # Group by question and correctness
    question_correctness_categories = defaultdict(lambda: defaultdict(list))

    for row in training_data:
        is_correct = row.question_id in correct_answers and row.mc_answer == correct_answers[row.question_id]
        question_correctness_categories[row.question_id][is_correct].append(row.category)

    # Build frequency-ordered lists
    result = {}
    for question_id, correctness_map in question_correctness_categories.items():
        result[question_id] = {}
        for is_correct, categories in correctness_map.items():
            # Count frequencies and sort by most common
            category_counts = Counter(categories)
            ordered_categories = [category for category, _ in category_counts.most_common()]
            result[question_id][is_correct] = ordered_categories

    logger.debug(f"Built category frequencies for {len(result)} questions")
    assert isinstance(result, dict), "Result must be a dictionary"
    return result


def extract_misconceptions_by_popularity(
    training_data: list[TrainingRow],
) -> dict[QuestionId, list[Misconception]]:
    """Extract misconceptions per question ordered by popularity (most common first)."""
    assert training_data, "Training data cannot be empty"

    # Get all unique question IDs
    all_questions = {row.question_id for row in training_data}

    question_misconceptions = defaultdict(list)
    for row in training_data:
        if row.misconception not in ("NA", None):
            question_misconceptions[row.question_id].append(row.misconception)

    result = {
        question_id: [m for m, _ in Counter(question_misconceptions.get(question_id, [])).most_common()]
        for question_id in all_questions
    }

    logger.debug(f"Extracted misconceptions by popularity for {len(result)} questions")
    return result


def extract_most_common_misconceptions(
    training_data: list[TrainingRow],
) -> dict[QuestionId, Misconception]:
    """Extract the most common misconception per question using popularity-ordered results."""
    misconceptions_by_popularity = extract_misconceptions_by_popularity(training_data)

    result = {}
    for question_id, misconceptions in misconceptions_by_popularity.items():
        if misconceptions:
            result[question_id] = misconceptions[0]  # Most popular is first
        else:
            result[question_id] = "NA"

    logger.debug(f"Extracted most common misconceptions for {len(result)} questions")
    assert isinstance(result, dict), "Result must be a dictionary"
    return result
