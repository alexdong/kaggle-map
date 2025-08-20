"""Dataset analysis and processing utilities for student misconception prediction.

This module contains shared functions for understanding the nature of the dataset,
including parsing training data, extracting patterns, and analyzing misconceptions.
These utilities are used by multiple strategy implementations.
"""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

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

    try:
        training_df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as e:
        msg = "Training CSV cannot be empty"
        raise AssertionError(msg) from e

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


def extract_most_common_misconceptions(
    training_data: list[TrainingRow],
) -> dict[QuestionId, Misconception | None]:
    assert training_data, "Training data cannot be empty"

    # Get all unique question IDs
    all_questions = {row.question_id for row in training_data}

    question_misconceptions = defaultdict(list)

    for row in training_data:
        if row.misconception != "NA":
            question_misconceptions[row.question_id].append(row.misconception)

    result = {}
    for question_id in all_questions:
        misconceptions = question_misconceptions.get(question_id, [])
        if misconceptions:
            most_common = Counter(misconceptions).most_common(1)[0][0]
            result[question_id] = most_common
        else:
            result[question_id] = "NA"

    logger.debug(f"Extracted most common misconceptions for {len(result)} questions")
    assert isinstance(result, dict), "Result must be a dictionary"
    return result


def get_training_data_with_correct_answers(
    training_data: list[TrainingRow], correct_answers: dict[QuestionId, Answer]
) -> list[tuple[TrainingRow, Answer]]:
    filtered_data = []

    for row in training_data:
        # Skip if we don't know the correct answer
        if row.question_id not in correct_answers:
            continue

        filtered_data.append((row, correct_answers[row.question_id]))

    logger.debug(f"Filtered to {len(filtered_data)} training rows with correct answers")
    return filtered_data


def analyze_dataset(csv_path: Path) -> dict[str, Any]:
    """Perform comprehensive dataset analysis.

    Args:
        csv_path: Path to the training CSV file

    Returns:
        Dictionary containing dataset analysis results including:
        - Basic statistics (number of rows, questions, etc.)
        - Category distribution
        - Misconception patterns
        - Question complexity metrics
    """
    training_data = parse_training_data(csv_path)
    correct_answers = extract_correct_answers(training_data)

    # Basic statistics
    unique_questions = len({row.question_id for row in training_data})
    unique_misconceptions = len({row.misconception for row in training_data if row.misconception != "NA"})

    # Category distribution
    category_counts = Counter(row.category for row in training_data)

    # Misconception analysis
    misconception_counts = Counter(row.misconception for row in training_data if row.misconception != "NA")

    # Question complexity (number of unique answers per question)
    question_answer_counts = defaultdict(set)
    for row in training_data:
        question_answer_counts[row.question_id].add(row.mc_answer)

    return {
        "total_rows": len(training_data),
        "unique_questions": unique_questions,
        "questions_with_correct_answers": len(correct_answers),
        "unique_misconceptions": unique_misconceptions,
        "category_distribution": dict(category_counts),
        "top_misconceptions": dict(misconception_counts.most_common(10)),
        "avg_answers_per_question": sum(len(answers) for answers in question_answer_counts.values()) / unique_questions,
        "max_answers_per_question": max(len(answers) for answers in question_answer_counts.values()),
    }
