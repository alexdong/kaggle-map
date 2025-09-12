"""Shared utilities for LLM evaluation."""

from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Template
from loguru import logger

from kaggle_map.core.models import Category, EvaluationResult, EvaluationRow, Prediction
from kaggle_map.utils.logger_config import configure_logger
from kaggle_map.utils.metrics import calculate_map_at_3

configure_logger(__name__)


def build_prediction_prompt(eval_row: EvaluationRow, template_path: Path) -> str:
    """Build a prediction prompt from evaluation row and template."""
    template = Template(template_path.read_text())
    return template.render(
        question_text=eval_row.question_text,
        mc_answer=eval_row.mc_answer,
        student_explanation=eval_row.student_explanation,
    )


def parse_predictions(response: str) -> list[Prediction]:
    """Parse LLM response to extract predictions.

    The LLM returns three predictions on ONE line separated by spaces.
    Format: "Category1:Misconception1 Category2:Misconception2 Category3:Misconception3"
    Example: "True_Correct:NA True_Neither:NA True_Misconception:Division"

    Args:
        response: Raw LLM response containing predictions

    Returns:
        List of up to 3 Prediction objects
    """
    predictions = []

    # The response should be a single line with three space-separated predictions
    response_clean = response.strip()

    # Handle case where LLM might return multiple lines - take the first non-empty line
    for line in response_clean.split("\n"):
        if line.strip() and ":" in line:
            response_clean = line.strip()
            break

    # Split by spaces to get individual predictions
    prediction_parts = response_clean.split()

    for part in prediction_parts:
        if ":" not in part:
            continue

        try:
            prediction = Prediction.from_string(part)
            predictions.append(prediction)

            max_predictions = 3
            if len(predictions) >= max_predictions:
                break
        except Exception as e:
            logger.debug(f"Failed to parse prediction '{part}': {e}")
            continue

    # Pad with default predictions if needed
    max_predictions = 3
    while len(predictions) < max_predictions:
        predictions.append(Prediction(category=Category.TRUE_CORRECT, misconception="NA"))

    return predictions[:max_predictions]


def evaluate_dataframe(
    df: pd.DataFrame,
    template: Template | str,
    llm: Any,  # noqa: ANN401
    stop_tokens: list[str] | None = None,
) -> tuple[list[EvaluationResult], float]:
    """Evaluate rows in a DataFrame using an LLM.

    Args:
        df: DataFrame with columns: row_id, QuestionId, QuestionText, MC_Answer,
            StudentExplanation, Category, Misconception
        template: Jinja2 template object or template string
        llm: Pre-loaded LLM instance for generating predictions
        stop_tokens: Optional list of stop tokens for the LLM

    Returns:
        Tuple of (list of EvaluationResult objects, average MAP@3 score)
    """
    assert not df.empty, "DataFrame cannot be empty"

    # Convert string template to Template if needed
    if isinstance(template, str):
        template = Template(template)

    evaluation_results = []
    scores = []

    logger.debug(f"Evaluating {len(df)} rows with LLM")

    for _, row in df.iterrows():
        # Create evaluation row
        eval_row = EvaluationRow(
            row_id=int(row["row_id"]),
            question_id=int(row["QuestionId"]),
            question_text=str(row["QuestionText"]),
            mc_answer=str(row["MC_Answer"]),
            student_explanation=str(row["StudentExplanation"]),
        )

        # Create ground truth prediction
        ground_truth = Prediction(
            category=(
                row["Category"]
                if isinstance(row["Category"], Category)
                else Category.from_csv_string(row["Category"])
            ),
            misconception=row["Misconception"] if pd.notna(row["Misconception"]) else "NA",
        )

        # Build prompt
        user_prompt = template.render(
            question_text=eval_row.question_text,
            mc_answer=eval_row.mc_answer,
            student_explanation=eval_row.student_explanation,
        )

        logger.debug(f"Prompt for row {eval_row.row_id}:\n{user_prompt}\n")

        # Generate predictions
        response = llm(
            user_prompt,
            max_tokens=256,
            temperature=0.1,
            top_p=0.95,
            stop=stop_tokens if stop_tokens else [],
            echo=False,
        )

        response_text = response["choices"][0]["text"]

        # Parse predictions
        predictions = parse_predictions(response_text)
        logger.debug(f"Predictions for row {eval_row.row_id}: {predictions}")

        # Calculate MAP@3
        score = calculate_map_at_3(ground_truth, predictions)
        scores.append(score)

        # Create evaluation result
        result = EvaluationResult(
            row_id=eval_row.row_id,
            question_id=eval_row.question_id,
            mc_answer=eval_row.mc_answer,
            explanation=eval_row.student_explanation,
            ground_truth=ground_truth,
            predictions=predictions,
            score=score,
        )

        evaluation_results.append(result)

    # Calculate average score
    avg_score = sum(scores) / len(scores) if scores else 0.0

    logger.info(f"Evaluated {len(evaluation_results)} rows, average MAP@3: {avg_score:.4f}")

    return evaluation_results, avg_score
