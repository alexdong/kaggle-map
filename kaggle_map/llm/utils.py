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


def _extract_prediction_line(response: str) -> str:
    """Extract the first valid prediction line from response."""
    for line in response.strip().split("\n"):
        if line.strip() and ":" in line:
            return line.strip()
    return response.strip()


def _parse_single_prediction(part: str) -> Prediction | None:
    """Parse a single prediction string."""
    if ":" not in part:
        return None
    try:
        return Prediction.from_string(part)
    except Exception as e:
        logger.debug(f"Failed to parse prediction '{part}': {e}")
        return None


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
    max_predictions = 3
    default_prediction = Prediction(category=Category.TRUE_CORRECT, misconception="NA")

    # Extract and parse predictions
    prediction_line = _extract_prediction_line(response)
    predictions = [pred for part in prediction_line.split() if (pred := _parse_single_prediction(part)) is not None][
        :max_predictions
    ]

    # Pad with defaults if needed
    while len(predictions) < max_predictions:
        predictions.append(default_prediction)

    return predictions


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
                else Category.from_csv_string(str(row["Category"]))
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
        logger.debug(f"LLM response for row {eval_row.row_id}:\n{response_text}\n")

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
