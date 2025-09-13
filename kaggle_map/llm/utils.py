"""Shared utilities for LLM evaluation."""

from pathlib import Path

from jinja2 import Template
from loguru import logger

from kaggle_map.core.models import Category, EvaluationRow, Prediction


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
