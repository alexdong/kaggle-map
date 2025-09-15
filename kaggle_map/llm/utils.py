"""Shared utilities for LLM evaluation."""

from pathlib import Path

from jinja2 import Template

from kaggle_map.core.models import EvaluationRow, Prediction
from kaggle_map.llm.robust_parser import parse_predictions_with_fuzzy_matching


def build_prediction_prompt(eval_row: EvaluationRow, template_path: Path) -> str:
    """Build a prediction prompt from evaluation row and template."""
    assert isinstance(eval_row, EvaluationRow), f"eval_row must be EvaluationRow, got {type(eval_row)}"
    assert isinstance(template_path, Path), f"template_path must be Path, got {type(template_path)}"
    assert template_path.exists(), f"Template file does not exist: {template_path}"
    assert template_path.suffix in (".j2", ".jinja", ".jinja2", ".txt"), (
        f"Unexpected template file extension: {template_path.suffix}"
    )

    # Validate required fields are present
    assert eval_row.question_text.strip(), "question_text cannot be empty"
    assert eval_row.mc_answer.strip(), "mc_answer cannot be empty"
    assert eval_row.student_explanation.strip(), "student_explanation cannot be empty"

    template = Template(template_path.read_text())
    rendered = template.render(
        question_text=eval_row.question_text,
        mc_answer=eval_row.mc_answer,
        student_explanation=eval_row.student_explanation,
    )

    assert rendered.strip(), "Template rendered to empty content"
    return rendered


def parse_predictions(response: str) -> list[Prediction]:
    """Parse LLM response to extract predictions.

    The LLM returns three predictions on ONE line separated by spaces.
    Format: "Category1:Misconception1 Category2:Misconception2 Category3:Misconception3"
    Example: "True_Correct:NA True_Neither:NA True_Misconception:Division"

    Args:
        response: Raw LLM response containing predictions

    Returns:
        List of exactly 3 Prediction objects
    """
    assert isinstance(response, str), f"response must be string, got {type(response)}"
    assert response.strip(), "response cannot be empty or whitespace-only"

    # Use the robust parser that handles typos and format issues
    return parse_predictions_with_fuzzy_matching(response)
