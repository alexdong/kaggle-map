"""Shared utilities for LLM evaluation."""

from pathlib import Path

from jinja2 import Template

from kaggle_map.core.models import EvaluationRow


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
