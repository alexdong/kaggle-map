"""Core reranking functionality using local GGUF models.

This module provides the core logic for reranking predictions using LLMs,
including prompt building, response parsing, and prediction reordering.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Template
from llama_cpp import Llama

from kaggle_map.core.models import EvaluationRow, Prediction
from kaggle_map.reranker.models import LLMResponse, PromptTemplate


@dataclass(frozen=True)
class RerankingRequest:
    """Complete request for reranking predictions."""

    evaluation_row: EvaluationRow
    candidate_predictions: list[Prediction]

    @property
    def top_prediction(self) -> Prediction | None:
        """Get the current top prediction."""
        return self.candidate_predictions[0] if self.candidate_predictions else None


EXPECTED_PREDICTIONS = 3


def build_reranking_prompt(request: RerankingRequest) -> PromptTemplate:
    """Build a concise prompt for reranking predictions."""
    n_predictions = len(request.candidate_predictions)
    assert n_predictions == EXPECTED_PREDICTIONS, (
        f"Expected exactly {EXPECTED_PREDICTIONS} predictions, got {n_predictions}"
    )

    # Load Jinja2 template
    template_path = Path(__file__).parent / "prompt.j2"
    template = Template(template_path.read_text())

    row = request.evaluation_row
    return template.render(
        mc_answer=row.mc_answer, student_explanation=row.student_explanation, predictions=request.candidate_predictions
    )


def parse_reranking_response(response: LLMResponse, original_predictions: list[Prediction]) -> list[Prediction]:
    """Parse LLM response to reorder predictions.

    Args:
        response: Raw LLM response containing comma-separated numbers
        original_predictions: Original list of predictions to reorder

    Returns:
        Reordered list of predictions based on LLM response

    Raises:
        AssertionError: If response format is invalid or indices are out of range
    """
    numbers = re.findall(r"\d+", response)
    assert numbers, "No numbers found in reranking response"

    indices = [int(n) - 1 for n in numbers]
    valid_indices = all(0 <= i < len(original_predictions) for i in indices)
    assert valid_indices, (
        f"Invalid indices in reranking response: {indices} for {len(original_predictions)} predictions"
    )

    # Ensure all indices are present (no missing predictions)
    unique = dict.fromkeys(indices)
    assert len(unique) == len(original_predictions), (
        f"Missing indices in reranking: expected {len(original_predictions)}, got {len(unique)}"
    )

    # Simple reordering since all indices are guaranteed to be present
    return [original_predictions[i] for i in unique]


def rerank_predictions(
    llm: Llama,
    request: RerankingRequest,
) -> list[Prediction]:
    """Rerank predictions using an LLM model.

    Args:
        llm: Loaded LLM model
        request: Reranking request with evaluation row and candidate predictions

    Returns:
        Reordered list of predictions based on LLM judgment
    """
    prompt = build_reranking_prompt(request)

    output = llm(
        prompt,
        max_tokens=50,  # Increased to ensure we get the full response
        temperature=0.01,  # Very low temperature for deterministic output
        stop=["\n", ".", ";"],  # Stop at newline, period, or semicolon
        echo=False,
    )
    response = output["choices"][0]["text"].strip()  # type: ignore

    return parse_reranking_response(response, request.candidate_predictions)
