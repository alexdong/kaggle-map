"""Simplified LLM reranker using direct llama-cpp-python calls.

This module provides reranking functionality using local GGUF models,
replacing the complex HTTP/async implementation with direct model calls.
"""

import re
from dataclasses import dataclass

from llama_cpp import Llama
from loguru import logger

from kaggle_map.core.models import (
    EvaluationRow,
    LLMResponse,
    Prediction,
    PromptTemplate,
)


@dataclass(frozen=True)
class RerankingRequest:
    """Complete request for reranking predictions."""

    evaluation_row: EvaluationRow
    candidate_predictions: list[Prediction]

    @property
    def top_prediction(self) -> Prediction | None:
        """Get the current top prediction."""
        return self.candidate_predictions[0] if self.candidate_predictions else None


def build_reranking_prompt(request: RerankingRequest) -> PromptTemplate:
    """Build a concise prompt for reranking predictions."""
    # Format predictions as numbered list
    predictions_text = "\n".join(f"{i + 1}. {pred!s}" for i, pred in enumerate(request.candidate_predictions))

    row = request.evaluation_row
    return f"""Analyze this student's math work and reorder the predictions by likelihood.

Question: {row.question_text}
Correct Answer: {row.correct_answer or "Not provided"}
Student Answer: {row.mc_answer}
Student Explanation: {row.student_explanation}

Predictions to reorder:
{predictions_text}

Reply with ONLY the reordered numbers separated by commas. Like "3,1,2".
Most likely first."""


def parse_reranking_response(response: LLMResponse, original_predictions: list[Prediction]) -> list[Prediction]:
    numbers = re.findall(r"\d+", response)
    assert numbers, "No numbers found in reranking response"

    indices = [int(n) - 1 for n in numbers]
    valid_indices = all(0 <= i < len(original_predictions) for i in indices)
    assert valid_indices, "Invalid indices in reranking response"

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
    logger.debug(f"Reranking {len(request.candidate_predictions)} predictions")
    prompt = build_reranking_prompt(request)
    logger.debug(f"Reranking prompt: {prompt}")
    output = llm(
        prompt,
        max_tokens=20,  # Just need numbers like "3,1,2"
        temperature=0.1,  # Low temperature for consistency
        stop=["\n"],
        echo=False,
    )

    response = output["choices"][0]["text"].strip()  # type: ignore
    logger.debug(f"Reranking response: {response}")
    return parse_reranking_response(response, request.candidate_predictions)
