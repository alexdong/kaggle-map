"""Simplified LLM reranker using direct llama-cpp-python calls.

This module provides reranking functionality using local GGUF models,
replacing the complex HTTP/async implementation with direct model calls.
"""

import re

import pandas as pd
from llama_cpp import Llama
from loguru import logger

from kaggle_map.core.models import (
    EvaluationRow,
    LLMResponse,
    Prediction,
    PromptTemplate,
)
from kaggle_map.reranker.models import RerankingRequest


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

Reply with ONLY the reordered numbers separated by commas (e.g., "3,1,2").
Most likely first."""


def parse_reranking_response(response: LLMResponse, original_predictions: list[Prediction]) -> list[Prediction]:
    """Parse LLM response to reorder predictions.

    Expected format: "3,1,2" indicating new order.
    Falls back to original order if parsing fails.
    """
    try:
        # Extract numbers from response
        numbers = re.findall(r"\d+", response)

        if not numbers:
            logger.warning(f"No numbers found in reranking response: {response}")
            return original_predictions

        # Convert to 0-based indices
        indices = [int(n) - 1 for n in numbers]

        # Validate indices
        valid_indices = all(0 <= i < len(original_predictions) for i in indices)
        if not valid_indices:
            logger.warning(f"Invalid indices in response: {indices}")
            return original_predictions

        # Reorder predictions
        reordered = []
        seen = set()

        for idx in indices:
            if idx not in seen:
                reordered.append(original_predictions[idx])
                seen.add(idx)

        # Add any missing predictions at the end
        for i, pred in enumerate(original_predictions):
            if i not in seen:
                reordered.append(pred)

        return reordered

    except Exception as e:
        logger.error(f"Failed to parse reranking response: {e}")
        return original_predictions


def rerank_predictions(
    llm: Llama,
    request: RerankingRequest,
) -> list[Prediction]:
    """Rerank predictions using direct LLM inference.

    Args:
        llm: Loaded llama-cpp model
        request: Complete reranking request with context

    Returns:
        Reordered list of predictions
    """
    logger.debug(f"Reranking {len(request.candidate_predictions)} predictions")

    prompt = build_reranking_prompt(request)

    # Direct LLM call
    output = llm(
        prompt,
        max_tokens=20,  # Just need numbers like "3,1,2"
        temperature=0.1,  # Low temperature for consistency
        stop=["\n"],
        echo=False,
    )

    # Extract response text
    response = output["choices"][0]["text"].strip() if isinstance(output, dict) else str(output).strip()

    logger.debug(f"Reranking response: {response}")

    # Parse and return reordered predictions
    return parse_reranking_response(response, request.candidate_predictions)


def process_dataframe_simple(
    llm: Llama,
    df: pd.DataFrame,
    sample_size: int = 100,
) -> pd.DataFrame:
    """Process DataFrame with LLM reranking using direct calls.

    Simplified version without async/HTTP complexity.
    """
    assert not df.empty, "DataFrame cannot be empty"

    # Required columns
    required = ["QuestionText", "MC_Answer", "StudentExplanation", "top_3_predictions_formatted", "Category"]
    missing = [col for col in required if col not in df.columns]
    assert not missing, f"Missing required columns: {missing}"

    # Sample data
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42).copy()

    # Add result columns
    df_sample["LLM_top_1"] = ""
    df_sample["LLM_top_3"] = ""
    df_sample["LLM_correct"] = False

    logger.info(f"Processing {len(df_sample)} rows")

    for idx, row in df_sample.iterrows():
        try:
            # Parse predictions
            predictions_str = str(row["top_3_predictions_formatted"])
            prediction_labels = [p.strip() for p in predictions_str.split("|")]
            predictions = [Prediction.from_string(label) for label in prediction_labels]

            # Build request
            eval_row = EvaluationRow(
                row_id=idx,
                question_id=row.get("QuestionId", 0),
                question_text=str(row["QuestionText"]),
                mc_answer=str(row["MC_Answer"]),
                student_explanation=str(row["StudentExplanation"]),
            )

            request = RerankingRequest(
                evaluation_row=eval_row,
                candidate_predictions=predictions,
            )

            # Rerank
            reranked = rerank_predictions(llm, request)

            # Update results
            df_sample.loc[idx, "LLM_top_1"] = str(reranked[0]) if reranked else ""
            df_sample.loc[idx, "LLM_top_3"] = "|".join(str(p) for p in reranked[:3])

            # Check accuracy if ground truth available
            if "actual_misconception" in row:
                actual = f"{row['Category']}:{row.get('actual_misconception', 'NA')}"
                predicted = str(reranked[0]) if reranked else ""
                df_sample.loc[idx, "LLM_correct"] = compare_labels(actual, predicted)

        except Exception as e:
            logger.error(f"Failed to process row {idx}: {e}")
            continue

    # Calculate stats
    if "LLM_correct" in df_sample.columns:
        accuracy = df_sample["LLM_correct"].mean()
        logger.info(f"Reranking accuracy: {accuracy:.2%}")

    return df_sample
