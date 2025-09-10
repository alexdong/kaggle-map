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


if __name__ == "__main__":
    import pandas as pd
    from loguru import logger
    from pathlib import Path

    from kaggle_map.core.dataset import extract_correct_answers
    from kaggle_map.core.models import Category, EvaluationRow, Prediction
    from kaggle_map.reranker.models import RerankerLLMLoadConfig, RerankerModelName, RerankerModelQuantizationLevel
    from kaggle_map.reranker.utils import load_llm_model

    logger.info("Testing rerank_predictions function")

    # Use default model configuration from benchmark.py
    load_config = RerankerLLMLoadConfig(
        model_name=RerankerModelName.GEMMA_3_12B_IT,
        quantization=RerankerModelQuantizationLevel.Q4_K_XL,
    )

    logger.info(f"Loading model: {load_config.model_name} with {load_config.quantization} quantization")
    llm = load_llm_model(load_config)
    logger.success("Model loaded successfully")

    # Load real data from train.csv
    logger.info("Loading real data from datasets/train.csv")
    train_df = pd.read_csv("datasets/train.csv")
    from kaggle_map.core.dataset import load_training_data
    training_rows = load_training_data(Path("datasets/train.csv"))
    correct_answers = extract_correct_answers(training_rows)
    
    # Get a sample row that has a misconception
    sample_row = train_df[train_df["Category"] == "True_Misconception"].iloc[0]
    
    # Create evaluation row from real data
    eval_row = EvaluationRow(
        row_id=int(sample_row["row_id"]),
        question_id=int(sample_row["QuestionId"]),
        question_text=str(sample_row["QuestionText"]),
        mc_answer=str(sample_row["MC_Answer"]),
        student_explanation=str(sample_row["StudentExplanation"]),
        correct_answer=correct_answers.get(int(sample_row["QuestionId"]), ""),
    )
    
    logger.info(f"Using question {eval_row.question_id}: {eval_row.question_text[:100]}...")
    logger.info(f"Student answer: {eval_row.mc_answer}")
    logger.info(f"Student explanation: {eval_row.student_explanation[:100]}...")

    # Create sample candidate predictions using real Category enum values
    # Mix of misconception and non-misconception categories
    candidate_predictions = [
        Prediction(
            category=Category.TRUE_MISCONCEPTION,
            misconception=str(sample_row["Misconception"]) if pd.notna(sample_row["Misconception"]) else "Student misunderstands the concept",
        ),
        Prediction(
            category=Category.TRUE_CORRECT,
            misconception="NA",
        ),
        Prediction(
            category=Category.TRUE_NEITHER,
            misconception="NA",
        ),
    ]

    logger.info("Original predictions order:")
    for i, pred in enumerate(candidate_predictions, 1):
        logger.info(f"  {i}. {pred}")

    # Create reranking request
    request = RerankingRequest(
        evaluation_row=eval_row,
        candidate_predictions=candidate_predictions,
    )

    # Test prompt generation
    logger.info("\nGenerating prompt...")
    prompt = build_reranking_prompt(request)
    logger.debug(f"Prompt preview (first 200 chars): {prompt[:200]}...")

    # Perform reranking
    logger.info("\nPerforming reranking...")
    from kaggle_map.reranker.utils import format_chat_prompt
    
    # Build prompt and wrap with chat format (like in benchmark.py)
    base_prompt = build_reranking_prompt(request)
    full_prompt = format_chat_prompt(load_config.model_name, base_prompt)
    
    # Call LLM directly to see what response we get
    response = llm(
        full_prompt,
        max_tokens=50,
        temperature=0.01,
        stop=["\n", ".", ";"],
        echo=False,
    )
    response_text = response["choices"][0]["text"].strip()  # type: ignore
    logger.info(f"Raw LLM response: {response_text!r}")
    
    # Now parse the response
    from kaggle_map.reranker.rerank import parse_reranking_response
    reranked = parse_reranking_response(response_text, request.candidate_predictions)

    logger.info("\nReranked predictions:")
    for i, pred in enumerate(reranked, 1):
        logger.info(f"  {i}. {pred}")

    # Verify reranking worked
    assert len(reranked) == len(candidate_predictions), "Reranking should preserve all predictions"
    # Check that all original predictions are present (can't use set with unhashable Prediction objects)
    for pred in candidate_predictions:
        assert pred in reranked, f"Original prediction {pred} not found in reranked list"

    logger.success("\nReranking test completed successfully!")
