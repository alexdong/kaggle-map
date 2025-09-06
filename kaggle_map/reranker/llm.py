"""Simplified LLM reranker using direct llama-cpp-python calls.

This module provides reranking functionality using local GGUF models,
replacing the complex HTTP/async implementation with direct model calls.
"""

import re
import time
from dataclasses import dataclass

from llama_cpp import Llama, llama_supports_gpu_offload
from loguru import logger
from rich.console import Console

from kaggle_map.core.models import (
    GGUF_MODELS,
    MODEL_OPTIONS,
    EvaluationRow,
    InferenceConfig,
    LLMModelLoadConfig,
    LLMResponse,
    Prediction,
    PromptTemplate,
)
from kaggle_map.reranker.utils import load_llm_model


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


if __name__ == "__main__":
    import pandas as pd
    from rich.table import Table

    from kaggle_map.core.models import Category
    from kaggle_map.utils.metrics import calculate_map_at_3

    # Check GPU support but don't require it
    has_gpu = llama_supports_gpu_offload()
    if not has_gpu:
        logger.warning("GPU support not available, running on CPU (will be slower)")

    console = Console()
    console.print("🚀 LLM Model Benchmarking Tool", style="bold cyan")
    console.print("=" * 50)

    # Load evaluation dataset
    console.print("Loading evaluation dataset...", style="cyan")
    eval_df = pd.read_csv("datasets/error_prediction.csv")
    console.print(f"Loaded {len(eval_df)} rows from error_prediction.csv", style="green")

    # Prepare benchmark results
    results = []

    # Download and benchmark all model variants
    for model_name in MODEL_OPTIONS:
        gguf_repo_spec = GGUF_MODELS[model_name]
        for quantization in gguf_repo_spec.available_quantizations:
            console.print(f"\nBenchmarking {model_name} with {quantization} quantization...", style="yellow")

            load_config = LLMModelLoadConfig(
                model_name=model_name,
                quantization=quantization,
            )
            inference_config = InferenceConfig(
                max_tokens=400,
                temperature=0.1,
                echo=False,
            )

            # Load model with context manager
            benchmark_start = time.time()
            with load_llm_model(load_config) as llm:
                total_map_score = 0.0
                valid_rows = 0
                idx = 0

                # Process each row
                for _, row in eval_df.iterrows():
                    # Parse predictions from top_3_predictions_formatted column
                    predictions_str = row["top_3_predictions_formatted"]
                    if pd.isna(predictions_str) or not predictions_str or not isinstance(predictions_str, str):
                        continue

                    # Parse the predictions (format: "Category:misconception | Category:misconception | ...")
                    candidate_predictions = [
                        Prediction.from_string(pred_str) for pred_str in predictions_str.split(" | ") if ":" in pred_str
                    ]
                    # Create ground truth prediction from Category and actual_misconception
                    category = Category.from_csv_string(row["Category"])
                    misconception = row["actual_misconception"] if pd.notna(row["actual_misconception"]) else "NA"
                    ground_truth = Prediction(category=category, misconception=misconception)

                    # Create evaluation row for reranking
                    eval_row = EvaluationRow(
                        row_id=row["row_id"],
                        question_id=row["QuestionId"],
                        question_text=row["QuestionText"],
                        mc_answer=row["MC_Answer"],
                        student_explanation=row["StudentExplanation"],
                        correct_answer=row["CorrectAnswer"] if pd.notna(row["CorrectAnswer"]) else None,
                    )

                    # Create reranking request
                    request = RerankingRequest(evaluation_row=eval_row, candidate_predictions=candidate_predictions)

                    # Rerank predictions - skip row if reranking fails
                    response = llm(
                        build_reranking_prompt(request),
                        max_tokens=20,
                        temperature=0.1,
                        stop=["\n"],
                        echo=False,
                    )
                    response_text = response["choices"][0]["text"].strip()  # type: ignore

                    # Parse reranking response - skip if invalid
                    numbers = re.findall(r"\d+", response_text)
                    indices = [int(n) - 1 for n in numbers]
                    reranked_predictions = [candidate_predictions[i] for i in indices]

                    # Calculate MAP@3 score
                    map_score = calculate_map_at_3(ground_truth, reranked_predictions)
                    total_map_score += map_score
                    valid_rows += 1
                    idx += 1

                    # Log progress every 10 rows
                    if idx % 10 == 0:
                        console.print(f"  Processed {idx}/{len(eval_df)} rows", style="dim")

                # Calculate average MAP@3
                assert valid_rows > 0, "No valid rows processed for MAP@3 calculation"
                avg_map_score = total_map_score / valid_rows
                results.append(
                    {
                        "Model": model_name,
                        "Quantization": quantization,
                        "MAP@3": avg_map_score,
                        "Valid Rows": valid_rows,
                        "Time (s)": round(time.time() - benchmark_start, 2),
                    }
                )
                console.print(f"  Completed: MAP@3 = {avg_map_score:.4f} on {valid_rows} rows", style="green")

    # Display results table
    console.print("\n" + "=" * 50, style="bold cyan")
    console.print("📊 Benchmark Results", style="bold cyan")
    console.print("=" * 50, style="bold cyan")

    table = Table(title="Model Performance Comparison")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Quantization", style="magenta")
    table.add_column("MAP@3", style="green", justify="right")

    # Sort results by MAP@3 score (descending)
    results.sort(key=lambda x: x["MAP@3"], reverse=True)

    for result in results:
        table.add_row(result["Model"], result["Quantization"], f"{result['MAP@3']:.4f}")

    console.print(table)
