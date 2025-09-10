"""Benchmarking tools for evaluating LLM reranking performance.

This module provides functionality to benchmark different LLM models and
quantization levels for the reranking task, measuring MAP@3 scores.
"""

import re
import time
from pathlib import Path

import click
import pandas as pd
from llama_cpp import llama_supports_gpu_offload
from loguru import logger
from rich.console import Console
from rich.table import Table

from kaggle_map.core.dataset import extract_correct_answers, load_training_data
from kaggle_map.core.models import Category, EvaluationRow, Prediction
from kaggle_map.reranker.models import (
    MODEL_OPTIONS,
    RerankerLLMLoadConfig,
    RerankerModelName,
    RerankerModelQuantizationLevel,
)
from kaggle_map.reranker.rerank import RerankingRequest, build_reranking_prompt
from kaggle_map.reranker.utils import format_chat_prompt, load_llm_model
from kaggle_map.utils.metrics import calculate_map_at_3


def benchmark_single_model(
    model_name: RerankerModelName,
    quantization: RerankerModelQuantizationLevel,
    eval_df: pd.DataFrame,
    correct_answers: dict,
) -> dict:
    """Benchmark a single model/quantization combination.

    Args:
        model_name: Name of the model to benchmark
        quantization: Quantization level to use
        eval_df: DataFrame with evaluation data
        correct_answers: Dictionary of correct answers by question ID

    Returns:
        Dictionary with benchmark results including MAP@3 score
    """
    console = Console()
    console.print(f"\nBenchmarking {model_name} with {quantization} quantization...", style="yellow")

    load_config = RerankerLLMLoadConfig(
        model_name=model_name,
        quantization=quantization,
    )

    # Load model with context manager
    benchmark_start = time.time()
    llm = load_llm_model(load_config)
    total_map_score = 0.0
    valid_rows = 0

    # Process each row
    for idx, (_index, row) in enumerate(eval_df.iterrows()):
        # Parse predictions from top_3_predictions_formatted column
        predictions_str = row["top_3_predictions_formatted"]
        assert not pd.isna(predictions_str), f"Row {idx}: predictions_str is NaN"
        assert isinstance(predictions_str, str), f"Row {idx}: predictions_str is not a string"

        candidate_predictions = [
            Prediction.from_string(pred_str) for pred_str in predictions_str.split(" | ") if ":" in pred_str
        ]
        assert candidate_predictions, f"Row {idx}: No valid predictions found"

        # Create ground truth prediction from Category and actual_misconception
        category = Category.from_csv_string(str(row["Category"]))
        misconception = row["actual_misconception"] if pd.notna(row["actual_misconception"]) else "NA"
        ground_truth = Prediction(category=category, misconception=misconception)

        # Create evaluation row for reranking
        eval_row = EvaluationRow(
            row_id=row["row_id"],
            question_id=row["QuestionId"],
            question_text=row["QuestionText"],
            mc_answer=row["MC_Answer"],
            student_explanation=row["StudentExplanation"],
            correct_answer=correct_answers.get(row["QuestionId"], ""),
        )

        # Create reranking request
        request = RerankingRequest(evaluation_row=eval_row, candidate_predictions=candidate_predictions)

        # Build prompt and wrap with chat format
        base_prompt = build_reranking_prompt(request)
        full_prompt = format_chat_prompt(model_name, base_prompt)

        response = llm(
            full_prompt,
            max_tokens=50,
            temperature=0.01,
            stop=["\n", ".", ";"],
            echo=False,
        )
        response_text = response["choices"][0]["text"].strip()  # type: ignore

        # Parse reranking response
        numbers = re.findall(r"\d+", response_text)
        assert numbers, f"Row {idx}: No numbers found in response"

        indices = [int(n) - 1 for n in numbers]
        assert len(indices) == len(candidate_predictions), (
            f"Row {idx}: Got {len(indices)} indices but have {len(candidate_predictions)} predictions"
        )

        reranked_predictions = [candidate_predictions[i] for i in indices if 0 <= i < len(candidate_predictions)]

        # Calculate MAP@3 score
        map_score = calculate_map_at_3(ground_truth, reranked_predictions)

        total_map_score += map_score
        valid_rows += 1

        # Log progress every 10 rows
        if (idx + 1) % 10 == 0:
            console.print(f"  Processed {idx}/{len(eval_df)} rows", style="dim")

    # Calculate average MAP@3
    assert valid_rows, "  Failed: No valid rows processed"
    avg_map_score = total_map_score / valid_rows
    console.print(f"  Completed: MAP@3 = {avg_map_score:.4f} on {valid_rows} rows", style="green")
    return {
        "Model": model_name,
        "Quantization": quantization,
        "MAP@3": avg_map_score,
        "Valid Rows": valid_rows,
        "Time (s)": round(time.time() - benchmark_start, 2),
    }


@click.command()
@click.option("--model", type=click.Choice(MODEL_OPTIONS), default="gemma-3-12b-it", help="Model to benchmark")
@click.option(
    "--quantization",
    type=click.Choice(["Q2_K_XL", "Q3_K_XL", "Q4_K_XL", "Q5_K_XL", "Q6_K_XL"]),
    default="Q4_K_XL",
    help="Quantization level to use",
)
@click.option(
    "--sample-ratio",
    type=click.FloatRange(0.001, 1.0),
    default=0.01,
    help="Ratio of dataset to sample (0.01 = 1%, 1.0 = 100%)",
)
def main(model: str, quantization: str, sample_ratio: float) -> None:
    """Benchmark LLM model reranking performance."""
    # Check GPU support but don't require it
    has_gpu = llama_supports_gpu_offload()
    if not has_gpu:
        logger.warning("GPU support not available, running on CPU (will be slower)")

    console = Console()
    logger.info("🚀 LLM Model Benchmarking Tool")
    logger.info("=" * 50)

    # Load evaluation dataset and sample requested rows
    console.print("Loading evaluation dataset...", style="cyan")
    eval_df = pd.read_csv("datasets/error_prediction.csv")
    console.print(f"Loaded {len(eval_df)} rows from error_prediction.csv", style="green")

    # Calculate sample size from ratio and sample with fixed seed for consistency
    sample_size = max(1, int(len(eval_df) * sample_ratio))
    eval_df = eval_df.sample(n=min(sample_size, len(eval_df)), random_state=42).reset_index(drop=True)
    console.print(
        f"Using {len(eval_df)} sampled rows ({sample_ratio * 100:.1f}% of dataset) for benchmarking", style="yellow"
    )

    # Prepare benchmark results
    correct_answers = extract_correct_answers(load_training_data(Path("datasets/train.csv")))

    # Convert string arguments to enum values
    # The model comes as enum name from Click, quantization as string value
    model_enum = RerankerModelName[model]
    quantization_enum = RerankerModelQuantizationLevel(quantization)

    # Run benchmark for single model/quantization combination
    result = benchmark_single_model(model_enum, quantization_enum, eval_df, correct_answers)
    results = [result]

    # Display results table
    logger.info("=" * 50)
    logger.info("📊 Benchmark Results")
    logger.info("=" * 50)

    table = Table(title="Model Performance Comparison")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Quantization", style="magenta")
    table.add_column("MAP@3", style="green", justify="right")

    # Sort results by MAP@3 score (descending)
    results.sort(key=lambda x: x["MAP@3"], reverse=True)

    for result in results:
        table.add_row(result["Model"], result["Quantization"], f"{result['MAP@3']:.4f}")

    console.print(table)


if __name__ == "__main__":
    main()
