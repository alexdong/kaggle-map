"""LLM-based evaluator for student misconception predictions."""

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from llama_cpp import Llama
from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from kaggle_map.core.models import EvaluationRow, Prediction
from kaggle_map.dataloader import load_validation_data, stratified_sample
from kaggle_map.llm.utils import build_prediction_prompt
from kaggle_map.utils.gguf_model import (
    GGUFModelLoadConfig,
    GGUFModelName,
    GGUFModelQuantizationLevel,
    format_chat_prompt,
    get_stop_tokens,
    load_llm_model,
    parse_llm_response,
    suggest_ctx_length,
    suggest_max_tokens,
)
from kaggle_map.utils.logger_config import configure_logger
from kaggle_map.utils.metrics import calculate_map_at_3

# Configuration constants
# Set to -1 for unlimited generation (will generate until context window limit or stop token)
# Set to a positive value to limit response length
MAX_RESPONSE_TOKENS = -1  # Unlimited generation within context window


@dataclass
class EvaluationConfig:
    template_path: Path
    data_path: Path
    sample_ratio: float
    row_ids: list[int] | None
    model_name: GGUFModelName
    quantization: GGUFModelQuantizationLevel


def save_evaluation_results_to_csv(
    evaluation_results: list[dict],
    output_dir: Path = Path("logs"),
) -> Path:
    """Save evaluation results to CSV file.

    Args:
        evaluation_results: List of evaluation result dictionaries
        output_dir: Directory to save the CSV file

    Returns:
        Path to the saved CSV file
    """
    assert evaluation_results, "Cannot save empty evaluation results to CSV"

    # Create logs directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"llm_evaluation_{timestamp}.csv"

    # Convert results to DataFrame
    df_data = []
    for result in evaluation_results:
        # Convert predictions to string format
        predictions_str = " | ".join([str(pred) for pred in result["predictions"]])

        df_data.append(
            {
                "row_id": result["row_id"],
                "mc_answer": result["mc_answer"],
                "explanation": result["explanation"],
                "true_category": result["category"],
                "true_misconception": result["misconception"],
                "predicted_labels": predictions_str,
                "map_at_3": result["score"],
            }
        )

    df = pd.DataFrame(df_data)
    df.to_csv(output_path, index=False)

    logger.info(f"Evaluation results saved to: {output_path}")
    return output_path


def display_evaluation_details(
    evaluation_results: list[dict],
) -> None:
    """Display detailed evaluation results for each row.

    Args:
        evaluation_results: List of evaluation result dictionaries
    """
    assert evaluation_results, "Cannot display empty evaluation results"

    console = Console()

    # Configuration
    max_explanation_length = 80
    max_llm_labels_length = 100

    # Create rich table
    table = Table(
        title="\n📊 Detailed Evaluation Results",
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
        width=None,
    )

    # Add columns
    table.add_column("Row ID", style="cyan", no_wrap=True)
    table.add_column("MC Answer", style="yellow", overflow="fold")
    table.add_column("Explanation", style="white", overflow="fold", max_width=max_explanation_length)
    table.add_column("Category:Misconception", style="green", overflow="fold")
    table.add_column("LLM Labels", style="dim", overflow="fold", max_width=max_llm_labels_length)
    table.add_column("MAP@3", style="red bold", justify="center")

    # Add each row
    for result in evaluation_results:
        # Truncate explanation if needed
        explanation = result["explanation"]
        if len(explanation) > max_explanation_length:
            explanation = explanation[: max_explanation_length - 3] + "..."

        # Format LLM predictions
        llm_labels = " | ".join([str(pred) for pred in result["predictions"]])
        if len(llm_labels) > max_llm_labels_length:
            llm_labels = llm_labels[: max_llm_labels_length - 3] + "..."

        # Format MAP@3 score
        score_str = f"{result['score']:.2f}"

        # Combine category and misconception
        category_misconception = f"{result['category']}:{result['misconception']}"

        # Add row
        table.add_row(
            str(result["row_id"]),
            result["mc_answer"],
            explanation,
            category_misconception,
            llm_labels,
            score_str,
        )

    # Display the table
    console.print(table)
    console.print(f"\nTotal rows evaluated: {len(evaluation_results)}")
    avg_score = sum(r["score"] for r in evaluation_results) / len(evaluation_results) if evaluation_results else 0
    console.print(f"Average MAP@3: {avg_score:.4f}\n")


def _prepare_dataframe(validation_pairs: list) -> pd.DataFrame:
    """Convert validation pairs to DataFrame."""
    data_rows = []
    for eval_row, ground_truth in validation_pairs:
        data_rows.append(
            {
                "row_id": eval_row.row_id,
                "QuestionId": eval_row.question_id,
                "QuestionText": eval_row.question_text,
                "MC_Answer": eval_row.mc_answer,
                "StudentExplanation": eval_row.student_explanation,
                "Category": ground_truth.category.value,
                "Misconception": ground_truth.misconception,
            }
        )
    return pd.DataFrame(data_rows)


def _sample_dataframe(
    df: pd.DataFrame,
    row_ids: list[int] | None,
    sample_ratio: float,
) -> pd.DataFrame:
    """Sample DataFrame by row_ids or stratified sampling."""
    if row_ids:
        logger.info(f"Filtering data to specified row IDs: {row_ids}")
        sampled_df = df[df["row_id"].isin(row_ids)]
        if len(sampled_df) != len(row_ids):
            found_ids = set(sampled_df["row_id"].tolist())
            missing_ids = set(row_ids) - found_ids
            logger.warning(f"Some row IDs not found in data: {missing_ids}")
        logger.info(f"Selected {len(sampled_df)} samples for evaluation")
        return pd.DataFrame(sampled_df)

    # Perform stratified sampling
    logger.info(f"Sampling {sample_ratio:.1%} of data with stratification")
    sampled_df = stratified_sample(
        df,
        sample_ratio=sample_ratio,
        stratify_cols=["QuestionId", "Category", "MC_Answer"],
        min_samples_per_stratum=2,
        random_seed=42,
    )
    logger.info(f"Selected {len(sampled_df)} samples for evaluation")
    return sampled_df


def _get_optimal_context(model_name: GGUFModelName, quantization: GGUFModelQuantizationLevel) -> int:
    """Calculate optimal context size for model/quantization."""
    optimal_ctx = suggest_ctx_length(
        vram_gb=16.0,  # RTX 2000 Ada has 16GB VRAM
        model_name=model_name,
        quantization=quantization,
        desktop_overhead_gb=0.7,
        safety_margin_gb=1.0,
    )

    # Model-specific minimum context requirements
    is_gpt_oss = model_name == GGUFModelName.GPT_OSS_20B
    min_context = 16384 if is_gpt_oss else 2048  # OpenAI recommends 16k for GPT-OSS
    standard_context = 16384 if is_gpt_oss else 4096

    if optimal_ctx <= 0:
        logger.warning(
            f"Model {model_name.value} {quantization.value} "
            f"may not fit in 16GB VRAM. Using minimum context of {min_context}"
        )
        return min_context

    if optimal_ctx < standard_context:
        if is_gpt_oss:
            logger.warning(
                f"Limited context of {optimal_ctx} tokens for {model_name.value} {quantization.value}. "
                f"OpenAI recommends minimum {standard_context}"
            )
        else:
            logger.warning(f"Limited context of {optimal_ctx} tokens for {model_name.value} {quantization.value}")

    return optimal_ctx


def _evaluate_single_sample(
    row: pd.Series, config: EvaluationConfig, llm: Llama, stop_tokens: list[str]
) -> tuple[float, dict[str, Any]]:
    """Evaluate a single sample and return score and result dictionary."""
    eval_row = EvaluationRow(
        row_id=int(row["row_id"]),
        question_id=int(row["QuestionId"]),
        question_text=str(row["QuestionText"]),
        mc_answer=str(row["MC_Answer"]),
        student_explanation=str(row["StudentExplanation"]),
    )

    ground_truth = Prediction(
        category=row["Category"],
        misconception=row["Misconception"] if pd.notna(row["Misconception"]) else "NA",
    )

    user_prompt = build_prediction_prompt(eval_row, config.template_path)
    logger.info(f"Prompt for row {eval_row.row_id}:\n{user_prompt}\n")
    full_prompt = format_chat_prompt(config.model_name, user_prompt)

    is_gpt_oss = config.model_name == GGUFModelName.GPT_OSS_20B
    temperature = 1.0 if is_gpt_oss else 0.1
    top_p = 1.0 if is_gpt_oss else 0.95

    llm_kwargs: dict[str, Any] = {
        "max_tokens": MAX_RESPONSE_TOKENS,
        "temperature": temperature,
        "top_p": top_p,
        "stop": stop_tokens,
        "echo": False,
    }

    if is_gpt_oss:
        llm_kwargs["reasoning_effort"] = "high"

    # Limit tokens to prevent repetitive reasoning patterns
    llm_kwargs = suggest_max_tokens(llm_kwargs)

    response = llm(full_prompt, **llm_kwargs)
    response_text = response["choices"][0]["text"]  # type: ignore[index]
    logger.debug(f"LLM response for row {eval_row.row_id}:\n{response_text}\n")

    result = parse_llm_response(response_text, Prediction.parse)
    predictions = result.predictions

    if result.thinking_trace:
        logger.info(f"LLM Thinking Trace for row {eval_row.row_id}:\n{result.thinking_trace}")

    logger.debug(f"Predictions for row {eval_row.row_id}: {predictions}")

    score = calculate_map_at_3(ground_truth, predictions)

    result_dict = {
        "row_id": eval_row.row_id,
        "mc_answer": eval_row.mc_answer,
        "explanation": eval_row.student_explanation,
        "category": ground_truth.category.value,
        "misconception": ground_truth.misconception,
        "predictions": predictions,
        "score": score,
    }

    return score, result_dict


def _setup_model(config: EvaluationConfig) -> Llama:
    """Setup and configure the LLM model."""
    optimal_ctx = _get_optimal_context(config.model_name, config.quantization)

    model_config = GGUFModelLoadConfig(
        model_name=config.model_name,
        quantization=config.quantization,
        n_ctx=optimal_ctx,
    )

    is_gpt_oss = config.model_name == GGUFModelName.GPT_OSS_20B
    logger.info(f"Loading {config.model_name.value} with {config.quantization.value} quantization")
    logger.info(f"Using dynamic context size: {optimal_ctx} tokens")
    logger.info(f"GPU layers: {model_config.n_gpu_layers} (-1 means use all available)")

    if is_gpt_oss:
        logger.info("Using OpenAI recommended parameters: temperature=1.0, top_p=1.0")
    else:
        logger.info("Using standard parameters: temperature=0.1, top_p=0.95")

    return load_llm_model(model_config)


def _finalize_results(scores: list[float], evaluation_results: list[dict[str, Any]], start_time: float) -> float:
    """Calculate final metrics and display results."""
    avg_score = sum(scores) / len(scores) if scores else 0.0
    elapsed_time = time.time() - start_time

    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        time_str = f"{int(minutes)}m {int(seconds)}s"
    else:
        time_str = f"{seconds:.1f}s"

    logger.success(f"\n{'=' * 50}")
    logger.success("Evaluation Complete")
    logger.success(f"{'=' * 50}")
    logger.success(f"Samples evaluated: {len(scores)}")
    logger.success(f"Average MAP@3: {avg_score:.4f}")
    logger.success(f"Total time: {time_str}")
    logger.success(f"Time per sample: {elapsed_time / len(scores):.2f}s")
    logger.success(f"{'=' * 50}")

    display_evaluation_details(evaluation_results)
    save_evaluation_results_to_csv(evaluation_results)

    return avg_score


def evaluate_with_llm(config: EvaluationConfig) -> float:
    logger.info(f"Loading validation data from {config.data_path}")
    validation_pairs = load_validation_data(config.data_path)
    logger.info(f"Loaded {len(validation_pairs)} validation samples")

    df = _prepare_dataframe(validation_pairs)
    sampled_df = _sample_dataframe(df, config.row_ids, config.sample_ratio)

    llm = _setup_model(config)

    # Evaluate each sample
    scores = []
    stop_tokens = get_stop_tokens(config.model_name)

    # Track all evaluation results for detailed output
    evaluation_results = []

    # Track timing
    start_time = time.time()

    # Create progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[cyan]MAP@3: {task.fields[current_map]:.3f} (avg: {task.fields[avg_map]:.3f})"),
        TimeRemainingColumn(),
        console=Console(stderr=True),  # Output to stderr to not interfere with logs
        refresh_per_second=2,
    ) as progress:
        task = progress.add_task(
            "[green]Evaluating samples...",
            total=len(sampled_df),
            current_map=0.0,
            avg_map=0.0,
        )

        for _row_number, (_, row) in enumerate(sampled_df.iterrows()):
            score, result_dict = _evaluate_single_sample(row, config, llm, stop_tokens)
            scores.append(score)
            evaluation_results.append(result_dict)

            # Update progress bar with latest metrics
            avg_score = sum(scores) / len(scores) if scores else 0.0
            progress.update(
                task,
                advance=1,
                current_map=score,
                avg_map=avg_score,
            )

    return _finalize_results(scores, evaluation_results, start_time)


if __name__ == "__main__":
    """Run evaluation with default settings."""

    import click

    @click.command()
    @click.option(
        "--sample-ratio",
        type=click.FloatRange(0.0, 1.0),
        default=1.0,
        help="Ratio of validation data to sample (0.0-1.0). Ignored if --row-ids is provided.",
    )
    @click.option(
        "--row-ids",
        type=str,
        default=None,
        help="Comma-separated list of specific row IDs to evaluate. Overrides --sample-ratio.",
    )
    @click.option(
        "--data-path",
        type=click.Path(exists=True, path_type=Path),
        default=Path("datasets/33474_focus_group.csv"),
        help="Path to CSV file",
    )
    @click.option(
        "--template-path",
        type=click.Path(exists=True, path_type=Path),
        default=Path("kaggle_map/llm/prompts/predict.j2"),
        help="Custom prompt template path",
    )
    def main(sample_ratio: float, row_ids: str | None, data_path: Path, template_path: Path) -> None:
        """Evaluate LLM predictions on validation data."""
        # Configure logging with DEBUG level to see LLM responses
        configure_logger(__name__, console_level="DEBUG")

        # Parse row_ids if provided
        row_ids_list = None
        if row_ids:
            row_ids_list = [int(rid.strip()) for rid in row_ids.split(",")]
            logger.info(f"Will evaluate specific row IDs: {row_ids_list}")

        # Run evaluation
        config = EvaluationConfig(
            template_path=template_path,
            data_path=data_path,
            sample_ratio=sample_ratio,
            row_ids=row_ids_list,
            # model_name=GGUFModelName.GEMMA_3_27B_IT,
            # quantization=GGUFModelQuantizationLevel.Q3_K_XL,
            model_name=GGUFModelName.GPT_OSS_20B,
            quantization=GGUFModelQuantizationLevel.Q2_K_L,
        )
        avg_map_score = evaluate_with_llm(config)

        print(f"\nFinal MAP@3 Score: {avg_map_score:.4f}")

    main()
