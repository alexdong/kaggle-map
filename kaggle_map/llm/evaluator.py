"""LLM-based evaluator for student misconception predictions."""

from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Template
from loguru import logger
from rich.console import Console
from rich.table import Table

from kaggle_map.core.models import EvaluationResult
from kaggle_map.dataloader import load_validation_data, stratified_sample
from kaggle_map.llm.utils import evaluate_dataframe
from kaggle_map.utils.gguf_model import (
    GGUFModelLoadConfig,
    GGUFModelName,
    GGUFModelQuantizationLevel,
    format_chat_prompt,
    get_stop_tokens,
    load_llm_model,
)
from kaggle_map.utils.logger_config import configure_logger

# Configure module-specific logging
# When run as a script, explicitly use the module path
module_name = "kaggle_map.llm.evaluator" if __name__ == "__main__" else __name__
configure_logger(module_name)


def display_evaluation_details(
    evaluation_results: list[EvaluationResult],
) -> None:
    """Display detailed evaluation results for each row.

    Args:
        evaluation_results: List of EvaluationResult objects
    """
    if not evaluation_results:
        return

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
        explanation = result.explanation
        if len(explanation) > max_explanation_length:
            explanation = explanation[: max_explanation_length - 3] + "..."

        # Format LLM predictions
        llm_labels = " | ".join([str(pred) for pred in result.predictions])
        if len(llm_labels) > max_llm_labels_length:
            llm_labels = llm_labels[: max_llm_labels_length - 3] + "..."

        # Format MAP@3 score
        score_str = f"{result.score:.2f}"

        # Combine category and misconception
        category_misconception = str(result.ground_truth)

        # Add row
        table.add_row(
            str(result.row_id),
            result.mc_answer,
            explanation,
            category_misconception,
            llm_labels,
            score_str,
        )

    # Display the table
    console.print(table)
    console.print(f"\nTotal rows evaluated: {len(evaluation_results)}")
    avg_score = sum(r.score for r in evaluation_results) / len(evaluation_results) if evaluation_results else 0
    console.print(f"Average MAP@3: {avg_score:.4f}\n")


def evaluate_with_llm(
    template_path: Path,
    data_path: Path,
    sample_ratio: float = 0.2,
    model_name: GGUFModelName = GGUFModelName.GEMMA_3_12B_IT,
    quantization: GGUFModelQuantizationLevel = GGUFModelQuantizationLevel.Q4_K_XL,
) -> float:
    logger.info(f"Loading validation data from {data_path}")
    validation_pairs = load_validation_data(data_path)
    logger.info(f"Loaded {len(validation_pairs)} validation samples")

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

    df = pd.DataFrame(data_rows)

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

    # Load the model
    config = GGUFModelLoadConfig(
        model_name=model_name,
        quantization=quantization,
        n_ctx=4096,
        n_batch=512,
        n_gpu_layers=-1,  # Use all GPU layers
        verbose=False,
    )

    logger.info(f"Loading {model_name.value} with {quantization.value} quantization")
    logger.info(f"GPU layers: {config.n_gpu_layers} (-1 means use all available)")
    llm = load_llm_model(config)

    # Prepare template
    template = Template(template_path.read_text())

    # Create wrapper for llm to handle format_chat_prompt
    stop_tokens = get_stop_tokens(model_name)

    def llm_wrapper(prompt: str, **kwargs: Any) -> Any:  # noqa: ANN401
        full_prompt = format_chat_prompt(model_name, prompt)
        return llm(full_prompt, **kwargs)

    # Evaluate using the new utility function
    evaluation_results, avg_score = evaluate_dataframe(sampled_df, template, llm_wrapper, stop_tokens=stop_tokens)

    logger.debug(f"\n{'=' * 50}")
    logger.debug("Evaluation Complete")
    logger.debug(f"{'=' * 50}")
    logger.debug(f"Samples evaluated: {len(evaluation_results)}")
    logger.debug(f"Average MAP@3: {avg_score:.4f}")
    logger.debug(f"{'=' * 50}")

    # Display detailed evaluation results
    display_evaluation_details(evaluation_results)

    return avg_score


if __name__ == "__main__":
    """Run evaluation with default settings."""

    import click

    @click.command()
    @click.option(
        "--sample-ratio",
        type=click.FloatRange(0.0, 1.0),
        default=0.2,
        help="Ratio of validation data to sample (0.0-1.0)",
    )
    @click.option(
        "--data-path",
        type=click.Path(exists=True, path_type=Path),
        default=Path("datasets/33474_train.csv"),
        help="Path to CSV file",
    )
    @click.option(
        "--template-path",
        type=click.Path(exists=True, path_type=Path),
        default=Path("kaggle_map/llm/prompts/predict.j2"),
        help="Custom prompt template path",
    )
    def main(sample_ratio: float, data_path: Path, template_path: Path) -> None:
        """Evaluate LLM predictions on validation data."""
        # Run evaluation
        avg_map_score = evaluate_with_llm(
            template_path=template_path,
            data_path=data_path,
            sample_ratio=sample_ratio,
            model_name=GGUFModelName.GEMMA_3_12B_IT,
            quantization=GGUFModelQuantizationLevel.Q4_K_XL,
        )

        print(f"\nFinal MAP@3 Score: {avg_map_score:.4f}")

    main()
