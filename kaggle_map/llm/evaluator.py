"""LLM-based evaluator for student misconception predictions."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Template
from loguru import logger
from rich.console import Console
from rich.table import Table

from kaggle_map.core.models import Category, EvaluationRow, Prediction
from kaggle_map.dataloader import load_validation_data, stratified_sample
from kaggle_map.utils.gguf_model import (
    GGUFModelLoadConfig,
    GGUFModelName,
    GGUFModelQuantizationLevel,
    format_chat_prompt,
    get_stop_tokens,
    load_llm_model,
)
from kaggle_map.utils.logger_config import configure_logger
from kaggle_map.utils.metrics import calculate_map_at_3


@dataclass
class EvaluationConfig:
    template_path: Path
    data_path: Path
    sample_ratio: float
    row_ids: list[int] | None
    model_name: GGUFModelName
    quantization: GGUFModelQuantizationLevel


def build_prediction_prompt(eval_row: EvaluationRow, template_path: Path) -> str:
    template = Template(template_path.read_text())
    return template.render(
        question_text=eval_row.question_text,
        mc_answer=eval_row.mc_answer,
        student_explanation=eval_row.student_explanation,
    )


def parse_predictions(response: str) -> list[Prediction]:  # noqa: C901
    """Parse LLM response to extract predictions.

    The LLM returns three predictions on ONE line separated by spaces.
    Format: "Category1:Misconception1 Category2:Misconception2 Category3:Misconception3"
    Example: "True_Correct:NA True_Neither:NA True_Misconception:Division"

    Also supports categories without colons (assumes NA misconception):
    Example: "True_Correct True_Misconception:Division True_Neither"

    Args:
        response: Raw LLM response containing predictions

    Returns:
        List of up to 3 Prediction objects
    """
    predictions = []

    # The response should be a single line with three space-separated predictions
    response_clean = response.strip()

    # Handle case where LLM might return multiple lines - take the first non-empty line
    # Look for lines that contain category names
    for line in response_clean.split("\n"):
        line_stripped = line.strip()
        if line_stripped and any(cat.value in line_stripped for cat in Category):
            response_clean = line_stripped
            break

    # Split by spaces to get individual predictions
    prediction_parts = response_clean.split()

    for part in prediction_parts:
        try:
            if ":" in part:
                # Standard format: Category:Misconception
                prediction = Prediction.from_string(part)
            else:
                # Check if it's a valid category without a colon
                # Try to match against Category enum values
                category = None
                for cat in Category:
                    if part == cat.value:
                        category = cat
                        break

                if category is None:
                    logger.debug(f"Skipping invalid prediction part: '{part}'")
                    continue

                # Create prediction with NA misconception for categories without colons
                prediction = Prediction(category=category, misconception="NA")

            predictions.append(prediction)

            max_predictions = 3
            if len(predictions) >= max_predictions:
                break
        except Exception as e:
            logger.debug(f"Failed to parse prediction '{part}': {e}")
            continue

    # Pad with default predictions if needed
    max_predictions = 3
    while len(predictions) < max_predictions:
        predictions.append(Prediction(category=Category.TRUE_CORRECT, misconception="NA"))

    return predictions[:max_predictions]


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


def evaluate_with_llm(config: EvaluationConfig) -> float:
    logger.info(f"Loading validation data from {config.data_path}")
    validation_pairs = load_validation_data(config.data_path)
    logger.info(f"Loaded {len(validation_pairs)} validation samples")

    df = _prepare_dataframe(validation_pairs)
    sampled_df = _sample_dataframe(df, config.row_ids, config.sample_ratio)

    # Load the model
    model_config = GGUFModelLoadConfig(
        model_name=config.model_name,
        quantization=config.quantization,
        n_ctx=4096,
        n_batch=512,
        n_gpu_layers=-1,  # Use all GPU layers
        verbose=False,
    )

    logger.info(f"Loading {config.model_name.value} with {config.quantization.value} quantization")
    logger.info(f"GPU layers: {model_config.n_gpu_layers} (-1 means use all available)")
    llm = load_llm_model(model_config)

    # Evaluate each sample
    scores = []
    stop_tokens = get_stop_tokens(config.model_name)

    # Track all evaluation results for detailed output
    evaluation_results = []

    for row_number, (_, row) in enumerate(sampled_df.iterrows()):
        # Reconstruct evaluation row
        eval_row = EvaluationRow(
            row_id=int(row["row_id"]),
            question_id=int(row["QuestionId"]),
            question_text=str(row["QuestionText"]),
            mc_answer=str(row["MC_Answer"]),
            student_explanation=str(row["StudentExplanation"]),
        )

        # Reconstruct ground truth
        ground_truth = Prediction(
            category=row["Category"],
            misconception=row["Misconception"] if pd.notna(row["Misconception"]) else "NA",
        )

        # Build prompt
        user_prompt = build_prediction_prompt(eval_row, config.template_path)
        logger.info(f"Prompt for row {eval_row.row_id}:\n{user_prompt}\n")
        full_prompt = format_chat_prompt(config.model_name, user_prompt)

        # Generate predictions
        response = llm(
            full_prompt,
            max_tokens=256,
            temperature=0.1,
            top_p=0.95,
            stop=stop_tokens,
            echo=False,
        )

        response_text = response["choices"][0]["text"]  # type: ignore[index]
        logger.debug(f"LLM response for row {eval_row.row_id}:\n{response_text}\n")

        # Parse predictions
        predictions = parse_predictions(response_text)
        logger.debug(f"Predictions for row {eval_row.row_id}: {predictions}")

        # Calculate MAP@3
        score = calculate_map_at_3(ground_truth, predictions)
        scores.append(score)

        # Store detailed result for every row
        evaluation_results.append(
            {
                "row_id": eval_row.row_id,
                "mc_answer": eval_row.mc_answer,
                "explanation": eval_row.student_explanation,
                "category": ground_truth.category.value,
                "misconception": ground_truth.misconception,
                "predictions": predictions,
                "score": score,
            }
        )

        if (row_number + 1) % 10 == 0:
            current_avg = sum(scores) / len(scores)
            logger.info(f"Progress: {row_number + 1}/{len(sampled_df)} | Current MAP@3: {current_avg:.4f}")

    # Calculate average score
    avg_score = sum(scores) / len(scores) if scores else 0.0

    logger.success(f"\n{'=' * 50}")
    logger.success("Evaluation Complete")
    logger.success(f"{'=' * 50}")
    logger.success(f"Samples evaluated: {len(scores)}")
    logger.success(f"Average MAP@3: {avg_score:.4f}")
    logger.success(f"{'=' * 50}")

    # Display detailed evaluation results
    display_evaluation_details(evaluation_results)

    # Save results to CSV
    save_evaluation_results_to_csv(evaluation_results)

    return avg_score


if __name__ == "__main__":
    """Run evaluation with default settings."""

    import click

    @click.command()
    @click.option(
        "--sample-ratio",
        type=click.FloatRange(0.0, 1.0),
        default=0.2,
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
        default=Path("datasets/33474_train.csv"),
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
            model_name=GGUFModelName.QWEN3_30B_Thinking,
            quantization=GGUFModelQuantizationLevel.Q2_K_XL,
        )
        avg_map_score = evaluate_with_llm(config)

        print(f"\nFinal MAP@3 Score: {avg_map_score:.4f}")

    main()
