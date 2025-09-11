"""LLM-based evaluator for student misconception predictions."""

from pathlib import Path

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
from kaggle_map.utils.metrics import calculate_map_at_3


def build_prediction_prompt(eval_row: EvaluationRow, template_path: Path | None = None) -> str:
    """Build a prompt for predicting student misconceptions.

    Args:
        eval_row: Evaluation row with question and student response
        template_path: Optional path to custom Jinja2 template

    Returns:
        Rendered prompt string
    """
    if template_path is None:
        template_path = Path(__file__).parent / "prompts" / "predict.j2"

    assert template_path.exists(), f"Template not found: {template_path}"
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

    Args:
        response: Raw LLM response containing predictions

    Returns:
        List of up to 3 Prediction objects
    """
    predictions = []

    # The response should be a single line with three space-separated predictions
    response_clean = response.strip()

    # Handle case where LLM might return multiple lines - take the first non-empty line
    for line in response_clean.split("\n"):
        if line.strip() and ":" in line:
            response_clean = line.strip()
            break

    # Split by spaces to get individual predictions
    prediction_parts = response_clean.split()

    for part in prediction_parts:
        if ":" not in part:
            continue

        try:
            prediction = Prediction.from_string(part)
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
        predictions.append(Prediction(category=Category.FALSE_NEITHER, misconception="NA"))

    return predictions[:max_predictions]


def display_evaluation_details(
    evaluation_results: list[dict],
) -> None:
    """Display detailed evaluation results for each row.

    Args:
        evaluation_results: List of evaluation result dictionaries
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
    table.add_column("Category", style="green")
    table.add_column("Misconception", style="blue", overflow="fold")
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

        # Add row
        table.add_row(
            str(result["row_id"]),
            result["mc_answer"],
            explanation,
            result["category"],
            result["misconception"],
            llm_labels,
            score_str,
        )

    # Display the table
    console.print(table)
    console.print(f"\nTotal rows evaluated: {len(evaluation_results)}")
    avg_score = sum(r["score"] for r in evaluation_results) / len(evaluation_results) if evaluation_results else 0
    console.print(f"Average MAP@3: {avg_score:.4f}\n")


def evaluate_with_llm(
    validation_path: Path = Path("datasets/33474_train.csv"),
    sample_ratio: float = 0.2,
    model_name: GGUFModelName = GGUFModelName.GEMMA_3_12B_IT,
    quantization: GGUFModelQuantizationLevel = GGUFModelQuantizationLevel.Q4_K_XL,
    template_path: Path | None = None,
) -> float:
    """Evaluate LLM predictions on validation data.

    Args:
        validation_path: Path to validation CSV file
        sample_ratio: Ratio of data to sample for evaluation
        model_name: GGUF model to use
        quantization: Quantization level for the model
        template_path: Optional custom prompt template

    Returns:
        Average MAP@3 score across all samples
    """
    # Load validation data
    logger.info(f"Loading validation data from {validation_path}")
    validation_pairs = load_validation_data(validation_path)
    logger.info(f"Loaded {len(validation_pairs)} validation samples")

    # Convert to DataFrame for stratified sampling
    import pandas as pd  # noqa: PLC0415

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

    # Evaluate each sample
    scores = []
    stop_tokens = get_stop_tokens(model_name)

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
        user_prompt = build_prediction_prompt(eval_row, template_path)
        full_prompt = format_chat_prompt(model_name, user_prompt)

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

        # Parse predictions
        predictions = parse_predictions(response_text)

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

    return avg_score


if __name__ == "__main__":
    """Run evaluation with default settings."""
    import sys

    import click

    @click.command()
    @click.option(
        "--sample-ratio",
        type=click.FloatRange(0.0, 1.0),
        default=0.2,
        help="Ratio of validation data to sample (0.0-1.0)",
    )
    @click.option(
        "--validation-path",
        type=click.Path(exists=True, path_type=Path),
        default=Path("datasets/33474_train.csv"),
        help="Path to validation CSV file",
    )
    @click.option(
        "--template-path",
        type=click.Path(exists=True, path_type=Path),
        default=None,
        help="Custom prompt template path",
    )
    def main(
        sample_ratio: float,
        validation_path: Path,
        template_path: Path | None,
    ) -> None:
        """Evaluate LLM predictions on validation data."""
        # Configure logging
        logger.remove()
        logger.add(sys.stderr, level="INFO")

        # Run evaluation
        avg_map_score = evaluate_with_llm(
            validation_path=validation_path,
            sample_ratio=sample_ratio,
            model_name=GGUFModelName.GEMMA_3_12B_IT,
            quantization=GGUFModelQuantizationLevel.Q4_K_XL,
            template_path=template_path,
        )

        print(f"\nFinal MAP@3 Score: {avg_map_score:.4f}")

    main()
