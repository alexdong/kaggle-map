"""LLM-based evaluator for student misconception predictions."""

from pathlib import Path

from jinja2 import Template
from loguru import logger

from kaggle_map.core.models import Category, EvaluationRow, Prediction
from kaggle_map.dataloader import load_validation_data, stratified_sample
from kaggle_map.embeddings.sampler import select_diverse_samples
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

    Args:
        response: Raw LLM response containing predictions

    Returns:
        List of up to 3 Prediction objects
    """
    predictions = []

    for raw_line in response.strip().split("\n"):
        line = raw_line.strip()
        if not line or ":" not in line:
            continue

        try:
            prediction = Prediction.from_string(line)
            predictions.append(prediction)

            max_predictions = 3
            if len(predictions) >= max_predictions:
                break
        except Exception as e:
            logger.debug(f"Failed to parse prediction line '{line}': {e}")
            continue

    # Pad with default predictions if needed
    max_predictions = 3
    while len(predictions) < max_predictions:
        predictions.append(Prediction(category=Category.FALSE_NEITHER, misconception="NA"))

    return predictions[:max_predictions]


def analyze_errors(error_cases: dict) -> None:
    """Analyze and log error patterns with diverse examples.

    Args:
        error_cases: Dictionary mapping (category, misconception) to list of errors
    """
    if not error_cases:
        return

    logger.info(f"\n{'=' * 50}")
    logger.info("Error Analysis Summary")
    logger.info(f"{'=' * 50}")

    # Sort by frequency
    sorted_errors = sorted(error_cases.items(), key=lambda x: len(x[1]), reverse=True)

    for (category, misconception), errors in sorted_errors[:10]:  # Top 10 error types
        logger.info(f"\n[{category}] {misconception}")
        logger.info(f"  Errors: {len(errors)}")

        # Select diverse examples
        if len(errors) > 1:
            explanations = [e["explanation"] for e in errors]
            n_examples = min(3, len(errors))
            diverse_indices, diversity_score = select_diverse_samples(explanations, n_samples=n_examples)

            logger.info(f"  Example explanations (diversity: {diversity_score:.2f}):")
            for idx in diverse_indices:
                error = errors[idx]
                logger.info(f"    • {error['explanation'][:100]}...")
                logger.info(f"      (Q: {error['question'][:50]}...)")
                logger.info(f"      Predicted: [{error['predicted'].category}] {error['predicted'].misconception}")
        else:
            error = errors[0]
            logger.info(f"  Example: {error['explanation'][:100]}...")
            logger.info(f"    (Q: {error['question'][:50]}...)")
            logger.info(f"    Predicted: [{error['predicted'].category}] {error['predicted'].misconception}")

    logger.info(f"\n{'=' * 50}")


def evaluate_with_llm(  # noqa: C901
    validation_path: Path = Path("datasets/33474_validation.csv"),
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

    # Track errors for analysis
    from collections import defaultdict  # noqa: PLC0415

    error_cases = defaultdict(list)  # (category, misconception) -> list of error details

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

        # Track errors if prediction was wrong
        if score < 1.0:
            # Check if ground truth was in top 3 predictions
            predicted_correctly = any(
                pred.category == ground_truth.category and pred.misconception == ground_truth.misconception
                for pred in predictions
            )

            if not predicted_correctly:
                key = (ground_truth.category, ground_truth.misconception)
                error_cases[key].append(
                    {
                        "question": eval_row.question_text,
                        "mc_answer": eval_row.mc_answer,
                        "explanation": eval_row.student_explanation,
                        "predicted": predictions[0],  # Top prediction
                        "row_id": eval_row.row_id,
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

    # Generate error analysis summary
    analyze_errors(error_cases)

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
        default=Path("datasets/33474_validation.csv"),
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
