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

from kaggle_map.core.models import EvaluationRow, Prediction, default_mlp_training_config
from kaggle_map.dataloader import MAPDataset
from kaggle_map.dataloader.sampling import stratified_sample
from kaggle_map.llm.utils import build_prediction_prompt
from kaggle_map.utils.gguf_model import (
    GGUFModelInferenceConfig,
    GGUFModelName,
    GGUFModelQuantizationLevel,
    format_chat_prompt,
    get_llm_predictions,
    load_llm_model,
)
from kaggle_map.utils.logger_config import configure_logger
from kaggle_map.utils.metrics import calculate_map_at_3

# Configuration constants
# Set to -1 for unlimited generation (will generate until context window limit or stop token)
# Set to a positive value to limit response length
MAX_RESPONSE_TOKENS = -1  # Unlimited generation within context window


@dataclass
class EvaluationConfig:
    data_path: Path
    sample_ratio: float
    row_ids: list[int] | None
    template_path: Path
    model_name: GGUFModelName = GGUFModelName.GPT_OSS_20B
    quantization: GGUFModelQuantizationLevel = GGUFModelQuantizationLevel.Q2_K_L


def save_evaluation_results_to_csv(
    evaluation_results: list[dict],
    output_dir: Path = Path("logs"),
) -> Path:
    assert evaluation_results, "Cannot save empty evaluation results to CSV"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"llm_evaluation_{timestamp}.csv"

    df_data = []
    for result in evaluation_results:
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
    assert evaluation_results, "Cannot display empty evaluation results"

    console = Console()

    max_explanation_length = 80
    max_llm_labels_length = 100

    table = Table(
        title="\n📊 Detailed Evaluation Results",
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
        width=None,
    )

    table.add_column("Row ID", style="cyan", no_wrap=True)
    table.add_column("MC Answer", style="yellow", overflow="fold")
    table.add_column("Explanation", style="white", overflow="fold", max_width=max_explanation_length)
    table.add_column("Category:Misconception", style="green", overflow="fold")
    table.add_column("LLM Labels", style="dim", overflow="fold", max_width=max_llm_labels_length)
    table.add_column("MAP@3", style="red bold", justify="center")

    for result in evaluation_results:
        explanation = result["explanation"]
        if len(explanation) > max_explanation_length:
            explanation = explanation[: max_explanation_length - 3] + "..."

        llm_labels = " | ".join([str(pred) for pred in result["predictions"]])
        if len(llm_labels) > max_llm_labels_length:
            llm_labels = llm_labels[: max_llm_labels_length - 3] + "..."

        score_str = f"{result['score']:.2f}"

        category_misconception = f"{result['category']}:{result['misconception']}"

        table.add_row(
            str(result["row_id"]),
            result["mc_answer"],
            explanation,
            category_misconception,
            llm_labels,
            score_str,
        )

    console.print(table)
    console.print(f"\nTotal rows evaluated: {len(evaluation_results)}")
    avg_score = sum(r["score"] for r in evaluation_results) / len(evaluation_results) if evaluation_results else 0
    console.print(f"Average MAP@3: {avg_score:.4f}\n")


def _prepare_dataframe(validation_pairs: list) -> pd.DataFrame:
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
    if row_ids:
        logger.info(f"Filtering data to specified row IDs: {row_ids}")
        sampled_df = df[df["row_id"].isin(row_ids)]
        if len(sampled_df) != len(row_ids):
            found_ids = set(sampled_df["row_id"].tolist())
            missing_ids = set(row_ids) - found_ids
            logger.warning(f"Some row IDs not found in data: {missing_ids}")
        logger.info(f"Selected {len(sampled_df)} samples for evaluation")
        return pd.DataFrame(sampled_df)

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


def _evaluate(
    row: pd.Series,
    config: EvaluationConfig,
    llm: Llama,
) -> tuple[float, dict[str, Any]]:
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
    full_prompt = format_chat_prompt(config.model_name, user_prompt)
    logger.info(f"Prompt for row {eval_row.row_id}:\n{user_prompt}\n")

    predictions = get_llm_predictions(llm, full_prompt, config.model_name)
    logger.debug(f"Predictions for row {eval_row.row_id}: {predictions}")

    score = calculate_map_at_3(ground_truth, predictions)

    return score, {
        "row_id": eval_row.row_id,
        "mc_answer": eval_row.mc_answer,
        "explanation": eval_row.student_explanation,
        "category": ground_truth.category.value,
        "misconception": ground_truth.misconception,
        "predictions": predictions,
        "score": score,
    }


def _finalize_results(scores: list[float], evaluation_results: list[dict[str, Any]], start_time: float) -> float:
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
    assert config.data_path.exists(), f"Validation CSV not found: {config.data_path}"
    dataset_config = default_mlp_training_config().model_copy(update={"train_csv_path": config.data_path})
    dataset = MAPDataset(csv_path=config.data_path, config=dataset_config)
    assert len(dataset) > 0, "Dataset cannot be empty"
    validation_pairs = dataset.evaluation_pairs()
    logger.info(f"Loaded {len(validation_pairs)} validation samples")

    df = _prepare_dataframe(validation_pairs)
    sampled_df = _sample_dataframe(df, config.row_ids, config.sample_ratio)

    logger.info(f"Loading model: {config.model_name}")
    llm = load_llm_model(config.model_name)

    inference_config = GGUFModelInferenceConfig.get_default_config(config.model_name)
    logger.info(f"Max tokens: {inference_config.max_tokens}, Temperature: {inference_config.temperature}")

    scores = []
    evaluation_results = []

    start_time = time.time()

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
            score, result_dict = _evaluate(row, config, llm)
            scores.append(score)
            evaluation_results.append(result_dict)

            avg_score = sum(scores) / len(scores) if scores else 0.0
            progress.update(
                task,
                advance=1,
                current_map=score,
                avg_map=avg_score,
            )

    return _finalize_results(scores, evaluation_results, start_time)


if __name__ == "__main__":
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
        configure_logger(__name__, console_level="DEBUG")

        row_ids_list = None
        if row_ids:
            row_ids_list = [int(rid.strip()) for rid in row_ids.split(",")]
            logger.info(f"Will evaluate specific row IDs: {row_ids_list}")

        config = EvaluationConfig(
            template_path=template_path,
            data_path=data_path,
            sample_ratio=sample_ratio,
            row_ids=row_ids_list,
        )
        avg_map_score = evaluate_with_llm(config)

        print(f"\nFinal MAP@3 Score: {avg_map_score:.4f}")

    main()
