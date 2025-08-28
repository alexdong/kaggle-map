"""Cross-validation implementation for model evaluation on original training data."""

import time
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import KFold

from kaggle_map.eval import evaluate
from kaggle_map.models import TestRow
from kaggle_map.strategies import get_strategy


def prepare_fold_data(
    train_df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray
) -> tuple[Path, list[TestRow], pd.DataFrame]:
    """Prepare training and validation data for a single fold.

    Args:
        train_df: Full training dataframe
        train_idx: Indices for training samples in this fold
        val_idx: Indices for validation samples in this fold

    Returns:
        Tuple of (temp_train_path, validation_test_rows, validation_ground_truth)
    """
    # Split data
    fold_train_df = train_df.iloc[train_idx].copy()
    fold_val_df = train_df.iloc[val_idx].copy()

    # Save fold training data to temporary file
    temp_train_path = Path("dataset/cv_train_fold.csv")
    fold_train_df.to_csv(temp_train_path, index=False)

    # Convert validation data to TestRow format
    val_test_rows = []
    for _, row in fold_val_df.iterrows():
        test_row = TestRow(
            row_id=int(row["row_id"]),
            question_id=int(row["QuestionId"]),
            question_text=str(row["QuestionText"]),
            mc_answer=str(row["MC_Answer"]),
            student_explanation=str(row["StudentExplanation"]),
        )
        val_test_rows.append(test_row)

    # Prepare validation ground truth
    val_ground_truth = fold_val_df[["row_id", "Category", "Misconception"]].copy()

    logger.debug(f"Fold: {len(fold_train_df)} train, {len(fold_val_df)} validation samples")

    return temp_train_path, val_test_rows, val_ground_truth


def run_cross_validation(
    strategy_name: str,
    train_path: Path,
    n_folds: int = 5,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run k-fold cross-validation for a strategy.

    Args:
        strategy_name: Name of the strategy to evaluate
        train_path: Path to training CSV file
        n_folds: Number of cross-validation folds
        verbose: Whether to show detailed output

    Returns:
        Dictionary with CV results and statistics
    """
    console = Console()

    console.print(f"\n[bold blue]Running {n_folds}-Fold Cross-Validation[/bold blue]")
    console.print(f"Strategy: [cyan]{strategy_name}[/cyan]")
    console.print(f"Training data: [cyan]{train_path}[/cyan]")

    # Load full training data
    train_df = pd.read_csv(train_path)
    logger.info(f"Loaded {len(train_df)} training samples from {train_path}")

    # Remove any rows with missing critical values
    initial_len = len(train_df)
    train_df = train_df.dropna(subset=["row_id", "QuestionId", "Category"])
    if len(train_df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(train_df)} rows with missing values")

    # Initialize k-fold splitter
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_scores = []
    fold_times = []
    perfect_predictions_per_fold = []

    # Run cross-validation
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_df), 1):
        console.print(f"\n[bold green]Fold {fold_idx}/{n_folds}[/bold green]")

        with console.status(f"[bold]Processing fold {fold_idx}..."):
            fold_start = time.time()

            # Prepare fold data
            temp_train_path, val_test_rows, val_ground_truth = prepare_fold_data(
                train_df, train_idx, val_idx
            )

            try:
                # Fit strategy on fold training data
                strategy_class = get_strategy(strategy_name)
                model = strategy_class.fit(temp_train_path)

                # Make predictions on validation data
                predictions = model.predict(val_test_rows)

                # Save predictions and ground truth for evaluation
                temp_submission_path = Path("dataset/cv_submission.csv")
                temp_ground_truth_path = Path("dataset/cv_ground_truth.csv")

                # Format submission data
                submission_data = []
                for pred_row in predictions:
                    pred_strings = [p.value for p in pred_row.predicted_categories]
                    submission_data.append({
                        "row_id": pred_row.row_id,
                        "predictions": " ".join(pred_strings),
                    })

                pd.DataFrame(submission_data).to_csv(temp_submission_path, index=False)
                val_ground_truth.to_csv(temp_ground_truth_path, index=False)

                # Evaluate fold
                result = evaluate(temp_ground_truth_path, temp_submission_path)

                fold_time = time.time() - fold_start
                fold_scores.append(result.map_score)
                fold_times.append(fold_time)
                perfect_predictions_per_fold.append(result.perfect_predictions)

                console.print(
                    f"  MAP@3: [magenta]{result.map_score:.4f}[/magenta] | "
                    f"Perfect: [cyan]{result.perfect_predictions}/{result.total_observations}[/cyan] | "
                    f"Time: [yellow]{fold_time:.1f}s[/yellow]"
                )

            finally:
                # Clean up temporary files
                for temp_file in [
                    temp_train_path,
                    temp_submission_path,
                    temp_ground_truth_path,
                ]:
                    if temp_file.exists():
                        temp_file.unlink()

            logger.debug(f"Fold {fold_idx} completed: MAP@3={result.map_score:.4f}")

    # Calculate statistics
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    min_score = np.min(fold_scores)
    max_score = np.max(fold_scores)
    total_time = sum(fold_times)

    # Display results
    console.print("\n[bold green]Cross-Validation Complete![/bold green]\n")

    # Summary table
    summary_table = Table(title="Cross-Validation Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="magenta")

    summary_table.add_row("Mean MAP@3", f"{mean_score:.4f}")
    summary_table.add_row("Std Dev", f"{std_score:.4f}")
    summary_table.add_row("Min Score", f"{min_score:.4f}")
    summary_table.add_row("Max Score", f"{max_score:.4f}")
    summary_table.add_row("95% CI", f"[{mean_score - 1.96*std_score:.4f}, {mean_score + 1.96*std_score:.4f}]")
    summary_table.add_row("Total Time", f"{total_time:.1f}s")

    console.print(summary_table)

    if verbose:
        # Detailed fold results
        fold_table = Table(title="Fold-by-Fold Results")
        fold_table.add_column("Fold", style="yellow")
        fold_table.add_column("MAP@3", style="magenta")
        fold_table.add_column("Perfect", style="cyan")
        fold_table.add_column("Time (s)", style="green")

        for i, (score, perfect, fold_time) in enumerate(
            zip(fold_scores, perfect_predictions_per_fold, fold_times, strict=False), 1
        ):
            fold_table.add_row(
                str(i),
                f"{score:.4f}",
                str(perfect),
                f"{fold_time:.1f}",
            )

        console.print("\n")
        console.print(fold_table)

    # Performance assessment
    excellent_threshold = 0.8
    good_threshold = 0.6
    moderate_threshold = 0.4

    console.print("\n[bold]Performance Assessment:[/bold]")
    if mean_score >= excellent_threshold:
        console.print("🎉 [bold green]Excellent performance![/bold green]")
    elif mean_score >= good_threshold:
        console.print("✅ [bold blue]Good performance[/bold blue]")
    elif mean_score >= moderate_threshold:
        console.print("⚠️  [bold yellow]Moderate performance[/bold yellow]")
    else:
        console.print("❌ [bold red]Poor performance - needs improvement[/bold red]")

    return {
        "mean_score": mean_score,
        "std_score": std_score,
        "min_score": min_score,
        "max_score": max_score,
        "fold_scores": fold_scores,
        "fold_times": fold_times,
        "total_time": total_time,
    }


@click.command()
@click.option(
    "--strategy",
    "-s",
    default="baseline",
    help="Strategy to evaluate (baseline, probabilistic)",
)
@click.option(
    "--data",
    "-d",
    type=click.Choice(["original", "synth", "train"], case_sensitive=False),
    default="original",
    help="Dataset to use for cross-validation",
)
@click.option(
    "--folds",
    "-k",
    default=5,
    type=int,
    help="Number of cross-validation folds",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed fold-by-fold results",
)
def main(strategy: str, data: str, folds: int, *, verbose: bool) -> None:
    """Run cross-validation for model evaluation.

    This script performs k-fold cross-validation on the specified dataset
    using the selected strategy.
    """
    console = Console()

    # Determine data path
    data_paths = {
        "original": Path("dataset/train_original.csv"),
        "synth": Path("dataset/synth.csv"),
        "train": Path("dataset/train.csv"),
    }

    train_path = data_paths[data.lower()]

    if not train_path.exists():
        console.print(f"[bold red]Error: Dataset not found at {train_path}[/bold red]")
        return

    console.print("\n[bold]Kaggle MAP Cross-Validation[/bold]")
    console.print("=" * 50)

    try:
        results = run_cross_validation(
            strategy_name=strategy,
            train_path=train_path,
            n_folds=folds,
            verbose=verbose,
        )

        # Log final results
        logger.info(
            f"CV completed for {strategy} on {data}: "
            f"Mean MAP@3={results['mean_score']:.4f} ± {results['std_score']:.4f}"
        )

    except Exception as e:
        console.print(f"[bold red]Error during cross-validation: {e}[/bold red]")
        logger.exception("Cross-validation failed")
        raise


if __name__ == "__main__":
    main()
