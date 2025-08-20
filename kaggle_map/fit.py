"""Model fitting script for the Kaggle student misconception prediction competition."""

from pathlib import Path

import click
from loguru import logger
from rich.console import Console
from rich.table import Table

from kaggle_map.strategies.baseline import BaselineStrategy


def fit_model(
    train_csv_path: Path = Path("dataset/train.csv"),
    model_output_path: Path = Path("models/baseline.json"),
) -> BaselineStrategy:
    """Fit a baseline model from training data and save it.

    Args:
        train_csv_path: Path to the training CSV file
        model_output_path: Path where to save the fitted model

    Returns:
        Fitted BaselineStrategy
    """
    logger.info(f"Starting model fitting from {train_csv_path}")

    # Fit the model
    model = BaselineStrategy.fit(train_csv_path)
    logger.info("Model fitting completed successfully")

    # Display model statistics
    logger.info(f"Model trained on {len(model.correct_answers)} questions")
    logger.info(f"Found correct answers for {len(model.correct_answers)} questions")
    logger.info(
        f"Built category patterns for {len(model.category_frequencies)} questions"
    )

    misconception_count = sum(
        1 for m in model.common_misconceptions.values() if m is not None
    )
    logger.info(f"Extracted misconceptions for {misconception_count} questions")

    # Save the model
    model.save(model_output_path)
    logger.info(f"Model saved to {model_output_path}")

    return model


@click.command()
@click.option(
    "--train-path",
    "-t",
    type=click.Path(exists=True, path_type=Path),
    default=Path("dataset/train.csv"),
    help="Path to training CSV file",
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("models/baseline.json"),
    help="Path to save the fitted model",
)
def main(train_path: Path, output_path: Path) -> None:
    """Fit baseline model and save it for prediction use."""
    console = Console()

    console.print("[bold blue]🚀 Starting Model Fitting[/bold blue]")

    # Display input parameters
    params_table = Table(title="Configuration")
    params_table.add_column("Parameter", style="cyan")
    params_table.add_column("Value", style="magenta")
    params_table.add_row("Training data", str(train_path))
    params_table.add_row("Output model", str(output_path))
    console.print(params_table)

    # Fit and save the model
    with console.status("[bold green]Fitting model..."):
        fitted_model = fit_model(train_path, output_path)

    # Display results
    console.print("\n[bold green]✅ MODEL FITTING COMPLETED[/bold green]")

    results_table = Table(title="Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="magenta")

    results_table.add_row("Questions processed", str(len(fitted_model.correct_answers)))

    if output_path.exists():
        model_size_kb = output_path.stat().st_size / 1024
        results_table.add_row("Model size", f"{model_size_kb:.1f} KB")
        results_table.add_row("Status", "[green]✅ Successfully saved[/green]")
    else:
        results_table.add_row("Status", "[red]❌ File not found after saving[/red]")
        console.print("[bold red]Error: Model file not found after saving[/bold red]")
        raise click.Abort()

    console.print(results_table)
    logger.info("Model fitting script completed successfully")


if __name__ == "__main__":
    main()
