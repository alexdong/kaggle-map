"""Command-line interface for kaggle-map prediction strategies."""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from kaggle_map.core.dataset import load_training_data
from kaggle_map.core.models import EvaluationRow
from kaggle_map.mlp import Predictor
from kaggle_map.mlp.trainer import TrainingConfig


@click.group()
def cli() -> None:
    """Kaggle MAP student misconception prediction toolkit."""


@click.command()
@click.argument("strategy", type=click.Choice(["mlp"], case_sensitive=False))
@click.argument("action", type=click.Choice(["fit", "eval", "predict"], case_sensitive=False))
@click.option("--train-split", type=float, default=0.7, help="Fraction of data for training (default: 0.7)")
@click.option("--random-seed", type=int, default=42, help="Random seed for reproducible results (default: 42)")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed model information")
@click.option("--model-path", type=click.Path(), help="Path to saved model file")
@click.option("--output-path", type=click.Path(), help="Path for output files")
@click.option("--train-data", type=click.Path(exists=True), help="Path to training data CSV")
@click.option("--epochs", type=int, default=50, help="Number of training epochs")
@click.option("--batch-size", type=int, default=256, help="Batch size for training")
@click.option("--learning-rate", type=float, default=1e-4, help="Learning rate")
def run(
    strategy: str,
    action: str,
    train_split: float,
    random_seed: int,
    verbose: bool,
    model_path: str | None,
    output_path: str | None,
    train_data: str | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    """Run a strategy with the specified action.

    STRATEGY: Name of the prediction strategy to use (currently only 'mlp')
    ACTION: Action to perform (fit, eval, predict)
    """
    console = Console()

    if strategy != "mlp":
        console.print(f"[red]Unknown strategy: {strategy}[/red]")
        return

    # Set up paths
    train_csv_path = Path(train_data) if train_data else Path("datasets/train.csv")
    model_filepath = Path(model_path) if model_path else Path(f"models/{strategy}.pkl")
    output_filepath = Path(output_path) if output_path else Path(f"outputs/{strategy}_output.csv")

    if action == "fit":
        console.print(f"\n[bold blue]Training {strategy.upper()} Strategy[/bold blue]")
        console.print(f"Training data: {train_csv_path}")
        console.print(f"Train split: {train_split:.1%}")
        console.print(f"Random seed: {random_seed}")
        console.print(f"Epochs: {epochs}")
        console.print(f"Batch size: {batch_size}")
        console.print(f"Learning rate: {learning_rate}")

        # Create training config
        config = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            train_split=train_split,
            random_seed=random_seed,
            train_csv_path=train_csv_path,
        )

        # Train model
        predictor = Predictor.fit(config)

        # Save model
        console.print(f"\n[green]Saving model to {model_filepath}[/green]")
        predictor.save(model_filepath)

        # Evaluate on validation set
        metrics = predictor.evaluate()
        console.print("\n[bold green]Training Complete![/bold green]")
        console.print(f"Validation MAP@3: {metrics['validation_map@3']:.4f}")

    elif action == "eval":
        console.print(f"\n[bold blue]Evaluating {strategy.upper()} Strategy[/bold blue]")

        if not model_filepath.exists():
            console.print(f"[red]Model file not found: {model_filepath}[/red]")
            console.print("Please train a model first using 'fit' action")
            return

        # Load model
        predictor = Predictor.load(model_filepath)

        # Evaluate
        metrics = predictor.evaluate(train_csv_path=train_csv_path)

        # Display results
        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("MAP@3", f"{metrics['validation_map@3']:.4f}")
        table.add_row("Validation Samples", str(metrics["validation_samples"]))
        console.print(table)

    elif action == "predict":
        console.print(f"\n[bold blue]Generating Predictions with {strategy.upper()} Strategy[/bold blue]")

        if not model_filepath.exists():
            console.print(f"[red]Model file not found: {model_filepath}[/red]")
            console.print("Please train a model first using 'fit' action")
            return

        # Load model
        predictor = Predictor.load(model_filepath)

        # Load test data (for now, use validation split as example)
        training_data = load_training_data(train_csv_path)

        # Make predictions on first 10 samples as example
        console.print("\n[yellow]Generating predictions for first 10 samples...[/yellow]")

        predictions = []
        for row in training_data[:10]:
            eval_row = EvaluationRow(
                row_id=row.row_id,
                question_id=row.question_id,
                question_text=row.question_text,
                mc_answer=row.mc_answer,
                student_explanation=row.student_explanation,
            )
            submission = predictor.predict(eval_row)
            predictions.append(submission)

        console.print(f"[green]Generated {len(predictions)} predictions[/green]")
        console.print(f"Output would be saved to: {output_filepath}")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")


@click.command()
def list_strategies() -> None:
    """List all available prediction strategies."""
    console = Console()

    table = Table(title="Available Strategies")
    table.add_column("Strategy", style="cyan")
    table.add_column("Description", style="white")

    table.add_row("mlp", "Multi-layer perceptron with question-specific heads")

    console.print(table)


# Register commands
cli.add_command(run)
cli.add_command(list_strategies)


if __name__ == "__main__":
    cli()
