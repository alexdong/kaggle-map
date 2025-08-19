"""Command-line interface for kaggle-map prediction strategies."""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import click
from loguru import logger
from rich.console import Console
from rich.table import Table

from .strategies import get_all_strategies, get_strategy, list_strategies
from .strategies.base import Strategy


@click.group()
def cli() -> None:
    """Kaggle MAP student misconception prediction toolkit."""


@click.command()
@click.argument("strategy", type=click.Choice(list_strategies(), case_sensitive=False))
@click.argument(
    "action", type=click.Choice(["fit", "eval", "predict"], case_sensitive=False)
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed model information")
@click.option("--model-path", type=click.Path(), help="Path to saved model file")
@click.option("--output-path", type=click.Path(), help="Path for output files")
@click.option(
    "--train-split",
    type=float,
    default=0.8,
    help="Fraction of data for training (default: 0.8)",
)
@click.option(
    "--random-seed",
    type=int,
    default=42,
    help="Random seed for reproducible results (default: 42)",
)
@click.option(
    "--embeddings-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to pre-computed embeddings .npz file (optional)",
)
def run(
    strategy: str,
    action: str,
    *,
    verbose: bool,
    model_path: str | None,
    output_path: str | None,
    train_split: float,
    random_seed: int,
    embeddings_path: Path | None,
) -> None:
    """Run a strategy with the specified action.

    STRATEGY: Name of the prediction strategy to use
    ACTION: Action to perform (fit, eval, predict)
    """
    console = Console()

    try:
        strategy_class = get_strategy(strategy)
        logger.info(f"Using strategy: {strategy} - {strategy_class}")

        handlers = {
            "fit": lambda: _handle_fit(
                strategy,
                strategy_class,
                console,
                verbose=verbose,
                output_path=output_path,
                train_split=train_split,
                random_seed=random_seed,
                embeddings_path=embeddings_path,
            ),
            "eval": lambda: _handle_eval(
                strategy,
                strategy_class,
                console,
                model_path=model_path,
                train_split=train_split,
                random_seed=random_seed,
            ),
            "predict": lambda: _handle_predict(
                strategy,
                strategy_class,
                console,
                model_path=model_path,
                output_path=output_path,
            ),
        }

        handlers[action]()

    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.Abort() from e
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        logger.exception(f"Failed to run {strategy} {action}")
        raise click.Abort() from e


@click.command()
def list_strategies_cmd() -> None:
    """List all available strategies with descriptions."""
    console = Console()

    strategies = get_all_strategies()

    if not strategies:
        console.print("[yellow]No strategies found in strategies/ directory[/yellow]")
        return

    table = Table(title="Available Strategies")
    table.add_column("Strategy", style="cyan", no_wrap=True)
    table.add_column("Description", style="magenta")

    for strategy_name, strategy_class in sorted(strategies.items()):
        # Create instance to get description (strategies might need to be instantiated)
        try:
            # Try to get description from class if available
            if hasattr(strategy_class, "description") and isinstance(
                strategy_class.description, str
            ):
                description = strategy_class.description
            else:
                # Try to create a temporary instance to get description
                temp_instance = strategy_class.__new__(strategy_class)
                description = temp_instance.description
        except Exception:
            description = "Description not available"

        table.add_row(strategy_name, description)

    console.print(table)


def _handle_fit(
    strategy: str,
    strategy_class: type[Strategy],
    console: Console,
    *,
    verbose: bool,
    output_path: str | None,
    train_split: float,
    random_seed: int,
    embeddings_path: Path | None,
) -> None:
    """Handle the fit action."""
    console.print(
        f"[bold blue]Training {strategy} strategy with {train_split:.0%} of data (seed: {random_seed})[/bold blue]"
    )

    with console.status(f"[bold green]Fitting {strategy} strategy..."):
        # Check if strategy supports enhanced parameters
        fit_method = strategy_class.fit
        if (
            hasattr(fit_method, "__code__")
            and "train_split" in fit_method.__code__.co_varnames
        ):
            # Build kwargs based on what the strategy supports
            fit_kwargs = {"train_split": train_split, "random_seed": random_seed}
            if (
                "embeddings_path" in fit_method.__code__.co_varnames
                and embeddings_path is not None
            ):
                fit_kwargs["embeddings_path"] = embeddings_path
                console.print(
                    f"[dim]Using pre-computed embeddings from {embeddings_path}[/dim]"
                )
            model = strategy_class.fit(**fit_kwargs)  # type: ignore[call-arg]
        else:
            # Fallback for strategies that don't support split
            model = strategy_class.fit()

    console.print(
        f"✅ [bold green]{strategy.title()} strategy completed successfully![/bold green]"
    )

    # Display results
    model.display_stats(console)

    if verbose:
        model.display_detailed_info(console)

    model.demonstrate_predictions(console)

    # Save model
    model_path = Path(output_path) if output_path else Path(f"{strategy}_model.pkl")

    model.save(model_path)
    console.print(f"✅ [bold green]Model saved to {model_path}[/bold green]")


def _handle_eval(
    strategy: str,
    strategy_class: type[Strategy],
    console: Console,
    *,
    model_path: str | None,
    train_split: float,
    random_seed: int,  # noqa: ARG001
) -> None:
    """Handle the eval action."""
    # Determine model path
    model_file = Path(model_path) if model_path else Path(f"{strategy}_model.pkl")

    if not model_file.exists():
        console.print(f"[bold red]Model file not found: {model_file}[/bold red]")
        console.print(
            f"[yellow]Hint: Run '{strategy} fit --train-split {train_split}' first, or specify --model-path[/yellow]"
        )
        raise click.Abort()

    with console.status(f"[bold blue]Loading {strategy} model..."):
        model = strategy_class.load(model_file)

    console.print(
        f"✅ [bold green]Loaded {strategy} model from {model_file}[/bold green]"
    )

    # Run evaluation if the strategy supports it
    if hasattr(strategy_class, "evaluate_on_split"):
        try:
            with console.status(f"[bold blue]Evaluating {strategy} model..."):
                eval_results = strategy_class.evaluate_on_split(model)  # type: ignore[misc]

            console.print("✅ [bold green]Evaluation completed![/bold green]")

            # Display evaluation results
            _display_eval_results(console, eval_results)

            # Update performance history
            _update_performance_history(strategy, eval_results)

        except ValueError as e:
            console.print(f"[bold red]Evaluation error:[/bold red] {e}")
            console.print(
                f"[yellow]Hint: Train with --train-split {train_split} to create validation split[/yellow]"
            )
    else:
        console.print("[yellow]Strategy does not support evaluation[/yellow]")
        console.print(
            "[dim]Only available for strategies with evaluate_on_split method[/dim]"
        )


def _handle_predict(
    strategy: str,
    strategy_class: type[Strategy],
    console: Console,
    *,
    model_path: str | None,
    output_path: str | None,
) -> None:
    """Handle the predict action."""
    # Determine model path
    model_file = Path(model_path) if model_path else Path(f"{strategy}_model.json")

    if not model_file.exists():
        console.print(f"[bold red]Model file not found: {model_file}[/bold red]")
        console.print(
            f"[yellow]Hint: Run '{strategy} fit' first, or specify --model-path[/yellow]"
        )
        raise click.Abort()

    with console.status(f"[bold blue]Loading {strategy} model..."):
        strategy_class.load(model_file)

    console.print(
        f"✅ [bold green]Loaded {strategy} model from {model_file}[/bold green]"
    )

    # TODO: Implement prediction logic for test data
    # Suppress unused-arg warning until implemented
    _ = output_path
    console.print("[yellow]Prediction functionality not yet implemented[/yellow]")
    console.print("[dim]This would generate predictions for test.csv[/dim]")


def _display_eval_results(console: Console, eval_results: dict[str, float]) -> None:
    """Display evaluation results in a formatted table."""
    eval_table = Table(title="Evaluation Results")
    eval_table.add_column("Metric", style="cyan", no_wrap=True)
    eval_table.add_column("Value", style="magenta")
    eval_table.add_column("Description", style="dim")

    eval_table.add_row(
        "MAP@3 Score", f"{eval_results['map_score']:.4f}", "Mean Average Precision at 3"
    )
    eval_table.add_row(
        "Total Observations",
        str(eval_results["total_observations"]),
        "Validation set size",
    )
    eval_table.add_row(
        "Perfect Predictions",
        f"{eval_results['perfect_predictions']} ({eval_results['accuracy']:.1%})",
        "Correct in 1st position",
    )

    console.print(eval_table)

    # Performance assessment thresholds
    excellent_threshold = 0.8
    good_threshold = 0.6
    moderate_threshold = 0.4

    map_score = eval_results["map_score"]
    if map_score >= excellent_threshold:
        console.print(
            "\n🎉 [bold green]Excellent performance![/bold green] Model shows strong validation results."
        )
    elif map_score >= good_threshold:
        console.print(
            "\n✅ [bold blue]Good performance.[/bold blue] Model shows reasonable validation results."
        )
    elif map_score >= moderate_threshold:
        console.print(
            "\n⚠️  [bold yellow]Moderate performance.[/bold yellow] Model has room for improvement."
        )
    else:
        console.print(
            "\n⚠️  [bold red]Poor performance.[/bold red] Model needs significant improvement."
        )


def _update_performance_history(strategy: str, eval_results: dict[str, float]) -> None:
    """Update performance_history.json with evaluation results."""
    performance_file = Path("performance_history.json")

    # Load existing history
    performance_history = []
    if performance_file.exists():
        with performance_file.open() as f:
            performance_history = json.load(f)

    # Get current git commit hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        commit_hash = result.stdout.strip()
    except subprocess.CalledProcessError:
        commit_hash = "unknown"

    # Create new entry
    timestamp = datetime.now(UTC).isoformat()
    new_entry = {
        "timestamp": timestamp,
        "commit_hash": commit_hash,
        "strategy": strategy,
        "map_score": eval_results["map_score"],
        "total_observations": eval_results["total_observations"],
        "perfect_predictions": eval_results["perfect_predictions"],
        "total_execution_time": 0.0,  # CLI doesn't track execution time
    }

    # Add new entry and sort by score (best first)
    performance_history.append(new_entry)
    performance_history.sort(key=lambda x: x["map_score"], reverse=True)

    # Save updated history
    with performance_file.open("w") as f:
        json.dump(performance_history, f, indent=2)

    logger.info(
        f"Updated performance history: MAP@3={eval_results['map_score']:.4f} for {strategy}"
    )


# Add commands to CLI group
cli.add_command(run)
cli.add_command(list_strategies_cmd, name="list-strategies")


if __name__ == "__main__":
    cli()
