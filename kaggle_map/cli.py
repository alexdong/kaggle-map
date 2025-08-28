"""Command-line interface for kaggle-map prediction strategies."""

from pathlib import Path

import click
from loguru import logger
from rich.console import Console
from rich.table import Table

from .core.dataset import parse_training_data
from .strategies import get_all_strategies, get_strategy, list_strategies
from .strategies.base import Strategy
from .strategies.utils import TRAIN_RATIO, ModelParameters, get_split_indices


@click.group()
def cli() -> None:
    """Kaggle MAP student misconception prediction toolkit."""


@click.command()
@click.argument("strategy", type=click.Choice(list_strategies(), case_sensitive=False))
@click.argument("action", type=click.Choice(["fit", "eval", "predict"], case_sensitive=False))
@click.option(
    "--train-split",
    type=float,
    default=TRAIN_RATIO,
    help=f"Fraction of data for training (default: {TRAIN_RATIO})",
)
@click.option(
    "--random-seed",
    type=int,
    default=42,
    help="Random seed for reproducible results (default: 42)",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed model information")
@click.option("--model-path", type=click.Path(), help="Path to saved model file")
@click.option("--output-path", type=click.Path(), help="Path for output files")
def run(
    strategy: str,
    action: str,
    train_split: float,
    random_seed: int,
    verbose: bool,
    model_path: str | None,
    output_path: str | None,
) -> None:
    """Run a strategy with the specified action.

    STRATEGY: Name of the prediction strategy to use
    ACTION: Action to perform (fit, eval, predict)
    """
    console = Console()

    try:
        strategy_class = get_strategy(strategy)
        logger.info(f"Using strategy: {strategy} - {strategy_class}")

        if action == "fit":
            _handle_fit(strategy, strategy_class, console, train_split, random_seed, verbose, output_path)
        elif action == "eval":
            _handle_eval(strategy, strategy_class, console, train_split, random_seed, model_path, verbose)
        elif action == "predict":
            _handle_predict(strategy, strategy_class, console, model_path, output_path)

    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.Abort()
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        logger.exception(f"Failed to run {strategy} {action}")
        raise click.Abort()


@click.command()
def list_strategies_cmd() -> None:
    """List all available strategies with descriptions."""
    console = Console()
    strategies = get_all_strategies()

    assert strategies, "No strategies found in strategies/ directory"

    table = Table(title="Available Strategies")
    table.add_column("Strategy", style="cyan", no_wrap=True)
    table.add_column("Description", style="magenta")

    for strategy_name, strategy_class in sorted(strategies.items()):
        # Create instance to get description (strategies might need to be instantiated)
        try:
            # Try to get description from class if available
            if hasattr(strategy_class, "description") and isinstance(strategy_class.description, str):
                description = strategy_class.description
            else:
                # Try to create a temporary instance to get description
                temp_instance = strategy_class.__new__(strategy_class)
                description = getattr(temp_instance, "description", "Description not available")
        except Exception:
            description = "Description not available"

        table.add_row(strategy_name, description)

    console.print(table)


def _handle_fit(
    strategy: str,
    strategy_class: type[Strategy],
    console: Console,
    train_split: float,
    random_seed: int,
    verbose: bool,
    output_path: str | None,
) -> None:
    """Handle the fit action."""
    with console.status(f"[bold green]Fitting {strategy} strategy..."):
        # Fit the model with parameters
        model = strategy_class.fit(train_split=train_split, random_seed=random_seed)

    console.print(f"✅ [bold green]{strategy.title()} strategy completed successfully![/bold green]")

    # Display results if model has display methods
    if hasattr(model, "display_stats"):
        model.display_stats(console)

    if verbose and hasattr(model, "display_detailed_info"):
        model.display_detailed_info(console)

    if hasattr(model, "demonstrate_predictions"):
        model.demonstrate_predictions(console)

    # Prepare save path - MLP uses .pkl, others use .json
    if output_path:
        model_path = Path(output_path)
    else:
        model_path = Path(f"models/{strategy}.pkl") if strategy == "mlp" else Path(f"models/{strategy}.json")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Create model parameters for saving
    # Get the actual data size from the training data
    all_data = parse_training_data(Path("datasets/train.csv"))
    train_indices, val_indices, test_indices = get_split_indices(
        len(all_data), train_ratio=train_split, random_seed=random_seed
    )

    ModelParameters.create(
        train_split=train_split,
        random_seed=random_seed,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        total_samples=len(all_data),
    )

    # Save model with parameters
    model.save(model_path)

    console.print(f"Model trained and saved to {model_path}")
    console.print(f"Training split: {train_split}, Random seed: {random_seed}")


def _handle_eval(
    strategy: str,
    strategy_class: type[Strategy],
    console: Console,
    train_split: float,
    random_seed: int,
    model_path: str | None,
    verbose: bool,
) -> None:
    """Handle the eval action."""
    # Determine model path
    if model_path:
        model_file = Path(model_path)
    else:
        # Try both .json and .pkl extensions
        model_file = Path(f"models/{strategy}.json")
        if not model_file.exists():
            model_file = Path(f"models/{strategy}.pkl")

    # Check if model file exists
    if model_file.exists():
        with console.status(f"[bold blue]Loading {strategy} model..."):
            # Load model and parameters
            model = strategy_class.load(model_file)
            if hasattr(model, "parameters") and model.parameters:
                params = model.parameters
                console.print(f"Using saved parameters: train_split={params.train_split}, random_seed={params.random_seed}")
                train_split = params.train_split
                random_seed = params.random_seed
        console.print(f"✅ [bold green]Loaded {strategy} model from {model_file}[/bold green]")
    # For MLP, try to use checkpoint if no model file exists
    elif strategy == "mlp":
        console.print("Model file not found, looking for checkpoints...")
        model = None  # Will trigger checkpoint loading in evaluate_on_split
    else:
        console.print(f"[bold red]Model file not found: {model_file}[/bold red]")
        console.print(f"[yellow]Hint: Run '{strategy} fit' first, or specify --model-path[/yellow]")
        raise click.Abort()

    assert hasattr(strategy_class, "evaluate_on_split"), f"Strategy {strategy} does not support evaluation"

    eval_results = strategy_class.evaluate_on_split(model, train_split=train_split, random_seed=random_seed)

    console.print("\n[bold]Evaluation results:[/bold]")
    for key, value in eval_results.items():
        if isinstance(value, float):
            console.print(f"  {key}: {value:.4f}")
        else:
            console.print(f"  {key}: {value}")


def _handle_predict(strategy: str, strategy_class, console: Console, model_path: str | None, output_path: str | None) -> None:
    """Handle the predict action."""
    # Determine model path
    model_file = Path(model_path) if model_path else Path(f"models/{strategy}.json")
    if not model_file.exists() and strategy == "mlp":
        model_file = Path(f"models/{strategy}.pkl")

    if not model_file.exists():
        console.print(f"[bold red]Model file not found: {model_file}[/bold red]")
        console.print(f"[yellow]Hint: Run '{strategy} fit' first, or specify --model-path[/yellow]")
        raise click.Abort()

    with console.status(f"[bold blue]Loading {strategy} model..."):
        strategy_class.load(model_file)

    console.print(f"✅ [bold green]Loaded {strategy} model from {model_file}[/bold green]")

    # TODO: Implement prediction logic for test data
    console.print("[yellow]Prediction functionality not yet implemented[/yellow]")
    console.print("[dim]This would generate predictions for test.csv[/dim]")


# Add commands to CLI group
cli.add_command(run)
cli.add_command(list_strategies_cmd, name="list-strategies")


if __name__ == "__main__":
    cli()
