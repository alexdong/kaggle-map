"""Command-line interface for kaggle-map prediction strategies."""

from dataclasses import dataclass
from pathlib import Path

import click
from loguru import logger
from rich.console import Console
from rich.table import Table

from .core.dataset import parse_training_data
from .strategies import get_all_strategies, get_strategy, list_strategies
from .strategies.base import Strategy
from .strategies.utils import TRAIN_RATIO, ModelParameters, get_split_indices


@dataclass
class CLIParams:
    """CLI parameters for strategy operations."""
    strategy: str
    strategy_class: type[Strategy]
    console: Console
    train_split: float
    random_seed: int
    train_csv_path: Path
    verbose: bool = False
    model_path: str | None = None
    output_path: str | None = None


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
@click.option(
    "--train-data",
    type=click.Path(exists=True),
    help="Path to training data CSV (default: datasets/train.csv)"
)
def run(
    strategy: str,
    action: str,
    train_split: float,
    random_seed: int,
    *,
    verbose: bool,
    model_path: str | None,
    output_path: str | None,
    train_data: str | None,
) -> None:
    """Run a strategy with the specified action.

    STRATEGY: Name of the prediction strategy to use
    ACTION: Action to perform (fit, eval, predict)
    """
    console = Console()

    try:
        strategy_class = get_strategy(strategy)
        logger.info(f"Using strategy: {strategy} - {strategy_class}")

        # Convert train_data to Path if provided
        train_csv_path = Path(train_data) if train_data else Path("datasets/train.csv")

        _dispatch_action(
            action=action,
            strategy=strategy,
            strategy_class=strategy_class,
            console=console,
            train_split=train_split,
            random_seed=random_seed,
            verbose=verbose,
            model_path=model_path,
            output_path=output_path,
            train_csv_path=train_csv_path,
        )

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


def _dispatch_action(action: str, params: CLIParams) -> None:
    """Dispatch the action to the appropriate handler."""
    if action == "fit":
        _handle_fit(params)
    elif action == "eval":
        _handle_eval(params)
    elif action == "predict":
        _handle_predict(params)
    else:
        msg = f"Unknown action: {action}"
        raise ValueError(msg)


def _handle_fit(params: CLIParams) -> None:
    """Handle the fit action."""
    model = _train_model(params)
    _display_model_info(model, params.console, verbose=params.verbose)
    _save_trained_model(model, params)


def _train_model(params: CLIParams) -> Strategy:
    """Train the model with given parameters."""
    status_msg = f"[bold green]Fitting {params.strategy} strategy using {params.train_csv_path}..."
    with params.console.status(status_msg):
        model = params.strategy_class.fit(
            train_split=params.train_split,
            random_seed=params.random_seed,
            train_csv_path=params.train_csv_path
        )
    success_msg = f"✅ [bold green]{params.strategy.title()} strategy completed successfully![/bold green]"
    params.console.print(success_msg)
    return model


def _display_model_info(model: Strategy, console: Console, *, verbose: bool) -> None:
    """Display model information and statistics."""
    if hasattr(model, "display_stats"):
        model.display_stats(console)

    if verbose and hasattr(model, "display_detailed_info"):
        model.display_detailed_info(console)

    if hasattr(model, "demonstrate_predictions"):
        model.demonstrate_predictions(console)


def _save_trained_model(model: Strategy, params: CLIParams) -> Path:
    """Save the trained model and return the save path."""
    # Prepare save path - MLP uses .pkl, others use .json
    if params.output_path:
        model_path = Path(params.output_path)
    else:
        ext = "pkl" if params.strategy == "mlp" else "json"
        model_path = Path(f"models/{params.strategy}.{ext}")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Create model parameters for saving
    all_data = parse_training_data(params.train_csv_path)
    train_indices, val_indices, test_indices = get_split_indices(
        len(all_data), train_ratio=params.train_split, random_seed=params.random_seed
    )

    ModelParameters.create(
        train_split=params.train_split,
        random_seed=params.random_seed,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        total_samples=len(all_data),
    )

    # Save model with parameters
    model.save(model_path)

    params.console.print(f"Model trained and saved to {model_path}")
    params.console.print(f"Training split: {params.train_split}, Random seed: {params.random_seed}")
    return model_path


def _handle_eval(params: CLIParams) -> None:
    """Handle the eval action."""
    model, updated_params = _load_model_for_eval(params)
    _run_evaluation(model, updated_params)


def _load_model_for_eval(
    strategy: str,
    strategy_class: type[Strategy],
    console: Console,
    model_path: str | None,
    train_split: float,
    random_seed: int,
) -> tuple[Strategy | None, float, int]:
    """Load model for evaluation and return updated parameters."""
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
        model, updated_train_split, updated_random_seed = _load_existing_model(
            strategy, strategy_class, console, model_file, train_split, random_seed
        )
        return model, updated_train_split, updated_random_seed
    # For MLP, try to use checkpoint if no model file exists
    if strategy == "mlp":
        console.print("Model file not found, looking for checkpoints...")
        return None, train_split, random_seed  # Will trigger checkpoint loading in evaluate_on_split
    console.print(f"[bold red]Model file not found: {model_file}[/bold red]")
    console.print(f"[yellow]Hint: Run '{strategy} fit' first, or specify --model-path[/yellow]")
    raise click.Abort()


def _load_existing_model(
    strategy: str,
    strategy_class: type[Strategy],
    console: Console,
    model_file: Path,
    train_split: float,
    random_seed: int,
) -> tuple[Strategy, float, int]:
    """Load an existing model and extract parameters."""
    with console.status(f"[bold blue]Loading {strategy} model..."):
        loaded_result = strategy_class.load(model_file)

    # Handle different return types from load method
    if isinstance(loaded_result, tuple):
        model, params = loaded_result
        updated_train_split, updated_random_seed = _extract_params_from_tuple(
            params, console, train_split, random_seed
        )
    else:
        model = loaded_result
        updated_train_split, updated_random_seed = _extract_params_from_model(
            model, console, train_split, random_seed
        )

    console.print(f"✅ [bold green]Loaded {strategy} model from {model_file}[/bold green]")
    return model, updated_train_split, updated_random_seed


def _extract_params_from_tuple(
    params: ModelParameters | None,
    console: Console,
    train_split: float,
    random_seed: int,
) -> tuple[float, int]:
    """Extract parameters from tuple result."""
    if params:
        console.print(
            f"Using saved parameters: train_split={params.train_split}, "
            f"random_seed={params.random_seed}"
        )
        return params.train_split, params.random_seed
    return train_split, random_seed


def _extract_params_from_model(
    model: Strategy,
    console: Console,
    train_split: float,
    random_seed: int,
) -> tuple[float, int]:
    """Extract parameters from model object."""
    if hasattr(model, "parameters") and model.parameters:
        params = model.parameters
        console.print(
            f"Using saved parameters: train_split={params.train_split}, "
            f"random_seed={params.random_seed}"
        )
        return params.train_split, params.random_seed
    return train_split, random_seed


def _run_evaluation(
    strategy: str,
    strategy_class: type[Strategy],
    model: Strategy | None,
    console: Console,
    train_split: float,
    random_seed: int,
    train_csv_path: Path,
) -> None:
    """Run evaluation and display results."""
    assert hasattr(strategy_class, "evaluate_on_split"), (
        f"Strategy {strategy} does not support evaluation"
    )

    eval_results = strategy_class.evaluate_on_split(
        model, train_split=train_split, random_seed=random_seed, train_csv_path=train_csv_path
    )

    console.print("\n[bold]Evaluation results:[/bold]")
    for key, value in eval_results.items():
        if isinstance(value, float):
            console.print(f"  {key}: {value:.4f}")
        else:
            console.print(f"  {key}: {value}")


def _handle_predict(params: CLIParams) -> None:
    """Handle the predict action."""
    model_file = _determine_model_file_for_predict(params)

    with params.console.status(f"[bold blue]Loading {params.strategy} model..."):
        params.strategy_class.load(model_file)

    params.console.print(f"✅ [bold green]Loaded {params.strategy} model from {model_file}[/bold green]")

    # TODO: Implement prediction logic for test data
    params.console.print("[yellow]Prediction functionality not yet implemented[/yellow]")
    params.console.print("[dim]This would generate predictions for test.csv[/dim]")


def _determine_model_file_for_predict(
    strategy: str, model_path: str | None, console: Console
) -> Path:
    """Determine the model file path for prediction."""
    model_file = Path(model_path) if model_path else Path(f"models/{strategy}.json")
    if not model_file.exists() and strategy == "mlp":
        model_file = Path(f"models/{strategy}.pkl")

    if not model_file.exists():
        console.print(f"[bold red]Model file not found: {model_file}[/bold red]")
        console.print(f"[yellow]Hint: Run '{strategy} fit' first, or specify --model-path[/yellow]")
        raise click.Abort()

    return model_file


# Add commands to CLI group
cli.add_command(run)
cli.add_command(list_strategies_cmd, name="list-strategies")


if __name__ == "__main__":
    cli()
