"""Command-line interface for kaggle-map prediction strategies."""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .core.dataset import parse_training_data
from .strategies import get_all_strategies, get_strategy, list_strategies
from .strategies.base import Strategy
from .strategies.utils import TRAIN_RATIO, ModelParameters, get_split_indices


@click.group()
def cli() -> None:
    pass


@click.command()
@click.argument("strategy", type=click.Choice(list_strategies(), case_sensitive=False))
@click.argument("action", type=click.Choice(["fit", "eval"], case_sensitive=False))
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
def run(
    strategy: str,
    action: str,
    train_split: float,
    random_seed: int,
) -> None:
    strategy_class = get_strategy(strategy)

    if action == "fit":
        _handle_fit(strategy, strategy_class, train_split, random_seed)
    elif action == "eval":
        _handle_eval(strategy, strategy_class, train_split, random_seed)


@click.command()
def list_strategies_cmd() -> None:
    console = Console()
    strategies = get_all_strategies()

    assert strategies, "No strategies found in strategies/ directory"

    table = Table(title="Available Strategies")
    table.add_column("Strategy", style="cyan", no_wrap=True)
    table.add_column("Description", style="magenta")

    for strategy_name, strategy_class in sorted(strategies.items()):
        if hasattr(strategy_class, "description") and isinstance(strategy_class.description, str):
            description = strategy_class.description
        else:
            temp_instance = strategy_class.__new__(strategy_class)
            description = getattr(temp_instance, "description", "Description not available")
        table.add_row(strategy_name, description)

    console.print(table)


def _handle_fit(
    strategy: str,
    strategy_class: type[Strategy],
    train_split: float,
    random_seed: int,
) -> None:
    # Fit the model
    model = strategy_class.fit(train_split=train_split, random_seed=random_seed)

    # Prepare save path - MLP uses .pkl, others use .json
    model_path = Path(f"models/{strategy}.pkl") if strategy == "mlp" else Path(f"models/{strategy}.json")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Create model parameters for saving
    # Get the actual data size from the training data
    all_data = parse_training_data(Path("datasets/train.csv"))
    train_indices, val_indices, test_indices = get_split_indices(
        len(all_data), train_ratio=train_split, random_seed=random_seed
    )

    params = ModelParameters.create(
        train_split=train_split,
        random_seed=random_seed,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        total_samples=len(all_data),
    )

    # Save model with parameters
    if hasattr(model, "save") and "parameters" in model.save.__code__.co_varnames:
        model.save(model_path, parameters=params)
    else:
        model.save(model_path)

    print(f"Model trained and saved to {model_path}")
    print(f"Training split: {train_split}, Random seed: {random_seed}")


def _handle_eval(
    strategy: str,
    strategy_class: type[Strategy],
    train_split: float,
    random_seed: int,
) -> None:
    # Try both .json and .pkl extensions
    model_file = Path(f"models/{strategy}.json")
    if not model_file.exists():
        model_file = Path(f"models/{strategy}.pkl")

    # Check if model file exists
    if model_file.exists():
        # Load model and parameters
        load_result = strategy_class.load(model_file)
        if isinstance(load_result, tuple):
            model, params = load_result
            if params:
                # Use saved parameters if available
                print(f"Using saved parameters: train_split={params.train_split}, random_seed={params.random_seed}")
                train_split = params.train_split
                random_seed = params.random_seed
        else:
            # Backward compatibility for models without parameter support
            model = load_result
    # For MLP, try to use checkpoint if no model file exists
    elif strategy == "mlp":
        print("Model file not found, looking for checkpoints...")
        model = None  # Will trigger checkpoint loading in evaluate_on_split
    else:
        msg = f"Model file not found: {model_file}"
        raise FileNotFoundError(msg)

    assert hasattr(strategy_class, "evaluate_on_split"), f"Strategy {strategy} does not support evaluation"

    eval_results = strategy_class.evaluate_on_split(model, train_split=train_split, random_seed=random_seed)

    print("\nEvaluation results:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")



# Add commands to CLI group
cli.add_command(run)
cli.add_command(list_strategies_cmd, name="list-strategies")


if __name__ == "__main__":
    cli()
