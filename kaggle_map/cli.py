"""Command-line interface for kaggle-map prediction strategies."""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .strategies import get_all_strategies, get_strategy, list_strategies
from .strategies.base import Strategy


@click.group()
def cli() -> None:
    pass


@click.command()
@click.argument("strategy", type=click.Choice(list_strategies(), case_sensitive=False))
@click.argument("action", type=click.Choice(["fit", "eval"], case_sensitive=False))
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
    model = strategy_class.fit(train_split=train_split, random_seed=random_seed)

    model_path = Path(f"models/{strategy}.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    print(f"Model trained and saved to {model_path}")


def _handle_eval(
    strategy: str,
    strategy_class: type[Strategy],
    train_split: float,
    random_seed: int,
) -> None:
    model_file = Path(f"models/{strategy}.pkl")
    assert model_file.exists(), f"Model file not found: {model_file}"

    model = strategy_class.load(model_file)

    assert hasattr(strategy_class, "evaluate_on_split"), f"Strategy {strategy} does not support evaluation"

    eval_results = strategy_class.evaluate_on_split(model, train_split=train_split, random_seed=random_seed)

    print(f"Evaluation results: {eval_results}")



# Add commands to CLI group
cli.add_command(run)
cli.add_command(list_strategies_cmd, name="list-strategies")


if __name__ == "__main__":
    cli()
