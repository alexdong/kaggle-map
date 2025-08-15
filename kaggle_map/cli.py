"""Command-line interface for kaggle-map prediction strategies."""

import json
from pathlib import Path

import click
from loguru import logger
from rich.console import Console
from rich.table import Table

from .strategies import get_all_strategies, get_strategy, list_strategies


@click.group()
def cli():
    """Kaggle MAP student misconception prediction toolkit."""
    pass


@click.command()
@click.argument("strategy", type=click.Choice(list_strategies(), case_sensitive=False))
@click.argument("action", type=click.Choice(["fit", "eval", "predict"], case_sensitive=False))
@click.option("--verbose", "-v", is_flag=True, help="Show detailed model information")
@click.option("--model-path", type=click.Path(), help="Path to saved model file")
@click.option("--output-path", type=click.Path(), help="Path for output files")
def run(strategy: str, action: str, verbose: bool, model_path: str | None, output_path: str | None):
    """Run a strategy with the specified action.
    
    STRATEGY: Name of the prediction strategy to use
    ACTION: Action to perform (fit, eval, predict)
    """
    console = Console()
    
    try:
        strategy_class = get_strategy(strategy)
        logger.info(f"Using strategy: {strategy} - {strategy_class}")
        
        if action == "fit":
            _handle_fit(strategy, strategy_class, console, verbose, output_path)
        elif action == "eval":
            _handle_eval(strategy, strategy_class, console, model_path, verbose)
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
def list_strategies_cmd():
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
            if hasattr(strategy_class, 'description') and isinstance(strategy_class.description, str):
                description = strategy_class.description
            else:
                # Try to create a temporary instance to get description
                temp_instance = strategy_class.__new__(strategy_class)
                description = temp_instance.description
        except Exception:
            description = "Description not available"
            
        table.add_row(strategy_name, description)
    
    console.print(table)


def _handle_fit(strategy: str, strategy_class, console: Console, verbose: bool, output_path: str | None):
    """Handle the fit action."""
    with console.status(f"[bold green]Fitting {strategy} strategy..."):
        model = strategy_class.fit()
        
    console.print(f"✅ [bold green]{strategy.title()} strategy completed successfully![/bold green]")
    
    # Display results
    model.display_stats(console)
    
    if verbose:
        model.display_detailed_info(console)
        
    model.demonstrate_predictions(console)
    
    # Save model
    if output_path:
        model_path = Path(output_path)
    else:
        model_path = Path(f"{strategy}_model.json")
        
    model.save(model_path)
    console.print(f"✅ [bold green]Model saved to {model_path}[/bold green]")


def _handle_eval(strategy: str, strategy_class, console: Console, model_path: str | None, verbose: bool):
    """Handle the eval action."""
    # Determine model path
    if model_path:
        model_file = Path(model_path)
    else:
        model_file = Path(f"{strategy}_model.json")
    
    if not model_file.exists():
        console.print(f"[bold red]Model file not found: {model_file}[/bold red]")
        console.print(f"[yellow]Hint: Run '{strategy} fit' first, or specify --model-path[/yellow]")
        raise click.Abort()
    
    with console.status(f"[bold blue]Loading {strategy} model..."):
        model = strategy_class.load(model_file)
    
    console.print(f"✅ [bold green]Loaded {strategy} model from {model_file}[/bold green]")
    
    # TODO: Implement evaluation logic
    console.print("[yellow]Evaluation functionality not yet implemented[/yellow]")
    console.print("[dim]This would run cross-validation or test set evaluation[/dim]")


def _handle_predict(strategy: str, strategy_class, console: Console, model_path: str | None, output_path: str | None):
    """Handle the predict action."""
    # Determine model path
    if model_path:
        model_file = Path(model_path)
    else:
        model_file = Path(f"{strategy}_model.json")
    
    if not model_file.exists():
        console.print(f"[bold red]Model file not found: {model_file}[/bold red]")
        console.print(f"[yellow]Hint: Run '{strategy} fit' first, or specify --model-path[/yellow]")
        raise click.Abort()
    
    with console.status(f"[bold blue]Loading {strategy} model..."):
        model = strategy_class.load(model_file)
    
    console.print(f"✅ [bold green]Loaded {strategy} model from {model_file}[/bold green]")
    
    # TODO: Implement prediction logic for test data
    console.print("[yellow]Prediction functionality not yet implemented[/yellow]")
    console.print("[dim]This would generate predictions for test.csv[/dim]")


# Add commands to CLI group
cli.add_command(run)
cli.add_command(list_strategies_cmd, name="list-strategies")


if __name__ == "__main__":
    cli()