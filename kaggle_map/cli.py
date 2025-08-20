"""Command-line interface for kaggle-map prediction strategies."""

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loguru import Logger

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
    start_time = time.time()

    # Create context logger with all parameters for debugging
    context_logger = logger.bind(
        strategy=strategy,
        action=action,
        verbose=verbose,
        model_path=model_path,
        output_path=output_path,
        train_split=train_split,
        random_seed=random_seed,
        embeddings_path=str(embeddings_path) if embeddings_path else None,
    )

    context_logger.info(
        "Starting CLI run command", extra={"command": "run", "status": "started"}
    )
    context_logger.debug("Run parameters validated and parsed")

    console = Console()

    try:
        context_logger.debug("Looking up strategy class", strategy_name=strategy)
        strategy_class = get_strategy(strategy)
        context_logger.info(
            "Strategy class resolved",
            strategy_class_name=strategy_class.__name__,
            strategy_module=strategy_class.__module__,
        )
        context_logger.debug("Strategy class ready for instantiation")

        # Log action routing decision
        context_logger.debug("Routing to action handler", action=action)

        handlers = {
            "fit": lambda: _handle_fit(
                strategy,
                strategy_class,
                console,
                context_logger,
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
                context_logger,
                model_path=model_path,
                train_split=train_split,
                random_seed=random_seed,
            ),
            "predict": lambda: _handle_predict(
                strategy,
                strategy_class,
                console,
                context_logger,
                model_path=model_path,
                output_path=output_path,
            ),
        }

        context_logger.info("Executing action", action=action, action_start=True)
        action_result = handlers[action]()

        execution_time = time.time() - start_time
        context_logger.info(
            "CLI run completed successfully",
            execution_time_seconds=execution_time,
            action_timing_info=action_result
            if isinstance(action_result, dict)
            else None,
            extra={"command": "run", "status": "completed"},
        )

    except ValueError as e:
        execution_time = time.time() - start_time
        context_logger.error(
            "Validation error in CLI run",
            error_type="ValueError",
            error_message=str(e),
            execution_time_seconds=execution_time,
            extra={"command": "run", "status": "failed"},
        )
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.Abort() from e
    except Exception as e:
        execution_time = time.time() - start_time
        context_logger.error(
            "Unexpected error in CLI run",
            error_type=type(e).__name__,
            error_message=str(e),
            execution_time_seconds=execution_time,
            extra={"command": "run", "status": "failed"},
        )
        console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        logger.exception(f"Failed to run {strategy} {action}")
        raise click.Abort() from e


@click.command()
def list_strategies_cmd() -> None:
    """List all available strategies with descriptions."""
    logger.info(
        "Starting list-strategies command",
        extra={"command": "list-strategies", "status": "started"},
    )
    console = Console()

    logger.debug("Fetching all available strategies")
    strategies = get_all_strategies()
    logger.debug(f"Found {len(strategies)} strategies", strategy_count=len(strategies))

    if not strategies:
        logger.warning("No strategies found in strategies/ directory")
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
    logger.info(
        "List-strategies command completed successfully",
        strategy_count=len(strategies),
        extra={"command": "list-strategies", "status": "completed"},
    )


def _handle_fit(
    strategy: str,
    strategy_class: type[Strategy],
    console: Console,
    context_logger: "Logger",
    *,
    verbose: bool,
    output_path: str | None,
    train_split: float,
    random_seed: int,
    embeddings_path: Path | None,
) -> dict[str, float]:
    """Handle the fit action and return timing information."""
    fit_start_time = time.time()

    context_logger.info(
        "Starting fit operation",
        fit_parameters={
            "strategy": strategy,
            "train_split": train_split,
            "random_seed": random_seed,
            "verbose": verbose,
            "output_path": output_path,
            "embeddings_path": str(embeddings_path) if embeddings_path else None,
        },
        extra={"action": "fit", "phase": "start"},
    )

    console.print(
        f"[bold blue]Training {strategy} strategy with {train_split:.0%} of data (seed: {random_seed})[/bold blue]"
    )

    with console.status(f"[bold green]Fitting {strategy} strategy..."):
        # Check if strategy supports enhanced parameters
        context_logger.debug("Analyzing strategy fit method capabilities")
        fit_method = strategy_class.fit

        supports_enhanced_params = (
            hasattr(fit_method, "__code__")
            and "train_split" in fit_method.__code__.co_varnames
        )
        context_logger.debug(
            "Strategy parameter support analyzed",
            supports_enhanced_params=supports_enhanced_params,
        )

        if supports_enhanced_params:
            # Build kwargs based on what the strategy supports
            fit_kwargs = {"train_split": train_split, "random_seed": random_seed}

            supports_embeddings = (
                "embeddings_path" in fit_method.__code__.co_varnames
                and embeddings_path is not None
            )
            context_logger.debug(
                "Strategy embedding support analyzed",
                supports_embeddings=supports_embeddings,
            )

            if supports_embeddings:
                fit_kwargs["embeddings_path"] = embeddings_path
                context_logger.info(
                    "Using pre-computed embeddings",
                    embeddings_path=str(embeddings_path),
                )
                console.print(
                    f"[dim]Using pre-computed embeddings from {embeddings_path}[/dim]"
                )

            context_logger.debug(
                "Calling strategy fit method with enhanced parameters",
                fit_kwargs=fit_kwargs,
            )
            model_fit_start = time.time()
            model = strategy_class.fit(**fit_kwargs)  # type: ignore[call-arg]
            model_fit_time = time.time() - model_fit_start
            context_logger.info(
                "Strategy fit completed",
                model_fit_time_seconds=model_fit_time,
                extra={"phase": "model_trained"},
            )
        else:
            # Fallback for strategies that don't support split
            context_logger.debug("Using basic fit method (no enhanced parameters)")
            model_fit_start = time.time()
            model = strategy_class.fit()
            model_fit_time = time.time() - model_fit_start
            context_logger.info(
                "Strategy fit completed (basic mode)",
                model_fit_time_seconds=model_fit_time,
                extra={"phase": "model_trained"},
            )

    console.print(
        f"✅ [bold green]{strategy.title()} strategy completed successfully![/bold green]"
    )

    # Display results
    context_logger.debug("Displaying model statistics")
    model.display_stats(console)

    if verbose:
        context_logger.debug("Displaying detailed model information (verbose mode)")
        model.display_detailed_info(console)

    context_logger.debug("Demonstrating model predictions")
    model.demonstrate_predictions(console)

    # Save model
    model_path = Path(output_path) if output_path else Path(f"{strategy}_model.pkl")
    context_logger.info(
        "Saving model to path",
        model_path=str(model_path),
        extra={"phase": "model_saving"},
    )

    save_start_time = time.time()
    model.save(model_path)
    save_time = time.time() - save_start_time

    total_fit_time = time.time() - fit_start_time
    context_logger.info(
        "Fit operation completed successfully",
        model_save_time_seconds=save_time,
        total_fit_time_seconds=total_fit_time,
        final_model_path=str(model_path),
        extra={"action": "fit", "phase": "completed"},
    )

    console.print(f"✅ [bold green]Model saved to {model_path}[/bold green]")

    # Return timing information for performance tracking
    timing_info = {
        "model_fit_time_seconds": model_fit_time,
        "model_save_time_seconds": save_time,
        "total_fit_time_seconds": total_fit_time,
    }
    context_logger.debug("Fit timing summary", timing_info=timing_info)
    return timing_info


def _handle_eval(
    strategy: str,
    strategy_class: type[Strategy],
    console: Console,
    context_logger: "Logger",
    *,
    model_path: str | None,
    train_split: float,
    random_seed: int,
) -> dict[str, float] | None:
    """Handle the eval action and return timing information."""
    eval_start_time = time.time()

    context_logger.info(
        "Starting eval operation",
        eval_parameters={
            "strategy": strategy,
            "model_path": model_path,
            "train_split": train_split,
            "random_seed": random_seed,
        },
        extra={"action": "eval", "phase": "start"},
    )

    # Determine model path
    model_file = Path(model_path) if model_path else Path(f"{strategy}_model.pkl")
    context_logger.debug("Resolved model file path", model_file=str(model_file))

    if not model_file.exists():
        context_logger.error(
            "Model file not found",
            model_file=str(model_file),
            suggested_command=f"{strategy} fit --train-split {train_split}",
            extra={"action": "eval", "phase": "failed", "error": "model_not_found"},
        )
        console.print(f"[bold red]Model file not found: {model_file}[/bold red]")
        console.print(
            f"[yellow]Hint: Run '{strategy} fit --train-split {train_split}' first, or specify --model-path[/yellow]"
        )
        raise click.Abort()

    with console.status(f"[bold blue]Loading {strategy} model..."):
        load_start_time = time.time()
        context_logger.debug("Loading model from file", model_file=str(model_file))
        model = strategy_class.load(model_file)
        load_time = time.time() - load_start_time
        context_logger.info(
            "Model loaded successfully",
            model_load_time_seconds=load_time,
            model_file=str(model_file),
            extra={"phase": "model_loaded"},
        )

    console.print(
        f"✅ [bold green]Loaded {strategy} model from {model_file}[/bold green]"
    )

    # Run evaluation if the strategy supports it
    supports_evaluation = hasattr(strategy_class, "evaluate_on_split")
    context_logger.debug(
        "Strategy evaluation support analyzed", supports_evaluation=supports_evaluation
    )

    if supports_evaluation:
        try:
            with console.status(f"[bold blue]Evaluating {strategy} model..."):
                eval_compute_start = time.time()
                context_logger.info(
                    "Starting model evaluation", extra={"phase": "evaluation_start"}
                )
                eval_results = strategy_class.evaluate_on_split(model)  # type: ignore[misc]
                eval_compute_time = time.time() - eval_compute_start
                context_logger.info(
                    "Model evaluation completed",
                    evaluation_time_seconds=eval_compute_time,
                    eval_results=eval_results,
                    extra={"phase": "evaluation_completed"},
                )

            console.print("✅ [bold green]Evaluation completed![/bold green]")

            # Display evaluation results
            context_logger.debug("Displaying evaluation results")
            _display_eval_results(console, eval_results)

            # Create timing information for performance tracking
            total_eval_time = time.time() - eval_start_time
            timing_info = {
                "model_load_time_seconds": load_time,
                "evaluation_time_seconds": eval_compute_time,
                "total_eval_time_seconds": total_eval_time,
            }

            # Update performance history with timing information
            context_logger.debug(
                "Updating performance history with timing", timing_info=timing_info
            )
            _update_performance_history(
                strategy, eval_results, timing_info, context_logger
            )

            context_logger.info(
                "Eval operation completed successfully",
                total_eval_time_seconds=total_eval_time,
                map_score=eval_results.get("map_score"),
                extra={"action": "eval", "phase": "completed"},
            )

            return timing_info

        except ValueError as e:
            total_eval_time = time.time() - eval_start_time
            context_logger.error(
                "Evaluation failed with ValueError",
                error_message=str(e),
                suggested_solution=f"Train with --train-split {train_split}",
                total_eval_time_seconds=total_eval_time,
                extra={
                    "action": "eval",
                    "phase": "failed",
                    "error": "validation_error",
                },
            )
            console.print(f"[bold red]Evaluation error:[/bold red] {e}")
            console.print(
                f"[yellow]Hint: Train with --train-split {train_split} to create validation split[/yellow]"
            )
            return None
    else:
        total_eval_time = time.time() - eval_start_time
        context_logger.warning(
            "Strategy does not support evaluation",
            strategy_class_name=strategy_class.__name__,
            total_eval_time_seconds=total_eval_time,
            extra={"action": "eval", "phase": "skipped", "reason": "not_supported"},
        )
        console.print("[yellow]Strategy does not support evaluation[/yellow]")
        console.print(
            "[dim]Only available for strategies with evaluate_on_split method[/dim]"
        )
        return None


def _handle_predict(
    strategy: str,
    strategy_class: type[Strategy],
    console: Console,
    context_logger: "Logger",
    *,
    model_path: str | None,
    output_path: str | None,
) -> None:
    """Handle the predict action."""
    time.time()

    context_logger.info(
        "Starting predict operation",
        predict_parameters={
            "strategy": strategy,
            "model_path": model_path,
            "output_path": output_path,
        },
        extra={"action": "predict", "phase": "start"},
    )

    # Determine model path
    model_file = Path(model_path) if model_path else Path(f"{strategy}_model.json")
    context_logger.debug("Resolved model file path", model_file=str(model_file))

    if not model_file.exists():
        context_logger.error(
            "Model file not found for prediction",
            model_file=str(model_file),
            suggested_command=f"{strategy} fit",
            extra={"action": "predict", "phase": "failed", "error": "model_not_found"},
        )
        console.print(f"[bold red]Model file not found: {model_file}[/bold red]")
        console.print(
            f"[yellow]Hint: Run '{strategy} fit' first, or specify --model-path[/yellow]"
        )
        raise click.Abort()

    with console.status(f"[bold blue]Loading {strategy} model..."):
        load_start_time = time.time()
        context_logger.debug("Loading model for prediction", model_file=str(model_file))
        strategy_class.load(model_file)
        load_time = time.time() - load_start_time
        context_logger.info(
            "Model loaded for prediction",
            model_load_time_seconds=load_time,
            extra={"phase": "model_loaded"},
        )

    console.print(
        f"✅ [bold green]Loaded {strategy} model from {model_file}[/bold green]"
    )

    # TODO: Implement prediction logic for test data
    # Suppress unused-arg warning until implemented
    _ = output_path
    context_logger.warning(
        "Prediction functionality not yet implemented",
        planned_output_path=output_path,
        extra={"action": "predict", "phase": "not_implemented"},
    )
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


def _update_performance_history(
    strategy: str,
    eval_results: dict[str, float],
    timing_info: dict[str, float],
    context_logger: "Logger",
) -> None:
    """Update performance_history.json with evaluation results and timing information."""
    performance_file = Path("performance_history.json")
    context_logger.debug(
        "Updating performance history file",
        performance_file=str(performance_file),
        timing_info=timing_info,
    )

    # Load existing history
    performance_history = []
    if performance_file.exists():
        context_logger.debug("Loading existing performance history")
        with performance_file.open() as f:
            performance_history = json.load(f)
        context_logger.debug(
            "Loaded existing performance entries", entry_count=len(performance_history)
        )
    else:
        context_logger.debug(
            "No existing performance history file found, creating new one"
        )

    # Get current git commit hash
    try:
        context_logger.debug("Retrieving git commit hash")
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        commit_hash = result.stdout.strip()
        context_logger.debug(
            "Retrieved git commit hash",
            commit_hash_short=commit_hash[:8] if len(commit_hash) >= 8 else commit_hash,  # noqa: PLR2004
        )
    except subprocess.CalledProcessError as e:
        context_logger.warning("Failed to retrieve git commit hash", error=str(e))
        commit_hash = "unknown"

    # Create new entry with actual timing information
    timestamp = datetime.now(UTC).isoformat()
    total_execution_time = timing_info.get("total_eval_time_seconds", 0.0)

    new_entry = {
        "timestamp": timestamp,
        "commit_hash": commit_hash,
        "strategy": strategy,
        "map_score": eval_results["map_score"],
        "total_observations": eval_results["total_observations"],
        "perfect_predictions": eval_results["perfect_predictions"],
        "total_execution_time": total_execution_time,
        # Add detailed timing breakdown for analysis
        "timing_breakdown": {
            "model_load_time_seconds": timing_info.get("model_load_time_seconds", 0.0),
            "evaluation_time_seconds": timing_info.get("evaluation_time_seconds", 0.0),
            "total_eval_time_seconds": total_execution_time,
        },
    }

    context_logger.debug(
        "Created new performance entry",
        new_entry_summary={
            "strategy": strategy,
            "map_score": eval_results["map_score"],
            "total_execution_time_seconds": total_execution_time,
            "commit_hash": commit_hash[:8] if commit_hash != "unknown" else "unknown",
        },
        timing_breakdown=new_entry["timing_breakdown"],
    )

    # Add new entry and sort by score (best first)
    performance_history.append(new_entry)
    performance_history.sort(key=lambda x: x["map_score"], reverse=True)

    context_logger.debug(
        "Performance history updated", total_entries=len(performance_history)
    )

    # Save updated history
    try:
        with performance_file.open("w") as f:
            json.dump(performance_history, f, indent=2)
        context_logger.info(
            "Performance history updated successfully",
            performance_file=str(performance_file),
            total_entries=len(performance_history),
        )
    except Exception as e:
        context_logger.error(
            "Failed to save performance history",
            error=str(e),
            performance_file=str(performance_file),
        )
        raise

    logger.info(
        f"Updated performance history: MAP@3={eval_results['map_score']:.4f} for {strategy} "
        f"(execution time: {total_execution_time:.3f}s)"
    )


# Add commands to CLI group
cli.add_command(run)
cli.add_command(list_strategies_cmd, name="list-strategies")


if __name__ == "__main__":
    cli()
