"""Command-line interface for kaggle-map prediction strategies."""

from pathlib import Path

import click
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table

from .strategies import get_all_strategies, get_strategy, list_strategies


@click.group()
def cli() -> None:
    """Kaggle MAP student misconception prediction toolkit."""


@click.command()
@click.argument("strategy", type=click.Choice(list_strategies(), case_sensitive=False))
@click.argument("action", type=click.Choice(["fit", "eval", "predict"], case_sensitive=False))
@click.option("--verbose", "-v", is_flag=True, help="Show detailed model information")
@click.option("--model-path", type=click.Path(), help="Path to saved model file")
@click.option("--output-path", type=click.Path(), help="Path for output files")
def run(strategy: str, action: str, verbose: bool, model_path: str | None, output_path: str | None) -> None:
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
            if hasattr(strategy_class, "description") and isinstance(strategy_class.description, str):
                description = strategy_class.description
            else:
                # Try to create a temporary instance to get description
                temp_instance = strategy_class.__new__(strategy_class)
                description = temp_instance.description
        except Exception:
            description = "Description not available"
            
        table.add_row(strategy_name, description)
    
    console.print(table)


def _handle_fit(strategy: str, strategy_class, console: Console, verbose: bool, output_path: str | None) -> None:
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
    model_path = Path(output_path) if output_path else Path(f"{strategy}_model.json")
        
    model.save(model_path)
    console.print(f"✅ [bold green]Model saved to {model_path}[/bold green]")


def _handle_eval(strategy: str, strategy_class, console: Console, model_path: str | None, verbose: bool) -> None:
    """Handle the eval action."""
    import tempfile
    import time

    from .eval import _display_performance_history, _log_model_performance, evaluate
    
    # Determine model path
    model_file = Path(model_path) if model_path else Path(f"{strategy}_model.json")
    
    if not model_file.exists():
        console.print(f"[bold red]Model file not found: {model_file}[/bold red]")
        console.print(f"[yellow]Hint: Run '{strategy} fit' first, or specify --model-path[/yellow]")
        raise click.Abort()
    
    start_time = time.time()
    
    with console.status(f"[bold blue]Loading {strategy} model..."):
        model = strategy_class.load(model_file)
    
    console.print(f"✅ [bold green]Loaded {strategy} model from {model_file}[/bold green]")
    
    if verbose:
        model.display_stats(console)
    
    # Run cross-validation evaluation
    console.print(f"\n[bold blue]🔄 Running Cross-Validation for {strategy.title()} Strategy[/bold blue]")
    
    train_csv_path = Path("dataset/train.csv")
    if not train_csv_path.exists():
        console.print(f"[bold red]Training data not found: {train_csv_path}[/bold red]")
        raise click.Abort()
    
    with console.status("[bold green]Preparing test data from training set..."):
        test_rows, ground_truth_data = _prepare_eval_data(train_csv_path)
        logger.info(f"Prepared {len(test_rows)} test rows for evaluation")
    
    console.print(f"✅ Prepared {len(test_rows)} test rows")
    
    with console.status(f"[bold green]Generating {strategy} predictions..."):
        predictions = model.predict(test_rows)
        logger.info(f"Generated predictions for {len(predictions)} rows")
    
    console.print(f"✅ Generated {len(predictions)} predictions")
    
    # Create temporary files and evaluate
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Save ground truth and submission files
        ground_truth_path = tmp_path / "ground_truth.csv"
        submission_path = tmp_path / "submission.csv"
        
        ground_truth_data.to_csv(ground_truth_path, index=False)
        _save_eval_submission_csv(predictions, submission_path)
        
        # Evaluate performance
        with console.status("[bold green]Calculating MAP@3 score..."):
            result = evaluate(ground_truth_path, submission_path)
    
    # Log performance if it's a new best
    total_execution_time = time.time() - start_time
    is_new_best = _log_model_performance(result, strategy, total_execution_time)
    
    # Display results
    _display_eval_results(console, result, strategy, total_execution_time)
    
    if is_new_best:
        console.print("\n🎉 [bold yellow]NEW BEST SCORE![/bold yellow]")
    
    _display_performance_history(console)
    
    if verbose:
        _display_detailed_eval_analysis(console, test_rows[:10], predictions[:10], ground_truth_data)
    
    console.print(f"\n✅ [bold green]Evaluation completed in {total_execution_time:.2f}s[/bold green]")


def _handle_predict(strategy: str, strategy_class, console: Console, model_path: str | None, output_path: str | None) -> None:
    """Handle the predict action."""
    # Determine model path
    model_file = Path(model_path) if model_path else Path(f"{strategy}_model.json")
    
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


def _prepare_eval_data(train_csv_path: Path) -> tuple[list, pd.DataFrame]:
    """Prepare test rows and ground truth from training data."""
    from .models import TestRow
    
    train_df = pd.read_csv(train_csv_path)
    assert not train_df.empty, "Training CSV cannot be empty"
    
    # Convert to TestRow objects
    test_rows = []
    for _, row in train_df.iterrows():
        test_row = TestRow(
            row_id=int(row["row_id"]),
            question_id=int(row["QuestionId"]),
            question_text=str(row["QuestionText"]),
            mc_answer=str(row["MC_Answer"]),
            student_explanation=str(row["StudentExplanation"]),
        )
        test_rows.append(test_row)
    
    # Prepare ground truth
    ground_truth_data = train_df[["row_id", "Category", "Misconception"]].copy()
    
    return test_rows, ground_truth_data


def _save_eval_submission_csv(predictions: list, submission_path: Path) -> None:
    """Save predictions in submission CSV format."""
    submission_data = []
    
    for submission_row in predictions:
        prediction_strings = [pred.value for pred in submission_row.predicted_categories]
        
        submission_data.append({
            "row_id": submission_row.row_id,
            "predictions": " ".join(prediction_strings),
        })
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(submission_path, index=False)


def _display_eval_results(console: Console, result, strategy: str, execution_time: float) -> None:
    """Display evaluation results in a nice table."""
    from rich.table import Table
    
    results_table = Table(title=f"{strategy.title()} Strategy Evaluation Results")
    results_table.add_column("Metric", style="cyan", no_wrap=True)
    results_table.add_column("Value", style="magenta")
    results_table.add_column("Description", style="dim")
    
    results_table.add_row(
        "MAP@3 Score", f"{result.map_score:.4f}", "Mean Average Precision at 3"
    )
    results_table.add_row(
        "Total Observations", str(result.total_observations), "Rows evaluated"
    )
    results_table.add_row(
        "Perfect Predictions",
        f"{result.perfect_predictions} ({result.perfect_predictions / result.total_observations:.1%})",
        "Correct in 1st position",
    )
    results_table.add_row(
        "Execution Time", f"{execution_time:.2f}s", "Total time for evaluation"
    )
    
    console.print(results_table)
    
    # Performance assessment
    if result.map_score >= 0.8:
        console.print("\n🎉 [bold green]Excellent performance![/bold green]")
    elif result.map_score >= 0.6:
        console.print("\n✅ [bold blue]Good performance.[/bold blue]")
    elif result.map_score >= 0.4:
        console.print("\n⚠️  [bold yellow]Moderate performance.[/bold yellow]")
    else:
        console.print("\n⚠️  [bold red]Poor performance.[/bold red]")


def _display_detailed_eval_analysis(console: Console, sample_test_rows: list,
                                   sample_predictions: list, ground_truth_data: pd.DataFrame) -> None:
    """Display detailed evaluation analysis when verbose mode is enabled."""
    from rich.table import Table
    
    console.print("\n[bold]Detailed Evaluation Analysis[/bold]")
    
    # Sample predictions table
    console.print("\n[cyan]Sample Predictions Analysis:[/cyan]")
    sample_table = Table()
    sample_table.add_column("Row ID", style="yellow")
    sample_table.add_column("Context", style="cyan")
    sample_table.add_column("Top Prediction", style="magenta")
    sample_table.add_column("Ground Truth", style="green")
    sample_table.add_column("Match", style="bold")
    
    for i, pred_row in enumerate(sample_predictions[:5]):
        if i >= len(sample_test_rows):
            break
            
        test_row = sample_test_rows[i]
        row_id = pred_row.row_id
        
        # Get ground truth
        gt_row = ground_truth_data[ground_truth_data["row_id"] == row_id].iloc[0]
        misconception = gt_row["Misconception"] if pd.notna(gt_row["Misconception"]) else "NA"
        ground_truth = f"{gt_row['Category']}:{misconception}"
        
        # Create context description
        context = f"Q{test_row.question_id}: '{test_row.mc_answer}'"
        
        # Get top prediction
        top_pred = pred_row.predicted_categories[0] if pred_row.predicted_categories else None
        top_pred_str = top_pred.value if top_pred else "No prediction"
        
        # Check if match
        match = "✅" if top_pred_str == ground_truth else "❌"
        
        sample_table.add_row(
            str(row_id),
            context,
            top_pred_str,
            ground_truth,
            match
        )
    
    console.print(sample_table)


# Add commands to CLI group
cli.add_command(run)
cli.add_command(list_strategies_cmd, name="list-strategies")


if __name__ == "__main__":
    cli()
