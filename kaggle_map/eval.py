"""MAP@3 evaluation metric implementation for kaggle competition."""

import json
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import click
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table

from kaggle_map.models import (
    Category,
    EvaluationResult,
    EvaluationRow,
    Prediction,
)
from kaggle_map.strategies.baseline import BaselineStrategy

# Performance assessment thresholds for MAP@3 scores
EXCELLENT_THRESHOLD = 0.8
GOOD_THRESHOLD = 0.6
MODERATE_THRESHOLD = 0.4

# Model performance log path
MODEL_PERFORMANCE_LOG = Path("performance_history.json")
MAX_HISTORY_DISPLAY = 5


def evaluate(ground_truth_path: Path, submission_path: Path) -> EvaluationResult:
    """Calculate MAP@3 score for kaggle submission.

    Args:
        ground_truth_path: Path to ground truth CSV with Category and Misconception columns
        submission_path: Path to submission CSV with predictions column

    Returns:
        EvaluationResult with MAP@3 score and detailed breakdown
    """
    logger.debug(f"Evaluating {submission_path} against {ground_truth_path}")

    # Load and parse files
    ground_truth = _load_ground_truth(ground_truth_path)
    submissions = _load_submissions(submission_path)

    # Log data loading results
    logger.info(f"📊 Ground truth loaded: {len(ground_truth)} rows")
    logger.info(f"📊 Submissions loaded: {len(submissions)} rows")

    # Calculate MAP@3 over common row_ids
    total_score = 0.0
    perfect_predictions = 0

    common_row_ids = set(ground_truth.keys()) & set(submissions.keys())

    # Log intersection analysis
    gt_only = set(ground_truth.keys()) - set(submissions.keys())
    sub_only = set(submissions.keys()) - set(ground_truth.keys())

    logger.info(f"📊 Common row_ids for evaluation: {len(common_row_ids)}")
    if gt_only:
        logger.warning(
            f"⚠️  {len(gt_only)} row_ids in ground truth but not in submissions"
        )
        logger.debug(f"Ground truth only row_ids (first 10): {sorted(gt_only)[:10]}")
    if sub_only:
        logger.warning(
            f"⚠️  {len(sub_only)} row_ids in submissions but not in ground truth"
        )
        logger.debug(f"Submissions only row_ids (first 10): {sorted(sub_only)[:10]}")

    for row_id in common_row_ids:
        gt_prediction = ground_truth[row_id]
        submission_predictions = submissions[row_id]

        # Calculate average precision for this observation
        ap = _calculate_average_precision(gt_prediction, submission_predictions)
        total_score += ap

        if ap == 1.0:
            perfect_predictions += 1

    total_observations = len(common_row_ids)
    map_score = total_score / total_observations if total_observations > 0 else 0.0

    logger.debug(f"MAP@3: {map_score:.4f} over {total_observations} observations")

    return EvaluationResult(
        map_score=map_score,
        total_observations=total_observations,
        perfect_predictions=perfect_predictions,
    )


def _get_git_commit_hash() -> str:
    """Get current git commit hash."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _log_model_performance(
    result: EvaluationResult,
    strategy: str = "baseline",
    total_execution_time: float = 0.0,
) -> bool:
    """Log model performance to performance history file only if it beats the current best.

    Returns:
        True if the model was logged (new best), False otherwise
    """
    # Load existing performance history
    performance_history = []
    if MODEL_PERFORMANCE_LOG.exists():
        with MODEL_PERFORMANCE_LOG.open() as f:
            performance_history = json.load(f)

    # Check if this beats the current best score
    current_best = performance_history[0]["map_score"] if performance_history else 0.0

    if result.map_score <= current_best:
        logger.info(
            f"Score {result.map_score:.4f} did not beat current best {current_best:.4f} - not logging"
        )
        return False

    # This is a new best! Log it
    commit_hash = _get_git_commit_hash()
    timestamp = datetime.now(UTC).isoformat()

    new_entry = {
        "timestamp": timestamp,
        "commit_hash": commit_hash,
        "strategy": strategy,
        "map_score": result.map_score,
        "total_observations": result.total_observations,
        "perfect_predictions": result.perfect_predictions,
        "total_execution_time": total_execution_time,
    }

    # Add new entry and sort by score (best first)
    performance_history.append(new_entry)
    performance_history.sort(key=lambda x: x["map_score"], reverse=True)

    # Save updated history
    with MODEL_PERFORMANCE_LOG.open("w") as f:
        json.dump(performance_history, f, indent=2)
    logger.info(
        f"🎉 NEW BEST! Logged performance: MAP@3={result.map_score:.4f} to {MODEL_PERFORMANCE_LOG}"
    )
    return True


def _display_performance_history(console: Console) -> None:
    """Display recent performance history if available."""
    if not MODEL_PERFORMANCE_LOG.exists():
        return

    with MODEL_PERFORMANCE_LOG.open() as f:
        history = json.load(f)

    if not history:
        return

    console.print("\n[bold cyan]Recent Performance History (Top 5)[/bold cyan]")

    history_table = Table()
    history_table.add_column("Rank", style="yellow", no_wrap=True)
    history_table.add_column("Score", style="magenta", no_wrap=True)
    history_table.add_column("Date", style="cyan", no_wrap=True)
    history_table.add_column("Commit", style="dim", no_wrap=True)
    history_table.add_column("Strategy", style="green", no_wrap=True)

    # Show top results
    for i, entry in enumerate(history[:MAX_HISTORY_DISPLAY], 1):
        timestamp = datetime.fromisoformat(entry["timestamp"])
        date_str = timestamp.strftime("%Y-%m-%d %H:%M")
        commit_short = entry["commit_hash"][:8]

        # Highlight current best score
        style = "bold green" if i == 1 else ""
        rank_str = f"🏆 {i}" if i == 1 else str(i)

        history_table.add_row(
            rank_str,
            f"{entry['map_score']:.4f}",
            date_str,
            commit_short,
            entry.get("strategy", "baseline"),
            style=style,
        )

    console.print(history_table)

    if len(history) > MAX_HISTORY_DISPLAY:
        console.print(
            f"[dim]... and {len(history) - MAX_HISTORY_DISPLAY} more entries in {MODEL_PERFORMANCE_LOG}[/dim]"
        )


def _load_ground_truth(path: Path) -> dict[int, Prediction]:
    """Load ground truth CSV into dict mapping row_id -> Prediction object."""
    assert path.exists(), f"Ground truth file not found: {path}"

    ground_truth_data = pd.read_csv(path)
    assert not ground_truth_data.empty, "Ground truth file cannot be empty"

    result = {}
    for _, row in ground_truth_data.iterrows():
        row_id = int(row["row_id"])
        category = Category(row["Category"])

        # Handle misconception - NA/None becomes None for Prediction object
        misconception = row["Misconception"]
        if pd.isna(misconception) or misconception == "NA":
            misconception = None

        prediction = Prediction(category=category, misconception=misconception)
        result[row_id] = prediction

    logger.debug(f"Loaded {len(result)} ground truth rows")
    return result


def _load_submissions(path: Path) -> dict[int, list[Prediction]]:
    """Load submission CSV into dict mapping row_id -> list of up to 3 Prediction objects."""
    assert path.exists(), f"Submission file not found: {path}"

    submission_data = pd.read_csv(path)

    result = {}
    for _, row in submission_data.iterrows():
        row_id = int(row["row_id"])
        predictions_str = str(row["predictions"]).strip()

        # Split by spaces and parse into Prediction objects
        predictions = []
        if predictions_str and predictions_str != "nan":
            prediction_strings = predictions_str.split()[:3]
            for pred_str in prediction_strings:
                try:
                    prediction = _parse_prediction_string(pred_str)
                    predictions.append(prediction)
                except ValueError:
                    # Invalid prediction format - skip it
                    logger.debug(f"Skipping invalid prediction: {pred_str}")

        result[row_id] = predictions

    logger.debug(f"Loaded {len(result)} submission rows")
    return result


def _parse_prediction_string(pred_str: str) -> Prediction:
    """Parse a prediction string into a Prediction object."""
    pred_str = pred_str.strip()

    if ":" in pred_str:
        category_part, misconception_part = pred_str.split(":", 1)
        category = Category(category_part.strip())
        misconception = (
            misconception_part.strip() if misconception_part.strip() else None
        )
    else:
        category = Category(pred_str)
        misconception = None

    return Prediction(category=category, misconception=misconception)


def _calculate_average_precision(
    ground_truth: Prediction, predictions: list[Prediction]
) -> float:
    """Calculate average precision for a single observation using MAP@3 formula."""
    if not predictions:
        return 0.0

    # Check each prediction position (1-indexed for precision calculation)
    for k, prediction in enumerate(predictions, 1):
        if _predictions_match(ground_truth, prediction):
            # Found correct prediction at position k, precision = 1/k
            return 1.0 / k

    # No correct prediction found
    return 0.0


def _predictions_match(ground_truth: Prediction, prediction: Prediction) -> bool:
    """Check if two predictions match using their value representation."""
    return ground_truth.value == prediction.value


@click.command()
def main() -> None:
    """Cross-validate baseline model against train.csv ground truth."""
    console = Console()

    console.print(
        "[bold blue]🔄 Running Cross-Validation on Baseline Model[/bold blue]"
    )

    _run_cross_validation(console)


def _run_cross_validation(console: Console) -> None:
    """Run cross-validation using baseline model against train.csv."""
    start_time = time.time()

    model_path = Path("baseline_model.json")
    train_csv_path = Path("dataset/train.csv")

    # Validate required files exist
    assert model_path.exists(), f"Baseline model not found: {model_path}"
    assert train_csv_path.exists(), f"Training data not found: {train_csv_path}"

    with console.status("[bold green]Loading baseline model..."):
        logger.info(f"Loading model from {model_path}")
        model = BaselineStrategy.load(model_path)
        logger.info("Model loaded successfully")

    console.print("✅ [bold green]Baseline model loaded[/bold green]")

    with console.status("[bold green]Preparing test data from train.csv..."):
        logger.info(f"Loading training data from {train_csv_path}")
        test_rows, ground_truth_data = _prepare_cross_validation_data(train_csv_path)
        logger.info(f"Prepared {len(test_rows)} test rows for cross-validation")

    console.print(f"✅ [bold green]Prepared {len(test_rows)} test rows[/bold green]")

    with console.status("[bold green]Generating predictions..."):
        logger.info("Generating predictions using baseline model")
        predictions = model.predict(test_rows)
        logger.info(f"Generated predictions for {len(predictions)} rows")

    console.print(
        f"✅ [bold green]Generated {len(predictions)} predictions[/bold green]"
    )

    # Create temporary files for evaluation pipeline
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        with console.status("[bold green]Preparing evaluation files..."):
            # Save ground truth CSV
            ground_truth_path = tmp_path / "ground_truth.csv"
            ground_truth_data.to_csv(ground_truth_path, index=False)

            # Save submission CSV
            submission_path = tmp_path / "submission.csv"
            _save_submission_csv(predictions, submission_path)

            logger.info(f"Created evaluation files in {tmp_path}")

        console.print("✅ [bold green]Evaluation files prepared[/bold green]")

        with console.status("[bold green]Running MAP@3 evaluation..."):
            result = evaluate(ground_truth_path, submission_path)
            logger.info(f"Cross-validation completed: MAP@3 = {result.map_score:.4f}")

        console.print("\n[bold green]✅ CROSS-VALIDATION COMPLETED[/bold green]")

        # Calculate total execution time
        total_execution_time = time.time() - start_time
        logger.info(f"Total execution time: {total_execution_time:.2f} seconds")

        # Log performance to history if it's a new best
        is_new_best = _log_model_performance(result, "baseline", total_execution_time)

        _display_cross_validation_results(console, result)

        if is_new_best:
            console.print(
                "\n🎉 [bold yellow]NEW BEST SCORE![/bold yellow] This result has been added to the performance history."
            )
        else:
            console.print(
                "\n[dim]Score did not beat current best - not added to history.[/dim]"
            )

        _display_performance_history(console)
        _display_detailed_cross_validation_analysis(
            console, test_rows, predictions, ground_truth_data
        )


def _log_data_quality_issues(train_df: pd.DataFrame) -> None:
    """Log data quality issues in the training DataFrame."""
    logger.info(f"📊 Loaded train.csv with {len(train_df)} total rows")
    logger.info(f"📊 DataFrame shape: {train_df.shape}")
    logger.info(f"📊 Columns: {list(train_df.columns)}")

    # Check for missing values in critical columns
    critical_columns = [
        "row_id",
        "QuestionId",
        "QuestionText",
        "MC_Answer",
        "StudentExplanation",
        "Category",
    ]
    for col in critical_columns:
        missing_count = train_df[col].isna().sum()
        if missing_count > 0:
            logger.warning(f"⚠️  Column '{col}' has {missing_count} missing values")

    # Check for duplicate row_ids
    duplicate_row_ids = train_df["row_id"].duplicated().sum()
    if duplicate_row_ids > 0:
        logger.warning(f"⚠️  Found {duplicate_row_ids} duplicate row_ids")


def _convert_to_test_rows(train_df: pd.DataFrame) -> list[EvaluationRow]:
    """Convert DataFrame rows to EvaluationRow objects."""
    test_rows = []
    failed_conversions = 0

    for idx, row in train_df.iterrows():
        # Validate required fields before conversion
        assert pd.notna(row["row_id"]), f"Row {idx}: row_id cannot be NaN"
        assert pd.notna(row["QuestionId"]), f"Row {idx}: QuestionId cannot be NaN"
        assert pd.notna(row["QuestionText"]), f"Row {idx}: QuestionText cannot be NaN"
        assert pd.notna(row["MC_Answer"]), f"Row {idx}: MC_Answer cannot be NaN"
        assert pd.notna(row["StudentExplanation"]), (
            f"Row {idx}: StudentExplanation cannot be NaN"
        )

        test_row = EvaluationRow(
            row_id=int(row["row_id"]),
            question_id=int(row["QuestionId"]),
            question_text=str(row["QuestionText"]),
            mc_answer=str(row["MC_Answer"]),
            student_explanation=str(row["StudentExplanation"]),
        )
        test_rows.append(test_row)

    if failed_conversions > 0:
        logger.warning(
            f"⚠️  Failed to convert {failed_conversions} rows to EvaluationRow objects"
        )

    return test_rows


def _prepare_cross_validation_data(
    train_csv_path: Path,
) -> tuple[list[EvaluationRow], pd.DataFrame]:
    """Prepare test rows and ground truth data from train.csv."""
    train_df = pd.read_csv(train_csv_path)
    assert not train_df.empty, "Training CSV cannot be empty"

    _log_data_quality_issues(train_df)
    test_rows = _convert_to_test_rows(train_df)

    # Prepare ground truth data for evaluation
    ground_truth_data = train_df[["row_id", "Category", "Misconception"]].copy()

    logger.info(f"✅ Successfully prepared {len(test_rows)} test rows")
    logger.info(f"✅ Ground truth data has {len(ground_truth_data)} rows")

    return test_rows, ground_truth_data


def _save_submission_csv(predictions: list, submission_path: Path) -> None:
    """Save predictions in submission CSV format."""
    logger.info(f"📊 Processing {len(predictions)} prediction rows for submission")

    submission_data = []
    row_ids_seen = set()
    total_individual_predictions = 0

    for submission_row in predictions:
        # Convert Prediction objects to space-separated string format
        prediction_strings = [
            pred.value for pred in submission_row.predicted_categories
        ]

        total_individual_predictions += len(prediction_strings)
        row_ids_seen.add(submission_row.row_id)

        submission_data.append(
            {
                "row_id": submission_row.row_id,
                "predictions": " ".join(prediction_strings),
            }
        )

    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(submission_path, index=False)

    logger.info(f"✅ Saved submission CSV with {len(submission_data)} rows")
    logger.info(f"📊 Total individual predictions: {total_individual_predictions}")
    logger.info(f"📊 Unique row_ids in submission: {len(row_ids_seen)}")
    logger.info(
        f"📊 Average predictions per row: {total_individual_predictions / len(submission_data):.2f}"
    )

    # Check for duplicate row_ids in predictions
    if len(row_ids_seen) != len(submission_data):
        logger.warning("⚠️  Duplicate row_ids detected in predictions!")

    logger.debug(f"Submission file saved to: {submission_path}")


def _display_cross_validation_results(
    console: Console, result: EvaluationResult
) -> None:
    """Display cross-validation results in formatted table."""
    results_table = Table(title="Cross-Validation Results")
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

    console.print(results_table)

    # Performance assessment
    if result.map_score >= EXCELLENT_THRESHOLD:
        console.print(
            "\n🎉 [bold green]Excellent performance![/bold green] Model shows strong cross-validation results."
        )
    elif result.map_score >= GOOD_THRESHOLD:
        console.print(
            "\n✅ [bold blue]Good performance.[/bold blue] Model shows reasonable cross-validation results."
        )
    elif result.map_score >= MODERATE_THRESHOLD:
        console.print(
            "\n⚠️  [bold yellow]Moderate performance.[/bold yellow] Model has room for improvement."
        )
    else:
        console.print(
            "\n⚠️  [bold red]Poor performance.[/bold red] Model needs significant improvement."
        )


def _display_detailed_cross_validation_analysis(
    console: Console,
    test_rows: list[EvaluationRow],
    predictions: list,
    ground_truth_data: pd.DataFrame,
) -> None:
    """Display detailed analysis when verbose mode is enabled."""
    console.print("\n[bold]Detailed Cross-Validation Analysis[/bold]")

    # Sample predictions analysis
    console.print("\n[cyan]Sample Predictions (first 5 rows):[/cyan]")
    sample_table = Table()
    sample_table.add_column("Row ID", style="yellow")
    sample_table.add_column("Question ID", style="cyan")
    sample_table.add_column("Predictions", style="magenta")
    sample_table.add_column("Ground Truth", style="green")

    for i in range(min(5, len(predictions))):
        pred_row = predictions[i]
        row_id = pred_row.row_id

        # Get ground truth for this row
        gt_row = ground_truth_data[ground_truth_data["row_id"] == row_id].iloc[0]
        misconception = (
            gt_row["Misconception"] if pd.notna(gt_row["Misconception"]) else "NA"
        )
        ground_truth = f"{gt_row['Category']}:{misconception}"

        # Format predictions
        pred_strings = [pred.value for pred in pred_row.predicted_categories]

        sample_table.add_row(
            str(row_id),
            str(test_rows[i].question_id),
            " | ".join(pred_strings[:3]),  # Show up to 3 predictions
            ground_truth,
        )

    console.print(sample_table)

    # Question distribution analysis
    question_counts = {}
    for test_row in test_rows:
        question_counts[test_row.question_id] = (
            question_counts.get(test_row.question_id, 0) + 1
        )

    console.print(
        f"\n[cyan]Dataset contains {len(question_counts)} unique questions[/cyan]"
    )
    console.print(
        f"[cyan]Average {sum(question_counts.values()) / len(question_counts):.1f} responses per question[/cyan]"
    )


def _demonstrate_map_evaluation(console: Console, *, verbose: bool) -> None:
    """Create and evaluate sample data to demonstrate MAP@3 calculation."""
    # Create sample data for demonstration using the rich type system
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        with console.status("[bold green]Creating sample evaluation data..."):
            # Sample ground truth data
            ground_truth_data = {
                "row_id": [1, 2, 3, 4, 5],
                "Category": [
                    "True_Correct",
                    "False_Misconception",
                    "True_Misconception",
                    "False_Neither",
                    "True_Neither",
                ],
                "Misconception": [
                    "Adding_across",
                    "Denominator-only_change",
                    "Incorrect_equivalent_fraction_addition",
                    "NA",
                    "NA",
                ],
            }

            # Sample submission data with various prediction accuracies
            submission_data = {
                "row_id": [1, 2, 3, 4, 5],
                "predictions": [
                    # Perfect prediction (1st position)
                    "True_Correct:Adding_across False_Neither:Other_tag True_Misconception:Wrong_tag",
                    # Correct in 2nd position
                    "False_Neither:Wrong_tag False_Misconception:Denominator-only_change True_Correct:Other_tag",
                    # Correct in 3rd position
                    "False_Neither:Wrong_tag True_Correct:Other_tag True_Misconception:Incorrect_equivalent_fraction_addition",
                    # Incorrect prediction
                    "True_Correct:Wrong_tag False_Misconception:Other_tag True_Misconception:Bad_tag",
                    # Correct prediction (category only, no misconception)
                    "True_Neither False_Correct True_Misconception:Some_tag",
                ],
            }

            # Save to temporary CSV files
            gt_path = tmp_path / "ground_truth.csv"
            sub_path = tmp_path / "submission.csv"

            pd.DataFrame(ground_truth_data).to_csv(gt_path, index=False)
            pd.DataFrame(submission_data).to_csv(sub_path, index=False)

        console.print("✅ [bold green]Sample data created[/bold green]")

        # Evaluate MAP@3
        with console.status("[bold green]Calculating MAP@3 score..."):
            result = evaluate(gt_path, sub_path)

        console.print("\n[bold green]✅ MAP@3 EVALUATION COMPLETED[/bold green]")

        # Display results in a table
        results_table = Table(title="MAP@3 Evaluation Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="magenta")

        results_table.add_row("MAP@3 Score", f"{result.map_score:.4f}")
        results_table.add_row("Total Observations", str(result.total_observations))
        results_table.add_row(
            "Perfect Predictions (1st position)", str(result.perfect_predictions)
        )

        console.print(results_table)

        if verbose:
            # Calculate breakdown by position
            console.print("\n[bold]Expected Breakdown by Position[/bold]")
            breakdown_table = Table()
            breakdown_table.add_column("Row", style="cyan")
            breakdown_table.add_column("Position", style="yellow")
            breakdown_table.add_column("Average Precision", style="magenta")

            breakdown_table.add_row("1", "1st (Perfect)", "1.0")
            breakdown_table.add_row("2", "2nd", "0.5")
            breakdown_table.add_row("3", "3rd", "0.333...")
            breakdown_table.add_row("4", "No match", "0.0")
            breakdown_table.add_row("5", "1st (Perfect)", "1.0")

            console.print(breakdown_table)

            expected_map = (1.0 + 0.5 + 1 / 3 + 0.0 + 1.0) / 5
            console.print(f"\n[bold]Expected MAP@3:[/bold] {expected_map:.4f}")
            console.print(f"[bold]Actual MAP@3:[/bold] {result.map_score:.4f}")

            # Verify calculation
            tolerance = 0.001
            if abs(result.map_score - expected_map) < tolerance:
                console.print("[bold green]✅ MAP@3 calculation verified![/bold green]")
            else:
                console.print("[bold red]❌ MAP@3 calculation mismatch![/bold red]")

            # Demonstrate the rich type system
            console.print("\n[bold]Prediction Object Demonstration[/bold]")

            # Show how ground truth is represented as Prediction objects
            gt_data = _load_ground_truth(gt_path)
            console.print("\n[cyan]Ground truth as Prediction objects:[/cyan]")
            for row_id, prediction in gt_data.items():
                console.print(f"  Row {row_id}: {prediction} -> '{prediction.value}'")

            # Show how submissions are parsed into Prediction objects
            sub_data = _load_submissions(sub_path)
            console.print("\n[cyan]Submissions as Prediction objects:[/cyan]")
            for row_id, predictions in sub_data.items():
                pred_strs = [f"'{pred.value}'" for pred in predictions]
                console.print(f"  Row {row_id}: {pred_strs}")

        logger.info("MAP@3 evaluation demonstration completed successfully")


if __name__ == "__main__":
    main()
