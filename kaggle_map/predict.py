"""Model prediction script for the Kaggle student misconception prediction competition."""

from pathlib import Path

import click
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table

from kaggle_map.models import TestRow, load_model


def make_predictions(
    model_path: Path = Path("baseline_model.json"),
    test_csv_path: Path = Path("dataset/test.csv"),
    output_path: Path = Path("submission.csv"),
) -> int:
    """Load model and make predictions on test data.

    Args:
        model_path: Path to the saved model JSON file
        test_csv_path: Path to the test CSV file
        output_path: Path where to save the submission CSV

    Returns:
        Number of predictions made
    """
    logger.info(f"Loading model from {model_path}")

    # Load the trained model
    model = load_model(model_path)
    logger.info("Model loaded successfully")

    # Load test data
    logger.info(f"Loading test data from {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    logger.info(f"Loaded {len(test_df)} test rows")

    # Convert to TestRow objects
    test_rows = []
    for _, row in test_df.iterrows():
        test_rows.append(
            TestRow(
                row_id=int(row["row_id"]),
                question_id=int(row["QuestionId"]),
                question_text=str(row["QuestionText"]),
                mc_answer=str(row["MC_Answer"]),
                student_explanation=str(row["StudentExplanation"]),
            )
        )

    # Make predictions
    logger.info("Making predictions...")
    predictions = model.predict(test_rows)
    logger.info(f"Generated predictions for {len(predictions)} rows")

    # Convert to submission format
    submission_data = []
    for pred in predictions:
        # Convert Prediction objects to space-separated string
        pred_strings = [str(p) for p in pred.predicted_categories]
        submission_data.append(
            {"row_id": pred.row_id, "predictions": " ".join(pred_strings)}
        )

    # Save submission file
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")

    return len(predictions)


def load_and_preview_test_data(test_csv_path: Path, console: Console) -> pd.DataFrame:
    """Load test data and show a preview.

    Args:
        test_csv_path: Path to test CSV file
        console: Rich console for output

    Returns:
        Test data DataFrame
    """
    logger.info(f"Loading test data from {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)

    # Display test data preview
    preview_table = Table(title="Test Data Preview")
    preview_table.add_column("Metric", style="cyan")
    preview_table.add_column("Value", style="magenta")

    preview_table.add_row("Total rows", str(len(test_df)))
    preview_table.add_row("Columns", str(list(test_df.columns)))
    preview_table.add_row("Unique questions", str(len(test_df["QuestionId"].unique())))

    console.print(preview_table)

    # Show sample rows
    console.print("\n[bold]Sample Test Rows[/bold]")
    for i, (_, row) in enumerate(test_df.head(3).iterrows()):
        console.print(
            f"Row {i + 1}: Question {row['QuestionId']} - {row['QuestionText'][:50]}..."
        )

    return test_df


@click.command()
@click.option(
    "--model-path",
    "-m",
    type=click.Path(exists=True, path_type=Path),
    default=Path("baseline_model.json"),
    help="Path to saved model JSON file",
)
@click.option(
    "--test-path",
    "-t",
    type=click.Path(exists=True, path_type=Path),
    default=Path("dataset/test.csv"),
    help="Path to test CSV file",
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("submission.csv"),
    help="Path to save submission CSV",
)
def main(model_path: Path, test_path: Path, output_path: Path) -> None:
    """Load trained model and generate predictions for test data."""
    console = Console()

    console.print("[bold blue]🔮 Starting Prediction Generation[/bold blue]")

    # Display configuration
    config_table = Table(title="Configuration")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="magenta")
    config_table.add_row("Model", str(model_path))
    config_table.add_row("Test data", str(test_path))
    config_table.add_row("Output", str(output_path))
    console.print(config_table)

    # Preview test data
    test_df = load_and_preview_test_data(test_path, console)

    # Make predictions
    with console.status("[bold green]Making predictions..."):
        num_predictions = make_predictions(model_path, test_path, output_path)

    # Display results
    console.print("\n[bold green]✅ PREDICTION COMPLETED[/bold green]")

    results_table = Table(title="Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="magenta")

    results_table.add_row("Test rows processed", str(len(test_df)))
    results_table.add_row("Predictions generated", str(num_predictions))

    if output_path.exists():
        submission_df = pd.read_csv(output_path)
        file_size_kb = output_path.stat().st_size / 1024
        results_table.add_row("Output file size", f"{file_size_kb:.1f} KB")
        results_table.add_row("Status", "[green]✅ Successfully created[/green]")

        console.print(results_table)

        # Show sample predictions
        console.print("\n[bold]Sample Predictions[/bold]")
        for _i, (_, row) in enumerate(submission_df.head(3).iterrows()):
            console.print(f"Row {row['row_id']}: {row['predictions']}")

    else:
        results_table.add_row("Status", "[red]❌ File not created[/red]")
        console.print(results_table)
        console.print("[bold red]Error: Submission file not created[/bold red]")
        raise click.Abort()

    logger.info("Prediction script completed successfully")


if __name__ == "__main__":
    main()
