"""Display, statistics, and evaluation methods for MLP strategy.

Handles model performance evaluation, statistics display, and demonstration
functionality. Provides rich console output and evaluation metrics
for understanding model behavior and performance.
"""

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table

from kaggle_map.eval import evaluate
from kaggle_map.models import EvaluationRow, TrainingRow
from kaggle_map.strategies.mlp.config import HIDDEN_DIMS

if TYPE_CHECKING:
    from kaggle_map.strategies.mlp.strategy import MLPStrategy


class MLPEvaluator:
    """Handles MLP model evaluation and display functionality.

    Provides methods for displaying model statistics, detailed information,
    and running evaluation metrics on held-out data.
    """

    def __init__(self, strategy: "MLPStrategy") -> None:
        """Initialize evaluator with strategy instance.

        Args:
            strategy: MLPStrategy instance to evaluate
        """
        self.strategy = strategy

    def display_stats(self, console: Console) -> None:
        """Display model statistics in a formatted table.

        Args:
            console: Rich console for output
        """
        stats_table = Table(title="MLP Model Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Count", style="magenta")

        stats_table.add_row(
            "Questions with correct answers", str(len(self.strategy.correct_answers))
        )
        stats_table.add_row(
            "Questions with misconceptions",
            str(len(self.strategy.question_misconceptions)),
        )

        # Get model device
        model_device = next(self.strategy.model.parameters()).device
        stats_table.add_row("Model device", str(model_device))

        total_params = sum(p.numel() for p in self.strategy.model.parameters())
        stats_table.add_row("Total model parameters", f"{total_params:,}")

        # Embedding model info
        stats_table.add_row("Embedding model", self.strategy.embedding_model.model_id)
        stats_table.add_row(
            "Embedding dimension", str(self.strategy.embedding_model.dim)
        )

        console.print(stats_table)

    def display_detailed_info(self, console: Console) -> None:
        """Display detailed model info for verbose mode.

        Args:
            console: Rich console for output
        """
        console.print("\n[bold]Detailed MLP Model Contents[/bold]")
        self._display_architecture(console)
        self._display_question_misconceptions(console)

    def _display_architecture(self, console: Console) -> None:
        """Display model architecture details.

        Args:
            console: Rich console for output
        """
        console.print("\n[cyan]Model Architecture:[/cyan]")
        console.print(f"  Embedding model: {self.strategy.embedding_model.model_id}")
        console.print(
            f"  Input dimensions: {self.strategy.embedding_model.dim} (dynamic from embedding model)"
        )

        # Get hidden dimensions from model
        hidden_dims_str = " → ".join(map(str, HIDDEN_DIMS))
        console.print(
            f"  Shared trunk: {self.strategy.embedding_model.dim} → {hidden_dims_str}"
        )
        console.print(
            f"  Question-specific heads: {len(self.strategy.question_misconceptions)}"
        )

        # Display misconception head info
        misconception_heads = len(self.strategy.model.misconception_heads)
        console.print(f"  Misconception heads: {misconception_heads}")

    def _display_question_misconceptions(self, console: Console) -> None:
        """Display question-specific misconceptions.

        Args:
            console: Rich console for output
        """
        console.print("\n[cyan]Question-specific misconceptions:[/cyan]")
        for qid, misconceptions in sorted(
            self.strategy.question_misconceptions.items()
        ):
            # Convert misconceptions to strings for display
            misconception_str = ", ".join([str(m) for m in misconceptions])
            console.print(f"  Q{qid}: [{misconception_str}]")

    def demonstrate_predictions(self, console: Console) -> None:
        """Show sample predictions to validate the model works.

        Args:
            console: Rich console for output
        """
        # Test prediction format with a sample
        sample_test_row = EvaluationRow(
            row_id=99999,
            question_id=next(
                iter(self.strategy.question_misconceptions.keys())
            ),  # Use first available question
            question_text="Sample question",
            mc_answer="Sample answer",
            student_explanation="Sample explanation",
        )

        sample_prediction = self.strategy.predict(sample_test_row)

        console.print("\n[bold]Sample MLP Prediction Test[/bold]")
        console.print(f"Row ID: {sample_prediction.row_id}")
        console.print(
            f"Predictions: {[str(pred) for pred in sample_prediction.predicted_categories]}"
        )

    @staticmethod
    def evaluate_on_split(
        strategy: "MLPStrategy", eval_data: list[TrainingRow] | None = None
    ) -> dict[str, float]:
        """Evaluate model performance on eval split.

        Args:
            strategy: Trained MLPStrategy model
            eval_data: Evaluation data (uses stored split if None)

        Returns:
            Dictionary with evaluation metrics

        Raises:
            ValueError: If no evaluation data is available
        """
        if eval_data is None:
            eval_data = getattr(strategy.__class__, "_eval_data", None)

        if eval_data is None:
            msg = "No evaluation data available. Train with train_split < 1.0 first."
            raise ValueError(msg)

        logger.info(f"Evaluating model on {len(eval_data)} validation samples")

        # Convert training rows to evaluation rows for prediction
        eval_rows = [
            EvaluationRow(
                row_id=row.row_id,
                question_id=row.question_id,
                question_text=row.question_text,
                mc_answer=row.mc_answer,
                student_explanation=row.student_explanation,
            )
            for row in eval_data
        ]

        # Make predictions
        predictions = [strategy.predict(row) for row in eval_rows]

        # Use existing evaluation pipeline with temporary files
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create ground truth CSV
            ground_truth_data = [
                {
                    "row_id": row.row_id,
                    "Category": row.category.value,
                    "Misconception": str(row.misconception)
                    if row.misconception is not None
                    else "NA",
                }
                for row in eval_data
            ]

            ground_truth_df = pd.DataFrame(ground_truth_data)
            ground_truth_path = tmp_path / "ground_truth.csv"
            ground_truth_df.to_csv(ground_truth_path, index=False)

            # Create submission CSV
            submission_data = []
            for pred in predictions:
                prediction_strs = [str(p) for p in pred.predicted_categories]
                submission_data.append(
                    {"row_id": pred.row_id, "predictions": " ".join(prediction_strs)}
                )

            submission_df = pd.DataFrame(submission_data)
            submission_path = tmp_path / "submission.csv"
            submission_df.to_csv(submission_path, index=False)

            # Use existing evaluate function
            eval_result = evaluate(ground_truth_path, submission_path)

            logger.info(
                "Evaluation completed",
                map_score=eval_result.map_score,
                total_observations=eval_result.total_observations,
                perfect_predictions=eval_result.perfect_predictions,
                accuracy=eval_result.perfect_predictions
                / eval_result.total_observations
                if eval_result.total_observations > 0
                else 0.0,
            )

            return {
                "map_score": eval_result.map_score,
                "total_observations": eval_result.total_observations,
                "perfect_predictions": eval_result.perfect_predictions,
                "accuracy": eval_result.perfect_predictions
                / eval_result.total_observations
                if eval_result.total_observations > 0
                else 0.0,
            }
