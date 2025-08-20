"""Baseline frequency-based strategy for student misconception prediction."""

import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.table import Table

from kaggle_map.dataset import (
    build_category_frequencies,
    extract_correct_answers,
    extract_most_common_misconceptions,
    is_answer_correct,
    parse_training_data,
)
from kaggle_map.models import (
    Answer,
    Category,
    EvaluationRow,
    Misconception,
    Prediction,
    QuestionId,
    SubmissionRow,
)

from .base import Strategy


@dataclass(frozen=True)
class BaselineStrategy(Strategy):
    """Baseline model for predicting student misconceptions.

    Uses frequency-based approach with category patterns by answer correctness
    and most common misconceptions per question.
    """

    correct_answers: dict[QuestionId, Answer]
    category_frequencies: dict[QuestionId, dict[bool, list[Category]]]
    common_misconceptions: dict[QuestionId, Misconception | None]

    @property
    def name(self) -> str:
        """Strategy name."""
        return "baseline"

    @property
    def description(self) -> str:
        """Strategy description."""
        return "Frequency-based model using category patterns and common misconceptions"

    @classmethod
    def fit(
        cls, train_csv_path: Path = Path("datasets/train.csv")
    ) -> "BaselineStrategy":
        """Build model from training data.

        Args:
            train_csv_path: Path to train.csv (default: dataset/train.csv)

        Returns:
            Trained BaselineStrategy
        """
        logger.info(f"Fitting baseline strategy from {train_csv_path}")
        training_data = parse_training_data(train_csv_path)
        logger.debug(f"Parsed {len(training_data)} training rows")

        correct_answers = extract_correct_answers(training_data)
        logger.debug(f"Found correct answers for {len(correct_answers)} questions")

        category_frequencies = build_category_frequencies(
            training_data, correct_answers
        )
        common_misconceptions = extract_most_common_misconceptions(training_data)

        return cls(
            correct_answers=correct_answers,
            category_frequencies=category_frequencies,
            common_misconceptions=common_misconceptions,
        )

    def predict(self, test_data: list[EvaluationRow]) -> list[SubmissionRow]:
        """Make predictions for test data.

        Args:
            test_data: List of test rows

        Returns:
            List of predictions with up to 3 categories each
        """
        logger.info(f"Making baseline predictions for {len(test_data)} test rows")
        predictions = []
        for row in test_data:
            prediction_strings = self._predict_categories_for_row(row)
            predictions.append(
                SubmissionRow(
                    row_id=row.row_id, predicted_categories=prediction_strings[:3]
                )
            )
        return predictions

    def save(self, filepath: Path) -> None:
        """Save model as JSON file."""
        logger.info(f"Saving baseline model to {filepath}")
        with filepath.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "BaselineStrategy":
        """Load model from JSON file."""
        logger.info(f"Loading baseline model from {filepath}")
        assert filepath.exists(), f"Model file not found: {filepath}"

        with filepath.open("r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def display_stats(self, console: Console) -> None:
        """Display model statistics."""
        stats_table = Table(title="Baseline Model Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Count", style="magenta")

        stats_table.add_row(
            "Questions with correct answers", str(len(self.correct_answers))
        )
        stats_table.add_row(
            "Questions with category patterns", str(len(self.category_frequencies))
        )
        stats_table.add_row(
            "Questions with misconception data", str(len(self.common_misconceptions))
        )

        console.print(stats_table)

    def display_detailed_info(self, console: Console) -> None:
        """Display detailed model info for verbose mode."""
        console.print("\\n[bold]Detailed Baseline Model Contents[/bold]")
        self._display_correct_answers(console)
        self._display_category_patterns(console)
        self._display_misconceptions_summary(console)

    def _display_correct_answers(self, console: Console) -> None:
        console.print("\\n[cyan]Questions with correct answers:[/cyan]")
        for qid, answer in sorted(self.correct_answers.items()):
            console.print(f"  Question {qid}: {answer}")

    def _display_category_patterns(self, console: Console) -> None:
        console.print(
            f"\\n[cyan]Category patterns for {len(self.category_frequencies)} questions:[/cyan]"
        )
        for qid, patterns in sorted(self.category_frequencies.items()):
            console.print(f"  Question {qid}:")
            if True in patterns:
                correct_cats = [cat.value for cat in patterns[True]]
                console.print(f"    When correct: {correct_cats}", style="green")
            if False in patterns:
                incorrect_cats = [cat.value for cat in patterns[False]]
                console.print(f"    When incorrect: {incorrect_cats}", style="red")

    def _display_misconceptions_summary(self, console: Console) -> None:
        console.print(
            f"\\n[cyan]Most common misconceptions for {len(self.common_misconceptions)} questions:[/cyan]"
        )
        misconception_count = 0
        for qid, misconception in sorted(self.common_misconceptions.items()):
            if misconception is not None:
                console.print(f"  Question {qid}: {misconception}")
                misconception_count += 1

        if misconception_count == 0:
            console.print("  (No misconceptions found in the data)", style="dim")
        else:
            console.print(
                f"  ({misconception_count} questions have misconceptions)",
                style="green",
            )

    def demonstrate_predictions(self, console: Console) -> None:
        """Show sample predictions."""
        # Test prediction format with a sample
        sample_test_row = EvaluationRow(
            row_id=99999,
            question_id=31772,
            question_text="Sample question",
            mc_answer="\\\\( \\\\frac{1}{3} \\\\)",
            student_explanation="Sample explanation",
        )
        sample_predictions = self.predict([sample_test_row])

        console.print("\\n[bold]Sample Baseline Prediction Test[/bold]")
        console.print(f"Row ID: {sample_predictions[0].row_id}")
        console.print(
            f"Predictions: {[str(pred) for pred in sample_predictions[0].predicted_categories]}"
        )

        # Test prediction creation with type safety
        test_pred1 = Prediction(category=Category.TRUE_CORRECT)
        test_pred2 = Prediction(
            category=Category.FALSE_MISCONCEPTION, misconception="TestError"
        )
        console.print("\\n[bold green]✅ Type-safe creation works:[/bold green]")
        console.print(f"  {test_pred1} -> {test_pred1.value}")
        console.print(f"  {test_pred2} -> {test_pred2.value}")

        # Test that misconception is ignored for non-misconception categories
        test_pred3 = Prediction(
            category=Category.TRUE_CORRECT, misconception="ShouldBeIgnored"
        )
        console.print(
            f"  {test_pred3} -> {test_pred3.value} (misconception ignored for non-misconception category)"
        )

    # Implementation methods

    def _predict_categories_for_row(self, row: EvaluationRow) -> list[Prediction]:
        """Predict ordered categories for a single test row."""
        is_correct = self._is_answer_correct(row.question_id, row.mc_answer)

        # Get ordered categories based on correctness
        assert row.question_id in self.category_frequencies, (
            f"Question {row.question_id} not found in training data"
        )
        categories = self.category_frequencies[row.question_id].get(is_correct, [])

        # Apply misconception suffix transformation
        return self._apply_misconception_suffix(
            categories, self.common_misconceptions.get(row.question_id)
        )

    def _is_answer_correct(
        self, question_id: QuestionId, student_answer: Answer
    ) -> bool:
        """Check if student answer matches the correct answer."""
        return is_answer_correct(question_id, student_answer, self.correct_answers)

    def _apply_misconception_suffix(
        self, categories: list[Category], misconception: Misconception | None
    ) -> list[Prediction]:
        """Create predictions in 'Category:Misconception' format for submission."""
        result = []
        for category in categories:
            if category.is_misconception and misconception is not None:
                # Misconception categories get the actual misconception name
                result.append(
                    Prediction(category=category, misconception=misconception)
                )
            else:
                # All other categories get :NA suffix
                result.append(Prediction(category=category))
        return result

    def to_dict(self) -> dict:
        """Convert to JSON-serializable format."""
        return {
            "correct_answers": self.correct_answers,
            "category_frequencies": {
                str(qid): {
                    str(is_correct): [cat.value for cat in cats]
                    for is_correct, cats in freq_map.items()
                }
                for qid, freq_map in self.category_frequencies.items()
            },
            "common_misconceptions": self.common_misconceptions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BaselineStrategy":
        """Load from JSON-serializable format."""
        # Convert category frequencies back
        category_frequencies = {}
        for qid_str, freq_map in data["category_frequencies"].items():
            qid = int(qid_str)
            category_frequencies[qid] = {}
            for is_correct_str, cat_values in freq_map.items():
                is_correct = is_correct_str == "True"
                categories = [Category(value) for value in cat_values]
                category_frequencies[qid][is_correct] = categories

        return cls(
            correct_answers={int(k): v for k, v in data["correct_answers"].items()},
            category_frequencies=category_frequencies,
            common_misconceptions={
                int(k): v for k, v in data["common_misconceptions"].items()
            },
        )
