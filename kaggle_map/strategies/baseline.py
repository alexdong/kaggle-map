"""Baseline frequency-based strategy for student misconception prediction."""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table

from ..models import (
    Answer,
    Category,
    Misconception,
    Prediction,
    QuestionId,
    SubmissionRow,
    TestRow,
    TrainingRow,
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
    def fit(cls, train_csv_path: Path = Path("dataset/train.csv")) -> "BaselineStrategy":
        """Build model from training data.

        Args:
            train_csv_path: Path to train.csv (default: dataset/train.csv)

        Returns:
            Trained BaselineStrategy
        """
        logger.info(f"Fitting baseline strategy from {train_csv_path}")
        training_data = cls._parse_training_data(train_csv_path)
        logger.debug(f"Parsed {len(training_data)} training rows")

        correct_answers = cls._extract_correct_answers(training_data)
        logger.debug(f"Found correct answers for {len(correct_answers)} questions")

        category_frequencies = cls._build_category_frequencies(
            training_data, correct_answers
        )
        common_misconceptions = cls._extract_most_common_misconceptions(training_data)

        return cls(
            correct_answers=correct_answers,
            category_frequencies=category_frequencies,
            common_misconceptions=common_misconceptions,
        )

    def predict(self, test_data: list[TestRow]) -> list[SubmissionRow]:
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

        # Correct answers
        console.print("\\n[cyan]Questions with correct answers:[/cyan]")
        for qid, answer in sorted(self.correct_answers.items()):
            console.print(f"  Question {qid}: {answer}")

        # Category patterns
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

        # Misconceptions
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
                f"  ({misconception_count} questions have misconceptions)", style="green"
            )

    def demonstrate_predictions(self, console: Console) -> None:
        """Show sample predictions."""
        # Test prediction format with a sample
        sample_test_row = TestRow(
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

    def _predict_categories_for_row(self, row: TestRow) -> list[Prediction]:
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
        correct_answer = self.correct_answers.get(question_id, "")
        return student_answer == correct_answer

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

    @staticmethod
    def _parse_training_data(csv_path: Path) -> list[TrainingRow]:
        """Parse CSV into strongly-typed training rows."""
        assert csv_path.exists(), f"Training file not found: {csv_path}"

        training_df = pd.read_csv(csv_path)
        logger.debug(f"Loaded CSV with columns: {list(training_df.columns)}")
        assert not training_df.empty, "Training CSV cannot be empty"

        training_rows = []
        for _, row in training_df.iterrows():
            # Handle NaN misconceptions (pandas converts "NA" to NaN)
            misconception = (
                row["Misconception"] if pd.notna(row["Misconception"]) else None
            )

            training_rows.append(
                TrainingRow(
                    row_id=int(row["row_id"]),
                    question_id=int(row["QuestionId"]),
                    question_text=str(row["QuestionText"]),
                    mc_answer=str(row["MC_Answer"]),
                    student_explanation=str(row["StudentExplanation"]),
                    category=Category(row["Category"]),
                    misconception=misconception,
                )
            )

        logger.debug(f"Parsed {len(training_rows)} training rows")
        assert training_rows, "Must parse at least one training row"
        return training_rows

    @staticmethod
    def _extract_correct_answers(training_data: list[TrainingRow]) -> dict[QuestionId, Answer]:
        """Extract the correct answer for each question."""
        assert training_data, "Training data cannot be empty"

        correct_answers = {}

        for row in training_data:
            if row.category == Category.TRUE_CORRECT:
                if row.question_id in correct_answers:
                    assert correct_answers[row.question_id] == row.mc_answer, (
                        f"Conflicting correct answers for question {row.question_id}"
                    )
                else:
                    correct_answers[row.question_id] = row.mc_answer

        logger.debug(f"Extracted correct answers for {len(correct_answers)} questions")
        assert correct_answers, "Must find at least one correct answer"
        assert all(isinstance(qid, int) for qid in correct_answers), (
            "Question IDs must be integers"
        )
        return correct_answers

    @staticmethod
    def _build_category_frequencies(
        training_data: list[TrainingRow], correct_answers: dict[QuestionId, Answer]
    ) -> dict[QuestionId, dict[bool, list[Category]]]:
        """Build frequency-ordered category lists for correct/incorrect answers."""
        assert training_data, "Training data cannot be empty"
        assert correct_answers, "Correct answers cannot be empty"

        # Group by question and correctness
        question_correctness_categories = defaultdict(lambda: defaultdict(list))

        for row in training_data:
            is_correct = (
                row.question_id in correct_answers
                and row.mc_answer == correct_answers[row.question_id]
            )
            question_correctness_categories[row.question_id][is_correct].append(
                row.category
            )

        # Build frequency-ordered lists
        result = {}
        for question_id, correctness_map in question_correctness_categories.items():
            result[question_id] = {}
            for is_correct, categories in correctness_map.items():
                # Count frequencies and sort by most common
                category_counts = Counter(categories)
                ordered_categories = [
                    category for category, _ in category_counts.most_common()
                ]
                result[question_id][is_correct] = ordered_categories

        logger.debug(f"Built category frequencies for {len(result)} questions")
        assert isinstance(result, dict), "Result must be a dictionary"
        return result

    @staticmethod
    def _extract_most_common_misconceptions(
        training_data: list[TrainingRow],
    ) -> dict[QuestionId, Misconception | None]:
        """Find most common misconception per question."""
        assert training_data, "Training data cannot be empty"

        question_misconceptions = defaultdict(list)

        for row in training_data:
            if row.misconception is not None:
                question_misconceptions[row.question_id].append(row.misconception)

        result = {}
        for question_id, misconceptions in question_misconceptions.items():
            if misconceptions:
                most_common = Counter(misconceptions).most_common(1)[0][0]
                result[question_id] = most_common
            else:
                result[question_id] = None

        logger.debug(
            f"Extracted most common misconceptions for {len(result)} questions"
        )
        assert isinstance(result, dict), "Result must be a dictionary"
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