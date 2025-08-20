"""Baseline frequency-based strategy for student misconception prediction."""

import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

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

    def predict(self, evaluation_row: EvaluationRow) -> SubmissionRow:
        """Make predictions for a single evaluation row.

        Args:
            evaluation_row: Single evaluation row to predict on

        Returns:
            Submission row with prediction (up to 3 categories)
        """
        logger.debug(f"Making baseline prediction for row {evaluation_row.row_id}")
        prediction_strings = self._predict_categories_for_row(evaluation_row)
        return SubmissionRow(
            row_id=evaluation_row.row_id, predicted_categories=prediction_strings[:3]
        )

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
