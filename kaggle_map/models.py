"""Core data structures for the Kaggle student misconception prediction competition."""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from loguru import logger
from pydantic import BaseModel

# Domain-specific type aliases
type QuestionId = int
type Answer = str
type Misconception = str


class Category(Enum):
    """All possible categories in the competition."""

    TRUE_CORRECT = "True_Correct"
    TRUE_NEITHER = "True_Neither"
    TRUE_MISCONCEPTION = "True_Misconception"
    FALSE_CORRECT = "False_Correct"
    FALSE_NEITHER = "False_Neither"
    FALSE_MISCONCEPTION = "False_Misconception"

    @property
    def is_misconception(self) -> bool:
        """Check if this category involves a misconception."""
        return self.value.endswith("_Misconception")

    @property
    def is_correct_answer(self) -> bool:
        """Check if this represents a correct answer."""
        return self.value.startswith("True_")


class Prediction(BaseModel):
    """A prediction in 'Category:Misconception' format for submission."""

    category: Category
    misconception: Misconception | None = None

    @property
    def value(self) -> str:
        """Return the prediction string in 'Category:Misconception' format."""
        if self.category.is_misconception and self.misconception is not None:
            return f"{self.category.value}:{self.misconception}"
        return f"{self.category.value}:NA"

    def __str__(self) -> str:
        """Return the prediction string for easy use."""
        return self.value

    def __repr__(self) -> str:
        """Return a clear representation."""
        return f"Prediction(category={self.category}, misconception={self.misconception!r})"


@dataclass(frozen=True)
class TrainingRow:
    """Single row from train.csv."""

    row_id: int
    question_id: QuestionId
    question_text: str
    mc_answer: Answer
    student_explanation: str
    category: Category
    misconception: Misconception | None


@dataclass(frozen=True)
class TestRow:
    """Single row from test.csv."""

    row_id: int
    question_id: QuestionId
    question_text: str
    mc_answer: Answer
    student_explanation: str


class SubmissionRow(NamedTuple):
    """Prediction result for MAP@3 evaluation."""

    row_id: int
    predicted_categories: list[Prediction]  # Max 3, ordered by confidence


@dataclass(frozen=True)
class MAPModel:
    """Baseline model for predicting student misconceptions."""

    correct_answers: dict[QuestionId, Answer]
    category_frequencies: dict[QuestionId, dict[bool, list[Category]]]
    common_misconceptions: dict[QuestionId, Misconception | None]

    @classmethod
    def fit(cls, train_csv_path: Path = Path("dataset/train.csv")) -> "MAPModel":
        """Build model from training data.

        Args:
            train_csv_path: Path to train.csv (default: dataset/train.csv)

        Returns:
            Trained MAPModel
        """
        logger.info(f"Fitting model from {train_csv_path}")
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
        logger.info(f"Making predictions for {len(test_data)} test rows")
        predictions = []
        for row in test_data:
            prediction_strings = self._predict_categories_for_row(row)
            predictions.append(
                SubmissionRow(
                    row_id=row.row_id, predicted_categories=prediction_strings[:3]
                )
            )
        return predictions

    def _predict_categories_for_row(self, row: TestRow) -> list[Prediction]:
        """Predict ordered categories for a single test row."""
        is_correct = self._is_answer_correct(row.question_id, row.mc_answer)

        # Get ordered categories based on correctness
        if row.question_id in self.category_frequencies:
            categories = self.category_frequencies[row.question_id].get(is_correct, [])
        else:
            # Fallback for unseen questions
            categories = list(Category)
            logger.debug(
                f"Using fallback categories for unseen question {row.question_id}"
            )

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
    def _extract_correct_answers(
        training_data: list[TrainingRow],
    ) -> dict[QuestionId, Answer]:
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
    def from_dict(cls, data: dict) -> "MAPModel":
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


def save_model(model: MAPModel, filepath: Path) -> None:
    """Save model as JSON file."""
    logger.info(f"Saving model to {filepath}")
    with filepath.open("w") as f:
        json.dump(model.to_dict(), f, indent=2)


def load_model(filepath: Path) -> MAPModel:
    """Load model from JSON file."""
    logger.info(f"Loading model from {filepath}")
    assert filepath.exists(), f"Model file not found: {filepath}"

    with filepath.open("r") as f:
        data = json.load(f)
    return MAPModel.from_dict(data)


if __name__ == "__main__":
    """Demonstrate model fitting and basic validation."""
    logger.info("Running models.py demonstration")

    # Fit the model using default training data
    try:
        model = MAPModel.fit()
        logger.info("Model fitting completed successfully")

        # Display detailed model information
        logger.info(
            f"Model contains {len(model.correct_answers)} questions with correct answers"
        )
        logger.info(
            f"Model contains {len(model.category_frequencies)} questions with category patterns"
        )
        logger.info(
            f"Model contains {len(model.common_misconceptions)} questions with misconception data"
        )

        # Print detailed model contents
        print("\n=== MODEL DETAILS ===")
        print(f"Questions with correct answers: {len(model.correct_answers)}")
        for qid, answer in sorted(model.correct_answers.items()):
            print(f"  Question {qid}: {answer}")

        print(f"\nCategory patterns for {len(model.category_frequencies)} questions:")
        for qid, patterns in sorted(model.category_frequencies.items()):
            print(f"  Question {qid}:")
            if True in patterns:
                correct_cats = [cat.value for cat in patterns[True]]
                print(f"    When correct: {correct_cats}")
            if False in patterns:
                incorrect_cats = [cat.value for cat in patterns[False]]
                print(f"    When incorrect: {incorrect_cats}")

        print(
            f"\nMost common misconceptions for {len(model.common_misconceptions)} questions:"
        )
        misconception_count = 0
        for qid, misconception in sorted(model.common_misconceptions.items()):
            if misconception is not None:
                print(f"  Question {qid}: {misconception}")
                misconception_count += 1
        if misconception_count == 0:
            print("  (No misconceptions found in the data)")
        else:
            print(f"  ({misconception_count} questions have misconceptions)")
        print("========================\n")

        # Save the model for testing
        model_path = Path("baseline_model.json")
        save_model(model, model_path)
        logger.info(f"Model saved to {model_path}")

        # Load it back to verify serialization works
        loaded_model = load_model(model_path)
        logger.info("Model loading verification successful")

        # Basic validation that the loaded model matches
        assert len(loaded_model.correct_answers) == len(model.correct_answers)
        assert len(loaded_model.category_frequencies) == len(model.category_frequencies)
        assert len(loaded_model.common_misconceptions) == len(
            model.common_misconceptions
        )
        logger.info("Model serialization validation passed")

        # Test prediction format with a sample
        sample_test_row = TestRow(
            row_id=99999,
            question_id=31772,
            question_text="Sample question",
            mc_answer="\\( \\frac{1}{3} \\)",
            student_explanation="Sample explanation",
        )
        sample_predictions = model.predict([sample_test_row])
        print("\nSample prediction for question 31772:")
        print(f"  Row ID: {sample_predictions[0].row_id}")
        print(
            f"  Predictions: {[str(pred) for pred in sample_predictions[0].predicted_categories]}"
        )
        print(f"  Prediction objects: {sample_predictions[0].predicted_categories}")
        print("  Format validation: All predictions are valid Pydantic objects")

        # Test prediction creation with type safety
        test_pred1 = Prediction(category=Category.TRUE_CORRECT)
        test_pred2 = Prediction(
            category=Category.FALSE_MISCONCEPTION, misconception="TestError"
        )
        print("  ✅ Type-safe creation works:")
        print(f"    {test_pred1} -> {test_pred1.value}")
        print(f"    {test_pred2} -> {test_pred2.value}")

        # Test that misconception is ignored for non-misconception categories
        test_pred3 = Prediction(
            category=Category.TRUE_CORRECT, misconception="ShouldBeIgnored"
        )
        print(
            f"    {test_pred3} -> {test_pred3.value} (misconception ignored for non-misconception category)"
        )

        print("========================")

    except Exception as e:
        logger.error(f"Error during model demonstration: {e}")
        raise
