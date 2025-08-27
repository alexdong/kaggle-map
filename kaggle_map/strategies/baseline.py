"""Baseline frequency-based strategy for student misconception prediction."""

import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from kaggle_map.core.dataset import (
    build_category_frequencies,
    extract_correct_answers,
    extract_most_common_misconceptions,
    is_answer_correct,
    parse_training_data,
)
from kaggle_map.core.metrics import calculate_map_at_3
from kaggle_map.core.models import (
    Answer,
    Category,
    EvaluationRow,
    Misconception,
    Prediction,
    QuestionId,
    SubmissionRow,
)

from .base import Strategy
from .utils import TRAIN_RATIO, ModelParameters, split_training_data


@dataclass(frozen=True)
class BaselineStrategy(Strategy):
    """Baseline model for predicting student misconceptions.

    Uses frequency-based approach with category patterns by answer correctness
    and most common misconceptions per question.
    """

    correct_answers: dict[QuestionId, Answer]
    category_frequencies: dict[QuestionId, dict[bool, list[Category]]]
    common_misconceptions: dict[QuestionId, Misconception]

    @property
    def name(self) -> str:
        return "baseline"

    @property
    def description(self) -> str:
        return "Frequency-based model using category patterns and common misconceptions"

    @classmethod
    def fit(
        cls,
        *,
        train_split: float = TRAIN_RATIO,
        random_seed: int = 42,
        train_csv_path: Path = Path("datasets/train.csv"),
    ) -> "BaselineStrategy":
        logger.info(f"Fitting baseline strategy from {train_csv_path}")
        logger.info(f"Using train_split={train_split}, random_seed={random_seed}")

        # Parse all training data
        all_training_data = parse_training_data(train_csv_path)
        logger.debug(f"Parsed {len(all_training_data)} total training rows")

        # Split the data
        train_data, val_data, test_data = split_training_data(
            all_training_data, train_ratio=train_split, random_seed=random_seed
        )
        logger.info(
            f"Data split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}"
        )

        # Train only on the training split
        correct_answers = extract_correct_answers(train_data)
        logger.debug(f"Found correct answers for {len(correct_answers)} questions")

        category_frequencies = build_category_frequencies(train_data, correct_answers)
        common_misconceptions = extract_most_common_misconceptions(train_data)

        return cls(
            correct_answers=correct_answers,
            category_frequencies=category_frequencies,
            common_misconceptions=common_misconceptions,
        )

    def predict(self, evaluation_row: EvaluationRow) -> SubmissionRow:
        logger.debug(f"Making baseline prediction for row {evaluation_row.row_id}")
        prediction_strings = self._predict_categories_for_row(evaluation_row)
        return SubmissionRow(row_id=evaluation_row.row_id, predicted_categories=prediction_strings[:3])

    def save(self, filepath: Path, parameters: ModelParameters | None = None) -> None:
        """Save model and optionally its training parameters."""
        logger.info(f"Saving baseline model to {filepath}")

        # Save the model pickle/json
        with filepath.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

        # Save parameters if provided
        if parameters:
            params_path = filepath.with_suffix(".params.json")
            logger.info(f"Saving model parameters to {params_path}")
            with params_path.open("w") as f:
                json.dump(parameters.model_dump(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> tuple["BaselineStrategy", ModelParameters | None]:
        """Load model and its parameters if available."""
        logger.info(f"Loading baseline model from {filepath}")
        assert filepath.exists(), f"Model file not found: {filepath}"

        with filepath.open("r") as f:
            data = json.load(f)
        model = cls.from_dict(data)

        # Try to load parameters
        params = None
        params_path = filepath.with_suffix(".params.json")
        if params_path.exists():
            logger.info(f"Loading model parameters from {params_path}")
            with params_path.open("r") as f:
                params_data = json.load(f)
                params = ModelParameters.model_validate(params_data)
        else:
            logger.warning(f"No parameters file found at {params_path}")

        return model, params

    # Implementation methods

    def _predict_categories_for_row(self, row: EvaluationRow) -> list[Prediction]:
        is_correct = self._is_answer_correct(row.question_id, row.mc_answer)

        # Get ordered categories based on correctness
        assert row.question_id in self.category_frequencies, f"Question {row.question_id} not found in training data"
        categories = self.category_frequencies[row.question_id].get(is_correct, [])

        # Apply misconception suffix transformation
        return self._apply_misconception_suffix(categories, self.common_misconceptions.get(row.question_id))

    def _is_answer_correct(self, question_id: QuestionId, student_answer: Answer) -> bool:
        return is_answer_correct(question_id, student_answer, self.correct_answers)

    @classmethod
    def evaluate_on_split(
        cls,
        model: "BaselineStrategy",
        *,
        train_split: float = TRAIN_RATIO,
        random_seed: int = 42,
        train_csv_path: Path = Path("datasets/train.csv"),
    ) -> dict[str, float]:
        """Evaluate model on validation split using MAP@3 metric."""
        logger.info("Evaluating baseline model on validation split")
        logger.info(f"Using train_split={train_split}, random_seed={random_seed}")

        # Parse and split data
        all_training_data = parse_training_data(train_csv_path)
        train_data, val_data, test_data = split_training_data(
            all_training_data, train_ratio=train_split, random_seed=random_seed
        )

        logger.info(f"Evaluating on {len(val_data)} validation samples")

        # Calculate MAP@3 for each validation sample
        map_scores = []
        for row in val_data:
            # Create evaluation row (without ground truth)
            eval_row = EvaluationRow(
                row_id=row.row_id,
                question_id=row.question_id,
                question_text=row.question_text,
                mc_answer=row.mc_answer,
                student_explanation=row.student_explanation,
            )

            # Get predictions
            submission = model.predict(eval_row)
            predictions = submission.predicted_categories

            # Calculate MAP@3
            ground_truth = row.prediction
            score = calculate_map_at_3(ground_truth, predictions)
            map_scores.append(score)

        # Calculate average MAP@3
        avg_map = sum(map_scores) / len(map_scores) if map_scores else 0.0

        results = {
            "validation_map@3": avg_map,
            "validation_samples": len(val_data),
        }

        logger.info(f"Validation MAP@3: {avg_map:.4f} on {len(val_data)} samples")

        return results

    def _apply_misconception_suffix(self, categories: list[Category], misconception: Misconception) -> list[Prediction]:
        result = []
        for category in categories:
            if category.is_misconception and misconception != "NA":
                # Misconception categories get the actual misconception name
                result.append(Prediction(category=category, misconception=misconception))
            else:
                # All other categories get :NA suffix
                result.append(Prediction(category=category))
        return result

    def to_dict(self) -> dict:
        return {
            "correct_answers": self.correct_answers,
            "category_frequencies": {
                str(qid): {str(is_correct): [cat.value for cat in cats] for is_correct, cats in freq_map.items()}
                for qid, freq_map in self.category_frequencies.items()
            },
            "common_misconceptions": self.common_misconceptions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BaselineStrategy":
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
            common_misconceptions={int(k): v for k, v in data["common_misconceptions"].items()},
        )
