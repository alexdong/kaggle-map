"""Core data structures for the Kaggle student misconception prediction competition."""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import click
import pandas as pd
from loguru import logger
from pydantic import BaseModel, field_validator
from rich.console import Console
from rich.table import Table

# Domain-specific type aliases
type QuestionId = int
type Answer = str
type Misconception = str

# Display constants
_MAX_DISPLAY_CONTEXTS = 5


class ResponseContext(NamedTuple):
    """The fundamental unit for probability modeling.

    Represents the context in which a student response occurs:
    - What question they were answering
    - What they selected as their answer
    - What the correct answer actually is

    This triple uniquely defines the "response state" for probability calculations.
    """

    question_id: QuestionId
    selected_answer: Answer
    correct_answer: Answer

    @property
    def is_correct_selection(self) -> bool:
        """True if the student selected the correct answer."""
        return self.selected_answer == self.correct_answer


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

    def __hash__(self) -> int:
        """Make Prediction hashable for use as dictionary keys."""
        return hash((self.category, self.misconception))

    def __eq__(self, other: object) -> bool:
        """Define equality for Prediction objects."""
        if not isinstance(other, Prediction):
            return False
        return (
            self.category == other.category
            and self.misconception == other.misconception
        )


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


class EvaluationResult(BaseModel):
    """MAP@3 evaluation result with detailed breakdown."""

    map_score: float
    total_observations: int
    perfect_predictions: int

    @field_validator("map_score")
    @classmethod
    def validate_map_score(cls, v: float) -> float:
        assert 0.0 <= v <= 1.0, f"MAP score must be between 0 and 1, got {v}"
        return v


# Probabilistic Model Data Structures

type CategoryDistribution = dict[Category, float]
type MisconceptionDistribution = dict[Misconception, float]
type StateCategory = tuple[ResponseContext, Category]


@dataclass(frozen=True)
class ProbabilisticMAPModel:
    """Two-stage probabilistic model for student misconception prediction.

    Models P(Category, Misconception | Question, Selected, Correct) as:
    P(Category | Context) * P(Misconception | Category, Context)

    Where Context = (QuestionId, Selected_Answer, Correct_Answer)
    """

    # Stage 1: P(Category | ResponseContext)
    category_distributions: dict[ResponseContext, CategoryDistribution]

    # Stage 2: P(Misconception | Category, ResponseContext)
    misconception_distributions: dict[StateCategory, MisconceptionDistribution]

    # Fallback distributions for unseen contexts
    global_category_prior: CategoryDistribution
    global_misconception_prior: dict[Category, MisconceptionDistribution]
    question_category_priors: dict[QuestionId, CategoryDistribution]

    @classmethod
    def fit(
        cls, train_csv_path: Path = Path("dataset/train.csv")
    ) -> "ProbabilisticMAPModel":
        """Build probabilistic model from training data.

        Args:
            train_csv_path: Path to train.csv (default: dataset/train.csv)

        Returns:
            Trained ProbabilisticMAPModel
        """
        logger.info(f"Fitting probabilistic model from {train_csv_path}")
        training_data = cls._parse_training_data(train_csv_path)
        logger.debug(f"Parsed {len(training_data)} training rows")

        # Extract correct answers first
        correct_answers = cls._extract_correct_answers(training_data)
        logger.debug(f"Found correct answers for {len(correct_answers)} questions")

        # Create ResponseContexts for all training data
        contexts_with_labels = cls._create_response_contexts(
            training_data, correct_answers
        )
        logger.debug(f"Created {len(contexts_with_labels)} response contexts")

        # Learn stage 1: P(Category | ResponseContext)
        category_distributions = cls._learn_category_distributions(contexts_with_labels)
        global_category_prior = cls._compute_global_category_prior(contexts_with_labels)
        question_category_priors = cls._compute_question_category_priors(
            contexts_with_labels
        )

        # Learn stage 2: P(Misconception | Category, ResponseContext)
        misconception_distributions = cls._learn_misconception_distributions(
            contexts_with_labels
        )
        global_misconception_prior = cls._compute_global_misconception_prior(
            contexts_with_labels
        )

        return cls(
            category_distributions=category_distributions,
            misconception_distributions=misconception_distributions,
            global_category_prior=global_category_prior,
            global_misconception_prior=global_misconception_prior,
            question_category_priors=question_category_priors,
        )

    def predict(self, test_data: list[TestRow]) -> list[SubmissionRow]:
        """Make probabilistic predictions for test data.

        For each test row, computes P(Category, Misconception | Context)
        and returns top 3 most likely predictions.

        Args:
            test_data: List of test rows

        Returns:
            List of predictions with up to 3 categories each, ordered by probability
        """
        logger.info(f"Making probabilistic predictions for {len(test_data)} test rows")
        predictions = []

        for row in test_data:
            # Create context (we need to find correct answer)
            context = self._create_test_context(row)

            # Get all possible category-misconception combinations with probabilities
            prediction_probs = self._compute_prediction_probabilities(context)

            # Sort by probability and take top 3
            sorted_predictions = sorted(
                prediction_probs.items(), key=lambda x: x[1], reverse=True
            )
            top_predictions = [pred for pred, _ in sorted_predictions[:3]]

            predictions.append(
                SubmissionRow(row_id=row.row_id, predicted_categories=top_predictions)
            )

        return predictions

    # Private implementation methods

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
        return correct_answers

    @staticmethod
    def _create_response_contexts(
        training_data: list[TrainingRow], correct_answers: dict[QuestionId, Answer]
    ) -> list[tuple[ResponseContext, Category, Misconception | None]]:
        """Create ResponseContext objects for all training data."""
        contexts_with_labels = []

        for row in training_data:
            # Skip if we don't know the correct answer
            if row.question_id not in correct_answers:
                continue

            context = ResponseContext(
                question_id=row.question_id,
                selected_answer=row.mc_answer,
                correct_answer=correct_answers[row.question_id],
            )

            contexts_with_labels.append((context, row.category, row.misconception))

        return contexts_with_labels

    @staticmethod
    def _learn_category_distributions(
        contexts_with_labels: list[
            tuple[ResponseContext, Category, Misconception | None]
        ],
    ) -> dict[ResponseContext, CategoryDistribution]:
        """Learn P(Category | ResponseContext) from training data."""
        # Group by context and count categories
        context_category_counts = defaultdict(lambda: defaultdict(int))

        for context, category, _ in contexts_with_labels:
            context_category_counts[context][category] += 1

        # Convert counts to probabilities
        category_distributions = {}
        for context, category_counts in context_category_counts.items():
            total = sum(category_counts.values())
            assert total > 0, f"No observations for context {context}"

            distribution = {
                category: count / total for category, count in category_counts.items()
            }
            category_distributions[context] = distribution

        logger.debug(
            f"Learned category distributions for {len(category_distributions)} contexts"
        )
        return category_distributions

    @staticmethod
    def _learn_misconception_distributions(
        contexts_with_labels: list[
            tuple[ResponseContext, Category, Misconception | None]
        ],
    ) -> dict[StateCategory, MisconceptionDistribution]:
        """Learn P(Misconception | Category, ResponseContext) from training data."""
        # Group by (context, category) and count misconceptions
        state_misconception_counts = defaultdict(lambda: defaultdict(int))

        for context, category, misconception in contexts_with_labels:
            state_category = (context, category)
            # Use "NA" for None misconceptions
            misconception_key = misconception if misconception is not None else "NA"
            state_misconception_counts[state_category][misconception_key] += 1

        # Convert counts to probabilities
        misconception_distributions = {}
        for state_category, misconception_counts in state_misconception_counts.items():
            total = sum(misconception_counts.values())
            assert total > 0, f"No observations for state {state_category}"

            distribution = {
                misconception: count / total
                for misconception, count in misconception_counts.items()
            }
            misconception_distributions[state_category] = distribution

        logger.debug(
            f"Learned misconception distributions for {len(misconception_distributions)} state-categories"
        )
        return misconception_distributions

    @staticmethod
    def _compute_global_category_prior(
        contexts_with_labels: list[
            tuple[ResponseContext, Category, Misconception | None]
        ],
    ) -> CategoryDistribution:
        """Compute global prior P(Category) for fallback."""
        category_counts = defaultdict(int)

        for _, category, _ in contexts_with_labels:
            category_counts[category] += 1

        total = sum(category_counts.values())
        return {category: count / total for category, count in category_counts.items()}

    @staticmethod
    def _compute_question_category_priors(
        contexts_with_labels: list[
            tuple[ResponseContext, Category, Misconception | None]
        ],
    ) -> dict[QuestionId, CategoryDistribution]:
        """Compute per-question priors P(Category | QuestionId) for fallback."""
        question_category_counts = defaultdict(lambda: defaultdict(int))

        for context, category, _ in contexts_with_labels:
            question_category_counts[context.question_id][category] += 1

        question_priors = {}
        for question_id, category_counts in question_category_counts.items():
            total = sum(category_counts.values())
            question_priors[question_id] = {
                category: count / total for category, count in category_counts.items()
            }

        return question_priors

    @staticmethod
    def _compute_global_misconception_prior(
        contexts_with_labels: list[
            tuple[ResponseContext, Category, Misconception | None]
        ],
    ) -> dict[Category, MisconceptionDistribution]:
        """Compute global misconception priors P(Misconception | Category) for fallback."""
        category_misconception_counts = defaultdict(lambda: defaultdict(int))

        for _, category, misconception in contexts_with_labels:
            misconception_key = misconception if misconception is not None else "NA"
            category_misconception_counts[category][misconception_key] += 1

        global_prior = {}
        for category, misconception_counts in category_misconception_counts.items():
            total = sum(misconception_counts.values())
            global_prior[category] = {
                misconception: count / total
                for misconception, count in misconception_counts.items()
            }

        return global_prior

    def _create_test_context(self, test_row: TestRow) -> ResponseContext:
        """Create ResponseContext for a test row by finding the correct answer."""
        # Find correct answer from our learned knowledge
        correct_answer = None

        # Look through all known contexts for this question to find the correct answer
        for context in self.category_distributions:
            if context.question_id == test_row.question_id:
                correct_answer = context.correct_answer
                break

        assert correct_answer is not None, (
            f"No correct answer found for question {test_row.question_id}"
        )

        return ResponseContext(
            question_id=test_row.question_id,
            selected_answer=test_row.mc_answer,
            correct_answer=correct_answer,
        )

    def _compute_prediction_probabilities(
        self, context: ResponseContext
    ) -> dict[Prediction, float]:
        """Compute P(Category, Misconception | Context) for all possible predictions."""
        # Stage 1: Get P(Category | Context) with fallbacks
        category_probs = self._get_category_probabilities(context)

        # Stage 2: For each category, get P(Misconception | Category, Context)
        prediction_probs = {}

        for category, category_prob in category_probs.items():
            misconception_probs = self._get_misconception_probabilities(
                context, category
            )

            for misconception, misconception_prob in misconception_probs.items():
                # Joint probability: P(Category, Misconception | Context)
                joint_prob = category_prob * misconception_prob

                # Create prediction object
                prediction = Prediction(
                    category=category,
                    misconception=misconception if misconception != "NA" else None,
                )

                prediction_probs[prediction] = joint_prob

        return prediction_probs

    def _get_category_probabilities(
        self, context: ResponseContext
    ) -> CategoryDistribution:
        """Get P(Category | Context) with graceful fallbacks."""
        # Try exact context match first
        if context in self.category_distributions:
            return self.category_distributions[context]

        # Fallback to question-specific prior
        if context.question_id in self.question_category_priors:
            logger.debug(f"Using question-specific prior for unseen context: {context}")
            return self.question_category_priors[context.question_id]

        # Fallback to global prior
        logger.debug(f"Using global prior for unseen question: {context.question_id}")
        return self.global_category_prior

    def _get_misconception_probabilities(
        self, context: ResponseContext, category: Category
    ) -> MisconceptionDistribution:
        """Get P(Misconception | Category, Context) with graceful fallbacks."""
        state_category = (context, category)

        # Try exact state match first
        if state_category in self.misconception_distributions:
            return self.misconception_distributions[state_category]

        # Fallback to global misconception prior for this category
        if category in self.global_misconception_prior:
            logger.debug(
                f"Using global misconception prior for unseen state: {state_category}"
            )
            return self.global_misconception_prior[category]

        # Ultimate fallback: everything is "NA"
        logger.debug(f"Using NA fallback for category {category}")
        return {"NA": 1.0}


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


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Show detailed model contents")
@click.option("--probabilistic", "-p", is_flag=True, help="Use new probabilistic model")
def main(*, verbose: bool, probabilistic: bool) -> None:
    """Demonstrate model fitting and basic validation."""
    console = Console()

    if probabilistic:
        with console.status("[bold green]Fitting probabilistic model..."):
            logger.info("Running probabilistic models.py demonstration")
            model = ProbabilisticMAPModel.fit()
            logger.info("Probabilistic model fitting completed successfully")

        console.print(
            "✅ [bold green]Probabilistic model fitting completed successfully[/bold green]"
        )
        _display_probabilistic_model_stats(console, model)

        if verbose:
            _display_detailed_probabilistic_contents(console, model)

        _demonstrate_probabilistic_predictions(console, model)
    else:
        with console.status("[bold green]Fitting baseline model..."):
            logger.info("Running baseline models.py demonstration")
            model = MAPModel.fit()
            logger.info("Baseline model fitting completed successfully")

        console.print(
            "✅ [bold green]Baseline model fitting completed successfully[/bold green]"
        )

        # Display model statistics
        _display_model_stats(console, model)

        if verbose:
            _display_detailed_model_contents(console, model)

        _save_and_validate_model(console, model)
        _demonstrate_prediction_types(console, model)


def _display_model_stats(console: Console, model: "MAPModel") -> None:
    """Display model statistics in a table."""
    stats_table = Table(title="Model Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Count", style="magenta")

    stats_table.add_row(
        "Questions with correct answers", str(len(model.correct_answers))
    )
    stats_table.add_row(
        "Questions with category patterns", str(len(model.category_frequencies))
    )
    stats_table.add_row(
        "Questions with misconception data", str(len(model.common_misconceptions))
    )

    console.print(stats_table)


def _display_detailed_model_contents(console: Console, model: "MAPModel") -> None:  # noqa: C901
    """Display detailed model contents when verbose is enabled."""
    # Print detailed model contents
    console.print("\n[bold]Detailed Model Contents[/bold]")

    # Correct answers
    console.print("\n[cyan]Questions with correct answers:[/cyan]")
    for qid, answer in sorted(model.correct_answers.items()):
        console.print(f"  Question {qid}: {answer}")

    # Category patterns
    console.print(
        f"\n[cyan]Category patterns for {len(model.category_frequencies)} questions:[/cyan]"
    )
    for qid, patterns in sorted(model.category_frequencies.items()):
        console.print(f"  Question {qid}:")
        if True in patterns:
            correct_cats = [cat.value for cat in patterns[True]]
            console.print(f"    When correct: {correct_cats}", style="green")
        if False in patterns:
            incorrect_cats = [cat.value for cat in patterns[False]]
            console.print(f"    When incorrect: {incorrect_cats}", style="red")

    # Misconceptions
    console.print(
        f"\n[cyan]Most common misconceptions for {len(model.common_misconceptions)} questions:[/cyan]"
    )
    misconception_count = 0
    for qid, misconception in sorted(model.common_misconceptions.items()):
        if misconception is not None:
            console.print(f"  Question {qid}: {misconception}")
            misconception_count += 1

    if misconception_count == 0:
        console.print("  (No misconceptions found in the data)", style="dim")
    else:
        console.print(
            f"  ({misconception_count} questions have misconceptions)", style="green"
        )


def _save_and_validate_model(console: Console, model: "MAPModel") -> Path:
    """Save model and validate serialization."""
    model_path = Path("baseline_model.json")

    with console.status("[bold blue]Saving and validating model..."):
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

    console.print(f"✅ [bold green]Model saved to {model_path}[/bold green]")
    console.print("✅ [bold green]Serialization validation passed[/bold green]")
    return model_path


def _demonstrate_prediction_types(console: Console, model: "MAPModel") -> None:
    """Demonstrate type-safe prediction creation and model usage."""
    # Test prediction format with a sample
    sample_test_row = TestRow(
        row_id=99999,
        question_id=31772,
        question_text="Sample question",
        mc_answer="\\( \\frac{1}{3} \\)",
        student_explanation="Sample explanation",
    )
    sample_predictions = model.predict([sample_test_row])

    console.print("\n[bold]Sample Prediction Test[/bold]")
    console.print(f"Row ID: {sample_predictions[0].row_id}")
    console.print(
        f"Predictions: {[str(pred) for pred in sample_predictions[0].predicted_categories]}"
    )

    # Test prediction creation with type safety
    test_pred1 = Prediction(category=Category.TRUE_CORRECT)
    test_pred2 = Prediction(
        category=Category.FALSE_MISCONCEPTION, misconception="TestError"
    )
    console.print("\n[bold green]✅ Type-safe creation works:[/bold green]")
    console.print(f"  {test_pred1} -> {test_pred1.value}")
    console.print(f"  {test_pred2} -> {test_pred2.value}")

    # Test that misconception is ignored for non-misconception categories
    test_pred3 = Prediction(
        category=Category.TRUE_CORRECT, misconception="ShouldBeIgnored"
    )
    console.print(
        f"  {test_pred3} -> {test_pred3.value} (misconception ignored for non-misconception category)"
    )


def _display_probabilistic_model_stats(
    console: Console, model: ProbabilisticMAPModel
) -> None:
    """Display probabilistic model statistics in a table."""
    stats_table = Table(title="Probabilistic Model Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Count", style="magenta")

    stats_table.add_row(
        "Unique response contexts", str(len(model.category_distributions))
    )
    stats_table.add_row(
        "State-category combinations", str(len(model.misconception_distributions))
    )
    stats_table.add_row(
        "Questions with priors", str(len(model.question_category_priors))
    )
    stats_table.add_row("Global categories", str(len(model.global_category_prior)))

    console.print(stats_table)


def _display_detailed_probabilistic_contents(
    console: Console, model: ProbabilisticMAPModel
) -> None:
    """Display detailed probabilistic model contents when verbose is enabled."""
    console.print("\n[bold]Detailed Probabilistic Model Contents[/bold]")

    # Show some example response contexts
    console.print(
        f"\n[cyan]Sample response contexts ({min(5, len(model.category_distributions))}):[/cyan]"
    )
    for i, (context, category_dist) in enumerate(model.category_distributions.items()):
        if i >= _MAX_DISPLAY_CONTEXTS:  # Show only first few
            break
        console.print(
            f"  Context: Q{context.question_id}, '{context.selected_answer}' (correct: '{context.correct_answer}')"
        )

        # Show top 2 categories for this context
        sorted_cats = sorted(category_dist.items(), key=lambda x: x[1], reverse=True)[
            :2
        ]
        for category, prob in sorted_cats:
            console.print(f"    {category.value}: {prob:.3f}")

    # Show global priors
    console.print("\n[cyan]Global category priors:[/cyan]")
    sorted_global = sorted(
        model.global_category_prior.items(), key=lambda x: x[1], reverse=True
    )
    for category, prob in sorted_global:
        console.print(f"  {category.value}: {prob:.3f}")


def _demonstrate_probabilistic_predictions(
    console: Console, model: ProbabilisticMAPModel
) -> None:
    """Demonstrate probabilistic prediction creation and model usage."""
    # Test prediction with a sample that should exist
    sample_test_row = TestRow(
        row_id=99999,
        question_id=31772,  # This should exist in our training data
        question_text="Sample question",
        mc_answer="\\( \\frac{1}{3} \\)",
        student_explanation="Sample explanation",
    )

    try:
        sample_predictions = model.predict([sample_test_row])

        console.print("\n[bold]Sample Probabilistic Prediction Test[/bold]")
        console.print(f"Row ID: {sample_predictions[0].row_id}")
        console.print("Top predictions with implicit probabilities:")
        for pred in sample_predictions[0].predicted_categories:
            console.print(f"  {pred}")

        console.print("\n[bold green]✅ Probabilistic prediction works![/bold green]")
    except Exception as e:
        console.print(f"\n[bold red]❌ Probabilistic prediction failed: {e}[/bold red]")


if __name__ == "__main__":
    main()
