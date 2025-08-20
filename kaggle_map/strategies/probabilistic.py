"""Probabilistic two-stage strategy for student misconception prediction."""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from kaggle_map.dataset import (
    create_response_contexts,
    extract_correct_answers,
    parse_training_data,
)
from kaggle_map.models import (
    Category,
    EvaluationRow,
    Misconception,
    Prediction,
    QuestionId,
    ResponseContext,
    SubmissionRow,
)

from .base import Strategy

# Type aliases for probabilistic model
type CategoryDistribution = dict[Category, float]
type MisconceptionDistribution = dict[Misconception, float]
type StateCategory = tuple[ResponseContext, Category]


# Limit for verbose display sections in dev tooling
_MAX_DISPLAY_CONTEXTS = 5


@dataclass(frozen=True)
class ProbabilisticStrategy(Strategy):
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

    @property
    def name(self) -> str:
        """Strategy name."""
        return "probabilistic"

    @property
    def description(self) -> str:
        """Strategy description."""
        return "Two-stage probabilistic model: P(Category|Context) x P(Misconception|Category,Context)"

    @classmethod
    def fit(
        cls, train_csv_path: Path = Path("dataset/train.csv")
    ) -> "ProbabilisticStrategy":
        """Build probabilistic model from training data.

        Args:
            train_csv_path: Path to train.csv (default: dataset/train.csv)

        Returns:
            Trained ProbabilisticStrategy
        """
        logger.info(f"Fitting probabilistic strategy from {train_csv_path}")
        training_data = parse_training_data(train_csv_path)
        logger.debug(f"Parsed {len(training_data)} training rows")

        # Extract correct answers first
        correct_answers = extract_correct_answers(training_data)
        logger.debug(f"Found correct answers for {len(correct_answers)} questions")

        # Create ResponseContexts for all training data
        contexts_with_labels = create_response_contexts(training_data, correct_answers)
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

    def predict(self, evaluation_row: EvaluationRow) -> SubmissionRow:
        """Make probabilistic predictions for a single evaluation row.

        Computes P(Category, Misconception | Context)
        and returns top 3 most likely predictions.

        Args:
            evaluation_row: Single evaluation row to predict on

        Returns:
            Submission row with up to 3 predictions, ordered by probability
        """
        logger.debug(f"Making probabilistic prediction for row {evaluation_row.row_id}")

        # Create context (we need to find correct answer)
        context = self._create_test_context(evaluation_row)

        # Get all possible category-misconception combinations with probabilities
        prediction_probs = self._compute_prediction_probabilities(context)

        # Sort by probability and take top 3
        sorted_predictions = sorted(
            prediction_probs.items(), key=lambda x: x[1], reverse=True
        )
        top_predictions = [pred for pred, _ in sorted_predictions[:3]]

        return SubmissionRow(
            row_id=evaluation_row.row_id, predicted_categories=top_predictions
        )

    def save(self, filepath: Path) -> None:
        """Save model as JSON file."""
        logger.info(f"Saving probabilistic model to {filepath}")

        # Convert ResponseContext objects and Category enums to strings for JSON serialization
        serializable_data = {
            "category_distributions": {
                f"{ctx.question_id}|{ctx.selected_answer}|{ctx.correct_answer}": {
                    cat.value: prob for cat, prob in dist.items()
                }
                for ctx, dist in self.category_distributions.items()
            },
            "misconception_distributions": {
                f"{ctx.question_id}|{ctx.selected_answer}|{ctx.correct_answer}|{cat.value}": dist
                for (ctx, cat), dist in self.misconception_distributions.items()
            },
            "global_category_prior": {
                cat.value: prob for cat, prob in self.global_category_prior.items()
            },
            "global_misconception_prior": {
                cat.value: dist for cat, dist in self.global_misconception_prior.items()
            },
            "question_category_priors": {
                str(qid): {cat.value: prob for cat, prob in dist.items()}
                for qid, dist in self.question_category_priors.items()
            },
        }

        with filepath.open("w") as f:
            json.dump(serializable_data, f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "ProbabilisticStrategy":
        """Load model from JSON file."""
        logger.info(f"Loading probabilistic model from {filepath}")
        assert filepath.exists(), f"Model file not found: {filepath}"

        with filepath.open("r") as f:
            data = json.load(f)

        # Reconstruct ResponseContext objects from string keys
        category_distributions = {}
        for key, dist in data["category_distributions"].items():
            parts = key.split("|")
            ctx = ResponseContext(int(parts[0]), parts[1], parts[2])
            category_distributions[ctx] = {
                Category(cat): prob for cat, prob in dist.items()
            }

        misconception_distributions = {}
        for key, dist in data["misconception_distributions"].items():
            parts = key.split("|")
            ctx = ResponseContext(int(parts[0]), parts[1], parts[2])
            cat = Category(parts[3])
            misconception_distributions[(ctx, cat)] = dist

        global_category_prior = {
            Category(cat): prob for cat, prob in data["global_category_prior"].items()
        }

        global_misconception_prior = {
            Category(cat): dist
            for cat, dist in data["global_misconception_prior"].items()
        }

        question_category_priors = {
            int(qid): {Category(cat): prob for cat, prob in dist.items()}
            for qid, dist in data["question_category_priors"].items()
        }

        return cls(
            category_distributions=category_distributions,
            misconception_distributions=misconception_distributions,
            global_category_prior=global_category_prior,
            global_misconception_prior=global_misconception_prior,
            question_category_priors=question_category_priors,
        )

    # Implementation methods

    def _create_test_context(self, test_row: EvaluationRow) -> ResponseContext:
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

    # Static methods for learning from data

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
