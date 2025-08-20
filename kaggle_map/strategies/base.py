"""Abstract base class for all prediction strategies."""

from abc import ABC, abstractmethod
from pathlib import Path

from kaggle_map.models import EvaluationRow, SubmissionRow


class Strategy(ABC):
    """Abstract base class for all prediction strategies.

    All strategies must implement the same interface for fitting, prediction,
    and persistence, allowing easy comparison and experimentation.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name for this strategy (e.g., 'baseline', 'probabilistic')."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this strategy does."""

    @classmethod
    @abstractmethod
    def fit(cls, train_csv_path: Path = Path("dataset/train.csv")) -> "Strategy":
        """Fit the strategy on training data.

        Args:
            train_csv_path: Path to training CSV file

        Returns:
            Fitted strategy instance
        """

    @abstractmethod
    def predict(self, test_data: list[EvaluationRow]) -> list[SubmissionRow]:
        """Make predictions on test data.

        Args:
            test_data: List of test rows to predict on

        Returns:
            List of submission rows with predictions
        """

    @abstractmethod
    def save(self, filepath: Path) -> None:
        """Save fitted model to disk.

        Args:
            filepath: Where to save the model
        """

    @classmethod
    @abstractmethod
    def load(cls, filepath: Path) -> "Strategy":
        """Load fitted model from disk.

        Args:
            filepath: Path to saved model file

        Returns:
            Loaded strategy instance
        """
