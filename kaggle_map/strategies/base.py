from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kaggle_map.core.models import EvaluationRow, SubmissionRow

if TYPE_CHECKING:
    from optuna.trial import Trial


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
    def fit(
        cls,
        *,
        train_split: float = 0.7,
        random_seed: int = 42,
        train_csv_path: Path = Path("dataset/train.csv"),
    ) -> "Strategy":
        """Fit the strategy on training data.

        Args:
            train_split: Fraction of data for training
            random_seed: Random seed for reproducible results
            train_csv_path: Path to training CSV file

        Returns:
            Fitted strategy instance
        """

    @abstractmethod
    def predict(self, evaluation_row: EvaluationRow) -> SubmissionRow:
        """Make predictions on a single evaluation row.

        Args:
            evaluation_row: Single evaluation row to predict on

        Returns:
            Submission row with prediction
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

    @classmethod
    @abstractmethod
    def evaluate_on_split(
        cls,
        model: "Strategy",
        *,
        train_split: float = 0.7,
        random_seed: int = 42,
    ) -> dict[str, float]:
        """Evaluate model on validation split.

        Args:
            model: Fitted strategy instance to evaluate
            train_split: Fraction of data used for training (rest for validation)
            random_seed: Random seed for reproducible split

        Returns:
            Dictionary with evaluation metrics
        """

    @classmethod
    def get_hyperparameter_search_space(cls, trial: "Trial") -> dict[str, Any]:  # noqa: ARG003
        """Define hyperparameter search space for Optuna.

        Override this method to define strategy-specific hyperparameters.

        Args:
            trial: Optuna trial object for suggesting parameters

        Returns:
            Dictionary of hyperparameter names to values

        Note:
            Default implementation returns empty dict.
            Override in subclasses to enable hyperparameter search.
        """
        return {}

    @classmethod
    def create_config_from_hyperparams(
        cls,
        hyperparams: dict[str, Any],
        **base_params: Any,
    ) -> dict[str, Any]:
        """Create configuration from hyperparameters.

        Override this method to convert hyperparameters to strategy-specific config.

        Args:
            hyperparams: Dictionary of hyperparameters from search
            **base_params: Additional base parameters

        Returns:
            Configuration dictionary or object for training

        Note:
            Default implementation merges hyperparams with base_params.
            Override for custom configuration handling.
        """
        return {**base_params, **hyperparams}
