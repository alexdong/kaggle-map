"""Compatibility layer for old MLPStrategy interface."""

from pathlib import Path
from typing import Any

from optuna import Trial

from kaggle_map.core.models import EvaluationRow, SubmissionRow
from kaggle_map.mlp import Predictor
from kaggle_map.mlp.trainer import TrainingConfig


class MLPStrategy:
    """Compatibility wrapper for Predictor to maintain old interface."""

    def __init__(self, predictor: Predictor) -> None:
        self._predictor = predictor
        self.model = predictor.model
        self.correct_answers = predictor.correct_answers
        self.device = predictor.device
        self.parameters = None  # For compatibility

    @property
    def name(self) -> str:
        return self._predictor.name

    @property
    def description(self) -> str:
        return self._predictor.description

    @classmethod
    def get_hyperparameter_search_space(cls, trial: Trial) -> dict[str, Any]:
        """Get hyperparameter search space for compatibility."""
        return {
            "learning_rate": trial.suggest_float("learning_rate", 8e-5, 3e-4, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [224, 256, 288, 320, 384, 448, 512]),
            "dropout": trial.suggest_float("dropout", 0.10, 0.42),
            "architecture_size": trial.suggest_categorical(
                "architecture_size", ["xlarge"] * 17 + ["large"] * 2 + ["medium"]
            ),
            "optimizer": trial.suggest_categorical("optimizer", ["adamw", "adam"]),
            "weight_decay": trial.suggest_float("weight_decay", 3e-3, 1.5e-2, log=True),
            "activation": trial.suggest_categorical("activation", ["gelu", "silu", "relu", "leaky_relu"]),
            "scheduler": trial.suggest_categorical("scheduler", ["cosine", "cosine", "onecycle", "none"]),
            "early_stopping_patience": trial.suggest_int("patience", 10, 22),
            "epochs": trial.suggest_int("epochs", 28, 180),
            "embedding_strategy": trial.suggest_categorical(
                "embedding_strategy", ["double_blind", "semantic"]
            ),
        }

    @classmethod
    def fit(cls, **kwargs: Any) -> "MLPStrategy":
        """Fit using new Predictor interface."""
        # Create config from kwargs
        config = TrainingConfig(
            epochs=kwargs.get("epochs", 50),
            batch_size=kwargs.get("batch_size", 256),
            learning_rate=kwargs.get("learning_rate", 1e-4),
            weight_decay=kwargs.get("weight_decay", 0.01),
            optimizer=kwargs.get("optimizer", "adamw"),
            scheduler=kwargs.get("scheduler", "cosine"),
            early_stopping_patience=kwargs.get("early_stopping_patience", 15),
            train_split=kwargs.get("train_split", 0.7),
            random_seed=kwargs.get("random_seed", 42),
            architecture_size=kwargs.get("architecture_size", "xlarge"),
            dropout=kwargs.get("dropout", 0.3),
            activation=kwargs.get("activation", "gelu"),
        )

        if "train_csv_path" in kwargs:
            config.train_csv_path = Path(kwargs["train_csv_path"])

        embedding_strategy = kwargs.get("embedding_strategy", "double_blind")
        predictor = Predictor.fit(config, embedding_strategy=embedding_strategy)
        return cls(predictor)

    def predict(self, evaluation_row: EvaluationRow) -> SubmissionRow:
        """Delegate to Predictor."""
        return self._predictor.predict(evaluation_row)

    @classmethod
    def evaluate(
        cls,
        model: "MLPStrategy | None" = None,
        *,
        train_split: float = 0.7,
        random_seed: int = 42,
        train_csv_path: Path = Path("datasets/train.csv"),
        checkpoint_path: Path | None = None,
    ) -> dict[str, float]:
        """Evaluate model with old interface."""
        if model is None:
            # Load from checkpoint
            if checkpoint_path is None:
                checkpoints = list(Path("checkpoints").glob("mlp_best_*.pt"))
                if not checkpoints:
                    msg = "No model provided and no checkpoints found!"
                    raise ValueError(msg)
                checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
            # For now, just train a new model
            config = TrainingConfig(train_split=train_split, random_seed=random_seed)
            predictor = Predictor.fit(config)
            model = cls(predictor)

        return model._predictor.evaluate(train_csv_path=train_csv_path)

    def save(self, filepath: Path) -> None:
        """Save model."""
        self._predictor.save(filepath)

    @classmethod
    def load(cls, filepath: Path) -> "MLPStrategy":
        """Load model."""
        predictor = Predictor.load(filepath)
        return cls(predictor)
